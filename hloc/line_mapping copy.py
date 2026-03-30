
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union

import h5py
import numpy as np
import pycolmap

from . import logger
from .matchers.line_nearest_neighbor import (
    center_alignment_error,
    endpoint_alignment_error,
    line_angle_diff_deg,
)
from .utils.io import get_line_matches, get_lines
from .utils.parsers import parse_retrieval


EPS = 1e-8


class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x):
        self.add(x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < EPS:
        return vec.astype(np.float64)
    return (vec / norm).astype(np.float64)


def _camera_center_world(image: pycolmap.Image) -> np.ndarray:
    world_from_cam = image.cam_from_world.inverse()
    return np.asarray(world_from_cam.translation, dtype=np.float64)


def _rotation_cam_to_world(image: pycolmap.Image) -> np.ndarray:
    world_from_cam = image.cam_from_world.inverse()
    return np.asarray(world_from_cam.rotation.matrix(), dtype=np.float64)


def _project_world_points(query_camera: pycolmap.Camera, cam_from_world: pycolmap.Rigid3d, points3d: np.ndarray):
    points3d = np.asarray(points3d, dtype=np.float64)
    if points3d.ndim == 1:
        points3d = points3d[None]
    rot = np.asarray(cam_from_world.rotation.matrix(), dtype=np.float64)
    trans = np.asarray(cam_from_world.translation, dtype=np.float64)
    points_cam = (rot @ points3d.T).T + trans[None]
    valid = points_cam[:, 2] > EPS
    if not np.any(valid):
        return np.zeros((points3d.shape[0], 2), dtype=np.float64), valid
    proj = np.zeros((points3d.shape[0], 2), dtype=np.float64)
    proj_valid = query_camera.img_from_cam(points_cam[valid])
    proj[valid] = np.asarray(proj_valid, dtype=np.float64)
    return proj, valid


def _camera_rays_from_line(camera: pycolmap.Camera, line_xyxy: np.ndarray) -> np.ndarray:
    pts = np.asarray(line_xyxy, dtype=np.float64).reshape(2, 2)
    cam_xy = np.asarray(camera.cam_from_img(pts), dtype=np.float64)
    rays = np.concatenate([cam_xy, np.ones((2, 1), dtype=np.float64)], axis=1)
    rays /= np.clip(np.linalg.norm(rays, axis=1, keepdims=True), EPS, None)
    return rays


def line_observation_to_plane(image: pycolmap.Image, camera: pycolmap.Camera, line_xyxy: np.ndarray):
    rays_cam = _camera_rays_from_line(camera, line_xyxy)
    normal_cam = np.cross(rays_cam[0], rays_cam[1])
    norm = np.linalg.norm(normal_cam)
    if norm < EPS:
        return None
    normal_cam = normal_cam / norm
    rot_c2w = _rotation_cam_to_world(image)
    center_w = _camera_center_world(image)
    normal_w = _normalize(rot_c2w @ normal_cam)
    offset = -float(np.dot(normal_w, center_w))
    return {
        'normal_world': normal_w,
        'offset': offset,
        'camera_center_world': center_w,
        'rot_cam_to_world': rot_c2w,
    }


def estimate_infinite_line_from_planes(observations: List[Dict]):
    normals = np.stack([obs['normal_world'] for obs in observations], axis=0)
    _, _, vt = np.linalg.svd(normals, full_matrices=False)
    direction = _normalize(vt[-1])

    offsets = np.asarray([obs['offset'] for obs in observations], dtype=np.float64)
    A = np.concatenate([normals, direction[None]], axis=0)
    b = np.concatenate([-offsets, np.zeros((1,), dtype=np.float64)], axis=0)
    anchor, *_ = np.linalg.lstsq(A, b, rcond=None)
    return anchor.astype(np.float64), direction.astype(np.float64)


def closest_point_on_line_to_ray(anchor: np.ndarray, direction: np.ndarray, center: np.ndarray, ray_dir: np.ndarray) -> np.ndarray:
    direction = _normalize(direction)
    ray_dir = _normalize(ray_dir)
    w0 = anchor - center
    a = float(np.dot(direction, direction))
    b = float(np.dot(direction, ray_dir))
    c = float(np.dot(ray_dir, ray_dir))
    rhs = np.array([-np.dot(direction, w0), np.dot(ray_dir, w0)], dtype=np.float64)
    lhs = np.array([[a, -b], [-b, c]], dtype=np.float64)
    try:
        ts = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        ts, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
    t = float(ts[0])
    return anchor + t * direction


def estimate_finite_segment(anchor: np.ndarray, direction: np.ndarray, observations: List[Dict]):
    support_points = []
    scalars = []
    for obs in observations:
        image = obs['image']
        camera = obs['camera']
        center_w = obs['camera_center_world']
        rot_c2w = obs['rot_cam_to_world']
        pts = obs['line_xyxy'].reshape(2, 2)
        cam_xy = np.asarray(camera.cam_from_img(pts), dtype=np.float64)
        rays_cam = np.concatenate([cam_xy, np.ones((2, 1), dtype=np.float64)], axis=1)
        for ray_cam in rays_cam:
            ray_dir_world = _normalize(rot_c2w @ _normalize(ray_cam))
            point_on_line = closest_point_on_line_to_ray(anchor, direction, center_w, ray_dir_world)
            support_points.append(point_on_line)
            scalars.append(float(np.dot(point_on_line - anchor, direction)))

    if len(scalars) < 2:
        return None
    scalars = np.asarray(scalars, dtype=np.float64)
    if len(scalars) >= 6:
        lo = float(np.percentile(scalars, 10))
        hi = float(np.percentile(scalars, 90))
    else:
        lo = float(np.min(scalars))
        hi = float(np.max(scalars))
    if hi - lo < 1e-4:
        hi = lo + 1e-4
    p0 = anchor + lo * direction
    p1 = anchor + hi * direction
    return np.stack([p0, p1], axis=0).astype(np.float64)


def project_line_segment(query_camera: pycolmap.Camera, cam_from_world: pycolmap.Rigid3d, segment_world: np.ndarray):
    proj, valid = _project_world_points(query_camera, cam_from_world, segment_world)
    if not np.all(valid):
        return None
    return proj.reshape(-1).astype(np.float64)


def compute_line_reprojection_error(segment_world: np.ndarray, observations: List[Dict]) -> float:
    errors = []
    for obs in observations:
        projected = project_line_segment(obs['camera'], obs['image'].cam_from_world, segment_world)
        if projected is None:
            continue
        errors.append(endpoint_alignment_error(projected, obs['line_xyxy']))
    if not errors:
        return float('inf')
    return float(np.mean(errors))


def aggregate_descriptor(descriptors: List[np.ndarray]) -> np.ndarray:
    if not descriptors:
        return np.zeros((0,), dtype=np.float32)
    desc = np.stack([np.asarray(d, dtype=np.float64) for d in descriptors], axis=0)
    mean_desc = np.mean(desc, axis=0)
    norm = np.linalg.norm(mean_desc)
    if norm < EPS:
        return mean_desc.astype(np.float32)
    return (mean_desc / norm).astype(np.float32)


def _pair_iter_from_retrieval(pairs_path: Path):
    pairs_dict = parse_retrieval(pairs_path)
    for name0, refs in pairs_dict.items():
        for name1 in refs:
            yield name0, name1


def build_line_tracks(
    reference_sfm: pycolmap.Reconstruction,
    features_path: Path,
    line_matches_path: Path,
    pairs_path: Path,
    min_support_count: int = 3,
    min_unique_images: int = 2,
    min_camera_center_distance: float = 0.1,
):
    db_name_to_id = {img.name: i for i, img in reference_sfm.images.items()}
    uf = UnionFind()
    touched_nodes = set()

    for name0, name1 in _pair_iter_from_retrieval(pairs_path):
        if name0 not in db_name_to_id or name1 not in db_name_to_id:
            continue
        try:
            pairs, _scores, verified, _stats = get_line_matches(line_matches_path, name0, name1)
        except Exception:
            continue
        if len(pairs) == 0:
            continue
        for pair, is_verified in zip(pairs.tolist(), verified.tolist()):
            if not is_verified:
                continue
            node0 = (name0, int(pair[0]))
            node1 = (name1, int(pair[1]))
            uf.union(node0, node1)
            touched_nodes.add(node0)
            touched_nodes.add(node1)

    components = defaultdict(list)
    for node in touched_nodes:
        components[uf.find(node)].append(node)

    line_cache = {}
    tracks = []
    for nodes in components.values():
        image_names = sorted({name for name, _ in nodes})
        if len(nodes) < min_support_count or len(image_names) < min_unique_images:
            continue

        observations = []
        descriptors = []
        camera_centers = []
        valid_track = True
        for image_name, line_idx in nodes:
            if image_name not in line_cache:
                line_cache[image_name] = get_lines(features_path, image_name)
            line_data = line_cache[image_name]
            if line_idx >= len(line_data['line_segments']):
                valid_track = False
                break
            image_id = db_name_to_id[image_name]
            image = reference_sfm.images[image_id]
            camera = reference_sfm.cameras[image.camera_id]
            line_xyxy = np.asarray(line_data['line_segments'][line_idx], dtype=np.float64)
            plane = line_observation_to_plane(image, camera, line_xyxy)
            if plane is None:
                valid_track = False
                break
            observation = {
                'image_name': image_name,
                'image_id': int(image_id),
                'line_idx': int(line_idx),
                'image': image,
                'camera': camera,
                'line_xyxy': line_xyxy,
                **plane,
            }
            observations.append(observation)
            descriptors.append(np.asarray(line_data['line_descriptors'][line_idx], dtype=np.float32))
            camera_centers.append(plane['camera_center_world'])

        if not valid_track or len(observations) < min_support_count:
            continue

        centers = np.stack(camera_centers, axis=0)
        baseline = float(np.max(np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1))) if len(centers) > 1 else 0.0
        if baseline < min_camera_center_distance:
            continue

        try:
            anchor, direction = estimate_infinite_line_from_planes(observations)
            segment = estimate_finite_segment(anchor, direction, observations)
        except np.linalg.LinAlgError:
            continue
        if segment is None:
            continue

        reproj_error = compute_line_reprojection_error(segment, observations)
        if not np.isfinite(reproj_error):
            continue

        tracks.append({
            'support_count': int(len(observations)),
            'support_image_names': [obs['image_name'] for obs in observations],
            'support_items': observations,
            'anchor_point_xyz': anchor.astype(np.float64),
            'direction_xyz': direction.astype(np.float64),
            'segment_endpoints_xyz': segment.astype(np.float64),
            'mean_descriptor': aggregate_descriptor(descriptors),
            'mean_reprojection_error': float(reproj_error),
        })

    return tracks


def write_line_map(line_map_path: Path, line_assignment_path: Path, tracks: List[Dict], features_path: Path):
    line_map_path.parent.mkdir(parents=True, exist_ok=True)
    line_assignment_path.parent.mkdir(parents=True, exist_ok=True)

    assignment_arrays = {}
    with h5py.File(str(features_path), 'r', libver='latest') as feat_h5:
        for image_name in feat_h5.keys():
            num_lines = feat_h5[image_name]['line_segments'].shape[0] if 'line_segments' in feat_h5[image_name] else 0
            assignment_arrays[image_name] = np.full((num_lines,), -1, dtype=np.int32)

    string_dtype = h5py.string_dtype(encoding='utf-8')
    with h5py.File(str(line_map_path), 'w', libver='latest') as hfile:
        lines_grp = hfile.create_group('lines')
        for line3d_id, track in enumerate(tracks):
            grp = lines_grp.create_group(str(line3d_id))
            grp.create_dataset('line3d_id', data=np.asarray([line3d_id], dtype=np.int32))
            grp.create_dataset('anchor_point_xyz', data=track['anchor_point_xyz'].astype(np.float32))
            grp.create_dataset('direction_xyz', data=track['direction_xyz'].astype(np.float32))
            grp.create_dataset('segment_endpoints_xyz', data=track['segment_endpoints_xyz'].astype(np.float32))
            grp.create_dataset('support_count', data=np.asarray([track['support_count']], dtype=np.int32))
            grp.create_dataset('mean_descriptor', data=track['mean_descriptor'].astype(np.float32))
            grp.create_dataset('mean_reprojection_error', data=np.asarray([track['mean_reprojection_error']], dtype=np.float32))
            grp.create_dataset('support_image_ids', data=np.asarray([obs['image_id'] for obs in track['support_items']], dtype=np.int32))
            grp.create_dataset('support_image_names', data=np.asarray(track['support_image_names'], dtype=object), dtype=string_dtype)
            grp.create_dataset('support_line_indices', data=np.asarray([obs['line_idx'] for obs in track['support_items']], dtype=np.int32))
            for obs in track['support_items']:
                image_name = obs['image_name']
                line_idx = obs['line_idx']
                if image_name in assignment_arrays and 0 <= line_idx < len(assignment_arrays[image_name]):
                    assignment_arrays[image_name][line_idx] = line3d_id
        hfile.attrs['num_lines'] = int(len(tracks))

    with h5py.File(str(line_assignment_path), 'w', libver='latest') as hfile:
        for image_name, assignments in assignment_arrays.items():
            grp = hfile.require_group(image_name)
            grp.create_dataset('line3d_ids', data=assignments.astype(np.int32))


def load_line_map(line_map_path: Path) -> Dict[int, Dict]:
    line_map = {}
    with h5py.File(str(line_map_path), 'r', libver='latest') as hfile:
        lines_grp = hfile['lines']
        for line3d_id_str, grp in lines_grp.items():
            line3d_id = int(line3d_id_str)
            line_map[line3d_id] = {
                'line3d_id': line3d_id,
                'anchor_point_xyz': grp['anchor_point_xyz'].__array__().astype(np.float64),
                'direction_xyz': grp['direction_xyz'].__array__().astype(np.float64),
                'segment_endpoints_xyz': grp['segment_endpoints_xyz'].__array__().astype(np.float64),
                'support_count': int(grp['support_count'].__array__()[0]),
                'mean_descriptor': grp['mean_descriptor'].__array__().astype(np.float32),
                'mean_reprojection_error': float(grp['mean_reprojection_error'].__array__()[0]),
                'support_image_ids': grp['support_image_ids'].__array__().astype(np.int32),
                'support_image_names': [name.decode('utf-8') if isinstance(name, bytes) else str(name) for name in grp['support_image_names'].__array__()],
                'support_line_indices': grp['support_line_indices'].__array__().astype(np.int32),
            }
    return line_map


def load_line_assignments(line_assignment_path: Path, image_name: str) -> np.ndarray:
    with h5py.File(str(line_assignment_path), 'r', libver='latest') as hfile:
        if image_name not in hfile:
            return np.zeros((0,), dtype=np.int32)
        return hfile[image_name]['line3d_ids'].__array__().astype(np.int32)


def summarize_line_map(line_map_path: Path) -> Dict[str, float]:
    line_map = load_line_map(line_map_path)
    if not line_map:
        return {
            'num_lines': 0,
            'mean_support_count': 0.0,
            'median_support_count': 0.0,
            'mean_reprojection_error': 0.0,
        }
    support_counts = np.asarray([item['support_count'] for item in line_map.values()], dtype=np.float32)
    reproj_errors = np.asarray([item['mean_reprojection_error'] for item in line_map.values()], dtype=np.float32)
    return {
        'num_lines': int(len(line_map)),
        'mean_support_count': float(np.mean(support_counts)),
        'median_support_count': float(np.median(support_counts)),
        'mean_reprojection_error': float(np.mean(reproj_errors)),
    }


def main(
    reference_sfm: Union[Path, pycolmap.Reconstruction],
    features: Path,
    line_matches: Path,
    pairs: Path,
    line_map_path: Path,
    line_assignment_path: Path,
    min_support_count: int = 3,
    min_unique_images: int = 2,
    min_camera_center_distance: float = 0.1,
):
    if not isinstance(reference_sfm, pycolmap.Reconstruction):
        reference_sfm = pycolmap.Reconstruction(reference_sfm)

    tracks = build_line_tracks(
        reference_sfm=reference_sfm,
        features_path=features,
        line_matches_path=line_matches,
        pairs_path=pairs,
        min_support_count=min_support_count,
        min_unique_images=min_unique_images,
        min_camera_center_distance=min_camera_center_distance,
    )
    write_line_map(line_map_path, line_assignment_path, tracks, features)
    summary = summarize_line_map(line_map_path)
    logger.info('Built 3D line map with %d tracks.', summary['num_lines'])
    logger.info('Line map summary: %s', summary)
    return line_map_path, line_assignment_path, summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_sfm', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--line_matches', type=Path, required=True)
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--line_map_path', type=Path, required=True)
    parser.add_argument('--line_assignment_path', type=Path, required=True)
    parser.add_argument('--min_support_count', type=int, default=3)
    parser.add_argument('--min_unique_images', type=int, default=2)
    parser.add_argument('--min_camera_center_distance', type=float, default=0.1)
    args = parser.parse_args()
    main(**args.__dict__)
