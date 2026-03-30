
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pycolmap
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from . import logger
from .line_mapping import load_line_assignments, load_line_map, project_line_segment, _project_world_points
from .localize_sfm import QueryLocalizer, do_covisibility_clustering, pose_from_cluster
from .matchers.line_nearest_neighbor import center_alignment_error, endpoint_alignment_error, line_angle_diff_deg
from .utils.io import get_line_matches, get_lines, get_matches
from .utils.parsers import parse_image_lists, parse_retrieval


EPS = 1e-8


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float32)))


def rerank_db_names_v2(
    qname,
    db_names,
    point_matches_path,
    line_matches_path,
    min_point_matches=15,
    min_verified_lines=4,
    min_verified_ratio=0.20,
    min_keep=10,
    endpoint_thresh=18.0,
    center_thresh=14.0,
    angle_thresh=10.0,
):
    support = []
    for db_name in db_names:
        try:
            point_matches, _ = get_matches(point_matches_path, qname, db_name)
            num_point_matches = int(len(point_matches))
        except Exception:
            num_point_matches = 0
        try:
            _, _, _verified_mask, line_stats = get_line_matches(line_matches_path, qname, db_name)
            candidate_lines = int(line_stats.get('line_candidate_matches', line_stats.get('candidate_matches', 0)))
            verified_lines = int(line_stats.get('line_verified_matches', line_stats.get('verified_matches', 0)))
            verified_ratio = float(line_stats.get('line_verified_ratio', verified_lines / max(candidate_lines, 1)))
            mean_similarity = float(line_stats.get('line_mean_verified_similarity', 0.0))
            mean_endpoint_error = float(line_stats.get('line_mean_verified_endpoint_error', endpoint_thresh))
            mean_center_error = float(line_stats.get('line_mean_verified_center_error', center_thresh))
            mean_angle_error = float(line_stats.get('line_mean_verified_angle_error_deg', angle_thresh))
        except Exception:
            candidate_lines = 0
            verified_lines = 0
            verified_ratio = 0.0
            mean_similarity = 0.0
            mean_endpoint_error = endpoint_thresh
            mean_center_error = center_thresh
            mean_angle_error = angle_thresh

        if verified_lines > 0:
            sim_quality = _clip01((mean_similarity - 0.5) / 0.5)
            endpoint_quality = _clip01(1.0 - mean_endpoint_error / max(endpoint_thresh, EPS))
            center_quality = _clip01(1.0 - mean_center_error / max(center_thresh, EPS))
            angle_quality = _clip01(1.0 - mean_angle_error / max(angle_thresh, EPS))
            line_geometry_quality = (
                0.40 * sim_quality
                + 0.25 * endpoint_quality
                + 0.20 * center_quality
                + 0.15 * angle_quality
            )
        else:
            line_geometry_quality = 0.0

        support.append({
            'db_name': db_name,
            'num_point_matches': num_point_matches,
            'num_candidate_line_matches': candidate_lines,
            'num_verified_line_matches': verified_lines,
            'verified_ratio': verified_ratio,
            'line_geometry_quality': float(line_geometry_quality),
        })

    max_points = max((item['num_point_matches'] for item in support), default=0)
    max_lines = max((item['num_verified_line_matches'] for item in support), default=0)
    for item in support:
        norm_points = item['num_point_matches'] / max(max_points, 1)
        norm_lines = item['num_verified_line_matches'] / max(max_lines, 1)
        item['pair_score'] = (
            0.45 * norm_points
            + 0.25 * norm_lines
            + 0.15 * item['verified_ratio']
            + 0.15 * item['line_geometry_quality']
        )

    filtered = [
        item for item in support
        if item['num_point_matches'] >= min_point_matches
        or (
            item['num_verified_line_matches'] >= min_verified_lines
            and item['verified_ratio'] >= min_verified_ratio
        )
    ]
    if len(filtered) < min_keep:
        keep_names = {item['db_name'] for item in filtered}
        for item in support:
            if item['db_name'] in keep_names:
                continue
            filtered.append(item)
            keep_names.add(item['db_name'])
            if len(filtered) >= min_keep:
                break

    filtered.sort(
        key=lambda item: (
            item['pair_score'],
            item['num_point_matches'],
            item['num_verified_line_matches'],
        ),
        reverse=True,
    )
    return [item['db_name'] for item in filtered], support


def make_candidate_subsets(db_ids: List[int], max_db_images_for_pnp: int = 20):
    if not db_ids:
        return []
    n = len(db_ids)
    sizes = [min(10, n), min(20, n), min(max_db_images_for_pnp, n), n]
    subsets = []
    seen = set()
    for size in sizes:
        if size <= 0:
            continue
        subset = tuple(db_ids[:size])
        if subset not in seen:
            subsets.append(list(subset))
            seen.add(subset)
    return subsets


def build_query_line_correspondences(
    qname: str,
    db_names: List[str],
    features_path: Path,
    line_matches_path: Path,
    line_map: Dict[int, Dict],
    line_assignment_path: Path,
):
    query_line_data = get_lines(features_path, qname)
    query_lines = query_line_data['line_segments']
    assignment_cache = {}
    best_per_query = {}

    for db_name in db_names:
        if db_name not in assignment_cache:
            assignment_cache[db_name] = load_line_assignments(line_assignment_path, db_name)
        assignments = assignment_cache[db_name]
        try:
            pairs, scores, verified, _stats = get_line_matches(line_matches_path, qname, db_name)
        except Exception:
            continue
        if len(pairs) == 0:
            continue
        for (q_idx, db_idx), score, is_verified in zip(pairs.tolist(), scores.tolist(), verified.tolist()):
            if not is_verified:
                continue
            if db_idx < 0 or db_idx >= len(assignments):
                continue
            line3d_id = int(assignments[db_idx])
            if line3d_id < 0 or line3d_id not in line_map:
                continue
            if q_idx < 0 or q_idx >= len(query_lines):
                continue
            line3d = line_map[line3d_id]
            candidate_score = float(score) + 0.01 * float(line3d.get('support_count', 0))
            current = best_per_query.get(int(q_idx))
            if current is None or candidate_score > current['candidate_score']:
                best_per_query[int(q_idx)] = {
                    'query_line_idx': int(q_idx),
                    'query_line_xyxy': np.asarray(query_lines[q_idx], dtype=np.float64),
                    'line3d_id': int(line3d_id),
                    'segment_endpoints_xyz': np.asarray(line3d['segment_endpoints_xyz'], dtype=np.float64),
                    'support_db_images': list(line3d['support_image_names']),
                    'support_count': int(line3d['support_count']),
                    'similarity': float(score),
                    'verified_flag': True,
                    'geometry_score': float(score),
                    'candidate_score': float(candidate_score),
                }

    correspondences = [best_per_query[idx] for idx in sorted(best_per_query.keys())]
    return correspondences


def evaluate_pose_with_line_correspondences(
    cam_from_world: pycolmap.Rigid3d,
    query_camera: pycolmap.Camera,
    correspondences: List[Dict],
    endpoint_thresh: float = 18.0,
    center_thresh: float = 14.0,
    angle_thresh: float = 10.0,
):
    endpoint_errors = []
    center_errors = []
    angle_errors = []
    inlier_mask = []
    valid_count = 0

    for corr in correspondences:
        projected = project_line_segment(query_camera, cam_from_world, corr['segment_endpoints_xyz'])
        if projected is None:
            inlier_mask.append(False)
            continue
        valid_count += 1
        endpoint_error = endpoint_alignment_error(projected, corr['query_line_xyxy'])
        center_error = center_alignment_error(projected, corr['query_line_xyxy'])
        angle_error = line_angle_diff_deg(projected, corr['query_line_xyxy'])
        endpoint_errors.append(float(endpoint_error))
        center_errors.append(float(center_error))
        angle_errors.append(float(angle_error))
        inlier_mask.append(
            endpoint_error <= endpoint_thresh
            and center_error <= center_thresh
            and angle_error <= angle_thresh
        )

    line_inlier_count = int(sum(inlier_mask))
    line_reproj_quality = _safe_mean([_clip01(1.0 - err / max(endpoint_thresh, EPS)) for err in endpoint_errors])
    line_angle_quality = _safe_mean([_clip01(1.0 - err / max(angle_thresh, EPS)) for err in angle_errors])
    combined_error = _safe_mean(
        [
            endpoint_errors[i] / max(endpoint_thresh, EPS)
            + center_errors[i] / max(center_thresh, EPS)
            + angle_errors[i] / max(angle_thresh, EPS)
            for i in range(len(endpoint_errors))
        ]
    )
    return {
        'line_correspondence_count': int(len(correspondences)),
        'line_valid_count': int(valid_count),
        'line_inlier_count': line_inlier_count,
        'line_reproj_quality': float(line_reproj_quality),
        'line_angle_quality': float(line_angle_quality),
        'mean_endpoint_error': _safe_mean(endpoint_errors),
        'mean_center_error': _safe_mean(center_errors),
        'mean_angle_error': _safe_mean(angle_errors),
        'mean_line_combined_error': float(combined_error),
        'line_inlier_mask': inlier_mask,
    }


def rigid3d_to_params(cam_from_world: pycolmap.Rigid3d) -> np.ndarray:
    rot = Rotation.from_matrix(np.asarray(cam_from_world.rotation.matrix(), dtype=np.float64)).as_rotvec()
    trans = np.asarray(cam_from_world.translation, dtype=np.float64)
    return np.concatenate([rot, trans], axis=0)


def params_to_rigid3d(params: np.ndarray) -> pycolmap.Rigid3d:
    rot = Rotation.from_rotvec(np.asarray(params[:3], dtype=np.float64)).as_matrix()
    trans = np.asarray(params[3:], dtype=np.float64)
    return pycolmap.Rigid3d(pycolmap.Rotation3d(rot), trans)


def build_point_refinement_data(log: Dict, reconstruction: pycolmap.Reconstruction, max_refine_points: int = 1024):
    points2d = np.asarray(log.get('keypoints_query', np.zeros((0, 2))), dtype=np.float64)
    point_ids = np.asarray(log.get('points3D_ids', []), dtype=np.int64)
    if len(points2d) == 0 or len(point_ids) == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
    points3d = np.stack([reconstruction.points3D[int(pid)].xyz for pid in point_ids], axis=0).astype(np.float64)
    if len(points2d) > max_refine_points:
        idx = np.linspace(0, len(points2d) - 1, max_refine_points).astype(np.int64)
        points2d = points2d[idx]
        points3d = points3d[idx]
    return points2d, points3d


def point_line_residuals(
    params: np.ndarray,
    query_camera: pycolmap.Camera,
    points2d: np.ndarray,
    points3d: np.ndarray,
    line_correspondences: List[Dict],
    endpoint_thresh: float,
    center_thresh: float,
    angle_thresh: float,
    point_scale: float = 10.0,
):
    cam_from_world = params_to_rigid3d(params)
    residuals = []

    if len(points3d) > 0:
        proj, valid = _project_world_points(query_camera, cam_from_world, points3d)
        for pred, obs, is_valid in zip(proj, points2d, valid.tolist()):
            if not is_valid:
                residuals.extend([2.0, 2.0])
            else:
                residuals.extend(((pred - obs) / max(point_scale, EPS)).tolist())

    for corr in line_correspondences:
        projected = project_line_segment(query_camera, cam_from_world, corr['segment_endpoints_xyz'])
        if projected is None:
            residuals.extend([2.0, 2.0, 2.0])
            continue
        endpoint_error = endpoint_alignment_error(projected, corr['query_line_xyxy']) / max(endpoint_thresh, EPS)
        center_error = center_alignment_error(projected, corr['query_line_xyxy']) / max(center_thresh, EPS)
        angle_error = line_angle_diff_deg(projected, corr['query_line_xyxy']) / max(angle_thresh, EPS)
        residuals.extend([float(endpoint_error), float(center_error), float(angle_error)])

    if not residuals:
        return np.zeros((0,), dtype=np.float64)
    return np.asarray(residuals, dtype=np.float64)


def refine_pose_with_points_and_lines(
    initial_pose: pycolmap.Rigid3d,
    query_camera: pycolmap.Camera,
    point_log: Dict,
    reconstruction: pycolmap.Reconstruction,
    line_correspondences: List[Dict],
    endpoint_thresh: float = 18.0,
    center_thresh: float = 14.0,
    angle_thresh: float = 10.0,
    max_refine_points: int = 1024,
    max_nfev: int = 50,
):
    points2d, points3d = build_point_refinement_data(point_log, reconstruction, max_refine_points=max_refine_points)
    if len(points2d) == 0 and len(line_correspondences) == 0:
        return initial_pose, False

    x0 = rigid3d_to_params(initial_pose)
    result = least_squares(
        point_line_residuals,
        x0,
        method='trf',
        max_nfev=max_nfev,
        args=(
            query_camera,
            points2d,
            points3d,
            line_correspondences,
            endpoint_thresh,
            center_thresh,
            angle_thresh,
        ),
    )
    refined_pose = params_to_rigid3d(result.x)
    return refined_pose, bool(result.success)


def build_cluster_candidates(db_ids: List[int], support: List[Dict], reconstruction: pycolmap.Reconstruction, max_clusters: int = 3):
    score_by_name = {item['db_name']: float(item['pair_score']) for item in support}
    clusters = do_covisibility_clustering(db_ids, reconstruction)
    scored = []
    for cluster in clusters:
        names = [reconstruction.images[image_id].name for image_id in cluster]
        scores = sorted([score_by_name.get(name, 0.0) for name in names], reverse=True)
        cluster_score = float(np.mean(scores[:3])) if scores else 0.0
        scored.append((cluster_score, cluster))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [cluster for _, cluster in scored[:max_clusters]]


def _extract_point_inliers(candidate: Dict) -> int:
    ret = candidate.get('point_ret')
    if ret is None:
        return 0
    if isinstance(ret, dict) and 'num_inliers' in ret:
        return int(ret['num_inliers'])
    return 0


def score_pose_candidates(candidates: List[Dict]):
    if not candidates:
        return []
    max_point = max((_extract_point_inliers(c) for c in candidates), default=0)
    max_line = max((c['line_metrics']['line_inlier_count'] for c in candidates), default=0)
    for candidate in candidates:
        norm_point = _extract_point_inliers(candidate) / max(max_point, 1)
        norm_line = candidate['line_metrics']['line_inlier_count'] / max(max_line, 1)
        candidate['combined_score'] = (
            0.50 * norm_point
            + 0.20 * norm_line
            + 0.15 * candidate['line_metrics']['line_reproj_quality']
            + 0.15 * candidate['line_metrics']['line_angle_quality']
        )
    candidates.sort(key=lambda item: item['combined_score'], reverse=True)
    return candidates


def main(
    reference_sfm: Union[Path, pycolmap.Reconstruction],
    queries: Path,
    retrieval: Path,
    features: Path,
    matches: Path,
    line_matches: Path,
    line_map: Path,
    line_track_assignments: Path,
    results: Path,
    ransac_thresh: float = 12.0,
    covisibility_clustering: bool = False,
    prepend_camera_name: bool = False,
    config: Dict = None,
    min_point_matches: int = 15,
    min_verified_lines: int = 4,
    min_verified_ratio: float = 0.20,
    max_db_images_for_pnp: int = 20,
    endpoint_thresh: float = 18.0,
    center_thresh: float = 14.0,
    angle_thresh: float = 10.0,
    max_refine_points: int = 1024,
):
    assert retrieval.exists(), retrieval
    assert features.exists(), features
    assert matches.exists(), matches
    assert line_matches.exists(), line_matches
    assert line_map.exists(), line_map
    assert line_track_assignments.exists(), line_track_assignments

    queries = parse_image_lists(queries, with_intrinsics=True)
    retrieval_dict = parse_retrieval(retrieval)

    logger.info('Reading the 3D model...')
    if not isinstance(reference_sfm, pycolmap.Reconstruction):
        reference_sfm = pycolmap.Reconstruction(reference_sfm)
    db_name_to_id = {img.name: i for i, img in reference_sfm.images.items()}
    line_map_dict = load_line_map(line_map)

    config = {'estimation': {'ransac': {'max_error': ransac_thresh}}, **(config or {})}
    localizer = QueryLocalizer(reference_sfm, config)

    cam_from_world = {}
    logs = {
        'features': features,
        'matches': matches,
        'line_matches': line_matches,
        'line_map': line_map,
        'line_track_assignments': line_track_assignments,
        'retrieval': retrieval,
        'loc': {},
    }

    logger.info('Starting point-line V2 localization...')
    for qname, qcam in tqdm(queries):
        if qname not in retrieval_dict:
            logger.warning(f'No images retrieved for query image {qname}. Skipping...')
            continue

        reranked_names, support = rerank_db_names_v2(
            qname,
            retrieval_dict[qname],
            matches,
            line_matches,
            min_point_matches=min_point_matches,
            min_verified_lines=min_verified_lines,
            min_verified_ratio=min_verified_ratio,
            endpoint_thresh=endpoint_thresh,
            center_thresh=center_thresh,
            angle_thresh=angle_thresh,
        )
        db_ids = [db_name_to_id[n] for n in reranked_names if n in db_name_to_id]
        if not db_ids:
            logger.warning(f'No database images available after V2 reranking for {qname}.')
            continue

        if covisibility_clustering:
            candidate_subsets = build_cluster_candidates(db_ids, support, reference_sfm, max_clusters=3)
        else:
            candidate_subsets = make_candidate_subsets(db_ids, max_db_images_for_pnp=max_db_images_for_pnp)

        candidates = []
        for subset in candidate_subsets:
            ret, log = pose_from_cluster(localizer, qname, qcam, subset, features, matches)
            if ret is None:
                continue
            subset_names = [reference_sfm.images[image_id].name for image_id in subset]
            line_corrs = build_query_line_correspondences(
                qname,
                subset_names,
                features,
                line_matches,
                line_map_dict,
                line_track_assignments,
            )
            line_metrics = evaluate_pose_with_line_correspondences(
                ret['cam_from_world'],
                qcam,
                line_corrs,
                endpoint_thresh=endpoint_thresh,
                center_thresh=center_thresh,
                angle_thresh=angle_thresh,
            )
            candidates.append({
                'db_ids': subset,
                'db_names': subset_names,
                'point_ret': ret,
                'point_log': log,
                'line_correspondences': line_corrs,
                'line_metrics': line_metrics,
            })

        pose_selected_by = 'point_only_fallback'
        selected_candidate = None
        line_refinement_applied = False
        line_residual_before = 0.0
        line_residual_after = 0.0

        if candidates:
            candidates = score_pose_candidates(candidates)
            selected_candidate = candidates[0]
            final_pose = selected_candidate['point_ret']['cam_from_world']
            pose_selected_by = 'point_line_score'
            line_residual_before = float(selected_candidate['line_metrics']['mean_line_combined_error'])
            line_residual_after = line_residual_before
            if selected_candidate['line_metrics']['line_correspondence_count'] >= 4:
                refined_pose, refine_ok = refine_pose_with_points_and_lines(
                    initial_pose=final_pose,
                    query_camera=qcam,
                    point_log=selected_candidate['point_log'],
                    reconstruction=reference_sfm,
                    line_correspondences=selected_candidate['line_correspondences'],
                    endpoint_thresh=endpoint_thresh,
                    center_thresh=center_thresh,
                    angle_thresh=angle_thresh,
                    max_refine_points=max_refine_points,
                )
                if refine_ok:
                    refined_metrics = evaluate_pose_with_line_correspondences(
                        refined_pose,
                        qcam,
                        selected_candidate['line_correspondences'],
                        endpoint_thresh=endpoint_thresh,
                        center_thresh=center_thresh,
                        angle_thresh=angle_thresh,
                    )
                    final_pose = refined_pose
                    line_refinement_applied = True
                    line_residual_after = float(refined_metrics['mean_line_combined_error'])
                    selected_candidate['refined_line_metrics'] = refined_metrics
                    pose_selected_by = 'point_line_refined'
            cam_from_world[qname] = final_pose
        else:
            closest = reference_sfm.images[db_ids[0]]
            cam_from_world[qname] = closest.cam_from_world
            pose_selected_by = 'closest_db_fallback'

        logs['loc'][qname] = {
            'db': db_ids,
            'reranked_db_names': reranked_names,
            'support': support,
            'candidate_logs': [
                {
                    'db_names': cand['db_names'],
                    'point_num_inliers': _extract_point_inliers(cand),
                    'line_correspondence_count': cand['line_metrics']['line_correspondence_count'],
                    'line_inlier_count': cand['line_metrics']['line_inlier_count'],
                    'line_reproj_quality': cand['line_metrics']['line_reproj_quality'],
                    'line_angle_quality': cand['line_metrics']['line_angle_quality'],
                    'combined_score': float(cand.get('combined_score', 0.0)),
                }
                for cand in candidates
            ],
            'selected_candidate_index': 0 if selected_candidate is not None else None,
            'selected_candidate_score': float(selected_candidate.get('combined_score', 0.0)) if selected_candidate is not None else 0.0,
            'line_correspondence_count': int(selected_candidate['line_metrics']['line_correspondence_count']) if selected_candidate is not None else 0,
            'line_inlier_count': int(selected_candidate['line_metrics']['line_inlier_count']) if selected_candidate is not None else 0,
            'line_residual_before_refine': float(line_residual_before),
            'line_residual_after_refine': float(line_residual_after),
            'line_refinement_applied': bool(line_refinement_applied),
            'pose_selected_by': pose_selected_by,
            'PnP_ret': selected_candidate['point_ret'] if selected_candidate is not None else None,
            'num_matches': int(selected_candidate['point_log']['num_matches']) if selected_candidate is not None else 0,
        }

    logger.info(f'Localized {len(cam_from_world)} / {len(queries)} images.')
    logger.info(f'Writing poses to {results}...')
    with open(results, 'w') as f:
        for query, t in cam_from_world.items():
            qvec = ' '.join(map(str, t.rotation.quat[[3, 0, 1, 2]]))
            tvec = ' '.join(map(str, t.translation))
            name = query.split('/')[-1]
            if prepend_camera_name:
                name = query.split('/')[-2] + '/' + name
            f.write(f"{name} {qvec} {tvec}\n")

    logs_path = f'{results}_logs.pkl'
    logger.info(f'Writing logs to {logs_path}...')
    with open(logs_path, 'wb') as f:
        pickle.dump(logs, f)
    logger.info('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_sfm', type=Path, required=True)
    parser.add_argument('--queries', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--matches', type=Path, required=True)
    parser.add_argument('--line_matches', type=Path, required=True)
    parser.add_argument('--line_map', type=Path, required=True)
    parser.add_argument('--line_track_assignments', type=Path, required=True)
    parser.add_argument('--retrieval', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--ransac_thresh', type=float, default=12.0)
    parser.add_argument('--min_point_matches', type=int, default=15)
    parser.add_argument('--min_verified_lines', type=int, default=4)
    parser.add_argument('--min_verified_ratio', type=float, default=0.20)
    parser.add_argument('--max_db_images_for_pnp', type=int, default=20)
    parser.add_argument('--endpoint_thresh', type=float, default=18.0)
    parser.add_argument('--center_thresh', type=float, default=14.0)
    parser.add_argument('--angle_thresh', type=float, default=10.0)
    parser.add_argument('--max_refine_points', type=int, default=1024)
    parser.add_argument('--covisibility_clustering', action='store_true')
    parser.add_argument('--prepend_camera_name', action='store_true')
    args = parser.parse_args()
    main(**args.__dict__)
