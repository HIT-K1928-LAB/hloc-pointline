import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
import pycolmap

from .line_mapping import load_line_map
from .utils.viz_3d import init_figure


def _to_rgb_strings(colors: np.ndarray) -> List[str]:
    colors = np.asarray(colors, dtype=np.uint8)
    return [f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in colors]


def _sample_indices(num_items: int, max_items: int, seed: int) -> np.ndarray:
    if max_items <= 0 or num_items <= max_items:
        return np.arange(num_items, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(num_items, size=max_items, replace=False))


def _to_homogeneous(points: np.ndarray) -> np.ndarray:
    pad = np.ones((points.shape[:-1] + (1,)), dtype=points.dtype)
    return np.concatenate([points, pad], axis=-1)


def _camera_center_and_corners(
    image: pycolmap.Image,
    camera: pycolmap.Camera,
    size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    world_t_camera = image.cam_from_world.inverse()
    rotation = np.asarray(world_t_camera.rotation.matrix(), dtype=np.float64)
    translation = np.asarray(world_t_camera.translation, dtype=np.float64)
    calibration = np.asarray(camera.calibration_matrix(), dtype=np.float64)

    width = calibration[0, 2] * 2.0
    height = calibration[1, 2] * 2.0
    corners = np.array(
        [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height], [0.0, 0.0]],
        dtype=np.float64,
    )
    image_extent = max(size * width / 1024.0, size * height / 1024.0)
    world_extent = max(width, height) / (calibration[0, 0] + calibration[1, 1]) / 0.5
    scale = 0.5 * image_extent / max(world_extent, 1e-8)
    corners = _to_homogeneous(corners) @ np.linalg.inv(calibration).T
    corners = (corners / 2.0 * scale) @ rotation.T + translation
    return translation, corners


def _camera_frustum_segments(
    image: pycolmap.Image,
    camera: pycolmap.Camera,
    size: float,
) -> np.ndarray:
    center, corners = _camera_center_and_corners(image, camera, size)
    segments = []
    for i in range(4):
        segments.append(np.stack([center, corners[i]], axis=0))
    for i in range(4):
        segments.append(np.stack([corners[i], corners[i + 1]], axis=0))
    return np.stack(segments, axis=0)


def load_filtered_points(
    reconstruction: pycolmap.Reconstruction,
    max_reproj_error: float,
    min_track_length: int,
    max_points: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    bbs = reconstruction.compute_bounding_box(0.001, 0.999)
    xyzs = []
    colors = []

    for _, point3d in reconstruction.points3D.items():
        xyz = np.asarray(point3d.xyz, dtype=np.float64)
        if not ((xyz >= bbs[0]).all() and (xyz <= bbs[1]).all()):
            continue
        if point3d.error > max_reproj_error:
            continue
        if point3d.track.length() < min_track_length:
            continue
        xyzs.append(xyz)
        colors.append(np.asarray(point3d.color, dtype=np.uint8))

    if not xyzs:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)

    xyzs = np.stack(xyzs, axis=0)
    colors = np.stack(colors, axis=0)
    indices = _sample_indices(len(xyzs), max_points, seed)
    return xyzs[indices], colors[indices]


def load_filtered_lines(
    line_map_path: Path,
    min_support_count: int,
    max_lines: int,
    max_line_reproj_error: float,
    seed: int,
) -> List[dict]:
    line_map = load_line_map(line_map_path)
    items = []
    for item in line_map.values():
        if int(item["support_count"]) < min_support_count:
            continue
        if float(item["mean_reprojection_error"]) > max_line_reproj_error:
            continue
        segment = np.asarray(item["segment_endpoints_xyz"], dtype=np.float64)
        if segment.shape != (2, 3):
            continue
        items.append(item)

    if not items:
        return []

    indices = _sample_indices(len(items), max_lines, seed + 1)
    return [items[i] for i in indices.tolist()]


def load_filtered_cameras(
    reconstruction: pycolmap.Reconstruction,
    max_cameras: int,
    seed: int,
) -> List[Tuple[pycolmap.Image, pycolmap.Camera]]:
    image_ids = sorted(reconstruction.images.keys())
    if not image_ids:
        return []
    indices = _sample_indices(len(image_ids), max_cameras, seed + 2)
    items = []
    for idx in indices.tolist():
        image = reconstruction.images[image_ids[idx]]
        camera = reconstruction.cameras[image.camera_id]
        items.append((image, camera))
    return items


def add_points_trace(
    fig: go.Figure,
    xyzs: np.ndarray,
    colors: np.ndarray,
    point_size: float,
) -> None:
    if len(xyzs) == 0:
        return
    fig.add_trace(
        go.Scatter3d(
            x=xyzs[:, 0],
            y=xyzs[:, 1],
            z=xyzs[:, 2],
            mode="markers",
            name="3D points",
            legendgroup="3D points",
            hoverinfo="skip",
            marker=dict(
                size=point_size,
                color=_to_rgb_strings(colors),
                opacity=0.85,
                line_width=0.0,
            ),
        )
    )


def add_lines_trace(
    fig: go.Figure,
    line_items: Sequence[dict],
    line_width: float,
) -> None:
    if not line_items:
        return

    xs, ys, zs, texts = [], [], [], []
    for item in line_items:
        p0, p1 = np.asarray(item["segment_endpoints_xyz"], dtype=np.float64)
        hover = (
            f"line3d_id: {int(item['line3d_id'])}<br>"
            f"support_count: {int(item['support_count'])}<br>"
            f"mean_reprojection_error: {float(item['mean_reprojection_error']):.3f}"
        )
        xs.extend([p0[0], p1[0], None])
        ys.extend([p0[1], p1[1], None])
        zs.extend([p0[2], p1[2], None])
        texts.extend([hover, hover, None])

    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            name="3D lines",
            legendgroup="3D lines",
            line=dict(color="rgb(34, 197, 94)", width=line_width),
            text=texts,
            hovertemplate="%{text}<extra></extra>",
        )
    )


def add_cameras_trace(
    fig: go.Figure,
    camera_items: Sequence[Tuple[pycolmap.Image, pycolmap.Camera]],
    camera_size: float,
    camera_line_width: float,
    camera_center_size: float,
) -> None:
    if not camera_items:
        return

    xs, ys, zs = [], [], []
    center_x, center_y, center_z, center_text = [], [], [], []
    for image, camera in camera_items:
        center, _ = _camera_center_and_corners(image, camera, camera_size)
        segments = _camera_frustum_segments(image, camera, camera_size)
        for segment in segments:
            xs.extend([segment[0, 0], segment[1, 0], None])
            ys.extend([segment[0, 1], segment[1, 1], None])
            zs.extend([segment[0, 2], segment[1, 2], None])
        center_x.append(center[0])
        center_y.append(center[1])
        center_z.append(center[2])
        center_text.append(image.name)

    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            name="Cameras",
            legendgroup="Cameras",
            hoverinfo="skip",
            line=dict(color="rgb(59, 130, 246)", width=camera_line_width),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=center_x,
            y=center_y,
            z=center_z,
            mode="markers",
            name="Camera centers",
            legendgroup="Cameras",
            text=center_text,
            hovertemplate="%{text}<extra></extra>",
            marker=dict(size=camera_center_size, color="rgb(37, 99, 235)", opacity=0.9),
        )
    )


def _sample_segment_points(segments: np.ndarray, num_samples: int) -> np.ndarray:
    if len(segments) == 0 or num_samples <= 1:
        return np.zeros((0, 3), dtype=np.float64)
    ts = np.linspace(0.0, 1.0, num_samples, dtype=np.float64)
    points = (
        segments[:, 0][:, None, :] * (1.0 - ts)[None, :, None]
        + segments[:, 1][:, None, :] * ts[None, :, None]
    )
    return points.reshape(-1, 3)


def _constant_colors(num_items: int, color: Tuple[int, int, int]) -> np.ndarray:
    if num_items <= 0:
        return np.zeros((0, 3), dtype=np.uint8)
    return np.tile(np.asarray(color, dtype=np.uint8)[None, :], (num_items, 1))


def write_binary_ply(path: Path, xyzs: np.ndarray, colors: np.ndarray) -> None:
    xyzs = np.asarray(xyzs, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    if len(xyzs) != len(colors):
        raise ValueError("xyzs and colors must have the same length.")

    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {len(xyzs)}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
            "",
        ]
    ).encode("ascii")
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    data = np.empty(len(xyzs), dtype=dtype)
    data["x"] = xyzs[:, 0]
    data["y"] = xyzs[:, 1]
    data["z"] = xyzs[:, 2]
    data["red"] = colors[:, 0]
    data["green"] = colors[:, 1]
    data["blue"] = colors[:, 2]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        data.tofile(f)


def export_ply_visualization(
    outputs: Path,
    point_xyzs: np.ndarray,
    point_colors: np.ndarray,
    line_items: Sequence[dict],
    camera_items: Sequence[Tuple[pycolmap.Image, pycolmap.Camera]],
    camera_size: float,
    line_samples_per_segment: int,
    camera_edge_samples: int,
) -> Tuple[Path, Path]:
    line_segments = []
    for item in line_items:
        line_segments.append(np.asarray(item["segment_endpoints_xyz"], dtype=np.float64))
    if line_segments:
        line_segments = np.stack(line_segments, axis=0)
        line_xyzs = _sample_segment_points(line_segments, line_samples_per_segment)
    else:
        line_xyzs = np.zeros((0, 3), dtype=np.float64)
    line_colors = _constant_colors(len(line_xyzs), (34, 197, 94))

    camera_segments = []
    camera_centers = []
    for image, camera in camera_items:
        center, _ = _camera_center_and_corners(image, camera, camera_size)
        camera_centers.append(center)
        camera_segments.append(_camera_frustum_segments(image, camera, camera_size))
    if camera_segments:
        camera_segments = np.concatenate(camera_segments, axis=0)
        camera_frustum_xyzs = _sample_segment_points(camera_segments, camera_edge_samples)
    else:
        camera_frustum_xyzs = np.zeros((0, 3), dtype=np.float64)
    if camera_centers:
        camera_centers = np.stack(camera_centers, axis=0)
    else:
        camera_centers = np.zeros((0, 3), dtype=np.float64)
    camera_frustum_colors = _constant_colors(len(camera_frustum_xyzs), (59, 130, 246))
    camera_center_colors = _constant_colors(len(camera_centers), (249, 115, 22))

    overlay_xyzs = np.concatenate(
        [point_xyzs, line_xyzs, camera_frustum_xyzs, camera_centers],
        axis=0,
    )
    overlay_colors = np.concatenate(
        [point_colors, line_colors, camera_frustum_colors, camera_center_colors],
        axis=0,
    )

    points_ply = outputs / "point_cloud_only.ply"
    overlay_ply = outputs / "point_line_camera_overlay.ply"
    write_binary_ply(points_ply, point_xyzs, point_colors)
    write_binary_ply(overlay_ply, overlay_xyzs, overlay_colors)
    return points_ply, overlay_ply


def build_figure(
    reconstruction: pycolmap.Reconstruction,
    line_map_path: Path,
    max_points: int,
    max_lines: int,
    max_cameras: int,
    max_reproj_error: float,
    min_track_length: int,
    min_line_support_count: int,
    max_line_reproj_error: float,
    point_size: float,
    line_width: float,
    camera_size: float,
    camera_line_width: float,
    camera_center_size: float,
    seed: int,
) -> Tuple[go.Figure, np.ndarray, np.ndarray, List[dict], List[Tuple[pycolmap.Image, pycolmap.Camera]]]:
    xyzs, colors = load_filtered_points(
        reconstruction=reconstruction,
        max_reproj_error=max_reproj_error,
        min_track_length=min_track_length,
        max_points=max_points,
        seed=seed,
    )
    line_items = load_filtered_lines(
        line_map_path=line_map_path,
        min_support_count=min_line_support_count,
        max_lines=max_lines,
        max_line_reproj_error=max_line_reproj_error,
        seed=seed,
    )
    camera_items = load_filtered_cameras(
        reconstruction=reconstruction,
        max_cameras=max_cameras,
        seed=seed,
    )

    fig = init_figure(height=900)
    add_points_trace(fig, xyzs, colors, point_size=point_size)
    add_lines_trace(fig, line_items, line_width=line_width)
    add_cameras_trace(
        fig,
        camera_items,
        camera_size=camera_size,
        camera_line_width=camera_line_width,
        camera_center_size=camera_center_size,
    )
    fig.update_layout(
        title=dict(
            text=(
                f"3D points ({len(xyzs)}) + 3D lines ({len(line_items)}) + cameras ({len(camera_items)})"
                "<br><sup>Interactive visualization exported from HLOC outputs</sup>"
            ),
            x=0.5,
        )
    )
    return fig, xyzs, colors, line_items, camera_items


def resolve_default_paths(
    outputs: Path,
    reference_sfm: Path,
    line_map: Path,
    output_html: Path,
):
    if reference_sfm is None:
        reference_sfm = outputs / "sfm_joint-xfeat-mlsd+NN"
    if line_map is None:
        line_map = outputs / "line_map.h5"
    if output_html is None:
        output_html = outputs / "point_line_map_visualization.html"
    return reference_sfm, line_map, output_html


def main(
    outputs: Path,
    reference_sfm: Path = None,
    line_map: Path = None,
    output_html: Path = None,
    max_points: int = 200000,
    max_lines: int = 20000,
    max_cameras: int = 0,
    max_reproj_error: float = 6.0,
    min_track_length: int = 2,
    min_line_support_count: int = 3,
    max_line_reproj_error: float = 20.0,
    point_size: float = 1.5,
    line_width: float = 3.0,
    camera_size: float = 1.0,
    camera_line_width: float = 2.0,
    camera_center_size: float = 2.5,
    export_ply: bool = False,
    line_samples_per_segment: int = 24,
    camera_edge_samples: int = 8,
    seed: int = 7,
):
    outputs = Path(outputs)
    reference_sfm, line_map, output_html = resolve_default_paths(
        outputs, reference_sfm, line_map, output_html
    )

    if not reference_sfm.exists():
        raise FileNotFoundError(f"Reference SfM not found: {reference_sfm}")
    if not line_map.exists():
        raise FileNotFoundError(f"Line map not found: {line_map}")

    reconstruction = pycolmap.Reconstruction(reference_sfm)
    fig, point_xyzs, point_colors, line_items, camera_items = build_figure(
        reconstruction=reconstruction,
        line_map_path=line_map,
        max_points=max_points,
        max_lines=max_lines,
        max_cameras=max_cameras,
        max_reproj_error=max_reproj_error,
        min_track_length=min_track_length,
        min_line_support_count=min_line_support_count,
        max_line_reproj_error=max_line_reproj_error,
        point_size=point_size,
        line_width=line_width,
        camera_size=camera_size,
        camera_line_width=camera_line_width,
        camera_center_size=camera_center_size,
        seed=seed,
    )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_html), include_plotlyjs="cdn")
    print(f"[OK] Wrote visualization to: {output_html}")
    print(
        f"[INFO] Plotted {len(point_xyzs)} 3D points, {len(line_items)} 3D lines, "
        f"and {len(camera_items)} cameras."
    )

    if export_ply:
        points_ply, overlay_ply = export_ply_visualization(
            outputs=outputs,
            point_xyzs=point_xyzs,
            point_colors=point_colors,
            line_items=line_items,
            camera_items=camera_items,
            camera_size=camera_size,
            line_samples_per_segment=line_samples_per_segment,
            camera_edge_samples=camera_edge_samples,
        )
        print(f"[OK] Wrote point cloud PLY to: {points_ply}")
        print(f"[OK] Wrote overlay PLY to: {overlay_ply}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Visualize a HLOC point map together with a 3D line map and camera poses "
            "as an interactive Plotly HTML, and optionally export PLY overlays."
        )
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        required=True,
        help="Path to the pipeline output directory, e.g. outputs/aachen_joint_point_line_v2_netvlad",
    )
    parser.add_argument(
        "--reference_sfm",
        type=Path,
        help="Optional path to the reference SfM directory. Defaults to <outputs>/sfm_joint-xfeat-mlsd+NN",
    )
    parser.add_argument(
        "--line_map",
        type=Path,
        help="Optional path to line_map.h5. Defaults to <outputs>/line_map.h5",
    )
    parser.add_argument(
        "--output_html",
        type=Path,
        help="Optional output HTML path. Defaults to <outputs>/point_line_map_visualization.html",
    )
    parser.add_argument("--max_points", type=int, default=200000)
    parser.add_argument("--max_lines", type=int, default=20000)
    parser.add_argument(
        "--max_cameras",
        type=int,
        default=0,
        help="Maximum number of cameras to visualize. Use 0 to keep all registered cameras.",
    )
    parser.add_argument("--max_reproj_error", type=float, default=6.0)
    parser.add_argument("--min_track_length", type=int, default=2)
    parser.add_argument("--min_line_support_count", type=int, default=3)
    parser.add_argument("--max_line_reproj_error", type=float, default=20.0)
    parser.add_argument("--point_size", type=float, default=1.5)
    parser.add_argument("--line_width", type=float, default=3.0)
    parser.add_argument("--camera_size", type=float, default=1.0)
    parser.add_argument("--camera_line_width", type=float, default=2.0)
    parser.add_argument("--camera_center_size", type=float, default=2.5)
    parser.add_argument(
        "--export_ply",
        action="store_true",
        help="Export point cloud and sampled point-line-camera overlays as PLY files.",
    )
    parser.add_argument("--line_samples_per_segment", type=int, default=24)
    parser.add_argument("--camera_edge_samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    main(**args.__dict__)
