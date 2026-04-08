import argparse
import importlib
import json
import pprint
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import h5py
import numpy as np
import torch
from tqdm import tqdm

from . import logger
from .joint_xfeat_mlsd.inference import load_cfg as load_joint_cfg
from .joint_xfeat_mlsd.inference import safe_torch_load
from .match_features import FeaturePairsDataset
from .matchers.line_nearest_neighbor import (
    estimate_homography_from_matches,
    evaluate_matches_against_homography,
    post_filter_line_result,
)
from .utils.parsers import (
    names_to_pair,
    names_to_pair_old,
    parse_retrieval,
)


confs = {
    "joint_wiregraph": {
        "point_output": "matches-joint-wiregraph",
        "line_output": "line-matches-joint-wiregraph",
        "model": {
            "config": "/home/hxy/doctor/feature dectect/linedectect/mlsd_pytorchv3/workdir/models/xfeat_mlsd_512_gt_plus_pred_plus_align/cfg.yaml",
            "variant": "structured_linegraph",
            "junction_feature_source": "line_descriptor_map",
            "use_points": True,
            "use_line_message_passing": True,
            "point_match_score_thresh": 0.20,
            "line_match_score_thresh": 0.20,
            "junction_match_score_thresh": 0.20,
            "max_keypoints": 1024,
            "max_lines": 256,
            "endpoint_merge_thresh_px": 4.0,
            "drop_keypoints_near_endpoints_px": 4.0,
            "ransac_reproj_thresh": 12.0,
            "point_correct_thresh_px": 6.0,
            "junction_correct_thresh_px": 6.0,
            "line_inlier_endpoint_thresh": 18.0,
            "line_inlier_center_thresh": 14.0,
            "line_inlier_angle_thresh_deg": 10.0,
            "line_inlier_length_ratio_min": 0.60,
        },
    },
    "joint_wiregraph_v2style_hybrid": {
        "point_output": "matches-joint-wiregraph-v2style-hybrid",
        "line_output": "line-matches-joint-wiregraph-v2style-hybrid",
        "model": {
            "config": "/home/hxy/doctor/feature dectect/linedectect/mlsd_pytorchv3/workdir/models/xfeat_mlsd_512_gt_plus_pred_plus_align/cfg.yaml",
            "variant": "structured_linegraph",
            "junction_feature_source": "line_descriptor_map",
            "use_points": True,
            "use_line_message_passing": True,
            "point_match_score_thresh": 0.001,
            "line_match_score_thresh": 0.001,
            "junction_match_score_thresh": 0.001,
            "max_keypoints": 4096,
            "max_lines": 500,
            "max_lines_per_image": 120,
            "endpoint_merge_thresh_px": 4.0,
            "drop_keypoints_near_endpoints_px": 0.0,
            "post_min_line_score": 0.10,
            "post_min_line_length_px": 30.0,
            "line_nms_center_thresh": 20.0,
            "line_nms_angle_thresh_deg": 8.0,
            "line_nms_endpoint_thresh": 24.0,
            "use_v2_style_line_prefilter": True,
            "filter_point_junction_matches_with_geometry": False,
            "allow_line_fallback_geometry": False,
            "ransac_reproj_thresh": 8.0,
            "point_correct_thresh_px": 6.0,
            "junction_correct_thresh_px": 6.0,
            "line_inlier_endpoint_thresh": 18.0,
            "line_inlier_center_thresh": 14.0,
            "line_inlier_angle_thresh_deg": 10.0,
            "line_inlier_length_ratio_min": 0.60,
        },
    },
}

JUNCTION_SIGNATURE_ATTR = "junction_builder_signature_json"
POINT_MATCH_SIGNATURE_ATTR = "junction_augmented_matching_signature_json"
POINT_MATCH_STATS_DATASET = "point_match_stats"
POINT_KEYPOINT_MATCHES_DATASET = "point_keypoint_matches"
POINT_JUNCTION_MATCHES_DATASET = "point_junction_matches"
POINT_TOTAL_MATCHES_DATASET = "point_total_matches"
POINT_KEYPOINT_CANDIDATE_MATCHES_DATASET = "point_keypoint_candidate_matches"
POINT_JUNCTION_CANDIDATE_MATCHES_DATASET = "point_junction_candidate_matches"
POINT_KEYPOINT_INLIER_MATCHES_DATASET = "point_keypoint_inlier_matches"
POINT_JUNCTION_INLIER_MATCHES_DATASET = "point_junction_inlier_matches"
POINT_GEOMETRY_SOURCE_DATASET = "point_geometry_source"
JUNCTION_GEOMETRY_SOURCE_DATASET = "junction_geometry_source"
LINE_GEOMETRY_SOURCE_DATASET = "line_geometry_source"


def _decode_h5_attr(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _serialize_signature(signature: Dict) -> str:
    return json.dumps(signature, sort_keys=True, separators=(",", ":"))


def build_junction_builder_signature(conf: Dict) -> Dict:
    return {
        "variant": str(conf_get(conf, "variant", "structured_linegraph")),
        "max_lines": int(conf_get(conf, "max_lines", 256)),
        "endpoint_merge_thresh_px": float(conf_get(conf, "endpoint_merge_thresh_px", 4.0)),
        "max_lines_per_image": int(conf_get(conf, "max_lines_per_image", 0)),
        "post_min_line_score": float(conf_get(conf, "post_min_line_score", 0.0)),
        "post_min_line_length_px": float(conf_get(conf, "post_min_line_length_px", 0.0)),
        "line_nms_center_thresh": float(conf_get(conf, "line_nms_center_thresh", 0.0)),
        "line_nms_angle_thresh_deg": float(
            conf_get(conf, "line_nms_angle_thresh_deg", 0.0)
        ),
        "line_nms_endpoint_thresh": float(
            conf_get(conf, "line_nms_endpoint_thresh", 0.0)
        ),
        "use_v2_style_line_prefilter": bool(
            conf_get(conf, "use_v2_style_line_prefilter", False)
        ),
    }


def build_augmented_point_match_signature(conf: Dict) -> Dict:
    return {
        "junction_augmented_points": True,
        **build_junction_builder_signature(conf),
        "use_points": bool(conf_get(conf, "use_points", True)),
        "use_line_message_passing": bool(conf_get(conf, "use_line_message_passing", True)),
        "point_match_score_thresh": float(conf_get(conf, "point_match_score_thresh", 0.20)),
        "line_match_score_thresh": float(conf_get(conf, "line_match_score_thresh", 0.20)),
        "junction_match_score_thresh": float(
            conf_get(conf, "junction_match_score_thresh", 0.20)
        ),
        "max_keypoints": int(conf_get(conf, "max_keypoints", 1024)),
        "drop_keypoints_near_endpoints_px": float(
            conf_get(conf, "drop_keypoints_near_endpoints_px", 4.0)
        ),
        "ransac_reproj_thresh": float(conf_get(conf, "ransac_reproj_thresh", 12.0)),
        "point_correct_thresh_px": float(conf_get(conf, "point_correct_thresh_px", 6.0)),
        "junction_correct_thresh_px": float(
            conf_get(conf, "junction_correct_thresh_px", 6.0)
        ),
        "filter_point_junction_matches_with_geometry": bool(
            conf_get(conf, "filter_point_junction_matches_with_geometry", True)
        ),
        "allow_line_fallback_geometry": bool(
            conf_get(conf, "allow_line_fallback_geometry", True)
        ),
        "line_inlier_endpoint_thresh": float(
            conf_get(conf, "line_inlier_endpoint_thresh", 18.0)
        ),
        "line_inlier_center_thresh": float(
            conf_get(conf, "line_inlier_center_thresh", 14.0)
        ),
        "line_inlier_angle_thresh_deg": float(
            conf_get(conf, "line_inlier_angle_thresh_deg", 10.0)
        ),
        "line_inlier_length_ratio_min": float(
            conf_get(conf, "line_inlier_length_ratio_min", 0.60)
        ),
    }


def _decode_h5_dataset_scalar(dataset) -> Optional[str]:
    return _decode_h5_attr(dataset[()])


def _encode_h5_string(value: str) -> np.bytes_:
    return np.bytes_(str(value))


def _prefilter_line_triplet(
    lines_xyxy,
    line_scores,
    line_descriptors,
    conf: Dict,
):
    lines_xyxy = np.asarray(lines_xyxy, dtype=np.float32).reshape(-1, 4)
    line_scores = np.asarray(line_scores, dtype=np.float32).reshape(-1)
    line_descriptors = np.asarray(line_descriptors, dtype=np.float32)
    if line_descriptors.ndim == 1:
        line_descriptors = line_descriptors.reshape(1, -1)
    if line_descriptors.size == 0:
        line_descriptors = np.zeros((lines_xyxy.shape[0], 0), dtype=np.float32)
    if not bool(conf_get(conf, "use_v2_style_line_prefilter", False)):
        keep = np.arange(lines_xyxy.shape[0], dtype=np.int64)
        return lines_xyxy, line_scores, line_descriptors, keep
    return post_filter_line_result(lines_xyxy, line_scores, line_descriptors, conf)


def _validate_junction_signature(feature_path: Path, conf: Dict):
    expected_signature = _serialize_signature(build_junction_builder_signature(conf))
    with h5py.File(str(feature_path), "r", libver="latest") as fd:
        actual_signature = _decode_h5_attr(fd.attrs.get(JUNCTION_SIGNATURE_ATTR))
        if actual_signature != expected_signature:
            raise RuntimeError(
                "Feature file {} does not contain v3 junction data matching the current "
                "wiregraph settings. Expected {}, found {}.".format(
                    feature_path,
                    expected_signature,
                    actual_signature,
                )
            )
        missing = []
        for name in _collect_group_names(
            fd,
            required_datasets=("image_size", "keypoints"),
        ):
            grp = fd[name]
            if "junctions" not in grp or "junction_scores" not in grp:
                missing.append(name)
                if len(missing) >= 3:
                    break
        if missing:
            raise RuntimeError(
                "Feature file {} is missing junction datasets for images like {}.".format(
                    feature_path,
                    ", ".join(missing),
                )
            )


def _invalidate_stale_match_file(match_path: Path, expected_signature: str, attr_name: str):
    if not match_path.exists():
        return
    with h5py.File(str(match_path), "r", libver="latest") as fd:
        actual_signature = _decode_h5_attr(fd.attrs.get(attr_name))
    if actual_signature == expected_signature:
        return
    logger.info(
        "Removing stale match file %s because the junction-aware signature changed.",
        match_path,
    )
    match_path.unlink()


def _set_match_file_signature(match_path: Path, attr_name: str, signature: str):
    with h5py.File(str(match_path), "a", libver="latest") as fd:
        fd.attrs[attr_name] = signature


def _collect_group_names(fd: h5py.File, required_datasets: Sequence[str]) -> List[str]:
    group_names = []

    def visitor(name, obj):
        if not isinstance(obj, h5py.Group):
            return
        if all(key in obj for key in required_datasets):
            group_names.append(str(name))

    fd.visititems(visitor)
    return group_names


def prepare_junction_features(
    conf: Dict,
    feature_path: Path,
    wiregraph_checkpoint: Path,
    overwrite: bool = False,
) -> Dict:
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_path}")

    _ensure_pointline_repo(conf, wiregraph_checkpoint)
    module = importlib.import_module("pointline.models.wiregraph_matcher")
    build_wireframe_graph = getattr(module, "build_wireframe_graph")
    structured_variant = getattr(module, "STRUCTURED_LINEGRAPH_VARIANT")

    signature = build_junction_builder_signature(conf)
    signature_json = _serialize_signature(signature)

    summary = {
        "feature_path": str(feature_path),
        "junction_builder_signature": signature,
        "num_images": 0,
        "total_junctions": 0,
        "mean_junctions_per_image": 0.0,
        "max_junctions_per_image": 0,
        "prepared": False,
    }

    with h5py.File(str(feature_path), "a", libver="latest") as fd:
        group_names = _collect_group_names(
            fd,
            required_datasets=("image_size", "keypoints", "line_segments"),
        )
        existing_signature = _decode_h5_attr(fd.attrs.get(JUNCTION_SIGNATURE_ATTR))
        has_complete_junctions = all(
            "junctions" in fd[name] and "junction_scores" in fd[name] for name in group_names
        )

        if not overwrite and existing_signature == signature_json and has_complete_junctions:
            counts = [int(fd[name]["junctions"].shape[0]) for name in group_names]
            summary.update(
                {
                    "num_images": int(len(group_names)),
                    "total_junctions": int(sum(counts)),
                    "mean_junctions_per_image": float(np.mean(counts)) if counts else 0.0,
                    "max_junctions_per_image": int(max(counts)) if counts else 0,
                    "prepared": False,
                }
            )
            logger.info("Reusing existing junction datasets in %s", feature_path)
            return summary

        junction_counts = []
        for name in tqdm(group_names, smoothing=0.1, desc="Preparing junctions"):
            grp = fd[name]
            image_size = grp["image_size"].__array__().reshape(-1)
            image_width = int(image_size[0])
            image_height = int(image_size[1])
            line_segments, line_scores, line_descriptors, _ = _prefilter_line_triplet(
                grp["line_segments"].__array__(),
                grp["line_scores"].__array__(),
                grp["line_descriptors"].__array__(),
                conf,
            )
            graph = build_wireframe_graph(
                keypoints_xy=torch.from_numpy(grp["keypoints"].__array__()).float(),
                keypoint_scores=torch.from_numpy(grp["scores"].__array__()).float(),
                keypoint_descriptors=torch.from_numpy(grp["descriptors"].__array__()).float(),
                line_segments_xyxy=torch.from_numpy(line_segments).float(),
                line_scores=torch.from_numpy(line_scores).float(),
                line_descriptors=torch.from_numpy(line_descriptors).float(),
                image_height=image_height,
                image_width=image_width,
                device="cpu",
                point_feature_map=torch.from_numpy(grp["point_feature_map"].__array__()).float(),
                point_feature_full_hw=grp["point_feature_full_hw"].__array__().reshape(-1),
                point_feature_original_to_model_scale_xy=grp[
                    "point_feature_original_to_model_scale_xy"
                ].__array__().reshape(-1),
                descriptor_map=torch.from_numpy(grp["descriptor_map"].__array__()).float(),
                descriptor_map_full_hw=grp["descriptor_map_full_hw"].__array__().reshape(-1),
                descriptor_map_original_to_model_scale_xy=grp[
                    "descriptor_map_original_to_model_scale_xy"
                ].__array__().reshape(-1),
                endpoint_merge_thresh_px=float(conf_get(conf, "endpoint_merge_thresh_px", 4.0)),
                drop_keypoints_near_endpoints_px=float(
                    conf_get(conf, "drop_keypoints_near_endpoints_px", 4.0)
                ),
                max_keypoints=int(conf_get(conf, "max_keypoints", 1024)),
                max_lines=int(conf_get(conf, "max_lines", 256)),
                use_points=bool(conf_get(conf, "use_points", True)),
                variant=str(conf_get(conf, "variant", structured_variant)),
                junction_feature_source=str(
                    conf_get(conf, "junction_feature_source", "line_descriptor_map")
                ),
            )

            junctions = graph.junction_points_xy.detach().cpu().numpy().astype(np.float32)
            if graph.junction_node_indices.numel() > 0:
                junction_scores = (
                    graph.node_scores[graph.junction_node_indices]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
            else:
                junction_scores = np.zeros((0,), dtype=np.float32)

            if "junctions" in grp:
                del grp["junctions"]
            if "junction_scores" in grp:
                del grp["junction_scores"]
            grp.create_dataset("junctions", data=junctions)
            grp.create_dataset("junction_scores", data=junction_scores)
            junction_counts.append(int(junctions.shape[0]))

        fd.attrs[JUNCTION_SIGNATURE_ATTR] = signature_json

    summary.update(
        {
            "num_images": int(len(junction_counts)),
            "total_junctions": int(sum(junction_counts)),
            "mean_junctions_per_image": float(np.mean(junction_counts))
            if junction_counts
            else 0.0,
            "max_junctions_per_image": int(max(junction_counts)) if junction_counts else 0,
            "prepared": True,
        }
    )
    logger.info(
        "Prepared junction datasets for %d images in %s",
        summary["num_images"],
        feature_path,
    )
    return summary


def summarize_augmented_point_matches(match_path: Path) -> Dict:
    summary = {
        "num_pairs": 0,
        "total_matches": 0,
        "total_keypoint_matches": 0,
        "total_junction_matches": 0,
        "total_keypoint_candidate_matches": 0,
        "total_junction_candidate_matches": 0,
        "total_keypoint_inlier_matches": 0,
        "total_junction_inlier_matches": 0,
        "mean_matches_per_pair": 0.0,
        "mean_keypoint_matches_per_pair": 0.0,
        "mean_junction_matches_per_pair": 0.0,
        "mean_keypoint_candidate_matches_per_pair": 0.0,
        "mean_junction_candidate_matches_per_pair": 0.0,
        "mean_keypoint_inlier_matches_per_pair": 0.0,
        "mean_junction_inlier_matches_per_pair": 0.0,
        "keypoint_inlier_ratio": 0.0,
        "junction_match_ratio": 0.0,
        "junction_inlier_ratio": 0.0,
        "point_geometry_source_counts": {},
        "junction_geometry_source_counts": {},
    }
    if not match_path.exists():
        return summary

    with h5py.File(str(match_path), "r", libver="latest") as fd:
        total_counts = []
        keypoint_counts = []
        junction_counts = []
        keypoint_candidate_counts = []
        junction_candidate_counts = []
        keypoint_inlier_counts = []
        junction_inlier_counts = []
        point_geometry_source_counts = {}
        junction_geometry_source_counts = {}
        for pair in _collect_group_names(fd, required_datasets=("matches0",)):
            grp = fd[pair]
            total = int(
                grp[POINT_TOTAL_MATCHES_DATASET].__array__().reshape(-1)[0]
            ) if POINT_TOTAL_MATCHES_DATASET in grp else int(
                np.count_nonzero(grp["matches0"].__array__() != -1)
            )
            keypoints = int(
                grp[POINT_KEYPOINT_MATCHES_DATASET].__array__().reshape(-1)[0]
            ) if POINT_KEYPOINT_MATCHES_DATASET in grp else total
            junctions = int(
                grp[POINT_JUNCTION_MATCHES_DATASET].__array__().reshape(-1)[0]
            ) if POINT_JUNCTION_MATCHES_DATASET in grp else 0
            keypoint_candidates = int(
                grp[POINT_KEYPOINT_CANDIDATE_MATCHES_DATASET].__array__().reshape(-1)[0]
            ) if POINT_KEYPOINT_CANDIDATE_MATCHES_DATASET in grp else keypoints
            junction_candidates = int(
                grp[POINT_JUNCTION_CANDIDATE_MATCHES_DATASET].__array__().reshape(-1)[0]
            ) if POINT_JUNCTION_CANDIDATE_MATCHES_DATASET in grp else junctions
            keypoint_inliers = int(
                grp[POINT_KEYPOINT_INLIER_MATCHES_DATASET].__array__().reshape(-1)[0]
            ) if POINT_KEYPOINT_INLIER_MATCHES_DATASET in grp else keypoints
            junction_inliers = int(
                grp[POINT_JUNCTION_INLIER_MATCHES_DATASET].__array__().reshape(-1)[0]
            ) if POINT_JUNCTION_INLIER_MATCHES_DATASET in grp else junctions
            total_counts.append(total)
            keypoint_counts.append(keypoints)
            junction_counts.append(junctions)
            keypoint_candidate_counts.append(keypoint_candidates)
            junction_candidate_counts.append(junction_candidates)
            keypoint_inlier_counts.append(keypoint_inliers)
            junction_inlier_counts.append(junction_inliers)
            if POINT_GEOMETRY_SOURCE_DATASET in grp:
                source = _decode_h5_dataset_scalar(grp[POINT_GEOMETRY_SOURCE_DATASET])
                point_geometry_source_counts[source] = (
                    point_geometry_source_counts.get(source, 0) + 1
                )
            if JUNCTION_GEOMETRY_SOURCE_DATASET in grp:
                source = _decode_h5_dataset_scalar(grp[JUNCTION_GEOMETRY_SOURCE_DATASET])
                junction_geometry_source_counts[source] = (
                    junction_geometry_source_counts.get(source, 0) + 1
                )

    summary.update(
        {
            "num_pairs": int(len(total_counts)),
            "total_matches": int(sum(total_counts)),
            "total_keypoint_matches": int(sum(keypoint_counts)),
            "total_junction_matches": int(sum(junction_counts)),
            "total_keypoint_candidate_matches": int(sum(keypoint_candidate_counts)),
            "total_junction_candidate_matches": int(sum(junction_candidate_counts)),
            "total_keypoint_inlier_matches": int(sum(keypoint_inlier_counts)),
            "total_junction_inlier_matches": int(sum(junction_inlier_counts)),
            "mean_matches_per_pair": float(np.mean(total_counts)) if total_counts else 0.0,
            "mean_keypoint_matches_per_pair": float(np.mean(keypoint_counts))
            if keypoint_counts
            else 0.0,
            "mean_junction_matches_per_pair": float(np.mean(junction_counts))
            if junction_counts
            else 0.0,
            "mean_keypoint_candidate_matches_per_pair": float(
                np.mean(keypoint_candidate_counts)
            )
            if keypoint_candidate_counts
            else 0.0,
            "mean_junction_candidate_matches_per_pair": float(
                np.mean(junction_candidate_counts)
            )
            if junction_candidate_counts
            else 0.0,
            "mean_keypoint_inlier_matches_per_pair": float(np.mean(keypoint_inlier_counts))
            if keypoint_inlier_counts
            else 0.0,
            "mean_junction_inlier_matches_per_pair": float(np.mean(junction_inlier_counts))
            if junction_inlier_counts
            else 0.0,
            "keypoint_inlier_ratio": float(
                sum(keypoint_inlier_counts) / max(sum(keypoint_candidate_counts), 1)
            ),
            "junction_match_ratio": float(sum(junction_counts) / max(sum(total_counts), 1)),
            "junction_inlier_ratio": float(
                sum(junction_inlier_counts) / max(sum(junction_candidate_counts), 1)
            ),
            "point_geometry_source_counts": point_geometry_source_counts,
            "junction_geometry_source_counts": junction_geometry_source_counts,
        }
    )
    return summary


def _pair_exists(fd: Optional[h5py.File], name0: str, name1: str) -> bool:
    if fd is None:
        return False
    for pair in (
        names_to_pair(name0, name1),
        names_to_pair(name1, name0),
        names_to_pair_old(name0, name1),
        names_to_pair_old(name1, name0),
    ):
        if pair in fd:
            return True
    return False


def find_unique_new_pairs_joint(
    pairs_all: List[Tuple[str, str]],
    point_match_path: Optional[Path] = None,
    line_match_path: Optional[Path] = None,
):
    pairs = set()
    for i, j in pairs_all:
        if (j, i) not in pairs:
            pairs.add((i, j))
    pairs = list(pairs)
    if point_match_path is None and line_match_path is None:
        return pairs

    point_fd = h5py.File(str(point_match_path), "r", libver="latest") if point_match_path and point_match_path.exists() else None
    line_fd = h5py.File(str(line_match_path), "r", libver="latest") if line_match_path and line_match_path.exists() else None
    try:
        filtered = []
        for i, j in pairs:
            if _pair_exists(point_fd, i, j) and _pair_exists(line_fd, i, j):
                continue
            filtered.append((i, j))
    finally:
        if point_fd is not None:
            point_fd.close()
        if line_fd is not None:
            line_fd.close()
    return filtered


def _mean_or_zero(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float32)))


def _estimate_homography_from_point_matches(matches, points0_xy, points1_xy, reproj_thresh: float):
    if len(matches) < 4:
        return None, np.zeros((len(matches),), dtype=bool)
    src_points = np.asarray(
        [np.asarray(points0_xy[int(match["idx1"])], dtype=np.float32) for match in matches],
        dtype=np.float32,
    )
    dst_points = np.asarray(
        [np.asarray(points1_xy[int(match["idx2"])], dtype=np.float32) for match in matches],
        dtype=np.float32,
    )
    homography, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, float(reproj_thresh))
    if mask is None:
        return homography, np.zeros((len(matches),), dtype=bool)
    return homography, mask.reshape(-1).astype(bool)


def _evaluate_pointlike_matches(
    matches,
    points0_xy: np.ndarray,
    points1_xy: np.ndarray,
    homography,
    thresh_px: float,
    inlier_mask=None,
):
    points0_xy = np.asarray(points0_xy, dtype=np.float32).reshape(-1, 2)
    points1_xy = np.asarray(points1_xy, dtype=np.float32).reshape(-1, 2)
    if homography is None:
        evaluated = []
        for match in matches:
            record = dict(match)
            record["reproj_error_px"] = None
            record["homography_inlier"] = False if inlier_mask is not None else None
            record["correct"] = False
            evaluated.append(record)
        return evaluated

    projected_points0 = cv2.perspectiveTransform(
        points0_xy.reshape(-1, 1, 2),
        np.asarray(homography, dtype=np.float32),
    ).reshape(-1, 2)
    evaluated = []
    for match_idx, match in enumerate(matches):
        idx0 = int(match["idx1"])
        idx1 = int(match["idx2"])
        projected = projected_points0[idx0]
        target = points1_xy[idx1]
        error = float(np.linalg.norm(projected - target))
        record = dict(match)
        record["reproj_error_px"] = error
        if inlier_mask is not None and match_idx < len(inlier_mask):
            homography_inlier = bool(inlier_mask[match_idx])
            record["homography_inlier"] = homography_inlier
        else:
            homography_inlier = None
            record["homography_inlier"] = None
        record["correct"] = bool(
            error <= float(thresh_px)
            and (homography_inlier is None or homography_inlier)
        )
        evaluated.append(record)
    return evaluated


def _mask_stats(mask, num_matches: int) -> Tuple[int, float]:
    if mask is None:
        return 0, 0.0
    inlier_count = int(np.count_nonzero(mask))
    inlier_ratio = float(inlier_count / max(int(num_matches), 1))
    return inlier_count, inlier_ratio


def _build_geometry_candidate(source: str, homography, mask, num_matches: int, priority: int):
    inlier_count, inlier_ratio = _mask_stats(mask, num_matches)
    return {
        "source": str(source),
        "homography": None if homography is None else np.asarray(homography, dtype=np.float32),
        "mask": None if mask is None else np.asarray(mask, dtype=bool),
        "num_inliers": int(inlier_count),
        "inlier_ratio": float(inlier_ratio),
        "priority": int(priority),
    }


def _select_best_geometry_candidate(candidates):
    valid = [candidate for candidate in candidates if candidate["homography"] is not None]
    if not valid:
        return None
    valid.sort(
        key=lambda item: (
            int(item["num_inliers"]),
            float(item["inlier_ratio"]),
            int(item.get("priority", 0)),
        ),
        reverse=True,
    )
    return valid[0]


def _resolve_eval_geometry(primitive_candidate, fallback_candidate):
    if primitive_candidate is not None and primitive_candidate["homography"] is not None:
        return (
            primitive_candidate["homography"],
            primitive_candidate.get("mask"),
            primitive_candidate["source"],
        )
    if fallback_candidate is not None and fallback_candidate["homography"] is not None:
        return (
            fallback_candidate["homography"],
            None,
            f"fallback_{fallback_candidate['source']}",
        )
    return None, None, "geometry_unavailable"


def _filter_pointlike_matches(
    matches,
    points0_xy: np.ndarray,
    points1_xy: np.ndarray,
    primitive_candidate,
    fallback_candidate,
    thresh_px: float,
    apply_filter: bool = True,
):
    eval_homography, eval_mask, eval_source = _resolve_eval_geometry(
        primitive_candidate,
        fallback_candidate,
    )
    if eval_homography is None:
        retained_matches = list(matches)
        return list(matches), {
            "candidate_count": int(len(matches)),
            "inlier_count": 0,
            "retained_count": int(len(retained_matches)),
            "geometry_source": str(eval_source),
            "retained_matches": retained_matches,
        }

    evaluated = _evaluate_pointlike_matches(
        matches,
        points0_xy,
        points1_xy,
        homography=eval_homography,
        thresh_px=float(thresh_px),
        inlier_mask=eval_mask,
    )
    filtered_matches = [
        dict(match)
        for match, record in zip(matches, evaluated)
        if bool(record.get("correct", False))
    ]
    retained_matches = filtered_matches if apply_filter else list(matches)
    return filtered_matches, {
        "candidate_count": int(len(matches)),
        "inlier_count": int(len(filtered_matches)),
        "retained_count": int(len(retained_matches)),
        "geometry_source": str(eval_source),
        "retained_matches": retained_matches,
    }


def _ensure_pointline_repo(conf: Dict, wiregraph_checkpoint: Path) -> Path:
    explicit_repo = conf.get("pointline_repo")
    candidates = []
    if explicit_repo:
        candidates.append(Path(explicit_repo).expanduser().resolve())

    config_path = Path(conf["config"]).expanduser().resolve()
    candidates.extend(config_path.parents)
    candidates.extend(wiregraph_checkpoint.expanduser().resolve().parents)
    for candidate in candidates:
        if (candidate / "pointline").is_dir():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate
    raise FileNotFoundError(
        "Could not infer the pointline repository root from config/checkpoint paths. "
        "Pass model.pointline_repo explicitly."
    )


class WireGraphPointLineRunner:
    def __init__(self, conf: Dict, wiregraph_checkpoint: Path, device: str):
        self.conf = conf = dict(conf)
        self.device = device
        wiregraph_checkpoint = Path(wiregraph_checkpoint).expanduser().resolve()
        if not wiregraph_checkpoint.is_file():
            raise FileNotFoundError(f"Wiregraph checkpoint not found: {wiregraph_checkpoint}")
        self.wiregraph_checkpoint = wiregraph_checkpoint

        self.pointline_repo = _ensure_pointline_repo(conf, wiregraph_checkpoint)
        module = importlib.import_module("pointline.models.wiregraph_matcher")
        self.build_wireframe_graph = getattr(module, "build_wireframe_graph")
        self.build_wiregraph_matcher_from_cfg = getattr(module, "build_wiregraph_matcher_from_cfg")
        self.structured_variant = getattr(module, "STRUCTURED_LINEGRAPH_VARIANT")

        cfg = load_joint_cfg(str(Path(conf["config"]).expanduser().resolve()))
        cfg.defrost()
        cfg.matcher.variant = str(conf.get("variant", cfg.matcher.variant))
        cfg.matcher.junction_feature_source = str(
            conf.get("junction_feature_source", cfg.matcher.junction_feature_source)
        )
        cfg.matcher.use_points = bool(conf.get("use_points", True))
        cfg.matcher.use_line_message_passing = bool(
            conf.get("use_line_message_passing", cfg.matcher.use_line_message_passing)
        )
        cfg.matcher.use_junction_head = bool(cfg.matcher.variant == self.structured_variant)
        cfg.freeze()
        self.cfg = cfg

        self.matcher = self.build_wiregraph_matcher_from_cfg(
            cfg,
            use_line_message_passing=bool(conf.get("use_line_message_passing", True)),
        ).to(device)
        checkpoint = safe_torch_load(str(wiregraph_checkpoint), map_location="cpu")
        self.matcher.load_state_dict(checkpoint.get("matcher", checkpoint), strict=True)
        self.matcher.eval()

    def _build_graph(self, data: Dict[str, torch.Tensor], suffix: str):
        image_size = data[f"image_size{suffix}"][0].detach().cpu().numpy().reshape(-1)
        image_width = int(image_size[0])
        image_height = int(image_size[1])
        line_segments_np = data[f"line_segments{suffix}"][0].detach().cpu().numpy().astype(
            np.float32
        )
        line_scores_np = data[f"line_scores{suffix}"][0].detach().cpu().numpy().astype(
            np.float32
        )
        line_descriptors_np = data[f"line_descriptors{suffix}"][0].detach().cpu().numpy().astype(
            np.float32
        )
        (
            filtered_line_segments_np,
            filtered_line_scores_np,
            filtered_line_descriptors_np,
            line_prefilter_indices_np,
        ) = _prefilter_line_triplet(
            line_segments_np,
            line_scores_np,
            line_descriptors_np,
            self.conf,
        )
        graph = self.build_wireframe_graph(
            keypoints_xy=data[f"keypoints{suffix}"][0],
            keypoint_scores=data[f"scores{suffix}"][0],
            keypoint_descriptors=data[f"descriptors{suffix}"][0],
            line_segments_xyxy=torch.from_numpy(filtered_line_segments_np).to(
                device=self.device
            ),
            line_scores=torch.from_numpy(filtered_line_scores_np).to(device=self.device),
            line_descriptors=torch.from_numpy(filtered_line_descriptors_np).to(
                device=self.device
            ),
            image_height=image_height,
            image_width=image_width,
            device=self.device,
            point_feature_map=data[f"point_feature_map{suffix}"][0],
            point_feature_full_hw=data[f"point_feature_full_hw{suffix}"][0].detach().cpu().numpy().reshape(-1),
            point_feature_original_to_model_scale_xy=data[
                f"point_feature_original_to_model_scale_xy{suffix}"
            ][0].detach().cpu().numpy().reshape(-1),
            descriptor_map=data[f"descriptor_map{suffix}"][0],
            descriptor_map_full_hw=data[f"descriptor_map_full_hw{suffix}"][0].detach().cpu().numpy().reshape(-1),
            descriptor_map_original_to_model_scale_xy=data[
                f"descriptor_map_original_to_model_scale_xy{suffix}"
            ][0].detach().cpu().numpy().reshape(-1),
            endpoint_merge_thresh_px=float(conf_get(self.conf, "endpoint_merge_thresh_px", 4.0)),
            drop_keypoints_near_endpoints_px=float(
                conf_get(self.conf, "drop_keypoints_near_endpoints_px", 4.0)
            ),
            max_keypoints=int(conf_get(self.conf, "max_keypoints", 1024)),
            max_lines=int(conf_get(self.conf, "max_lines", 256)),
            use_points=bool(conf_get(self.conf, "use_points", True)),
            variant=str(conf_get(self.conf, "variant", self.structured_variant)),
            junction_feature_source=str(
                conf_get(self.conf, "junction_feature_source", "line_descriptor_map")
            ),
        )
        line_prefilter_indices = torch.from_numpy(line_prefilter_indices_np).to(
            device=self.device, dtype=torch.long
        )
        return graph, line_prefilter_indices

    @staticmethod
    def _remap_matches(matches, source_indices0, source_indices1):
        remapped = []
        num0 = int(source_indices0.numel())
        num1 = int(source_indices1.numel())
        for match in matches:
            idx0 = int(match["idx1"])
            idx1 = int(match["idx2"])
            if idx0 < 0 or idx0 >= num0 or idx1 < 0 or idx1 >= num1:
                continue
            remapped.append(
                {
                    "idx1": int(source_indices0[idx0].item()),
                    "idx2": int(source_indices1[idx1].item()),
                    "score": float(match["score"]),
                    "similarity": float(match.get("similarity", match["score"])),
                }
            )
        return remapped

    @staticmethod
    def _get_augmented_points(data: Dict[str, torch.Tensor], suffix: str):
        keypoints = data[f"keypoints{suffix}"][0].detach().cpu().numpy().astype(np.float32)
        raw_keypoint_count = int(keypoints.shape[0])
        junction_key = f"junctions{suffix}"
        if junction_key not in data:
            return keypoints, raw_keypoint_count, 0
        junctions = data[junction_key][0].detach().cpu().numpy().astype(np.float32)
        if junctions.size == 0:
            return keypoints, raw_keypoint_count, 0
        augmented = np.concatenate([keypoints, junctions], axis=0).astype(np.float32)
        return augmented, raw_keypoint_count, int(junctions.shape[0])

    @torch.no_grad()
    def __call__(self, data: Dict[str, torch.Tensor]):
        keypoints0_xy = data["keypoints0"][0].detach().cpu().numpy().astype(np.float32)
        keypoints1_xy = data["keypoints1"][0].detach().cpu().numpy().astype(np.float32)
        graph0, line_prefilter_indices0 = self._build_graph(data, "0")
        graph1, line_prefilter_indices1 = self._build_graph(data, "1")
        outputs = self.matcher.predict(
            graph0,
            graph1,
            point_score_thresh=float(conf_get(self.conf, "point_match_score_thresh", 0.20)),
            line_score_thresh=float(conf_get(self.conf, "line_match_score_thresh", 0.20)),
            junction_score_thresh=float(conf_get(self.conf, "junction_match_score_thresh", 0.20)),
        )

        point_matches = self._remap_matches(
            outputs["point_matches"],
            graph0.point_source_indices,
            graph1.point_source_indices,
        )
        junction_matches = self._remap_matches(
            outputs.get("junction_matches", []),
            graph0.junction_source_indices,
            graph1.junction_source_indices,
        )
        line_matches = self._remap_matches(
            outputs["line_matches"],
            line_prefilter_indices0[graph0.line_source_indices],
            line_prefilter_indices1[graph1.line_source_indices],
        )
        points0_xy, raw_keypoint_count0, junction_count0 = self._get_augmented_points(data, "0")
        points1_xy, raw_keypoint_count1, _ = self._get_augmented_points(data, "1")

        junctions0_xy = points0_xy[raw_keypoint_count0:].astype(np.float32)
        junctions1_xy = points1_xy[raw_keypoint_count1:].astype(np.float32)
        ransac_reproj_thresh = float(conf_get(self.conf, "ransac_reproj_thresh", 12.0))
        point_correct_thresh = float(conf_get(self.conf, "point_correct_thresh_px", 6.0))
        junction_correct_thresh = float(
            conf_get(self.conf, "junction_correct_thresh_px", 6.0)
        )
        filter_point_junction = bool(
            conf_get(self.conf, "filter_point_junction_matches_with_geometry", True)
        )

        point_homography, point_inlier_mask = _estimate_homography_from_point_matches(
            point_matches,
            keypoints0_xy,
            keypoints1_xy,
            reproj_thresh=ransac_reproj_thresh,
        )
        point_geometry_candidate = _build_geometry_candidate(
            "estimated_from_point_matches",
            point_homography,
            point_inlier_mask,
            len(point_matches),
            priority=3,
        )

        junction_homography, junction_inlier_mask = _estimate_homography_from_point_matches(
            junction_matches,
            junctions0_xy,
            junctions1_xy,
            reproj_thresh=ransac_reproj_thresh,
        )
        junction_geometry_candidate = _build_geometry_candidate(
            "estimated_from_junction_matches",
            junction_homography,
            junction_inlier_mask,
            len(junction_matches),
            priority=2,
        )

        line_homography, line_inlier_mask = estimate_homography_from_matches(
            line_matches,
            data["line_segments0"][0].detach().cpu().numpy().astype(np.float32),
            data["line_segments1"][0].detach().cpu().numpy().astype(np.float32),
            reproj_thresh=ransac_reproj_thresh,
        )
        line_geometry_candidate = _build_geometry_candidate(
            "estimated_from_line_matches",
            line_homography,
            line_inlier_mask,
            len(line_matches),
            priority=1,
        )
        fallback_geometry_candidate = _select_best_geometry_candidate(
            [
                point_geometry_candidate,
                junction_geometry_candidate,
                line_geometry_candidate,
            ]
        )

        filtered_point_matches, point_filter_meta = _filter_pointlike_matches(
            point_matches,
            keypoints0_xy,
            keypoints1_xy,
            point_geometry_candidate,
            fallback_geometry_candidate,
            thresh_px=point_correct_thresh,
            apply_filter=filter_point_junction,
        )
        filtered_junction_matches, junction_filter_meta = _filter_pointlike_matches(
            junction_matches,
            junctions0_xy,
            junctions1_xy,
            junction_geometry_candidate,
            fallback_geometry_candidate,
            thresh_px=junction_correct_thresh,
            apply_filter=filter_point_junction,
        )
        retained_point_matches = list(point_filter_meta.pop("retained_matches"))
        retained_junction_matches = list(junction_filter_meta.pop("retained_matches"))
        augmented_point_matches = list(retained_point_matches)
        for match in retained_junction_matches:
            augmented_point_matches.append(
                {
                    "idx1": raw_keypoint_count0 + int(match["idx1"]),
                    "idx2": raw_keypoint_count1 + int(match["idx2"]),
                    "score": float(match["score"]),
                    "similarity": float(match.get("similarity", match["score"])),
                }
            )
        augmented_point_matches.sort(key=lambda item: (int(item["idx1"]), int(item["idx2"])))

        point_pred = build_point_prediction(
            augmented_point_matches,
            raw_keypoint_count0 + junction_count0,
            num_keypoint_matches=len(retained_point_matches),
            num_junction_matches=len(retained_junction_matches),
            num_keypoint_candidates=point_filter_meta["candidate_count"],
            num_junction_candidates=junction_filter_meta["candidate_count"],
            num_keypoint_inliers=point_filter_meta["inlier_count"],
            num_junction_inliers=junction_filter_meta["inlier_count"],
            point_geometry_source=point_filter_meta["geometry_source"],
            junction_geometry_source=junction_filter_meta["geometry_source"],
        )
        line_pred = build_line_prediction(
            line_matches=line_matches,
            lines0_xyxy=data["line_segments0"][0].detach().cpu().numpy().astype(np.float32),
            lines1_xyxy=data["line_segments1"][0].detach().cpu().numpy().astype(np.float32),
            num_lines0=int(data["line_segments0"][0].shape[0]),
            line_geometry_candidate=line_geometry_candidate,
            fallback_geometry_candidate=fallback_geometry_candidate,
            allow_fallback_geometry=bool(
                conf_get(self.conf, "allow_line_fallback_geometry", True)
            ),
            endpoint_thresh=float(conf_get(self.conf, "line_inlier_endpoint_thresh", 18.0)),
            center_thresh=float(conf_get(self.conf, "line_inlier_center_thresh", 14.0)),
            angle_thresh=float(conf_get(self.conf, "line_inlier_angle_thresh_deg", 10.0)),
            length_ratio_min=float(conf_get(self.conf, "line_inlier_length_ratio_min", 0.60)),
        )
        return point_pred, line_pred


def conf_get(conf: Dict, key: str, default):
    return conf[key] if key in conf else default


def build_point_prediction(
    point_matches,
    num_points0: int,
    num_keypoint_matches: int = 0,
    num_junction_matches: int = 0,
    num_keypoint_candidates: int = 0,
    num_junction_candidates: int = 0,
    num_keypoint_inliers: int = 0,
    num_junction_inliers: int = 0,
    point_geometry_source: str = "geometry_unavailable",
    junction_geometry_source: str = "geometry_unavailable",
):
    full_matches0 = np.full((num_points0,), -1, dtype=np.int32)
    full_scores0 = np.zeros((num_points0,), dtype=np.float32)
    for match in point_matches:
        idx0 = int(match["idx1"])
        idx1 = int(match["idx2"])
        if idx0 < 0 or idx0 >= num_points0:
            continue
        full_matches0[idx0] = idx1
        full_scores0[idx0] = float(match["score"])
    return {
        "matches0": full_matches0,
        "matching_scores0": full_scores0,
        POINT_MATCH_STATS_DATASET: np.asarray(
            [len(point_matches), num_keypoint_matches, num_junction_matches],
            dtype=np.int32,
        ),
        POINT_TOTAL_MATCHES_DATASET: np.asarray([len(point_matches)], dtype=np.int32),
        POINT_KEYPOINT_MATCHES_DATASET: np.asarray([num_keypoint_matches], dtype=np.int32),
        POINT_JUNCTION_MATCHES_DATASET: np.asarray([num_junction_matches], dtype=np.int32),
        POINT_KEYPOINT_CANDIDATE_MATCHES_DATASET: np.asarray(
            [num_keypoint_candidates], dtype=np.int32
        ),
        POINT_JUNCTION_CANDIDATE_MATCHES_DATASET: np.asarray(
            [num_junction_candidates], dtype=np.int32
        ),
        POINT_KEYPOINT_INLIER_MATCHES_DATASET: np.asarray(
            [num_keypoint_inliers], dtype=np.int32
        ),
        POINT_JUNCTION_INLIER_MATCHES_DATASET: np.asarray(
            [num_junction_inliers], dtype=np.int32
        ),
        POINT_GEOMETRY_SOURCE_DATASET: str(point_geometry_source),
        JUNCTION_GEOMETRY_SOURCE_DATASET: str(junction_geometry_source),
    }


def build_line_prediction(
    line_matches,
    lines0_xyxy: np.ndarray,
    lines1_xyxy: np.ndarray,
    num_lines0: int,
    line_geometry_candidate,
    fallback_geometry_candidate,
    allow_fallback_geometry: bool,
    endpoint_thresh: float,
    center_thresh: float,
    angle_thresh: float,
    length_ratio_min: float,
):
    full_matches0 = np.full((num_lines0,), -1, dtype=np.int32)
    full_scores0 = np.zeros((num_lines0,), dtype=np.float32)
    full_verified0 = np.zeros((num_lines0,), dtype=np.uint8)

    for match in line_matches:
        idx0 = int(match["idx1"])
        if idx0 < 0 or idx0 >= num_lines0:
            continue
        full_matches0[idx0] = int(match["idx2"])
        full_scores0[idx0] = float(match["score"])

    candidate_count = int(len(line_matches))
    verified_count = 0
    verified_similarities = []
    verified_endpoint_errors = []
    verified_center_errors = []
    verified_angle_errors = []
    if allow_fallback_geometry:
        (
            line_eval_homography,
            line_eval_inlier_mask,
            line_geometry_source,
        ) = _resolve_eval_geometry(
            line_geometry_candidate,
            fallback_geometry_candidate,
        )
    elif (
        line_geometry_candidate is not None
        and line_geometry_candidate.get("homography") is not None
    ):
        line_eval_homography = line_geometry_candidate["homography"]
        line_eval_inlier_mask = line_geometry_candidate.get("mask")
        line_geometry_source = str(line_geometry_candidate["source"])
    else:
        line_eval_homography = None
        line_eval_inlier_mask = None
        line_geometry_source = "geometry_unavailable"

    if line_eval_homography is None:
        evaluated = [dict(match) for match in line_matches]
        verified_flags = [False] * len(evaluated)
    else:
        evaluated = evaluate_matches_against_homography(
            line_matches,
            lines0_xyxy,
            lines1_xyxy,
            line_eval_homography,
            endpoint_thresh=endpoint_thresh,
            angle_thresh_deg=angle_thresh,
            center_thresh=center_thresh,
            length_ratio_min=length_ratio_min,
        )
        if (
            line_geometry_source == "estimated_from_line_matches"
            and line_eval_inlier_mask is not None
        ):
            verified_flags = [
                bool(line_eval_inlier_mask[idx]) and bool(record["correct"])
                for idx, record in enumerate(evaluated)
            ]
        else:
            verified_flags = [bool(record["correct"]) for record in evaluated]

    for match, record, is_verified in zip(line_matches, evaluated, verified_flags):
        idx0 = int(match["idx1"])
        if idx0 < 0 or idx0 >= num_lines0:
            continue
        full_verified0[idx0] = 1 if is_verified else 0
        if not is_verified:
            continue
        verified_count += 1
        verified_similarities.append(float(match["score"]))
        verified_endpoint_errors.append(float(record.get("endpoint_error", 0.0)))
        verified_center_errors.append(float(record.get("center_reproj_error", 0.0)))
        verified_angle_errors.append(float(record.get("angle_diff_deg", 0.0)))

    verified_ratio = float(verified_count / max(candidate_count, 1)) if candidate_count > 0 else 0.0
    return {
        "line_matches0": full_matches0,
        "line_matching_scores0": full_scores0,
        "line_verified_mask0": full_verified0,
        "line_match_stats": np.asarray([candidate_count, verified_count], dtype=np.int32),
        "line_candidate_matches": np.asarray([candidate_count], dtype=np.int32),
        "line_verified_matches": np.asarray([verified_count], dtype=np.int32),
        "line_verified_ratio": np.asarray([verified_ratio], dtype=np.float32),
        "line_mean_verified_similarity": np.asarray([_mean_or_zero(verified_similarities)], dtype=np.float32),
        "line_mean_verified_endpoint_error": np.asarray([_mean_or_zero(verified_endpoint_errors)], dtype=np.float32),
        "line_mean_verified_center_error": np.asarray([_mean_or_zero(verified_center_errors)], dtype=np.float32),
        "line_mean_verified_angle_error_deg": np.asarray([_mean_or_zero(verified_angle_errors)], dtype=np.float32),
        LINE_GEOMETRY_SOURCE_DATASET: str(line_geometry_source),
    }


def write_point_matches(pair: str, pred: Dict[str, np.ndarray], match_path: Path):
    with h5py.File(str(match_path), "a", libver="latest") as fd:
        if pair in fd:
            del fd[pair]
        grp = fd.create_group(pair)
        grp.create_dataset("matches0", data=pred["matches0"].astype(np.int32))
        grp.create_dataset("matching_scores0", data=pred["matching_scores0"].astype(np.float16))
        for key in (
            POINT_MATCH_STATS_DATASET,
            POINT_TOTAL_MATCHES_DATASET,
            POINT_KEYPOINT_MATCHES_DATASET,
            POINT_JUNCTION_MATCHES_DATASET,
            POINT_KEYPOINT_CANDIDATE_MATCHES_DATASET,
            POINT_JUNCTION_CANDIDATE_MATCHES_DATASET,
            POINT_KEYPOINT_INLIER_MATCHES_DATASET,
            POINT_JUNCTION_INLIER_MATCHES_DATASET,
        ):
            if key in pred:
                grp.create_dataset(key, data=pred[key].astype(np.int32))
        for key in (POINT_GEOMETRY_SOURCE_DATASET, JUNCTION_GEOMETRY_SOURCE_DATASET):
            if key in pred:
                grp.create_dataset(key, data=_encode_h5_string(pred[key]))


def write_line_matches(pair: str, pred: Dict[str, np.ndarray], match_path: Path):
    with h5py.File(str(match_path), "a", libver="latest") as fd:
        if pair in fd:
            del fd[pair]
        grp = fd.create_group(pair)
        grp.create_dataset("line_matches0", data=pred["line_matches0"].astype(np.int32))
        grp.create_dataset(
            "line_matching_scores0",
            data=pred["line_matching_scores0"].astype(np.float32),
        )
        grp.create_dataset(
            "line_verified_mask0",
            data=pred["line_verified_mask0"].astype(np.uint8),
        )
        grp.create_dataset(
            "line_match_stats",
            data=pred["line_match_stats"].astype(np.int32),
        )
        for key in (
            "line_candidate_matches",
            "line_verified_matches",
            "line_verified_ratio",
            "line_mean_verified_similarity",
            "line_mean_verified_endpoint_error",
            "line_mean_verified_center_error",
            "line_mean_verified_angle_error_deg",
        ):
            grp.create_dataset(key, data=pred[key])
        if LINE_GEOMETRY_SOURCE_DATASET in pred:
            grp.create_dataset(
                LINE_GEOMETRY_SOURCE_DATASET,
                data=_encode_h5_string(pred[LINE_GEOMETRY_SOURCE_DATASET]),
            )


def main(
    conf: Dict,
    pairs: Path,
    features: Union[Path, str],
    export_dir: Optional[Path] = None,
    point_matches: Optional[Path] = None,
    line_matches: Optional[Path] = None,
    features_ref: Optional[Path] = None,
    wiregraph_checkpoint: Optional[Union[Path, str]] = None,
    overwrite: bool = False,
) -> Tuple[Path, Path]:
    if wiregraph_checkpoint is None:
        raise ValueError("wiregraph_checkpoint is required for joint wiregraph matching.")

    if isinstance(features, Path) or Path(features).exists():
        features_q = Path(features)
        if point_matches is None or line_matches is None:
            raise ValueError(
                "Provide both point_matches and line_matches when features is given as a file path."
            )
    else:
        if export_dir is None:
            raise ValueError("Provide an export_dir if features is not a file path.")
        features_q = Path(export_dir, str(features) + ".h5")
        if point_matches is None:
            point_matches = Path(export_dir, f"{features}_{conf['point_output']}_{pairs.stem}.h5")
        if line_matches is None:
            line_matches = Path(export_dir, f"{features}_{conf['line_output']}_{pairs.stem}.h5")

    if features_ref is None:
        features_ref = features_q

    return match_from_paths(
        conf=conf,
        pairs_path=pairs,
        point_match_path=Path(point_matches),
        line_match_path=Path(line_matches),
        feature_path_q=features_q,
        feature_path_ref=Path(features_ref),
        wiregraph_checkpoint=Path(wiregraph_checkpoint),
        overwrite=overwrite,
    )


@torch.no_grad()
def match_from_paths(
    conf: Dict,
    pairs_path: Path,
    point_match_path: Path,
    line_match_path: Path,
    feature_path_q: Path,
    feature_path_ref: Path,
    wiregraph_checkpoint: Path,
    overwrite: bool = False,
) -> Tuple[Path, Path]:
    logger.info(
        "Matching point-line features with configuration:\n%s",
        pprint.pformat(conf),
    )
    if not feature_path_q.exists():
        raise FileNotFoundError(f"Query feature file {feature_path_q}.")
    if not feature_path_ref.exists():
        raise FileNotFoundError(f"Reference feature file {feature_path_ref}.")
    point_match_path.parent.mkdir(exist_ok=True, parents=True)
    line_match_path.parent.mkdir(exist_ok=True, parents=True)
    _validate_junction_signature(feature_path_q, conf["model"])
    if feature_path_ref != feature_path_q:
        _validate_junction_signature(feature_path_ref, conf["model"])

    match_signature = _serialize_signature(build_augmented_point_match_signature(conf["model"]))
    _invalidate_stale_match_file(point_match_path, match_signature, POINT_MATCH_SIGNATURE_ATTR)
    _invalidate_stale_match_file(line_match_path, match_signature, POINT_MATCH_SIGNATURE_ATTR)
    _set_match_file_signature(point_match_path, POINT_MATCH_SIGNATURE_ATTR, match_signature)
    _set_match_file_signature(line_match_path, POINT_MATCH_SIGNATURE_ATTR, match_signature)

    pairs = parse_retrieval(pairs_path)
    pairs = [(q, r) for q, rs in pairs.items() for r in rs]
    pairs = find_unique_new_pairs_joint(
        pairs,
        None if overwrite else point_match_path,
        None if overwrite else line_match_path,
    )
    if len(pairs) == 0:
        logger.info("Skipping joint point-line matching.")
        return point_match_path, line_match_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    runner = WireGraphPointLineRunner(conf["model"], wiregraph_checkpoint, device)

    dataset = FeaturePairsDataset(pairs, feature_path_q, feature_path_ref)
    loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )

    for idx, data in enumerate(tqdm(loader, smoothing=0.1)):
        data = {
            k: v if k.startswith("image") else v.to(device, non_blocking=True)
            for k, v in data.items()
        }
        point_pred, line_pred = runner(data)
        pair = names_to_pair(pairs[idx][0], pairs[idx][1])
        write_point_matches(pair, point_pred, point_match_path)
        write_line_matches(pair, line_pred, line_match_path)

    return point_match_path, line_match_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--export_dir", type=Path)
    parser.add_argument("--point_matches", type=Path)
    parser.add_argument("--line_matches", type=Path)
    parser.add_argument("--features_ref", type=Path)
    parser.add_argument("--wiregraph_checkpoint", type=Path, required=True)
    parser.add_argument(
        "--conf",
        type=str,
        default="joint_wiregraph",
        choices=list(confs.keys()),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(
        confs[args.conf],
        args.pairs,
        args.features,
        args.export_dir,
        args.point_matches,
        args.line_matches,
        args.features_ref,
        args.wiregraph_checkpoint,
        args.overwrite,
    )
