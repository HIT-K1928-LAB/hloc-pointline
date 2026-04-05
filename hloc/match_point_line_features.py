import argparse
import importlib
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
            "line_inlier_endpoint_thresh": 18.0,
            "line_inlier_center_thresh": 14.0,
            "line_inlier_angle_thresh_deg": 10.0,
            "line_inlier_length_ratio_min": 0.60,
        },
    },
}


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
        return self.build_wireframe_graph(
            keypoints_xy=data[f"keypoints{suffix}"][0],
            keypoint_scores=data[f"scores{suffix}"][0],
            keypoint_descriptors=data[f"descriptors{suffix}"][0],
            line_segments_xyxy=data[f"line_segments{suffix}"][0],
            line_scores=data[f"line_scores{suffix}"][0],
            line_descriptors=data[f"line_descriptors{suffix}"][0],
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

    @torch.no_grad()
    def __call__(self, data: Dict[str, torch.Tensor]):
        graph0 = self._build_graph(data, "0")
        graph1 = self._build_graph(data, "1")
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
        line_matches = self._remap_matches(
            outputs["line_matches"],
            graph0.line_source_indices,
            graph1.line_source_indices,
        )

        point_pred = build_point_prediction(point_matches, int(data["keypoints0"][0].shape[0]))
        line_pred = build_line_prediction(
            point_matches=point_matches,
            line_matches=line_matches,
            lines0_xyxy=data["line_segments0"][0].detach().cpu().numpy().astype(np.float32),
            lines1_xyxy=data["line_segments1"][0].detach().cpu().numpy().astype(np.float32),
            points0_xy=data["keypoints0"][0].detach().cpu().numpy().astype(np.float32),
            points1_xy=data["keypoints1"][0].detach().cpu().numpy().astype(np.float32),
            num_lines0=int(data["line_segments0"][0].shape[0]),
            ransac_reproj_thresh=float(conf_get(self.conf, "ransac_reproj_thresh", 12.0)),
            endpoint_thresh=float(conf_get(self.conf, "line_inlier_endpoint_thresh", 18.0)),
            center_thresh=float(conf_get(self.conf, "line_inlier_center_thresh", 14.0)),
            angle_thresh=float(conf_get(self.conf, "line_inlier_angle_thresh_deg", 10.0)),
            length_ratio_min=float(conf_get(self.conf, "line_inlier_length_ratio_min", 0.60)),
        )
        return point_pred, line_pred


def conf_get(conf: Dict, key: str, default):
    return conf[key] if key in conf else default


def build_point_prediction(point_matches, num_points0: int):
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
    }


def build_line_prediction(
    point_matches,
    line_matches,
    lines0_xyxy: np.ndarray,
    lines1_xyxy: np.ndarray,
    points0_xy: np.ndarray,
    points1_xy: np.ndarray,
    num_lines0: int,
    ransac_reproj_thresh: float,
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

    point_homography, _ = _estimate_homography_from_point_matches(
        point_matches,
        points0_xy,
        points1_xy,
        reproj_thresh=ransac_reproj_thresh,
    )

    if point_homography is not None:
        evaluated = evaluate_matches_against_homography(
            line_matches,
            lines0_xyxy,
            lines1_xyxy,
            point_homography,
            endpoint_thresh=endpoint_thresh,
            angle_thresh_deg=angle_thresh,
            center_thresh=center_thresh,
            length_ratio_min=length_ratio_min,
        )
        verified_flags = [bool(record["correct"]) for record in evaluated]
    else:
        line_homography, line_inlier_mask = estimate_homography_from_matches(
            line_matches,
            lines0_xyxy,
            lines1_xyxy,
            reproj_thresh=ransac_reproj_thresh,
        )
        if line_homography is not None:
            evaluated = evaluate_matches_against_homography(
                line_matches,
                lines0_xyxy,
                lines1_xyxy,
                line_homography,
                endpoint_thresh=endpoint_thresh,
                angle_thresh_deg=angle_thresh,
                center_thresh=center_thresh,
                length_ratio_min=length_ratio_min,
            )
            verified_flags = [
                bool(line_inlier_mask[idx]) and bool(record["correct"])
                for idx, record in enumerate(evaluated)
            ]
        else:
            evaluated = [dict(match) for match in line_matches]
            verified_flags = [False] * len(evaluated)

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
    }


def write_point_matches(pair: str, pred: Dict[str, np.ndarray], match_path: Path):
    with h5py.File(str(match_path), "a", libver="latest") as fd:
        if pair in fd:
            del fd[pair]
        grp = fd.create_group(pair)
        grp.create_dataset("matches0", data=pred["matches0"].astype(np.int32))
        grp.create_dataset("matching_scores0", data=pred["matching_scores0"].astype(np.float16))


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
