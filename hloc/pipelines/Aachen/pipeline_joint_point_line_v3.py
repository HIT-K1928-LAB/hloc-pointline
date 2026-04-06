import argparse
import copy
import json
import pickle
from pathlib import Path
from pprint import pformat
from statistics import mean, median
from typing import Dict

from ... import (
    colmap_from_nvm,
    extract_features,
    line_mapping,
    localize_sfm_point_line_v2,
    logger,
    match_point_line_features,
    pairs_from_covisibility,
    pairs_from_retrieval,
    triangulation,
)

# cd "/home/hxy/doctor/feature dectect/hloc/Hierarchical-Localization"
#
# python -m hloc.pipelines.Aachen.pipeline_joint_point_line_v3 \
#   --dataset datasets/aachen \
#   --outputs outputs/aachen_joint_point_line_v3 \
#   --wiregraph_checkpoint /path/to/wiregraph_matcher_best.pth \
#   --num_covis 20 \
#   --num_loc 50 \
#   --feature_conf joint_xfeat_mlsd_hloc_matcher \
#   --retrieval_conf gaussvladplusvgg \
#   --joint_matcher_conf joint_wiregraph


def _safe_mean(values):
    return float(mean(values)) if values else 0.0


def _safe_median(values):
    return float(median(values)) if values else 0.0


def summarize_v2_logs(logs_path: Path) -> Dict:
    with open(logs_path, "rb") as f:
        logs = pickle.load(f)

    loc_logs = logs.get("loc", {})
    successful_pnp = 0
    successful_refined = 0
    pnp_inliers = []
    total_matches = []
    line_corr_counts = []
    line_inlier_counts = []
    refinement_gains = []
    queries_with_line_support = 0

    for log in loc_logs.values():
        ret = log.get("PnP_ret")
        if ret is not None:
            successful_pnp += 1
            if isinstance(ret, dict) and "num_inliers" in ret:
                pnp_inliers.append(int(ret["num_inliers"]))
        total_matches.append(int(log.get("num_matches", 0)))
        line_corr = int(log.get("line_correspondence_count", 0))
        line_inliers = int(log.get("line_inlier_count", 0))
        line_corr_counts.append(line_corr)
        line_inlier_counts.append(line_inliers)
        if line_corr > 0:
            queries_with_line_support += 1
        if bool(log.get("line_refinement_applied", False)):
            successful_refined += 1
            before = float(log.get("line_residual_before_refine", 0.0))
            after = float(log.get("line_residual_after_refine", before))
            refinement_gains.append(before - after)

    return {
        "total_queries": int(len(loc_logs)),
        "successful_pnp_queries": int(successful_pnp),
        "successful_pnp_ratio": float(successful_pnp / max(len(loc_logs), 1)),
        "successful_pose_after_refinement_queries": int(successful_refined),
        "mean_pnp_inliers": _safe_mean(pnp_inliers),
        "median_pnp_inliers": _safe_median(pnp_inliers),
        "mean_total_2d3d_matches": _safe_mean(total_matches),
        "median_total_2d3d_matches": _safe_median(total_matches),
        "queries_with_verified_line_support": int(queries_with_line_support),
        "queries_with_verified_line_support_ratio": float(
            queries_with_line_support / max(len(loc_logs), 1)
        ),
        "mean_verified_line_correspondences": _safe_mean(line_corr_counts),
        "median_verified_line_correspondences": _safe_median(line_corr_counts),
        "mean_line_inliers": _safe_mean(line_inlier_counts),
        "median_line_inliers": _safe_median(line_inlier_counts),
        "mean_line_refinement_gain": _safe_mean(refinement_gains),
        "median_line_refinement_gain": _safe_median(refinement_gains),
    }


def run(args):
    dataset = args.dataset
    images = dataset / "images_upright"
    outputs = args.outputs
    outputs.mkdir(parents=True, exist_ok=True)

    sift_sfm = outputs / "sfm_sift"
    reference_sfm = outputs / "sfm_joint-xfeat-mlsd+wiregraph"
    sfm_pairs = outputs / f"pairs-db-covis{args.num_covis}.txt"
    loc_pairs = outputs / f"pairs-query-netvlad{args.num_loc}.txt"

    results_v3 = outputs / (
        f"Aachen_hloc_joint-xfeat-mlsd+wiregraph_netvlad{args.num_loc}_point-line-v3.txt"
    )
    summary_json = outputs / (
        f"Aachen_hloc_joint-xfeat-mlsd+wiregraph_netvlad{args.num_loc}_summary_v3.json"
    )
    line_map_path = outputs / "line_map.h5"
    line_assignment_path = outputs / "line_track_assignments.h5"

    logger.info("Configs for feature extractors\n%s", pformat(extract_features.confs))
    logger.info(
        "Configs for joint point-line matchers\n%s",
        pformat(match_point_line_features.confs),
    )

    retrieval_conf = extract_features.confs[args.retrieval_conf]
    feature_conf = extract_features.confs[args.feature_conf]
    joint_matcher_conf = copy.deepcopy(
        match_point_line_features.confs[args.joint_matcher_conf]
    )
    joint_matcher_model_conf = joint_matcher_conf["model"]
    if args.point_match_score_thresh is not None:
        joint_matcher_model_conf["point_match_score_thresh"] = float(
            args.point_match_score_thresh
        )
    if args.line_match_score_thresh is not None:
        joint_matcher_model_conf["line_match_score_thresh"] = float(
            args.line_match_score_thresh
        )
    if args.junction_match_score_thresh is not None:
        joint_matcher_model_conf["junction_match_score_thresh"] = float(
            args.junction_match_score_thresh
        )
    if args.point_correct_thresh_px is not None:
        joint_matcher_model_conf["point_correct_thresh_px"] = float(
            args.point_correct_thresh_px
        )
    if args.junction_correct_thresh_px is not None:
        joint_matcher_model_conf["junction_correct_thresh_px"] = float(
            args.junction_correct_thresh_px
        )
    if args.matcher_max_keypoints is not None:
        joint_matcher_model_conf["max_keypoints"] = int(args.matcher_max_keypoints)
    if args.matcher_max_lines is not None:
        joint_matcher_model_conf["max_lines"] = int(args.matcher_max_lines)
    if args.endpoint_merge_thresh_px is not None:
        joint_matcher_model_conf["endpoint_merge_thresh_px"] = float(
            args.endpoint_merge_thresh_px
        )
    if args.drop_keypoints_near_endpoints_px is not None:
        joint_matcher_model_conf["drop_keypoints_near_endpoints_px"] = float(
            args.drop_keypoints_near_endpoints_px
        )

    logger.info("Extracting matcher-compatible joint point-line features...")
    features = extract_features.main(feature_conf, images, outputs, overwrite=args.overwrite)
    logger.info("Preparing junction-aware point augmentation for V3...")
    junction_feature_summary = match_point_line_features.prepare_junction_features(
        joint_matcher_conf["model"],
        features,
        args.wiregraph_checkpoint,
        overwrite=args.overwrite,
    )

    logger.info("Preparing reference SfM scaffold from Aachen NVM...")
    colmap_from_nvm.main(
        dataset / "3D-models/aachen_cvpr2018_db.nvm",
        dataset / "3D-models/database_intrinsics.txt",
        dataset / "aachen.db",
        sift_sfm,
    )

    logger.info("Generating covisibility pairs for triangulation...")
    pairs_from_covisibility.main(sift_sfm, sfm_pairs, num_matched=args.num_covis)

    logger.info("Running joint wiregraph matcher for database-database pairs...")
    sfm_matches, sfm_line_matches = match_point_line_features.main(
        joint_matcher_conf,
        sfm_pairs,
        feature_conf["output"],
        outputs,
        wiregraph_checkpoint=args.wiregraph_checkpoint,
        overwrite=args.overwrite,
    )
    sfm_point_match_summary = match_point_line_features.summarize_augmented_point_matches(
        sfm_matches
    )

    logger.info("Triangulating reference SfM model from matcher point matches...")
    triangulation.main(reference_sfm, sift_sfm, images, sfm_pairs, features, sfm_matches)

    if args.overwrite or not line_map_path.exists() or not line_assignment_path.exists():
        logger.info("Building 3D line map...")
        _, _, line_map_summary = line_mapping.main(
            reference_sfm=reference_sfm,
            features=features,
            line_matches=sfm_line_matches,
            pairs=sfm_pairs,
            line_map_path=line_map_path,
            line_assignment_path=line_assignment_path,
            min_support_count=args.line_map_min_support_count,
            min_unique_images=args.line_map_min_unique_images,
            min_camera_center_distance=args.line_map_min_camera_center_distance,
        )
    else:
        logger.info("Reusing existing 3D line map at %s", line_map_path)
        line_map_summary = line_mapping.summarize_line_map(line_map_path)

    logger.info("Extracting retrieval descriptors...")
    global_descriptors = extract_features.main(
        retrieval_conf,
        images,
        outputs,
        overwrite=args.overwrite,
    )

    logger.info("Generating retrieval pairs for Aachen queries...")
    pairs_from_retrieval.main(
        global_descriptors,
        loc_pairs,
        args.num_loc,
        query_prefix="query",
        db_model=reference_sfm,
    )

    logger.info("Running joint wiregraph matcher for localization pairs...")
    loc_matches, loc_line_matches = match_point_line_features.main(
        joint_matcher_conf,
        loc_pairs,
        feature_conf["output"],
        outputs,
        wiregraph_checkpoint=args.wiregraph_checkpoint,
        overwrite=args.overwrite,
    )
    loc_point_match_summary = match_point_line_features.summarize_augmented_point_matches(
        loc_matches
    )

    query_list = dataset / "queries/*_time_queries_with_intrinsics.txt"

    logger.info("Running point-line V3 Aachen localization with 3D line map...")
    localize_sfm_point_line_v2.main(
        reference_sfm=reference_sfm,
        queries=query_list,
        retrieval=loc_pairs,
        features=features,
        matches=loc_matches,
        line_matches=loc_line_matches,
        line_map=line_map_path,
        line_track_assignments=line_assignment_path,
        results=results_v3,
        ransac_thresh=args.ransac_thresh,
        covisibility_clustering=args.covisibility_clustering,
        min_point_matches=args.min_point_matches,
        min_verified_lines=args.min_verified_lines,
        min_verified_ratio=args.min_verified_ratio,
        max_db_images_for_pnp=args.max_db_images_for_pnp,
        endpoint_thresh=args.endpoint_thresh,
        center_thresh=args.center_thresh,
        angle_thresh=args.angle_thresh,
        max_refine_points=args.max_refine_points,
    )

    v3_logs = Path(f"{results_v3}_logs.pkl")
    v3_summary = summarize_v2_logs(v3_logs)

    summary = {
        "dataset": str(dataset),
        "images": str(images),
        "outputs": str(outputs),
        "feature_conf": args.feature_conf,
        "retrieval_conf": args.retrieval_conf,
        "joint_matcher_conf": args.joint_matcher_conf,
        "joint_matcher_conf_effective": joint_matcher_conf,
        "junction_aware_points": True,
        "junction_feature_summary": junction_feature_summary,
        "sfm_point_match_summary": sfm_point_match_summary,
        "loc_point_match_summary": loc_point_match_summary,
        "wiregraph_checkpoint": str(args.wiregraph_checkpoint),
        "num_covis": int(args.num_covis),
        "num_loc": int(args.num_loc),
        "ransac_thresh": float(args.ransac_thresh),
        "line_map_summary": line_map_summary,
        "paths": {
            "features": str(features),
            "sfm_pairs": str(sfm_pairs),
            "loc_pairs": str(loc_pairs),
            "sfm_matches": str(sfm_matches),
            "sfm_line_matches": str(sfm_line_matches),
            "loc_matches": str(loc_matches),
            "loc_line_matches": str(loc_line_matches),
            "line_map": str(line_map_path),
            "line_track_assignments": str(line_assignment_path),
            "reference_sfm": str(reference_sfm),
            "results_v3": str(results_v3),
            "results_v3_logs": str(v3_logs),
        },
        "point_line_v3_summary": v3_summary,
        "note": (
            "Aachen official pose accuracy is not computed here. "
            "This script runs a joint wiregraph matcher for both point and line pairs, "
            "builds a 3D line map, and summarizes point-line-v3 localization logs."
        ),
    }

    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info("Wrote V3 summary to %s", summary_json)
    logger.info("Point-line V3 results: %s", results_v3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default="datasets/aachen",
        help="Path to the Aachen dataset, default: %(default)s",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default="outputs/aachen_joint_point_line_v3",
        help="Path to the output directory, default: %(default)s",
    )
    parser.add_argument(
        "--wiregraph_checkpoint",
        type=Path,
        required=True,
        help="Path to the wiregraph matcher checkpoint.",
    )
    parser.add_argument(
        "--num_covis",
        type=int,
        default=20,
        help="Number of image pairs for SfM triangulation, default: %(default)s",
    )
    parser.add_argument(
        "--num_loc",
        type=int,
        default=50,
        help="Number of retrieved database images per query, default: %(default)s",
    )
    parser.add_argument(
        "--feature_conf",
        type=str,
        default="joint_xfeat_mlsd_hloc_matcher",
        choices=list(extract_features.confs.keys()),
    )
    parser.add_argument(
        "--retrieval_conf",
        type=str,
        default="gaussvladplusvgg",
        choices=list(extract_features.confs.keys()),
    )
    parser.add_argument(
        "--joint_matcher_conf",
        type=str,
        default="joint_wiregraph",
        choices=list(match_point_line_features.confs.keys()),
    )
    parser.add_argument("--point_match_score_thresh", type=float, default=None)
    parser.add_argument("--line_match_score_thresh", type=float, default=None)
    parser.add_argument("--junction_match_score_thresh", type=float, default=None)
    parser.add_argument("--point_correct_thresh_px", type=float, default=None)
    parser.add_argument("--junction_correct_thresh_px", type=float, default=None)
    parser.add_argument("--matcher_max_keypoints", type=int, default=None)
    parser.add_argument("--matcher_max_lines", type=int, default=None)
    parser.add_argument("--endpoint_merge_thresh_px", type=float, default=None)
    parser.add_argument(
        "--drop_keypoints_near_endpoints_px", type=float, default=None
    )
    parser.add_argument("--ransac_thresh", type=float, default=12.0)
    parser.add_argument("--line_map_min_support_count", type=int, default=3)
    parser.add_argument("--line_map_min_unique_images", type=int, default=2)
    parser.add_argument("--line_map_min_camera_center_distance", type=float, default=0.1)
    parser.add_argument("--min_point_matches", type=int, default=15)
    parser.add_argument("--min_verified_lines", type=int, default=4)
    parser.add_argument("--min_verified_ratio", type=float, default=0.20)
    parser.add_argument("--max_db_images_for_pnp", type=int, default=20)
    parser.add_argument("--endpoint_thresh", type=float, default=18.0)
    parser.add_argument("--center_thresh", type=float, default=14.0)
    parser.add_argument("--angle_thresh", type=float, default=10.0)
    parser.add_argument("--max_refine_points", type=int, default=1024)
    parser.add_argument("--covisibility_clustering", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing features, matches, and line maps when supported.",
    )
    run(parser.parse_args())
