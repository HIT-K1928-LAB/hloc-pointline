import argparse
import copy
import json
from pathlib import Path
from pprint import pformat

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
from .pipeline_joint_point_line_v3 import summarize_v2_logs


def run(args):
    dataset = args.dataset
    images = dataset / "images_upright"
    outputs = args.outputs
    outputs.mkdir(parents=True, exist_ok=True)

    sift_sfm = outputs / "sfm_sift"
    reference_sfm = outputs / "sfm_joint-xfeat-mlsd+wiregraph-hybrid"
    sfm_pairs = outputs / f"pairs-db-covis{args.num_covis}.txt"
    loc_pairs = outputs / f"pairs-query-netvlad{args.num_loc}.txt"

    results_hybrid = outputs / (
        f"Aachen_hloc_joint-xfeat-mlsd+wiregraph_netvlad{args.num_loc}_point-line-v3-hybrid.txt"
    )
    summary_json = outputs / (
        f"Aachen_hloc_joint-xfeat-mlsd+wiregraph_netvlad{args.num_loc}_summary_v3_hybrid.json"
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
    if args.max_lines_per_image is not None:
        joint_matcher_model_conf["max_lines_per_image"] = int(args.max_lines_per_image)
    if args.endpoint_merge_thresh_px is not None:
        joint_matcher_model_conf["endpoint_merge_thresh_px"] = float(
            args.endpoint_merge_thresh_px
        )
    if args.drop_keypoints_near_endpoints_px is not None:
        joint_matcher_model_conf["drop_keypoints_near_endpoints_px"] = float(
            args.drop_keypoints_near_endpoints_px
        )
    if args.post_min_line_score is not None:
        joint_matcher_model_conf["post_min_line_score"] = float(
            args.post_min_line_score
        )
    if args.post_min_line_length_px is not None:
        joint_matcher_model_conf["post_min_line_length_px"] = float(
            args.post_min_line_length_px
        )
    if args.line_nms_center_thresh is not None:
        joint_matcher_model_conf["line_nms_center_thresh"] = float(
            args.line_nms_center_thresh
        )
    if args.line_nms_angle_thresh_deg is not None:
        joint_matcher_model_conf["line_nms_angle_thresh_deg"] = float(
            args.line_nms_angle_thresh_deg
        )
    if args.line_nms_endpoint_thresh is not None:
        joint_matcher_model_conf["line_nms_endpoint_thresh"] = float(
            args.line_nms_endpoint_thresh
        )

    logger.info("Extracting matcher-compatible joint point-line features...")
    features = extract_features.main(feature_conf, images, outputs, overwrite=args.overwrite)
    logger.info("Preparing junction-aware point augmentation for V3 hybrid...")
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

    logger.info("Running hybrid joint wiregraph matcher for database-database pairs...")
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

    logger.info("Triangulating reference SfM model from hybrid matcher point matches...")
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

    logger.info("Running hybrid joint wiregraph matcher for localization pairs...")
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

    logger.info("Running point-line V3 hybrid Aachen localization with 3D line map...")
    localize_sfm_point_line_v2.main(
        reference_sfm=reference_sfm,
        queries=query_list,
        retrieval=loc_pairs,
        features=features,
        matches=loc_matches,
        line_matches=loc_line_matches,
        line_map=line_map_path,
        line_track_assignments=line_assignment_path,
        results=results_hybrid,
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

    hybrid_logs = Path(f"{results_hybrid}_logs.pkl")
    hybrid_summary = summarize_v2_logs(hybrid_logs)

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
            "results_v3_hybrid": str(results_hybrid),
            "results_v3_hybrid_logs": str(hybrid_logs),
        },
        "point_line_v3_hybrid_summary": hybrid_summary,
        "note": (
            "This hybrid script keeps wiregraph matcher as the main point-line matcher, "
            "applies v2-style line prefiltering and line verification, weakens point/junction "
            "pre-filtering, and summarizes Aachen localization logs."
        ),
    }

    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info("Wrote V3 hybrid summary to %s", summary_json)
    logger.info("Point-line V3 hybrid results: %s", results_hybrid)


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
        default="outputs/aachen_joint_point_line_v3_hybrid",
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
        default="joint_wiregraph_v2style_hybrid",
        choices=list(match_point_line_features.confs.keys()),
    )
    parser.add_argument("--point_match_score_thresh", type=float, default=None)
    parser.add_argument("--line_match_score_thresh", type=float, default=None)
    parser.add_argument("--junction_match_score_thresh", type=float, default=None)
    parser.add_argument("--point_correct_thresh_px", type=float, default=None)
    parser.add_argument("--junction_correct_thresh_px", type=float, default=None)
    parser.add_argument("--matcher_max_keypoints", type=int, default=None)
    parser.add_argument("--matcher_max_lines", type=int, default=None)
    parser.add_argument("--max_lines_per_image", type=int, default=None)
    parser.add_argument("--endpoint_merge_thresh_px", type=float, default=None)
    parser.add_argument(
        "--drop_keypoints_near_endpoints_px", type=float, default=None
    )
    parser.add_argument("--post_min_line_score", type=float, default=None)
    parser.add_argument("--post_min_line_length_px", type=float, default=None)
    parser.add_argument("--line_nms_center_thresh", type=float, default=None)
    parser.add_argument("--line_nms_angle_thresh_deg", type=float, default=None)
    parser.add_argument("--line_nms_endpoint_thresh", type=float, default=None)
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
