
import argparse
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
    localize_sfm,
    localize_sfm_hybrid,
    localize_sfm_point_line_v2,
    logger,
    match_features,
    match_line_features,
    pairs_from_covisibility,
    pairs_from_retrieval,
    triangulation,
)

# cd "/home/hxy/doctor/feature dectect/hloc/Hierarchical-Localization"

# python -m hloc.pipelines.Aachen.pipeline_joint_point_line_v2 \
#   --dataset datasets/aachen \
#   --outputs outputs/aachen_joint_point_line_v2 \
#   --num_covis 20 \
#   --num_loc 50 \
#   --feature_conf joint_xfeat_mlsd_hloc \
#   --retrieval_conf gaussvladplusvgg \
#   --matcher_conf NN-mutual \
#   --line_matcher_conf joint_line_strict

# cd "/home/hxy/doctor/feature dectect/hloc/Hierarchical-Localization"

# python -m hloc.pipelines.Aachen.pipeline_joint_point_line_v2 \
#   --dataset datasets/aachen \
#   --outputs /media/hxy/PortableSSD/output \
#   --num_covis 20 \
#   --num_loc 50 \
#   --feature_conf joint_xfeat_mlsd_hloc \
#   --retrieval_conf gaussvladplusvgg \
#   --matcher_conf NN-mutual \
#   --line_matcher_conf joint_line_strict
def _safe_mean(values):
    return float(mean(values)) if values else 0.0


def _safe_median(values):
    return float(median(values)) if values else 0.0


def _flatten_localizer_log(log: Dict) -> Dict:
    if 'log_clusters' not in log:
        return log
    best_cluster = log.get('best_cluster')
    if best_cluster is None:
        clusters = log.get('log_clusters', [])
        return clusters[0] if clusters else {}
    clusters = log.get('log_clusters', [])
    if 0 <= best_cluster < len(clusters):
        return clusters[best_cluster]
    return {}


def summarize_localization_logs(logs_path: Path, include_hybrid: bool = False) -> Dict:
    with open(logs_path, 'rb') as f:
        logs = pickle.load(f)

    loc_logs = logs.get('loc', {})
    successful_pnp = 0
    pnp_inliers = []
    total_matches = []
    top_verified_lines = []
    top_candidate_lines = []
    queries_with_line_support = 0

    for log in loc_logs.values():
        main_log = _flatten_localizer_log(log)
        ret = main_log.get('PnP_ret')
        if ret is not None:
            successful_pnp += 1
            if isinstance(ret, dict) and 'num_inliers' in ret:
                pnp_inliers.append(int(ret['num_inliers']))
        if 'num_matches' in main_log:
            total_matches.append(int(main_log['num_matches']))

        if include_hybrid:
            support = log.get('hybrid_support', [])
            if support:
                top = support[0]
                top_verified_lines.append(int(top.get('num_verified_line_matches', 0)))
                top_candidate_lines.append(int(top.get('num_candidate_line_matches', 0)))
                if any(int(item.get('num_verified_line_matches', 0)) > 0 for item in support):
                    queries_with_line_support += 1

    summary = {
        'total_queries': int(len(loc_logs)),
        'successful_pnp_queries': int(successful_pnp),
        'successful_pnp_ratio': float(successful_pnp / max(len(loc_logs), 1)),
        'mean_pnp_inliers': _safe_mean(pnp_inliers),
        'median_pnp_inliers': _safe_median(pnp_inliers),
        'mean_total_2d3d_matches': _safe_mean(total_matches),
        'median_total_2d3d_matches': _safe_median(total_matches),
    }
    if include_hybrid:
        summary.update({
            'queries_with_verified_line_support': int(queries_with_line_support),
            'queries_with_verified_line_support_ratio': float(queries_with_line_support / max(len(loc_logs), 1)),
            'mean_top_reranked_verified_line_matches': _safe_mean(top_verified_lines),
            'median_top_reranked_verified_line_matches': _safe_median(top_verified_lines),
            'mean_top_reranked_candidate_line_matches': _safe_mean(top_candidate_lines),
            'median_top_reranked_candidate_line_matches': _safe_median(top_candidate_lines),
        })
    return summary


def summarize_v2_logs(logs_path: Path) -> Dict:
    with open(logs_path, 'rb') as f:
        logs = pickle.load(f)

    loc_logs = logs.get('loc', {})
    successful_pnp = 0
    successful_refined = 0
    pnp_inliers = []
    total_matches = []
    line_corr_counts = []
    line_inlier_counts = []
    refinement_gains = []
    queries_with_line_support = 0

    for log in loc_logs.values():
        ret = log.get('PnP_ret')
        if ret is not None:
            successful_pnp += 1
            if isinstance(ret, dict) and 'num_inliers' in ret:
                pnp_inliers.append(int(ret['num_inliers']))
        total_matches.append(int(log.get('num_matches', 0)))
        line_corr = int(log.get('line_correspondence_count', 0))
        line_inliers = int(log.get('line_inlier_count', 0))
        line_corr_counts.append(line_corr)
        line_inlier_counts.append(line_inliers)
        if line_corr > 0:
            queries_with_line_support += 1
        if bool(log.get('line_refinement_applied', False)):
            successful_refined += 1
            before = float(log.get('line_residual_before_refine', 0.0))
            after = float(log.get('line_residual_after_refine', before))
            refinement_gains.append(before - after)

    return {
        'total_queries': int(len(loc_logs)),
        'successful_pnp_queries': int(successful_pnp),
        'successful_pnp_ratio': float(successful_pnp / max(len(loc_logs), 1)),
        'successful_pose_after_refinement_queries': int(successful_refined),
        'mean_pnp_inliers': _safe_mean(pnp_inliers),
        'median_pnp_inliers': _safe_median(pnp_inliers),
        'mean_total_2d3d_matches': _safe_mean(total_matches),
        'median_total_2d3d_matches': _safe_median(total_matches),
        'queries_with_verified_line_support': int(queries_with_line_support),
        'queries_with_verified_line_support_ratio': float(queries_with_line_support / max(len(loc_logs), 1)),
        'mean_verified_line_correspondences': _safe_mean(line_corr_counts),
        'median_verified_line_correspondences': _safe_median(line_corr_counts),
        'mean_line_inliers': _safe_mean(line_inlier_counts),
        'median_line_inliers': _safe_median(line_inlier_counts),
        'mean_line_refinement_gain': _safe_mean(refinement_gains),
        'median_line_refinement_gain': _safe_median(refinement_gains),
    }


def compare_point_hybrid_v2(point_logs_path: Path, hybrid_logs_path: Path, v2_logs_path: Path, residual_improvement_eps: float = 1e-3) -> Dict:
    with open(point_logs_path, 'rb') as f:
        point_logs = pickle.load(f).get('loc', {})
    with open(hybrid_logs_path, 'rb') as f:
        hybrid_logs = pickle.load(f).get('loc', {})
    with open(v2_logs_path, 'rb') as f:
        v2_logs = pickle.load(f).get('loc', {})

    all_queries = sorted(set(point_logs.keys()) | set(hybrid_logs.keys()) | set(v2_logs.keys()))
    helped = []
    hurt = []

    for qname in all_queries:
        point_log = _flatten_localizer_log(point_logs.get(qname, {}))
        v2_log = v2_logs.get(qname, {})

        point_success = point_log.get('PnP_ret') is not None
        v2_success = v2_log.get('PnP_ret') is not None
        refined = bool(v2_log.get('line_refinement_applied', False))
        before = float(v2_log.get('line_residual_before_refine', 0.0))
        after = float(v2_log.get('line_residual_after_refine', before))

        improved = refined and (after + residual_improvement_eps < before)
        degraded = refined and (after > before + residual_improvement_eps)

        if (not point_success and v2_success) or (point_success and v2_success and improved):
            helped.append(qname)
        elif (point_success and not v2_success) or (point_success and v2_success and degraded):
            hurt.append(qname)

    return {
        'queries_helped_by_lines': helped,
        'queries_hurt_by_lines': hurt,
        'num_queries_helped_by_lines': int(len(helped)),
        'num_queries_hurt_by_lines': int(len(hurt)),
    }


def make_delta_dict(base_summary: Dict, target_summary: Dict) -> Dict:
    delta = {}
    keys = sorted(set(base_summary.keys()) & set(target_summary.keys()))
    for key in keys:
        base_value = base_summary[key]
        target_value = target_summary[key]
        if isinstance(base_value, (int, float)) and isinstance(target_value, (int, float)):
            delta[key] = target_value - base_value
    return delta


def run(args):
    dataset = args.dataset
    images = dataset / 'images_upright'
    outputs = args.outputs
    outputs.mkdir(parents=True, exist_ok=True)

    sift_sfm = outputs / 'sfm_sift'
    reference_sfm = outputs / 'sfm_joint-xfeat-mlsd+NN'
    sfm_pairs = outputs / f'pairs-db-covis{args.num_covis}.txt'
    loc_pairs = outputs / f'pairs-query-netvlad{args.num_loc}.txt'

    results_point = outputs / f'Aachen_hloc_joint-xfeat-mlsd+NN_netvlad{args.num_loc}_point.txt'
    results_hybrid = outputs / f'Aachen_hloc_joint-xfeat-mlsd+NN_netvlad{args.num_loc}_hybrid.txt'
    results_v2 = outputs / f'Aachen_hloc_joint-xfeat-mlsd+NN_netvlad{args.num_loc}_point-line-v2.txt'
    comparison_json = outputs / f'Aachen_hloc_joint-xfeat-mlsd+NN_netvlad{args.num_loc}_comparison_v2.json'
    line_map_path = outputs / 'line_map.h5'
    line_assignment_path = outputs / 'line_track_assignments.h5'

    logger.info('Configs for feature extractors\n%s', pformat(extract_features.confs))
    logger.info('Configs for feature matchers\n%s', pformat(match_features.confs))
    logger.info('Configs for line matchers\n%s', pformat(match_line_features.confs))

    retrieval_conf = extract_features.confs[args.retrieval_conf]
    feature_conf = extract_features.confs[args.feature_conf]
    matcher_conf = match_features.confs[args.matcher_conf]
    line_matcher_conf = match_line_features.confs[args.line_matcher_conf]

    logger.info('Extracting joint point-line features...')
    features = extract_features.main(feature_conf, images, outputs, overwrite=args.overwrite)

    logger.info('Preparing reference SfM scaffold from Aachen NVM...')
    colmap_from_nvm.main(
        dataset / '3D-models/aachen_cvpr2018_db.nvm',
        dataset / '3D-models/database_intrinsics.txt',
        dataset / 'aachen.db',
        sift_sfm,
    )

    logger.info('Generating covisibility pairs for triangulation...')
    pairs_from_covisibility.main(sift_sfm, sfm_pairs, num_matched=args.num_covis)

    logger.info('Matching point features for SfM triangulation...')
    sfm_matches = match_features.main(
        matcher_conf,
        sfm_pairs,
        feature_conf['output'],
        outputs,
        overwrite=args.overwrite,
    )

    logger.info('Triangulating reference SfM model from joint point features...')
    triangulation.main(reference_sfm, sift_sfm, images, sfm_pairs, features, sfm_matches)

    sfm_line_matches = outputs / f'{Path(features).stem}_{line_matcher_conf["output"]}_{sfm_pairs.stem}.h5'
    logger.info('Matching line features for database-database pairs...')
    match_line_features.main(
        line_matcher_conf,
        sfm_pairs,
        features,
        matches=sfm_line_matches,
        overwrite=args.overwrite,
    )

    if args.overwrite or not line_map_path.exists() or not line_assignment_path.exists():
        logger.info('Building 3D line map...')
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
        logger.info('Reusing existing 3D line map at %s', line_map_path)
        line_map_summary = line_mapping.summarize_line_map(line_map_path)

    logger.info('Extracting retrieval descriptors...')
    global_descriptors = extract_features.main(retrieval_conf, images, outputs, overwrite=args.overwrite)

    logger.info('Generating retrieval pairs for Aachen queries...')
    pairs_from_retrieval.main(
        global_descriptors,
        loc_pairs,
        args.num_loc,
        query_prefix='query',
        db_model=reference_sfm,
    )

    logger.info('Matching point features for localization pairs...')
    loc_matches = match_features.main(
        matcher_conf,
        loc_pairs,
        feature_conf['output'],
        outputs,
        overwrite=args.overwrite,
    )

    loc_line_matches = outputs / f'{Path(features).stem}_{line_matcher_conf["output"]}_{loc_pairs.stem}.h5'
    logger.info('Matching line features for localization pairs...')
    match_line_features.main(
        line_matcher_conf,
        loc_pairs,
        features,
        matches=loc_line_matches,
        overwrite=args.overwrite,
    )

    query_list = dataset / 'queries/*_time_queries_with_intrinsics.txt'

    # logger.info('Running point-only Aachen localization baseline...')
    # localize_sfm.main(
    #     reference_sfm,
    #     query_list,
    #     loc_pairs,
    #     features,
    #     loc_matches,
    #     results_point,
    #     ransac_thresh=args.ransac_thresh,
    #     covisibility_clustering=args.covisibility_clustering,
    # )

    # logger.info('Running point-line hybrid V1 Aachen localization...')
    # localize_sfm_hybrid.main(
    #     reference_sfm,
    #     query_list,
    #     loc_pairs,
    #     features,
    #     loc_matches,
    #     loc_line_matches,
    #     results_hybrid,
    #     ransac_thresh=args.ransac_thresh,
    #     covisibility_clustering=args.covisibility_clustering,
    #     point_weight=args.point_weight,
    #     line_weight=args.line_weight,
    # )

    logger.info('Running point-line V2 Aachen localization with 3D line map...')
    localize_sfm_point_line_v2.main(
        reference_sfm=reference_sfm,
        queries=query_list,
        retrieval=loc_pairs,
        features=features,
        matches=loc_matches,
        line_matches=loc_line_matches,
        line_map=line_map_path,
        line_track_assignments=line_assignment_path,
        results=results_v2,
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

    point_logs = Path(f'{results_point}_logs.pkl')
    hybrid_logs = Path(f'{results_hybrid}_logs.pkl')
    v2_logs = Path(f'{results_v2}_logs.pkl')

    point_summary = summarize_localization_logs(point_logs, include_hybrid=False)
    hybrid_summary = summarize_localization_logs(hybrid_logs, include_hybrid=True)
    v2_summary = summarize_v2_logs(v2_logs)
    compare_summary = compare_point_hybrid_v2(point_logs, hybrid_logs, v2_logs)

    summary = {
        'dataset': str(dataset),
        'images': str(images),
        'outputs': str(outputs),
        'feature_conf': args.feature_conf,
        'retrieval_conf': args.retrieval_conf,
        'matcher_conf': args.matcher_conf,
        'line_matcher_conf': args.line_matcher_conf,
        'num_covis': int(args.num_covis),
        'num_loc': int(args.num_loc),
        'ransac_thresh': float(args.ransac_thresh),
        'point_weight': float(args.point_weight),
        'line_weight': float(args.line_weight),
        'line_map_summary': line_map_summary,
        'paths': {
            'features': str(features),
            'sfm_pairs': str(sfm_pairs),
            'loc_pairs': str(loc_pairs),
            'sfm_matches': str(sfm_matches),
            'sfm_line_matches': str(sfm_line_matches),
            'loc_matches': str(loc_matches),
            'loc_line_matches': str(loc_line_matches),
            'line_map': str(line_map_path),
            'line_track_assignments': str(line_assignment_path),
            'reference_sfm': str(reference_sfm),
            'results_point': str(results_point),
            'results_hybrid': str(results_hybrid),
            'results_v2': str(results_v2),
            'results_point_logs': str(point_logs),
            'results_hybrid_logs': str(hybrid_logs),
            'results_v2_logs': str(v2_logs),
        },
        'point_only_summary': point_summary,
        'hybrid_v1_summary': hybrid_summary,
        'point_line_v2_summary': v2_summary,
        'delta_hybrid_v1_minus_point': make_delta_dict(point_summary, hybrid_summary),
        'delta_point_line_v2_minus_point': make_delta_dict(point_summary, v2_summary),
        'delta_point_line_v2_minus_hybrid_v1': make_delta_dict(hybrid_summary, v2_summary),
        'v2_line_effect_analysis': compare_summary,
        'note': 'Aachen official pose accuracy is not computed here. This script compares point-only, hybrid-v1, and point-line-v2 localization using HLOC logs, 3D line map statistics, and line-refinement diagnostics.',
    }

    comparison_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info('Wrote comparison summary to %s', comparison_json)
    logger.info('Point-only results: %s', results_point)
    logger.info('Hybrid V1 results: %s', results_hybrid)
    logger.info('Point-line V2 results: %s', results_v2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default='datasets/aachen', help='Path to the Aachen dataset, default: %(default)s')
    parser.add_argument('--outputs', type=Path, default='outputs/aachen_joint_point_line_v2', help='Path to the output directory, default: %(default)s')
    parser.add_argument('--num_covis', type=int, default=20, help='Number of image pairs for SfM triangulation, default: %(default)s')
    parser.add_argument('--num_loc', type=int, default=50, help='Number of retrieved database images per query, default: %(default)s')
    parser.add_argument('--feature_conf', type=str, default='joint_xfeat_mlsd_hloc', choices=list(extract_features.confs.keys()))
    parser.add_argument('--retrieval_conf', type=str, default='gaussvladplusvgg', choices=list(extract_features.confs.keys()))
    parser.add_argument('--matcher_conf', type=str, default='NN-mutual', choices=list(match_features.confs.keys()))
    parser.add_argument('--line_matcher_conf', type=str, default='joint_line_strict', choices=list(match_line_features.confs.keys()))
    parser.add_argument('--point_weight', type=float, default=0.6)
    parser.add_argument('--line_weight', type=float, default=0.4)
    parser.add_argument('--ransac_thresh', type=float, default=12.0)
    parser.add_argument('--line_map_min_support_count', type=int, default=3)
    parser.add_argument('--line_map_min_unique_images', type=int, default=2)
    parser.add_argument('--line_map_min_camera_center_distance', type=float, default=0.1)
    parser.add_argument('--min_point_matches', type=int, default=15)
    parser.add_argument('--min_verified_lines', type=int, default=4)
    parser.add_argument('--min_verified_ratio', type=float, default=0.20)
    parser.add_argument('--max_db_images_for_pnp', type=int, default=20)
    parser.add_argument('--endpoint_thresh', type=float, default=18.0)
    parser.add_argument('--center_thresh', type=float, default=14.0)
    parser.add_argument('--angle_thresh', type=float, default=10.0)
    parser.add_argument('--max_refine_points', type=int, default=1024)
    parser.add_argument('--covisibility_clustering', action='store_true')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing features, matches, and line maps when supported by the underlying HLOC entrypoints.')
    args = parser.parse_args()
    run(args)
