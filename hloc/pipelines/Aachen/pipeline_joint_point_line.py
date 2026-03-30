import argparse
import json
import pickle
from pathlib import Path
from pprint import pformat
from statistics import mean, median
from typing import Dict, Optional

from ... import (
    colmap_from_nvm,
    extract_features,
    localize_sfm,
    localize_sfm_hybrid,
    logger,
    match_features,
    match_line_features,
    pairs_from_covisibility,
    pairs_from_retrieval,
    triangulation,
)
# cd "/home/hxy/doctor/feature dectect/hloc/Hierarchical-Localization"

# python -m hloc.pipelines.Aachen.pipeline_joint_point_line \
#   --dataset datasets/aachen \
#   --outputs outputs/aachen_joint_point_line \
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


def make_delta_dict(point_summary: Dict, hybrid_summary: Dict) -> Dict:
    keys = [
        'successful_pnp_queries',
        'successful_pnp_ratio',
        'mean_pnp_inliers',
        'median_pnp_inliers',
        'mean_total_2d3d_matches',
        'median_total_2d3d_matches',
    ]
    delta = {}
    for key in keys:
        if key in point_summary and key in hybrid_summary:
            delta[key] = hybrid_summary[key] - point_summary[key]
    if 'queries_with_verified_line_support' in hybrid_summary:
        delta['queries_with_verified_line_support'] = hybrid_summary['queries_with_verified_line_support']
        delta['queries_with_verified_line_support_ratio'] = hybrid_summary['queries_with_verified_line_support_ratio']
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
    comparison_json = outputs / f'Aachen_hloc_joint-xfeat-mlsd+NN_netvlad{args.num_loc}_comparison.json'

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

    logger.info('Matching line features for localization pairs...')
    line_matches = match_line_features.main(
        line_matcher_conf,
        loc_pairs,
        feature_conf['output'],
        outputs,
        overwrite=args.overwrite,
    )

    query_list = dataset / 'queries/*_time_queries_with_intrinsics.txt'

    logger.info('Running point-only Aachen localization baseline...')
    localize_sfm.main(
        reference_sfm,
        query_list,
        loc_pairs,
        features,
        loc_matches,
        results_point,
        ransac_thresh=args.ransac_thresh,
        covisibility_clustering=args.covisibility_clustering,
    )

    logger.info('Running point-line hybrid Aachen localization...')
    localize_sfm_hybrid.main(
        reference_sfm,
        query_list,
        loc_pairs,
        features,
        loc_matches,
        line_matches,
        results_hybrid,
        ransac_thresh=args.ransac_thresh,
        covisibility_clustering=args.covisibility_clustering,
        point_weight=args.point_weight,
        line_weight=args.line_weight,
    )

    point_logs = Path(f'{results_point}_logs.pkl')
    hybrid_logs = Path(f'{results_hybrid}_logs.pkl')
    point_summary = summarize_localization_logs(point_logs, include_hybrid=False)
    hybrid_summary = summarize_localization_logs(hybrid_logs, include_hybrid=True)
    delta = make_delta_dict(point_summary, hybrid_summary)

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
        'paths': {
            'features': str(features),
            'sfm_pairs': str(sfm_pairs),
            'loc_pairs': str(loc_pairs),
            'sfm_matches': str(sfm_matches),
            'loc_matches': str(loc_matches),
            'line_matches': str(line_matches),
            'reference_sfm': str(reference_sfm),
            'results_point': str(results_point),
            'results_hybrid': str(results_hybrid),
            'results_point_logs': str(point_logs),
            'results_hybrid_logs': str(hybrid_logs),
        },
        'point_only_summary': point_summary,
        'hybrid_summary': hybrid_summary,
        'delta_hybrid_minus_point': delta,
        'note': 'Aachen official pose accuracy is not computed here. This script compares point-only and point-line hybrid localization using HLOC logs and hybrid support statistics.',
    }

    comparison_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info('Wrote comparison summary to %s', comparison_json)
    logger.info('Point-only results: %s', results_point)
    logger.info('Hybrid results: %s', results_hybrid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default='datasets/aachen', help='Path to the Aachen dataset, default: %(default)s')
    parser.add_argument('--outputs', type=Path, default='outputs/aachen_joint_point_line', help='Path to the output directory, default: %(default)s')
    parser.add_argument('--num_covis', type=int, default=20, help='Number of image pairs for SfM triangulation, default: %(default)s')
    parser.add_argument('--num_loc', type=int, default=50, help='Number of retrieved database images per query, default: %(default)s')
    parser.add_argument('--feature_conf', type=str, default='joint_xfeat_mlsd_hloc', choices=list(extract_features.confs.keys()))
    parser.add_argument('--retrieval_conf', type=str, default='gaussvladplusvgg', choices=list(extract_features.confs.keys()))
    parser.add_argument('--matcher_conf', type=str, default='NN-mutual', choices=list(match_features.confs.keys()))
    parser.add_argument('--line_matcher_conf', type=str, default='joint_line_strict', choices=list(match_line_features.confs.keys()))
    parser.add_argument('--point_weight', type=float, default=0.6)
    parser.add_argument('--line_weight', type=float, default=0.4)
    parser.add_argument('--ransac_thresh', type=float, default=12.0)
    parser.add_argument('--covisibility_clustering', action='store_true')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing features and matches when supported by the underlying HLOC entrypoints.')
    args = parser.parse_args()
    run(args)
