import argparse
from pathlib import Path
from pprint import pformat
import numpy as np
from scipy.spatial.transform import Rotation
from ... import (
    colmap_from_nvm,
    extract_features,
    localize_sfm,
    logger,
    match_features,
    pairs_from_covisibility,
    pairs_from_retrieval,
    triangulation,
)
from ...utils.io import get_keypoints, get_matches
from ...utils.geometry import compute_epipolar_errors
from collections import defaultdict
import cv2
import pycolmap
from ...utils.parsers import parse_retrieval

def spatial_consistency_based_recall(reconstruction, retrieval_dict, covis_pairs, features_path, matches_path, num_loc):
    spatial_pairs = {}
    
    for qname, db_names in retrieval_dict.items():
        all_db = set(db_names) | set(covis_pairs.get(qname, []))
        
        inlier_ratios = {}
        for dbname in all_db:
            try:
                matches = get_matches(matches_path, qname, dbname)[0]
            except ValueError:
                continue
                
            kp_q = get_keypoints(features_path, qname)
            kp_db = get_keypoints(features_path, dbname)
            
            if len(matches) > 5:
                E, mask = cv2.findEssentialMat(
                    kp_q[matches[:, 0]],
                    kp_db[matches[:, 1]],
                    focal=1000,
                    pp=(0, 0),
                    method=cv2.RANSAC,
                    prob=0.999,
                    threshold=1.0
                )
                inlier_ratios[dbname] = np.mean(mask) if mask is not None else 0
        
        sorted_spatial = sorted(inlier_ratios.items(), key=lambda x: x[1], reverse=True)
        spatial_pairs[qname] = [db for db, _ in sorted_spatial[:num_loc]]
    
    return spatial_pairs

def covisibility_based_recall(reconstruction, retrieval_dict, num_covis):
    covis_pairs = {}
    
    for qname, db_names in retrieval_dict.items():
        db_ids = [img_id for img_id, img in reconstruction.images.items() if img.name in db_names]
        
        point3D_ids = set()
        for img_id in db_ids:
            image = reconstruction.images[img_id]
            for point2D in image.points2D:
                if point2D.has_point3D():
                    point3D_ids.add(point2D.point3D_id)
        
        covis_counts = defaultdict(int)
        for pid in point3D_ids:
            if pid in reconstruction.points3D:
                for obs in reconstruction.points3D[pid].track.elements:
                    covis_counts[obs.image_id] += 1
        
        sorted_covis = sorted(covis_counts.items(), key=lambda x: x[1], reverse=True)
        new_db_ids = [img_id for img_id, _ in sorted_covis[:num_covis] if img_id not in db_ids]
        
        covis_pairs[qname] = [reconstruction.images[img_id].name for img_id in new_db_ids]
    
    return covis_pairs

def covisibility_spatial_recall(reconstruction, retrieval_dict, features_path, matches_path, num_covis, num_loc, outputs, feature_conf, matcher_conf):
    # 1. 基于共视性的召回
    covis_pairs = covisibility_based_recall(reconstruction, retrieval_dict, num_covis)
    
    # 2. 合并共视性图像对
    covis_pairs_dict = {}
    for q in set(retrieval_dict.keys()) | set(covis_pairs.keys()):
        dbs = set(retrieval_dict.get(q, [])) | set(covis_pairs.get(q, []))
        covis_pairs_dict[q] = list(dbs)
    
    # 3. 计算共视性图像对匹配
    covis_pairs_path = outputs / "pairs_covis.txt"
    with open(covis_pairs_path, 'w') as f:
        for q, db_list in covis_pairs_dict.items():
            for db in db_list:
                f.write(f"{q} {db}\n")
    
    # 关键修改：移除matches_path参数
    covis_matches_path = match_features.main(
        matcher_conf,
        covis_pairs_path,
        feature_conf["output"],
        outputs
    )
    
    # 4. 基于空间一致性的召回
    spatial_pairs = spatial_consistency_based_recall(
        reconstruction, 
        retrieval_dict, 
        covis_pairs,
        features_path,
        covis_matches_path,  # 使用新计算的匹配文件
        num_loc
    )
    
    # 5. 合并所有图像对
    all_pairs = {}
    all_queries = set(retrieval_dict.keys()) | set(covis_pairs.keys()) | set(spatial_pairs.keys())
    for q in all_queries:
        dbs = set()
        dbs.update(retrieval_dict.get(q, []))
        dbs.update(covis_pairs.get(q, []))
        dbs.update(spatial_pairs.get(q, []))
        all_pairs[q] = list(dbs)
    
    # 6. 生成最终匹配对
    new_pairs_path = outputs / "pairs_recall.txt"
    with open(new_pairs_path, 'w') as f:
        for q, db_list in all_pairs.items():
            for db in db_list:
                f.write(f"{q} {db}\n")
    
    # 关键修改：移除matches_path参数
    new_matches_path = match_features.main(
        matcher_conf,
        new_pairs_path,
        feature_conf["output"],
        outputs
    )
    
    return {
        'pairs': new_pairs_path,
        'matches': new_matches_path,
        'covis_pairs': covis_pairs,
        'spatial_pairs': spatial_pairs
    }
    
def run(args):
    dataset = args.dataset
    images = dataset / "images_upright/"

    outputs = args.outputs
    sift_sfm = outputs / "sfm_sift"
    reference_sfm = outputs / "sfm_xfeat+NN"
    sfm_pairs = outputs / f"pairs-db-covis{args.num_covis}.txt"
    loc_pairs = outputs / f"pairs-query-netvlad{args.num_loc}.txt"
    results = outputs / f"Aachen_hloc_xfeat+NN_netvlad{args.num_loc}.txt"

    logger.info("Configs for feature extractors:\n%s", pformat(extract_features.confs))
    logger.info("Configs for feature matchers:\n%s", pformat(match_features.confs))

    retrieval_conf = extract_features.confs["gaussvladplusvgg"]
    feature_conf = extract_features.confs["xfeatplus_aachen"]
    matcher_conf = match_features.confs["NN-mutual"]
    
    features = extract_features.main(feature_conf, images, outputs)
    
    colmap_from_nvm.main(
        dataset / "3D-models/aachen_cvpr2018_db.nvm",
        dataset / "3D-models/database_intrinsics.txt",
        dataset / "aachen.db",
        sift_sfm,
    )
    
    pairs_from_covisibility.main(sift_sfm, sfm_pairs, num_matched=args.num_covis)
    
    sfm_matches = match_features.main(
        matcher_conf, sfm_pairs, feature_conf["output"], outputs
    )
    
    triangulation.main(
        reference_sfm, sift_sfm, images, sfm_pairs, features, sfm_matches
    )
    
    global_descriptors = extract_features.main(retrieval_conf, images, outputs)
    pairs_from_retrieval.main(
        global_descriptors,
        loc_pairs,
        args.num_loc,
        query_prefix="query",
        db_model=reference_sfm,
    )
    
    loc_matches = match_features.main(
        matcher_conf, loc_pairs, feature_conf["output"], outputs
    )

    reference_sfm = pycolmap.Reconstruction(reference_sfm)
    retrieval_dict = parse_retrieval(loc_pairs) 
    
    recall_results = covisibility_spatial_recall(
        reference_sfm,
        retrieval_dict,
        features_path=features,
        matches_path=loc_matches,
        num_covis=args.num_covis,
        num_loc=args.num_loc,
        outputs=outputs,
        feature_conf=feature_conf,
        matcher_conf=matcher_conf
    )
    
    localize_sfm.main(
        reference_sfm,
        dataset / "queries/*_time_queries_with_intrinsics.txt",
        recall_results['pairs'],
        features,
        recall_results['matches'],
        results,
        covisibility_clustering=False,
        recall_data=recall_results
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default="datasets/aachen/",
        help="Path to the dataset, default: %(default)s",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default="outputs/aachen",
        help="Path to the output directory, default: %(default)s",
    )
    parser.add_argument(
        "--num_covis",
        type=int,
        default=20,
        help="Number of image pairs for SfM, default: %(default)s",
    )
    parser.add_argument(
        "--num_loc",
        type=int,
        default=50,
        help="Number of image pairs for loc, default: %(default)s",
    )
    args = parser.parse_args()
    run(args)