import argparse
import pickle
from pathlib import Path
from typing import Dict, Union

import pycolmap
from tqdm import tqdm

from . import logger
from .localize_sfm import QueryLocalizer, do_covisibility_clustering, pose_from_cluster
from .utils.io import get_line_matches
from .utils.parsers import parse_image_lists, parse_retrieval


def rerank_db_names(qname, db_names, point_matches_path, line_matches_path, point_weight=0.6, line_weight=0.4):
    support = []
    for db_name in db_names:
        try:
            from .utils.io import get_matches
            point_matches, _ = get_matches(point_matches_path, qname, db_name)
            num_point_matches = int(len(point_matches))
        except Exception:
            num_point_matches = 0
        try:
            _, _, verified_mask, stats = get_line_matches(line_matches_path, qname, db_name)
            num_verified_lines = int(verified_mask.sum())
            num_candidate_lines = int(stats.get("candidate_matches", 0))
        except Exception:
            num_verified_lines = 0
            num_candidate_lines = 0
        support.append({
            "db_name": db_name,
            "num_point_matches": num_point_matches,
            "num_candidate_line_matches": num_candidate_lines,
            "num_verified_line_matches": num_verified_lines,
        })

    max_points = max((item["num_point_matches"] for item in support), default=0)
    max_lines = max((item["num_verified_line_matches"] for item in support), default=0)
    for item in support:
        norm_points = item["num_point_matches"] / max(max_points, 1)
        norm_lines = item["num_verified_line_matches"] / max(max_lines, 1)
        item["hybrid_score"] = point_weight * norm_points + line_weight * norm_lines

    filtered = [item for item in support if item["num_point_matches"] > 0 or item["num_verified_line_matches"] > 0]
    if not filtered:
        filtered = support
    filtered.sort(key=lambda item: (item["hybrid_score"], item["num_point_matches"], item["num_verified_line_matches"]), reverse=True)
    return [item["db_name"] for item in filtered], support


def main(
    reference_sfm: Union[Path, pycolmap.Reconstruction],
    queries: Path,
    retrieval: Path,
    features: Path,
    matches: Path,
    line_matches: Path,
    results: Path,
    ransac_thresh: int = 12,
    covisibility_clustering: bool = False,
    prepend_camera_name: bool = False,
    config: Dict = None,
    point_weight: float = 0.6,
    line_weight: float = 0.4,
):
    assert retrieval.exists(), retrieval
    assert features.exists(), features
    assert matches.exists(), matches
    assert line_matches.exists(), line_matches

    queries = parse_image_lists(queries, with_intrinsics=True)
    retrieval_dict = parse_retrieval(retrieval)

    logger.info("Reading the 3D model...")
    if not isinstance(reference_sfm, pycolmap.Reconstruction):
        reference_sfm = pycolmap.Reconstruction(reference_sfm)
    db_name_to_id = {img.name: i for i, img in reference_sfm.images.items()}

    config = {"estimation": {"ransac": {"max_error": ransac_thresh}}, **(config or {})}
    localizer = QueryLocalizer(reference_sfm, config)

    cam_from_world = {}
    logs = {
        "features": features,
        "matches": matches,
        "line_matches": line_matches,
        "retrieval": retrieval,
        "loc": {},
    }
    logger.info("Starting hybrid localization...")
    for qname, qcam in tqdm(queries):
        if qname not in retrieval_dict:
            logger.warning(f"No images retrieved for query image {qname}. Skipping...")
            continue
        reranked_names, support = rerank_db_names(qname, retrieval_dict[qname], matches, line_matches, point_weight=point_weight, line_weight=line_weight)
        db_ids = [db_name_to_id[n] for n in reranked_names if n in db_name_to_id]
        if not db_ids:
            logger.warning(f"No database images available after hybrid reranking for {qname}.")
            continue

        if covisibility_clustering:
            clusters = do_covisibility_clustering(db_ids, reference_sfm)
            best_inliers = 0
            best_cluster = None
            logs_clusters = []
            for i, cluster_ids in enumerate(clusters):
                ret, log = pose_from_cluster(localizer, qname, qcam, cluster_ids, features, matches)
                if ret is not None and ret["num_inliers"] > best_inliers:
                    best_cluster = i
                    best_inliers = ret["num_inliers"]
                logs_clusters.append(log)
            if best_cluster is not None:
                ret = logs_clusters[best_cluster]["PnP_ret"]
                cam_from_world[qname] = ret["cam_from_world"]
            logs["loc"][qname] = {
                "db": db_ids,
                "best_cluster": best_cluster,
                "log_clusters": logs_clusters,
                "covisibility_clustering": covisibility_clustering,
                "hybrid_support": support,
            }
        else:
            ret, log = pose_from_cluster(localizer, qname, qcam, db_ids, features, matches)
            if ret is not None:
                cam_from_world[qname] = ret["cam_from_world"]
            else:
                closest = reference_sfm.images[db_ids[0]]
                cam_from_world[qname] = closest.cam_from_world
            log["covisibility_clustering"] = covisibility_clustering
            log["hybrid_support"] = support
            logs["loc"][qname] = log

    logger.info(f"Localized {len(cam_from_world)} / {len(queries)} images.")
    logger.info(f"Writing poses to {results}...")
    with open(results, "w") as f:
        for query, t in cam_from_world.items():
            qvec = " ".join(map(str, t.rotation.quat[[3, 0, 1, 2]]))
            tvec = " ".join(map(str, t.translation))
            name = query.split("/")[-1]
            if prepend_camera_name:
                name = query.split("/")[-2] + "/" + name
            f.write(f"{name} {qvec} {tvec}\n")

    logs_path = f"{results}_logs.pkl"
    logger.info(f"Writing logs to {logs_path}...")
    with open(logs_path, "wb") as f:
        pickle.dump(logs, f)
    logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_sfm", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--line_matches", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--ransac_thresh", type=float, default=12.0)
    parser.add_argument("--point_weight", type=float, default=0.6)
    parser.add_argument("--line_weight", type=float, default=0.4)
    parser.add_argument("--covisibility_clustering", action="store_true")
    parser.add_argument("--prepend_camera_name", action="store_true")
    args = parser.parse_args()
    main(**args.__dict__)
