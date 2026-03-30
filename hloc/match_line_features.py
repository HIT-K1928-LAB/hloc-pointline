import argparse
import pprint
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Union

import h5py
import numpy as np
import torch
from tqdm import tqdm

from . import logger, matchers
from .match_features import FeaturePairsDataset, WorkQueue, find_unique_new_pairs
from .utils.base_model import dynamic_load
from .utils.parsers import names_to_pair, parse_retrieval


confs = {
    "joint_line_strict": {
        "output": "line-matches-joint-strict",
        "model": {
            "name": "line_nearest_neighbor",
            "precision_preset": "strict",
            "ratio_threshold": 0.82,
            "min_similarity": 0.75,
        },
    },
    "joint_line_very_strict": {
        "output": "line-matches-joint-very-strict",
        "model": {
            "name": "line_nearest_neighbor",
            "precision_preset": "very_strict",
        },
    },
}


def writer_fn(inp, match_path):
    pair, pred = inp
    with h5py.File(str(match_path), "a", libver="latest") as fd:
        if pair in fd:
            del fd[pair]
        grp = fd.create_group(pair)
        grp.create_dataset(
            "line_matches0",
            data=pred["line_matches0"][0].cpu().numpy().astype(np.int32),
        )
        grp.create_dataset(
            "line_matching_scores0",
            data=pred["line_matching_scores0"][0].cpu().numpy().astype(np.float32),
        )
        grp.create_dataset(
            "line_verified_mask0",
            data=pred["line_verified_mask0"][0].cpu().numpy().astype(np.uint8),
        )
        grp.create_dataset(
            "line_match_stats",
            data=pred["line_match_stats"][0].cpu().numpy().astype(np.int32),
        )
        optional_fields = {
            "line_candidate_matches": np.int32,
            "line_verified_matches": np.int32,
            "line_verified_ratio": np.float32,
            "line_mean_verified_similarity": np.float32,
            "line_mean_verified_endpoint_error": np.float32,
            "line_mean_verified_center_error": np.float32,
            "line_mean_verified_angle_error_deg": np.float32,
        }
        for key, dtype in optional_fields.items():
            if key in pred:
                grp.create_dataset(key, data=pred[key][0].cpu().numpy().astype(dtype))


def main(
    conf: Dict,
    pairs: Path,
    features: Union[Path, str],
    export_dir: Optional[Path] = None,
    matches: Optional[Path] = None,
    features_ref: Optional[Path] = None,
    overwrite: bool = False,
) -> Path:
    if isinstance(features, Path) or Path(features).exists():
        features_q = Path(features)
        if matches is None:
            raise ValueError(
                "Either provide both features and matches as Path or both as names."
            )
    else:
        if export_dir is None:
            raise ValueError(
                "Provide an export_dir if features is not a file path."
            )
        features_q = Path(export_dir, str(features) + ".h5")
        if matches is None:
            matches = Path(export_dir, f"{features}_{conf['output']}_{pairs.stem}.h5")

    if features_ref is None:
        features_ref = features_q

    match_from_paths(
        conf,
        pairs,
        matches,
        features_q,
        Path(features_ref),
        overwrite,
    )
    return matches


@torch.no_grad()
def match_from_paths(
    conf: Dict,
    pairs_path: Path,
    match_path: Path,
    feature_path_q: Path,
    feature_path_ref: Path,
    overwrite: bool = False,
) -> Path:
    logger.info(
        "Matching line features with configuration:\n%s", pprint.pformat(conf)
    )
    if not feature_path_q.exists():
        raise FileNotFoundError(f"Query feature file {feature_path_q}.")
    if not feature_path_ref.exists():
        raise FileNotFoundError(f"Reference feature file {feature_path_ref}.")
    match_path.parent.mkdir(exist_ok=True, parents=True)

    pairs = parse_retrieval(pairs_path)
    pairs = [(q, r) for q, rs in pairs.items() for r in rs]
    pairs = find_unique_new_pairs(pairs, None if overwrite else match_path)
    if len(pairs) == 0:
        logger.info("Skipping line matching.")
        return match_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Model = dynamic_load(matchers, conf["model"]["name"])
    model = Model(conf["model"]).eval().to(device)

    dataset = FeaturePairsDataset(pairs, feature_path_q, feature_path_ref)
    loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=1,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    writer_queue = WorkQueue(partial(writer_fn, match_path=match_path), 2)

    for idx, data in enumerate(tqdm(loader, smoothing=0.1)):
        data = {
            k: v if k.startswith("image") else v.to(device, non_blocking=True)
            for k, v in data.items()
        }
        pred = model(data)
        pair = pairs[idx]
        writer_queue.put((names_to_pair(pair[0], pair[1]), pred))

    writer_queue.join()
    return match_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--export_dir", type=Path)
    parser.add_argument("--matches", type=Path)
    parser.add_argument("--features_ref", type=Path)
    parser.add_argument(
        "--conf",
        type=str,
        default="joint_line_strict",
        choices=list(confs.keys()),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(
        confs[args.conf],
        args.pairs,
        args.features,
        args.export_dir,
        args.matches,
        args.features_ref,
        args.overwrite,
    )
