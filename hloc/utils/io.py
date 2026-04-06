from pathlib import Path
from typing import Tuple

import cv2
import h5py
import numpy as np

from .parsers import names_to_pair, names_to_pair_old


def read_image(path, grayscale=False):
    if grayscale:
        mode = cv2.IMREAD_GRAYSCALE
    else:
        mode = cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError(f"Cannot read image {path}.")
    if not grayscale and len(image.shape) == 3:
        image = image[:, :, ::-1]  # BGR to RGB
    return image


def list_h5_names(path):
    names = []
    with h5py.File(str(path), "r", libver="latest") as fd:

        def visit_fn(_, obj):
            if isinstance(obj, h5py.Dataset):
                names.append(obj.parent.name.strip("/"))

        fd.visititems(visit_fn)
    return list(set(names))


def get_keypoints(
    path: Path, name: str, return_uncertainty: bool = False
) -> np.ndarray:
    with h5py.File(str(path), "r", libver="latest") as hfile:
        grp = hfile[name]
        dset = grp["keypoints"]
        p = dset.__array__()
        if "junctions" in grp:
            junctions = grp["junctions"].__array__()
            if junctions.size > 0:
                p = np.concatenate(
                    [
                        np.asarray(p, dtype=np.float32),
                        np.asarray(junctions, dtype=np.float32),
                    ],
                    axis=0,
                )
        uncertainty = dset.attrs.get("uncertainty")
    if return_uncertainty:
        return p, uncertainty
    return p


def find_pair(hfile: h5py.File, name0: str, name1: str):
    pair = names_to_pair(name0, name1)
    if pair in hfile:
        return pair, False
    pair = names_to_pair(name1, name0)
    if pair in hfile:
        return pair, True
    # older, less efficient format
    pair = names_to_pair_old(name0, name1)
    if pair in hfile:
        return pair, False
    pair = names_to_pair_old(name1, name0)
    if pair in hfile:
        return pair, True
    raise ValueError(
        f"Could not find pair {(name0, name1)}... "
        "Maybe you matched with a different list of pairs? "
    )


def get_matches(path: Path, name0: str, name1: str) -> Tuple[np.ndarray]:
    with h5py.File(str(path), "r", libver="latest") as hfile:
        pair, reverse = find_pair(hfile, name0, name1)
        matches = hfile[pair]["matches0"].__array__()
        scores = hfile[pair]["matching_scores0"].__array__()
    idx = np.where(matches != -1)[0]
    matches = np.stack([idx, matches[idx]], -1)
    if reverse:
        matches = np.flip(matches, -1)
    scores = scores[idx]
    return matches, scores



def get_lines(path: Path, name: str):
    with h5py.File(str(path), "r", libver="latest") as hfile:
        grp = hfile[name]
        return {
            "line_segments": grp["line_segments"].__array__(),
            "line_scores": grp["line_scores"].__array__() if "line_scores" in grp else np.zeros((0,), dtype=np.float32),
            "line_descriptors": grp["line_descriptors"].__array__(),
            "line_centers": grp["line_centers"].__array__() if "line_centers" in grp else np.zeros((0, 2), dtype=np.float32),
        }


def get_line_matches(path: Path, name0: str, name1: str):
    with h5py.File(str(path), "r", libver="latest") as hfile:
        pair, reverse = find_pair(hfile, name0, name1)
        grp = hfile[pair]
        matches = grp["line_matches0"].__array__()
        scores = grp["line_matching_scores0"].__array__()
        verified = grp["line_verified_mask0"].__array__().astype(bool)
        stats_raw = grp["line_match_stats"].__array__() if "line_match_stats" in grp else np.asarray([0, 0], dtype=np.int32)
        optional_scalars = {}
        for key in [
            "line_candidate_matches",
            "line_verified_matches",
            "line_verified_ratio",
            "line_mean_verified_similarity",
            "line_mean_verified_endpoint_error",
            "line_mean_verified_center_error",
            "line_mean_verified_angle_error_deg",
        ]:
            if key in grp:
                value = grp[key].__array__()
                optional_scalars[key] = float(value.reshape(-1)[0])
    idx = np.where(matches != -1)[0]
    pairs = np.stack([idx, matches[idx]], -1) if len(idx) > 0 else np.zeros((0, 2), dtype=np.int32)
    if reverse and len(pairs) > 0:
        pairs = np.flip(pairs, -1)
    scores = scores[idx]
    verified = verified[idx]
    stats = {
        "candidate_matches": int(stats_raw[0]) if len(stats_raw) > 0 else int(len(pairs)),
        "verified_matches": int(stats_raw[1]) if len(stats_raw) > 1 else int(verified.sum()),
    }
    if "line_candidate_matches" in optional_scalars:
        stats["line_candidate_matches"] = int(optional_scalars["line_candidate_matches"])
    if "line_verified_matches" in optional_scalars:
        stats["line_verified_matches"] = int(optional_scalars["line_verified_matches"])
    for key in [
        "line_verified_ratio",
        "line_mean_verified_similarity",
        "line_mean_verified_endpoint_error",
        "line_mean_verified_center_error",
        "line_mean_verified_angle_error_deg",
    ]:
        if key in optional_scalars:
            stats[key] = float(optional_scalars[key])
    return pairs, scores, verified, stats
