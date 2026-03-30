# Joint Point-Line HLOC Integration

## Overview

This HLOC integration embeds the checkpoint-only `XFeat_MLSD` joint model directly inside `hloc`.
A single forward pass now exports:

- `keypoints`
- `scores`
- `descriptors`
- `line_segments`
- `line_scores`
- `line_descriptors`
- `line_centers`

The final camera pose is still estimated with the standard point-based HLOC / pycolmap pipeline.
Line matches are used for:

- database candidate reranking
- weak pair filtering
- localization diagnostics

## New components

- `hloc/extractors/joint_xfeat_mlsd.py`
- `hloc/joint_xfeat_mlsd/`
- `hloc/matchers/line_nearest_neighbor.py`
- `hloc/match_line_features.py`
- `hloc/localize_sfm_hybrid.py`

## Feature extraction

Use the new extractor config `joint_xfeat_mlsd_hloc` from `hloc/extract_features.py`.
It loads the point-line model only from the MLSD checkpoint and does not read an external `xfeat.pt`.

Example:

```bash
python -m hloc.extract_features   --image_dir /path/to/images   --export_dir /path/to/outputs   --conf joint_xfeat_mlsd_hloc
```

The feature HDF5 now stores the standard point keys plus these line keys in the same image group:

- `line_segments`
- `line_scores`
- `line_descriptors`
- `line_centers`

## Point matching and line matching

Point matching stays unchanged and still uses `hloc.match_features`.

Line matching is handled by the new entrypoint:

```bash
python -m hloc.match_line_features   --pairs /path/to/pairs.txt   --features /path/to/features.h5   --matches /path/to/line_matches.h5   --conf joint_line_strict
```

The line match HDF5 stores:

- `line_matches0`
- `line_matching_scores0`
- `line_verified_mask0`
- `line_match_stats`

`joint_line_strict` is the default recommended preset for localization.
It prefers fewer but cleaner lines and stricter geometric verification.

## Hybrid localization

The new entrypoint `hloc/localize_sfm_hybrid.py` reranks retrieved database images with a hybrid support score:

- `0.6 * normalized point matches`
- `0.4 * normalized verified line matches`

The final pose is still solved with point correspondences only.

Example:

```bash
python -m hloc.localize_sfm_hybrid   --reference_sfm /path/to/sfm   --queries /path/to/queries.txt   --features /path/to/features.h5   --matches /path/to/point_matches.h5   --line_matches /path/to/line_matches.h5   --retrieval /path/to/pairs-query-db.txt   --results /path/to/results.txt
```

## Notes

- The first version does not build a 3D line map.
- Lines do not enter the pycolmap pose solver directly.
- The joint extractor automatically pads arbitrary image sizes to a valid network stride and clips outputs back to the original image region.
- If you do not run `match_line_features.py`, the point-only HLOC pipeline still works unchanged.
