import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from dataclasses import dataclass
from yacs.config import CfgNode as CN

from .cfg.default import get_cfg_defaults
from .models.build_model import build_model

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class SparseInterpolator2d(nn.Module):
    def __init__(self, mode="bicubic", align_corners=False):
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

    @staticmethod
    def normgrid(pos, height, width):
        if height <= 1 or width <= 1:
            return pos.new_zeros(pos.shape)
        scale = torch.tensor([width - 1, height - 1], device=pos.device, dtype=pos.dtype)
        return 2.0 * (pos / scale) - 1.0

    def forward(self, x, pos, height, width):
        if pos.numel() == 0:
            return x.new_zeros((x.shape[0], 0, x.shape[1]))
        grid = self.normgrid(pos, height, width).unsqueeze(-2).to(x.dtype)
        sampled = F.grid_sample(x, grid, mode=self.mode, align_corners=self.align_corners)
        return sampled.permute(0, 2, 3, 1).squeeze(-2)


class JointXFeatPostProcessor:
    def __init__(self):
        self.sparse_interpolator = SparseInterpolator2d("bicubic")
        self.nearest_interpolator = SparseInterpolator2d("nearest")
        self.bilinear_interpolator = SparseInterpolator2d("bilinear")

    @staticmethod
    def get_kpts_heatmap(keypoint_logits, softmax_temp=1.0):
        scores = F.softmax(keypoint_logits * softmax_temp, dim=1)[:, :64]
        batch, _, height, width = scores.shape
        heatmap = scores.permute(0, 2, 3, 1).reshape(batch, height, width, 8, 8)
        heatmap = heatmap.permute(0, 1, 3, 2, 4).reshape(batch, 1, height * 8, width * 8)
        return heatmap

    @staticmethod
    def nms(x, threshold=0.05, kernel_size=5):
        batch, _, _, _ = x.shape
        pad = kernel_size // 2
        local_max = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=pad)(x)
        pos_mask = (x == local_max) & (x > threshold)
        pos_batched = [coords.nonzero()[..., 1:].flip(-1) for coords in pos_mask]
        max_points = max((len(coords) for coords in pos_batched), default=0)
        pos = torch.zeros((batch, max_points, 2), dtype=torch.long, device=x.device)
        for batch_idx, coords in enumerate(pos_batched):
            if len(coords) > 0:
                pos[batch_idx, : len(coords), :] = coords
        return pos

    def decode(self, feats, keypoint_logits, heatmap, top_k=4096, detection_threshold=0.05):
        feats = F.normalize(feats, dim=1)
        heatmap_kpts = self.get_kpts_heatmap(keypoint_logits)
        mkpts = self.nms(heatmap_kpts, threshold=detection_threshold, kernel_size=5)
        batch, _, feat_h, feat_w = feats.shape
        full_h = feat_h * 8
        full_w = feat_w * 8

        if mkpts.shape[1] == 0:
            empty_outputs = []
            for _ in range(batch):
                empty_outputs.append({
                    "keypoints": feats.new_zeros((0, 2)),
                    "scores": feats.new_zeros((0,)),
                    "descriptors": feats.new_zeros((0, feats.shape[1])),
                })
            return empty_outputs

        scores = (
            self.nearest_interpolator(heatmap_kpts, mkpts, full_h, full_w)
            * self.bilinear_interpolator(heatmap, mkpts, full_h, full_w)
        ).squeeze(-1)
        scores[torch.all(mkpts == 0, dim=-1)] = -1

        idxs = torch.argsort(-scores, dim=-1)
        mkpts_x = torch.gather(mkpts[..., 0], -1, idxs)[:, :top_k]
        mkpts_y = torch.gather(mkpts[..., 1], -1, idxs)[:, :top_k]
        mkpts = torch.cat([mkpts_x[..., None], mkpts_y[..., None]], dim=-1).float()
        scores = torch.gather(scores, -1, idxs)[:, :top_k]

        # Match the original XFeat inference path: sparse descriptors are sampled
        # using full-resolution keypoint coordinates and normalized by the same
        # full-resolution image size, while grid_sample handles the mapping to
        # the lower-resolution feature map internally.
        descriptors = self.sparse_interpolator(feats, mkpts, full_h, full_w)
        descriptors = F.normalize(descriptors, dim=-1)

        valid = scores > 0
        outputs = []
        for batch_idx in range(batch):
            outputs.append({
                "keypoints": mkpts[batch_idx][valid[batch_idx]],
                "scores": scores[batch_idx][valid[batch_idx]],
                "descriptors": descriptors[batch_idx][valid[batch_idx]],
            })
        return outputs


@dataclass
class JointDecodeConfig:
    line_score_thresh: float
    line_len_thresh: float
    line_top_k: int
    point_top_k: int
    point_score_thresh: float
    descriptor_num_samples: int


def tensor_to_numpy(tensor):
    if tensor is None:
        return None
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def strip_module_prefix(state_dict):
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict.keys()):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def load_cfg(config_path):
    cfg = get_cfg_defaults()
    with open(config_path, "r") as f:
        raw_cfg = yaml.safe_load(f) or {}
    keep_keys = {"datasets", "model", "loss", "decode"}
    filtered = {key: value for key, value in raw_cfg.items() if key in keep_keys}
    cfg.merge_from_other_cfg(CN(filtered))
    return cfg


def build_joint_model_from_checkpoint(cfg, checkpoint_path):
    cfg = cfg.clone()
    cfg.defrost()
    cfg.model.pretrained_xfeat_backbone = False
    cfg.model.freeze_xfeat_backbone = True
    cfg.model.xfeat_weights = ""
    cfg.freeze()

    model = build_model(cfg)
    checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "net"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    checkpoint = strip_module_prefix(checkpoint)
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    return model


def make_decode_config(cfg, conf):
    return JointDecodeConfig(
        line_score_thresh=float(conf.get("line_score_thresh", cfg.decode.score_thresh)),
        line_len_thresh=float(conf.get("line_len_thresh", cfg.decode.len_thresh)),
        line_top_k=int(conf.get("line_top_k", cfg.decode.top_k)),
        point_top_k=int(conf.get("point_top_k", 4096)),
        point_score_thresh=float(conf.get("point_score_thresh", 0.05)),
        descriptor_num_samples=int(getattr(cfg.loss, "descriptor_num_samples", 5)),
    )




def pad_image_to_stride(image, stride=32):
    if image.ndim != 4:
        raise ValueError("Expected image tensor with shape [B, C, H, W].")
    height = int(image.shape[-2])
    width = int(image.shape[-1])
    pad_h = (stride - height % stride) % stride
    pad_w = (stride - width % stride) % stride
    if pad_h == 0 and pad_w == 0:
        return image, height, width
    padded = F.pad(image, (0, pad_w, 0, pad_h), mode="replicate")
    return padded, height, width


def clip_keypoints_to_image(keypoints, scores, descriptors, image_width, image_height):
    if keypoints.size == 0:
        return keypoints, scores, descriptors
    mask = (keypoints[:, 0] >= 0.0) & (keypoints[:, 0] <= image_width - 1.0) & (keypoints[:, 1] >= 0.0) & (keypoints[:, 1] <= image_height - 1.0)
    return keypoints[mask], scores[mask], descriptors[mask]


def clip_lines_to_image(line_segments, line_scores, line_descriptors, line_centers, image_width, image_height):
    if line_segments.size == 0:
        return line_segments, line_scores, line_descriptors, line_centers
    mask = (line_centers[:, 0] >= 0.0) & (line_centers[:, 0] <= image_width - 1.0) & (line_centers[:, 1] >= 0.0) & (line_centers[:, 1] <= image_height - 1.0)
    line_segments = line_segments[mask]
    line_scores = line_scores[mask]
    line_descriptors = line_descriptors[mask]
    line_centers = line_centers[mask]
    if line_segments.size == 0:
        return line_segments, line_scores, line_descriptors, line_centers
    line_segments[:, [0, 2]] = np.clip(line_segments[:, [0, 2]], 0.0, image_width - 1.0)
    line_segments[:, [1, 3]] = np.clip(line_segments[:, [1, 3]], 0.0, image_height - 1.0)
    line_centers[:, 0] = np.clip(line_centers[:, 0], 0.0, image_width - 1.0)
    line_centers[:, 1] = np.clip(line_centers[:, 1], 0.0, image_height - 1.0)
    return line_segments, line_scores, line_descriptors, line_centers

def normalize_image_for_joint_model(image):
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("Expected image tensor with shape [B, 3, H, W].")
    mean = IMAGENET_MEAN.to(device=image.device, dtype=image.dtype)
    std = IMAGENET_STD.to(device=image.device, dtype=image.dtype)
    return (image - mean) / std


def scale_lines_to_image(lines_model, centers_model, image_width, image_height, line_map_width, line_map_height):
    line_scale = np.asarray([
        image_width / max(float(line_map_width), 1.0),
        image_height / max(float(line_map_height), 1.0),
        image_width / max(float(line_map_width), 1.0),
        image_height / max(float(line_map_height), 1.0),
    ], dtype=np.float32)
    center_scale = np.asarray([
        image_width / max(float(line_map_width), 1.0),
        image_height / max(float(line_map_height), 1.0),
    ], dtype=np.float32)

    if lines_model.size == 0:
        line_segments_xyxy = np.zeros((0, 4), dtype=np.float32)
    else:
        line_segments_xyxy = lines_model.astype(np.float32) * line_scale[None, :]

    if centers_model.size == 0:
        line_centers_xy = np.zeros((0, 2), dtype=np.float32)
    else:
        line_centers_xy = centers_model.astype(np.float32) * center_scale[None, :]

    return line_centers_xy, line_segments_xyxy


def run_joint_inference(model, image, decode_cfg, point_decoder=None):
    if point_decoder is None:
        point_decoder = JointXFeatPostProcessor()

    image, original_height, original_width = pad_image_to_stride(image)
    normalized = normalize_image_for_joint_model(image)
    with torch.inference_mode():
        outputs = model(normalized, return_joint=True)

    required_keys = {"line_preds", "descriptor_map", "xfeat_feats", "xfeat_keypoints", "xfeat_heatmap"}
    if not required_keys.issubset(set(outputs.keys())):
        missing = sorted(required_keys - set(outputs.keys()))
        raise RuntimeError(f"Joint model outputs missing keys: {missing}")

    line_map_height = int(outputs["line_preds"].shape[-2])
    line_map_width = int(outputs["line_preds"].shape[-1])
    coord_scale = (float(line_map_width), float(line_map_height))

    center_pts, lines, _, line_scores, line_desc = model.decode_lines_with_descriptors(
        outputs["line_preds"],
        outputs["descriptor_map"],
        score_thresh=decode_cfg.line_score_thresh,
        len_thresh=decode_cfg.line_len_thresh,
        topk_n=decode_cfg.line_top_k,
        ksize=3,
        coord_scale=coord_scale,
        num_samples=decode_cfg.descriptor_num_samples,
    )

    point_outputs = point_decoder.decode(
        outputs["xfeat_feats"],
        outputs["xfeat_keypoints"],
        outputs["xfeat_heatmap"],
        top_k=decode_cfg.point_top_k,
        detection_threshold=decode_cfg.point_score_thresh,
    )[0]

    center_pts_np = tensor_to_numpy(center_pts).astype(np.float32) if center_pts is not None else np.zeros((0, 2), dtype=np.float32)
    lines_np = tensor_to_numpy(lines).astype(np.float32) if lines is not None else np.zeros((0, 4), dtype=np.float32)
    line_scores_np = tensor_to_numpy(line_scores).astype(np.float32) if line_scores is not None else np.zeros((0,), dtype=np.float32)
    if line_desc is None:
        line_desc_np = np.zeros((0, 64), dtype=np.float32)
    else:
        line_desc_np = tensor_to_numpy(line_desc).astype(np.float32)

    keypoints_np = tensor_to_numpy(point_outputs["keypoints"]).astype(np.float32)
    keypoint_scores_np = tensor_to_numpy(point_outputs["scores"]).astype(np.float32)
    keypoint_desc_np = tensor_to_numpy(point_outputs["descriptors"]).astype(np.float32)

    if keypoint_desc_np.size > 0:
        keypoint_desc_norms = np.linalg.norm(keypoint_desc_np, axis=1)
        if np.all(keypoint_desc_norms < 1e-6):
            raise RuntimeError(
                "Decoded point descriptors are all near zero. "
                "Please check point descriptor sampling coordinates and the "
                "point-branch weights in the joint checkpoint."
            )

    padded_height = int(image.shape[-2])
    padded_width = int(image.shape[-1])
    line_centers_xy, line_segments_xyxy = scale_lines_to_image(
        lines_np,
        center_pts_np,
        image_width=padded_width,
        image_height=padded_height,
        line_map_width=line_map_width,
        line_map_height=line_map_height,
    )
    line_segments_xyxy, line_scores_np, line_desc_np, line_centers_xy = clip_lines_to_image(
        line_segments_xyxy,
        line_scores_np,
        line_desc_np,
        line_centers_xy,
        image_width=original_width,
        image_height=original_height,
    )
    keypoints_np, keypoint_scores_np, keypoint_desc_np = clip_keypoints_to_image(
        keypoints_np,
        keypoint_scores_np,
        keypoint_desc_np,
        image_width=original_width,
        image_height=original_height,
    )

    return {
        "image_size": np.asarray([original_width, original_height], dtype=np.float32),
        "keypoints": keypoints_np,
        "scores": keypoint_scores_np,
        "descriptors": keypoint_desc_np,
        "line_segments": line_segments_xyxy.astype(np.float32),
        "line_scores": line_scores_np,
        "line_descriptors": line_desc_np,
        "line_centers": line_centers_xy.astype(np.float32),
        "line_map_size_hw": np.asarray([line_map_height, line_map_width], dtype=np.int32),
    }
