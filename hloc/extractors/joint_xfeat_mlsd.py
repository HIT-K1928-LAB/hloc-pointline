from pathlib import Path

import torch

from ..joint_xfeat_mlsd.inference import (
    JointXFeatPostProcessor,
    build_joint_model_from_checkpoint,
    load_cfg,
    make_decode_config,
    run_joint_inference,
)
from ..utils.base_model import BaseModel


class JointXfeatMlsd(BaseModel):
    default_conf = {
        "config": "/home/hxy/doctor/feature dectect/linedectect/mlsd_pytorchv3/workdir/models/xfeat_mlsd_512_gt_plus_pred_plus_align/cfg.yaml",
        "checkpoint": "/home/hxy/doctor/feature dectect/linedectect/mlsd_pytorchv3/workdir/models/xfeat_mlsd_512_gt_plus_pred_plus_align/best.pth",
        "line_score_thresh": 0.10,
        "line_len_thresh": 8.0,
        "line_top_k": 500,
        "point_top_k": 4096,
        "point_score_thresh": 0.05,
    }
    required_inputs = ["image"]
    detection_noise = 2.0

    def _init(self, conf):
        config_path = Path(conf["config"]).expanduser().resolve()
        checkpoint_path = Path(conf["checkpoint"]).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Joint MLSD config not found: {config_path}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Joint MLSD checkpoint not found: {checkpoint_path}")

        self.cfg = load_cfg(str(config_path))
        self.decode_cfg = make_decode_config(self.cfg, conf)
        self.point_decoder = JointXFeatPostProcessor()
        self.net = build_joint_model_from_checkpoint(self.cfg, str(checkpoint_path))

    def _forward(self, data):
        image = data["image"]
        if image.ndim == 3:
            image = image.unsqueeze(0)
        outputs = run_joint_inference(
            self.net,
            image,
            self.decode_cfg,
            point_decoder=self.point_decoder,
        )
        return {k: torch.from_numpy(v) for k, v in outputs.items() if k != "line_map_size_hw"}
