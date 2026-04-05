from yacs.config import CfgNode as CN
__all_ = ['get_cfg_defaults']
##
_C = CN()
_C.sys = CN()
_C.sys.cpu = False
_C.sys.gpus = 1
_C.sys.num_workers = 8
##
_C.datasets = CN()
_C.datasets.name = ''
_C.datasets.input_size = 512
_C.datasets.with_centermap_extend = False


##
_C.model = CN()
_C.model.model_name = ''
_C.model.with_deconv = False
_C.model.num_classes = 1
_C.model.xfeat_weights = ''
_C.model.pretrained_xfeat_backbone = True
_C.model.freeze_xfeat_backbone = True
_C.model.with_line_descriptor = False
_C.model.line_descriptor_dim = 64


##
_C.train = CN()
_C.train.do_train = True
_C.train.batch_size = 48
_C.train.save_dir = ''
_C.train.gradient_accumulation_steps = 1
_C.train.num_train_epochs = 170
_C.train.use_step_lr_policy = False
_C.train.warmup_steps = 200
_C.train.learning_rate = 0.0008
_C.train.dropout = 0.1
_C.train.milestones = [100, 150]
_C.train.milestones_in_epo = True
_C.train.lr_decay_gamma = 0.1
_C.train.weight_decay = 0.000001
_C.train.device_ids_str = "0"
_C.train.device_ids = [0]
_C.train.adam_epsilon = 1e-6
_C.train.early_stop_n = 200
_C.train.device_ids_str = "0"
_C.train.device_ids = [0]
_C.train.num_workers = 8
_C.train.log_steps = 50

_C.train.img_dir = ''
_C.train.label_fn = ''
_C.train.data_cache_dir = ''
_C.train.with_cache = False
_C.train.cache_to_mem = False


_C.train.load_from = ""
##
_C.val = CN()
_C.val.batch_size = 8

_C.val.img_dir = ''
_C.val.label_fn = ''

_C.val.val_after_epoch = 0

_C.loss = CN()
_C.loss.loss_weight_dict_list = []
_C.loss.loss_type = '1*L1'
_C.loss.with_sol_loss = True
_C.loss.with_match_loss = False
_C.loss.with_focal_loss = True
_C.loss.match_sap_thresh = 5.0
_C.loss.focal_loss_level = 0
_C.loss.with_descriptor_loss = False
_C.loss.descriptor_loss_weight = 0.05
_C.loss.gt_descriptor_loss_weight = 0.05
_C.loss.with_matched_pred_descriptor_loss = True
_C.loss.pred_descriptor_loss_weight = 0.20
_C.loss.with_pred_gt_alignment_loss = True
_C.loss.pred_gt_alignment_loss_weight = 0.03
_C.loss.descriptor_num_samples = 5
_C.loss.descriptor_temperature = 0.07
_C.loss.pred_descriptor_warmup_epochs = 5
_C.loss.pred_gt_alignment_warmup_epochs = 10
_C.loss.pred_descriptor_angle_thresh_deg = 15.0
_C.loss.pred_descriptor_center_thresh_scale = 1.5
_C.loss.pred_descriptor_length_ratio_min = 0.5
_C.loss.pred_descriptor_length_ratio_max = 2.0
_C.loss.pred_gt_alignment_min_pairs = 4
_C.loss.pred_gt_alignment_min_match_ratio = 0.2
_C.loss.pred_descriptor_score_thresh = 0.05

_C.decode = CN()
_C.decode.score_thresh = 0.05
_C.decode.len_thresh = 5
_C.decode.top_k = 500

_C.matcher = CN()
_C.matcher.hidden_dim = 256
_C.matcher.num_heads = 4
_C.matcher.num_blocks = 6
_C.matcher.max_keypoints = 1024
_C.matcher.max_lines = 256
_C.matcher.endpoint_merge_thresh_px = 4.0
_C.matcher.drop_keypoints_near_endpoints_px = 4.0
_C.matcher.use_points = True
_C.matcher.use_line_message_passing = True
_C.matcher.variant = 'structured_linegraph'
_C.matcher.junction_feature_source = 'line_descriptor_map'
_C.matcher.use_junction_head = True
_C.matcher.match_score_thresh = 0.20
_C.matcher.point_positive_thresh_px = 3.0
_C.matcher.gt_junction_merge_thresh_px = 4.0
_C.matcher.junction_assign_thresh_px = 6.0
_C.matcher.line_assign_endpoint_thresh_px = 8.0
_C.matcher.line_assign_center_thresh_px = 6.0
_C.matcher.line_assign_angle_thresh_deg = 10.0
_C.matcher.line_length_ratio_min = 0.5
_C.matcher.point_loss_weight = 1.0
_C.matcher.line_loss_weight = 1.0
_C.matcher.junction_loss_weight = 0.5
_C.matcher.descriptor_aux_loss_weight = 1.0
_C.matcher.pred_gt_alignment_loss_weight = 1.0
_C.matcher.learning_rate = 3e-4
_C.matcher.weight_decay = 0.000001
_C.matcher.stage_b_fraction = 0.2
_C.matcher.stage_b_lr_scale = 0.1
_C.matcher.dustbin_init = 1.0
_C.matcher.junction_score_thresh = 0.20


def get_cfg_defaults(merge_from=None):
  cfg = _C.clone()
  if merge_from is not None:
      cfg.merge_from_other_cfg(merge_from)
  return cfg
