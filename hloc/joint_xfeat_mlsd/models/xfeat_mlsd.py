import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import BlockTypeA, BlockTypeB, BlockTypeC
from .line_descriptor import LineDescriptorHead
from ..utils.decode import deccode_lines_TP_with_descriptor


DEFAULT_XFEAT_WEIGHTS = '/home/hxy/doctor/xfeat/weights/xfeat.pt'


class BasicLayer(nn.Module):
    """Conv2d -> BatchNorm2d -> ReLU block used by XFeat."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=False):
        super(BasicLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size,
                      padding=padding,
                      stride=stride,
                      dilation=dilation,
                      bias=bias),
            nn.BatchNorm2d(out_channels, affine=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)


class FrozenXFeatModel(nn.Module):
    """Original XFeat model kept intact and frozen for joint point-line extraction."""

    def __init__(self, weights_path=DEFAULT_XFEAT_WEIGHTS, pretrained=True, freeze_backbone=True):
        super(FrozenXFeatModel, self).__init__()
        self.freeze_backbone = freeze_backbone
        self.norm = nn.InstanceNorm2d(1)

        self.skip1 = nn.Sequential(
            nn.AvgPool2d(4, stride=4),
            nn.Conv2d(1, 24, 1, stride=1, padding=0),
        )

        self.block1 = nn.Sequential(
            BasicLayer(1, 4, stride=1),
            BasicLayer(4, 8, stride=2),
            BasicLayer(8, 8, stride=1),
            BasicLayer(8, 24, stride=2),
        )

        self.block2 = nn.Sequential(
            BasicLayer(24, 24, stride=1),
            BasicLayer(24, 24, stride=1),
        )

        self.block3 = nn.Sequential(
            BasicLayer(24, 64, stride=2),
            BasicLayer(64, 64, stride=1),
            BasicLayer(64, 64, 1, padding=0),
        )

        self.block4 = nn.Sequential(
            BasicLayer(64, 64, stride=2),
            BasicLayer(64, 64, stride=1),
            BasicLayer(64, 64, stride=1),
        )

        self.block5 = nn.Sequential(
            BasicLayer(64, 128, stride=2),
            BasicLayer(128, 128, stride=1),
            BasicLayer(128, 128, stride=1),
            BasicLayer(128, 64, 1, padding=0),
        )

        self.block_fusion = nn.Sequential(
            BasicLayer(64, 64, stride=1),
            BasicLayer(64, 64, stride=1),
            nn.Conv2d(64, 64, 1, padding=0),
        )

        self.heatmap_head = nn.Sequential(
            BasicLayer(64, 64, 1, padding=0),
            BasicLayer(64, 64, 1, padding=0),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )

        self.keypoint_head = nn.Sequential(
            BasicLayer(64, 64, 1, padding=0),
            BasicLayer(64, 64, 1, padding=0),
            BasicLayer(64, 64, 1, padding=0),
            nn.Conv2d(64, 65, 1),
        )

        self.fine_matcher = nn.Sequential(
            nn.Linear(128, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 64),
        )

        if pretrained:
            self._load_pretrained_model(weights_path)

        if self.freeze_backbone:
            self.freeze()

    @staticmethod
    def _load_torch_weights(weights_path):
        try:
            return torch.load(weights_path, map_location='cpu', weights_only=True)
        except TypeError:
            return torch.load(weights_path, map_location='cpu')

    def _load_pretrained_model(self, weights_path):
        if not weights_path:
            raise ValueError('weights_path must be provided when pretrained=True')
        if not os.path.isfile(weights_path):
            raise FileNotFoundError('XFeat weights not found: {}'.format(weights_path))

        state_dict = self._load_torch_weights(weights_path)
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=True)
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                'Error while loading XFeat weights. Missing keys: {}. Unexpected keys: {}.'.format(
                    missing_keys, unexpected_keys
                )
            )

    def freeze(self):
        self.freeze_backbone = True
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def unfreeze(self):
        self.freeze_backbone = False
        for param in self.parameters():
            param.requires_grad = True
        return self

    def _unfold2d(self, x, ws=2):
        b, c, h, w = x.shape
        x = x.unfold(2, ws, ws).unfold(3, ws, ws).reshape(b, c, h // ws, w // ws, ws ** 2)
        return x.permute(0, 1, 4, 2, 3).reshape(b, -1, h // ws, w // ws)

    def _forward_backbone(self, x):
        x = x.mean(dim=1, keepdim=True)
        x = self.norm(x)

        x1 = self.block1(x)
        x2 = self.block2(x1 + self.skip1(x))
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)

        return x, x2, x3, x4, x5

    def _forward_point_branch(self, x, x3, x4, x5):
        x4_up = F.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode='bilinear', align_corners=False)
        x5_up = F.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode='bilinear', align_corners=False)
        feats = self.block_fusion(x3 + x4_up + x5_up)
        keypoints = self.keypoint_head(self._unfold2d(x, ws=8))
        heatmap = self.heatmap_head(feats)
        return feats, keypoints, heatmap

    def _forward_impl(self, x, return_point_outputs=False):
        x, x2, x3, x4, x5 = self._forward_backbone(x)
        outputs = {
            'x2': x2,
            'x3': x3,
            'x4': x4,
            'x5': x5,
        }

        if return_point_outputs:
            feats, keypoints, heatmap = self._forward_point_branch(x, x3, x4, x5)
            outputs.update({
                'xfeat_feats': feats,
                'xfeat_keypoints': keypoints,
                'xfeat_heatmap': heatmap,
            })

        return outputs

    def forward(self, x, return_point_outputs=False):
        if self.freeze_backbone:
            with torch.no_grad():
                return self._forward_impl(x, return_point_outputs=return_point_outputs)
        return self._forward_impl(x, return_point_outputs=return_point_outputs)


class BilinearConvTranspose2d(nn.ConvTranspose2d):
    """A conv transpose initialized to bilinear interpolation."""

    def __init__(self, channels, stride, groups=1):
        if isinstance(stride, int):
            stride = (stride, stride)

        assert groups in (1, channels), "Must use no grouping, or one group per channel"

        kernel_size = (2 * stride[0] - 1, 2 * stride[1] - 1)
        padding = (stride[0] - 1, stride[1] - 1)
        super(BilinearConvTranspose2d, self).__init__(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=padding,
            groups=groups,
        )

    def reset_parameters(self):
        nn.init.constant_(self.bias, 0)
        nn.init.constant_(self.weight, 0)
        bilinear_kernel = self.bilinear_kernel(self.stride)
        for i in range(self.in_channels):
            j = i if self.groups == 1 else 0
            self.weight.data[i, j] = bilinear_kernel

    @staticmethod
    def bilinear_kernel(stride):
        num_dims = len(stride)
        shape = (1,) * num_dims
        bilinear_kernel = torch.ones(*shape)

        for channel in range(num_dims):
            channel_stride = stride[channel]
            kernel_size = 2 * channel_stride - 1
            delta = torch.arange(1 - channel_stride, channel_stride)
            channel_filter = 1 - torch.abs(delta / channel_stride)
            shape = [1] * num_dims
            shape[channel] = kernel_size
            bilinear_kernel = bilinear_kernel * channel_filter.view(shape)
        return bilinear_kernel


class XFeat_MLSD(nn.Module):
    """Frozen XFeat point branch plus a trainable MLSD line decoder."""

    def __init__(self, cfg):
        super(XFeat_MLSD, self).__init__()

        weights_path = DEFAULT_XFEAT_WEIGHTS
        if 'xfeat_weights' in cfg.model and cfg.model.xfeat_weights:
            weights_path = cfg.model.xfeat_weights

        self.pretrained_xfeat_backbone = getattr(cfg.model, 'pretrained_xfeat_backbone', True)
        self.freeze_xfeat_backbone = getattr(cfg.model, 'freeze_xfeat_backbone', True)
        self.backbone = FrozenXFeatModel(
            weights_path=weights_path,
            pretrained=self.pretrained_xfeat_backbone,
            freeze_backbone=self.freeze_xfeat_backbone,
        )

        self.block12 = BlockTypeA(in_c1=64, in_c2=64, out_c1=64, out_c2=64)
        self.block13 = BlockTypeB(128, 64)

        self.block14 = BlockTypeA(in_c1=64, in_c2=64, out_c1=64, out_c2=64)
        self.block15 = BlockTypeB(128, 64)

        self.block16 = BlockTypeA(in_c1=24, in_c2=64, out_c1=32, out_c2=32)
        self.block17 = BlockTypeB(64, 64)

        self.block18 = BlockTypeC(64, 16)

        self.with_deconv = cfg.model.with_deconv
        self.with_line_descriptor = getattr(cfg.model, 'with_line_descriptor', False)
        self.line_descriptor_dim = getattr(cfg.model, 'line_descriptor_dim', 64)
        descriptor_num_samples = getattr(getattr(cfg, 'loss', None), 'descriptor_num_samples', 5)
        self.line_descriptor_head = None
        if self.with_deconv:
            self.block19 = BilinearConvTranspose2d(16, 2, 1)
            self.block19.reset_parameters()

        if self.with_line_descriptor:
            self.line_descriptor_head = LineDescriptorHead(
                in_channels=16,
                descriptor_dim=self.line_descriptor_dim,
                num_samples=descriptor_num_samples,
            )

    def train(self, mode=True):
        super(XFeat_MLSD, self).train(mode)
        if self.freeze_xfeat_backbone:
            self.backbone.freeze()
        else:
            self.backbone.unfreeze()
        return self

    def _forward_line_decoder(self, c2, c3, c4, c5):
        x = self.block12(c4, c5)
        x = self.block13(x)

        x = self.block14(c3, x)
        x = self.block15(x)

        x = self.block16(c2, x)
        x = self.block17(x)
        x = self.block18(x)

        if self.with_deconv:
            x = self.block19(x)
        else:
            x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=True)

        return x

    def _build_line_outputs(self, line_preds):
        if not self.with_line_descriptor:
            return line_preds

        return {
            'line_preds': line_preds,
            'descriptor_map': self.line_descriptor_head.forward_map(line_preds),
        }

    @staticmethod
    def _get_tp_descriptor_input(line_preds):
        if line_preds.shape[1] >= 12:
            return line_preds[:, 7:12, :, :]
        return line_preds

    def extract_line_descriptors(self, descriptor_map, lines, coord_scale=512.0, num_samples=None):
        if self.line_descriptor_head is None:
            raise RuntimeError('Line descriptor head is disabled for this model.')
        return self.line_descriptor_head.extract_descriptors(
            descriptor_map,
            lines,
            coord_scale=coord_scale,
            num_samples=num_samples,
        )

    def decode_lines_with_descriptors(self, line_preds, descriptor_map, score_thresh=0.1, len_thresh=2,
                                      topk_n=1000, ksize=3, coord_scale=512.0, num_samples=None):
        tp_map = self._get_tp_descriptor_input(line_preds)
        return deccode_lines_TP_with_descriptor(
            tp_map,
            descriptor_map,
            self.extract_line_descriptors,
            score_thresh=score_thresh,
            len_thresh=len_thresh,
            topk_n=topk_n,
            ksize=ksize,
            coord_scale=coord_scale,
            num_samples=num_samples,
        )

    def forward_points(self, x):
        outputs = self.backbone(x, return_point_outputs=True)
        return outputs['xfeat_feats'], outputs['xfeat_keypoints'], outputs['xfeat_heatmap']

    def forward_joint(self, x):
        outputs = self.backbone(x, return_point_outputs=True)
        line_preds = self._forward_line_decoder(outputs['x2'], outputs['x3'], outputs['x4'], outputs['x5'])
        joint_outputs = {
            'xfeat_feats': outputs['xfeat_feats'],
            'xfeat_keypoints': outputs['xfeat_keypoints'],
            'xfeat_heatmap': outputs['xfeat_heatmap'],
        }
        line_outputs = self._build_line_outputs(line_preds)
        if isinstance(line_outputs, dict):
            joint_outputs.update(line_outputs)
        else:
            joint_outputs['line_preds'] = line_outputs
        return joint_outputs

    def forward(self, x, return_joint=False):
        if return_joint:
            return self.forward_joint(x)

        outputs = self.backbone(x, return_point_outputs=False)
        line_preds = self._forward_line_decoder(outputs['x2'], outputs['x3'], outputs['x4'], outputs['x5'])
        return self._build_line_outputs(line_preds)
