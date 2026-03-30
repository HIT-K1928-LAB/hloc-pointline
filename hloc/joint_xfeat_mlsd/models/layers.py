import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.ops import deform_conv2d

class Block7(nn.Module):
    def __init__(self,  in_c1, out_c1):
        super(Block7, self).__init__()
        self.conv_crossh = nn.Sequential(
            nn.Conv2d(in_c1, out_c1, (7, 1), 1, (3, 0)),
            nn.BatchNorm2d(out_c1),
            nn.ReLU(inplace=True)
        )
        self.conv_crossv = nn.Sequential(
            nn.Conv2d(in_c1, out_c1, (1, 7), 1, (0, 3)),
            nn.BatchNorm2d(out_c1),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Conv2d(2*out_c1, out_c1, kernel_size=1)

    def forward(self, x):
        b = self.conv_crossh(x)
        a = self.conv_crossv(x)
        x = self.conv1(torch.cat((a, b), dim=1))
        return x

# class BlockTypeA(nn.Module):
#     def __init__(self, in_c1, in_c2, out_c1, out_c2, upscale = True):
#         super(BlockTypeA, self).__init__()
#         self.conv1 = Block7(in_c2, out_c2)
#         self.conv2 = Block7(in_c1, out_c1)
#         self.upscale = upscale

#     def forward(self, a, b):
#         b = self.conv1(b)
#         a = self.conv2(a)
#         if self.upscale:
#             b = F.interpolate(b, scale_factor=2.0, mode='bilinear', align_corners=True)
#         return torch.cat((a, b), dim=1)

class BlockTypeA(nn.Module):
    def __init__(self, in_c1, in_c2, out_c1, out_c2, upscale = True):
        super(BlockTypeA, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c2, out_c2, kernel_size=1),
            nn.BatchNorm2d(out_c2),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c1, out_c1, kernel_size=1),
            nn.BatchNorm2d(out_c1),
            nn.ReLU(inplace=True)
        )
        self.upscale = upscale

    def forward(self, a, b):
        b = self.conv1(b)
        a = self.conv2(a)
        if self.upscale:
            b = F.interpolate(b, scale_factor=2.0, mode='bilinear', align_corners=True)
        return torch.cat((a, b), dim=1)
    
class DeformableLayer(nn.Module):
    """ 
    自适应可变形卷积层：包含偏移量学习网络
    适用于处理视角剧烈变化和几何形变 
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        
        # 偏移量学习分支：为每个采样点学习 (x, y) 偏移
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, 
                                     kernel_size=kernel_size, stride=stride, padding=padding)
        
        # 权重/掩码分支 (DCNv2)：学习采样点的权重
        self.mask_conv = nn.Conv2d(in_channels, kernel_size * kernel_size, 
                                   kernel_size=kernel_size, stride=stride, padding=padding)
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, affine=False)
        self.relu = nn.ReLU(inplace=True)
        self.onnx_export_mode = 'native'

    def forward(self, x):
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))

        if self.onnx_export_mode == 'approx_conv':
            x = F.conv2d(
                x,
                self.conv.weight,
                self.conv.bias,
                stride=self.stride,
                padding=self.padding,
            )
        else:
            x = deform_conv2d(
                x,
                offset,
                self.conv.weight,
                mask=mask,
                stride=self.stride,
                padding=self.padding,
            )
        return self.relu(self.bn(x))

class BlockTypeB(nn.Module):
    def __init__(self, in_c, out_c):
        super(BlockTypeB, self).__init__()
        self.preconv = Block7(in_c, in_c)
        self.conv1 = DeformableLayer(in_c, in_c, stride=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.preconv(x)+x
        x = self.conv1(x) + x
        x = self.conv2(x)
        return x

class BlockTypeC(nn.Module):
    def __init__(self, in_c, out_c):
        super(BlockTypeC, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, in_c,  kernel_size=3, padding=5, dilation=5),
            nn.BatchNorm2d(in_c),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c, in_c,  kernel_size=3, padding=1),
            nn.BatchNorm2d(in_c),
            nn.ReLU()
        )
        self.conv3 = nn.Conv2d(in_c, out_c, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

def set_deform_conv_onnx_export_mode(module, mode='native'):
    if mode not in ('native', 'approx_conv'):
        raise ValueError('Unsupported deform-conv export mode: {}'.format(mode))

    for submodule in module.modules():
        if isinstance(submodule, DeformableLayer):
            submodule.onnx_export_mode = mode
    return module
