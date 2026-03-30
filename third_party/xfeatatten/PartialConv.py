import torch
import torch.nn as nn
import torch.nn.functional as F
"""
CV缝合救星魔改创新2：引入空间注意力
全局信息利用不足：原始模块中，卷积操作仅限于局部空间特征提取，缺乏对全局特征的充分利用。这意味着模型在理解全局上下文时可能存在局限，
尤其在需要捕捉长距离依赖关系的任务中表现欠佳。
改进方法：
1. 在部分卷积后加入空间注意力模块，使得模型能够对特征图的空间位置进行加权，聚焦于对目标更有辨识度的区域。
2. 使用7×7卷积核提取全局空间信息：通过使用大卷积核（7×7）提取空间注意力，可以获取更大的感受野，使模型更加专注于全局空间模式。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class Basic(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, padding=0, bias=False):
        super(Basic, self).__init__()
        self.out_channels = out_planes
        groups = 1
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, groups=groups, bias=bias)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x
    
class ChannelPool(nn.Module):
    def __init__(self):
        super(ChannelPool, self).__init__()

    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SAB(nn.Module):
    def __init__(self):
        super(SAB, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = Basic(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return scale


class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.sigmoid=nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        y=self.gap(x)  # 在空间方向执行全局平均池化: (B,C,H,W)-->(B,C,1,1)
        y=y.squeeze(-1).permute(0,2,1)  # 将通道描述符去掉一维,便于在通道上执行卷积操作:(B,C,1,1)-->(B,C,1)-->(B,1,C)
        y=self.conv(y)  # 在通道维度上执行1D卷积操作,建模局部通道之间的相关性: (B,1,C)-->(B,1,C)
        y=self.sigmoid(y) # 生成权重表示: (B,1,C)
        y=y.permute(0,2,1).unsqueeze(-1)  # 重塑shape: (B,1,C)-->(B,C,1)-->(B,C,1,1)
        return y.expand_as(x)  # 权重对输入的通道进行重新加权: (B,C,H,W) * (B,C,1,1) = (B,C,H,W)

class PartialConv3WithSpatialAttention(nn.Module):
    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        # 空间注意力模块
        # self.spatial_attention = nn.Sequential(
        #     nn.Conv2d(dim, 1, kernel_size=7, padding=3, bias=False),
        #     nn.Sigmoid()
        # )
        self.spatial_attention = SAB()
        self.channel_attention = ECAAttention()

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError
    
    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        
        # reshape
        x = x.view(batchsize, groups, 
                   channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        
        # flatten
        x = x.view(batchsize, -1, height, width)
        return x


    def forward_slicing(self, x):
        x = x.clone()  # 保持原始输入不变
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        # 添加空间注意力
        attention_map = self.spatial_attention(x)
        x = x * attention_map

        return x

    def forward_split_cat(self, x):
        # print(self.dim_conv3)
        # print(self.dim_untouched)
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        attention_map_x1 = self.spatial_attention(x1)
        x1 = x1 * attention_map_x1
        x2 = self.channel_attention(x2) * x2
        x = torch.cat((x1, x2), 1)
        x = self.channel_shuffle(x, groups=2)

        # # 添加空间注意力
        # attention_map = self.spatial_attention(x)
        # x = x * attention_map

        return x


if __name__ == '__main__':
    block = PartialConv3WithSpatialAttention(24, 2, 'split_cat').cuda()
    input_tensor = torch.rand(1, 24, 128, 128).cuda()
    output = block(input_tensor)
    print(input_tensor.size(), output.size())


