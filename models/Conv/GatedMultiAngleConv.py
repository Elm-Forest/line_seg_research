# file: line_prior_conv.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedMultiAngleConv(nn.Module):
    """
    多角度方向卷积 + 门控融合
    用于加强对线/边缘等方向敏感特征的提取

    参数:
        in_channels:  输入通道数
        out_channels: 输出通道数
        n_angles:     多少个方向卷积核
        kernel_size:  卷积核尺寸 (默认3)
        gating:       是否启用门控注意力, 如果False则简单相加
    """

    def __init__(self, in_channels, out_channels, n_angles=8, kernel_size=3, gating=True):
        super(GatedMultiAngleConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_angles = n_angles
        self.kernel_size = kernel_size
        self.gating = gating

        # 基础点卷积，用于学到通用特征(非方向卷积)
        self.base_conv = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=kernel_size, padding=kernel_size // 2)

        # 多方向卷积核 (每个方向各一个conv)
        self.dir_convs = nn.ModuleList()
        for _ in range(n_angles):
            conv = nn.Conv2d(in_channels, out_channels,
                             kernel_size=kernel_size,
                             padding=kernel_size // 2,
                             bias=False)
            self.dir_convs.append(conv)

        # 学习角度(可选, 也可固定)
        # 这里简单用 angle offset 让网络学, 也可以更复杂 steerable filter
        # self.angles = nn.Parameter(torch.linspace(0, math.pi, n_angles))

        # 如果要门控, 就加一个通道注意力
        if gating:
            # gating网络: 先将多方向输出concat, 再1x1 conv => gating map
            self.gate_conv = nn.Sequential(
                nn.Conv2d(out_channels * n_angles, out_channels, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(out_channels, n_angles, kernel_size=1)
            )

    def forward(self, x):
        # 基础卷积
        base_out = self.base_conv(x)

        # 多方向
        dir_feats = []
        for conv_d in self.dir_convs:
            d_out = conv_d(x)
            dir_feats.append(d_out)  # shape(B, out_ch, H, W)

        # 堆叠 => shape(B, n_angles, out_ch, H, W)
        stack_dir = torch.stack(dir_feats, dim=1)
        # sum or gating
        if self.gating:
            # reshape => (B, out_ch*n_angles, H, W)
            cat_dir = stack_dir.view(x.size(0), -1, x.size(2), x.size(3))
            gating_map = self.gate_conv(cat_dir)  # => shape(B, n_angles, H, W)
            gating_map = torch.softmax(gating_map, dim=1)

            # gating => sum over angle
            # gating_map shape(B, n_angles, H, W)
            # stack_dir shape(B, n_angles, out_ch, H, W)
            out = 0
            for i in range(self.n_angles):
                # gating_map[:,i] => shape(B,1,H,W)
                gi = gating_map[:, i].unsqueeze(1)
                out += stack_dir[:, i] * gi
            # out shape (B,out_ch,H,W)
            out = out + base_out  # 叠加基础卷积
        else:
            # 简单相加
            out = base_out + stack_dir.sum(dim=1)
        return F.relu(out)
