#############################################################
# line_prior_conv.py
#############################################################

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiAngleGatedConv(nn.Module):
    """
    多角度方向卷积 + gating 注意力融合
    - 角度核可学习(steerable)或固定初始化
    - gating可对每个方向分支分配自适应权重
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 angle_list=None,
                 kernel_size=5,
                 use_global_gate=True):
        """
        参数:
        - angle_list: 预设若干角度(弧度), 如 [0, pi/4, pi/2, ...].
          若=None, 自动生成 0,45,90,135 deg. 也可设多一些
        - kernel_size: 每个方向卷积核的大小
        - use_global_gate: True则用 global gating(每通道一权重),
                          False则可改成pixel-wise注意力(需额外Conv)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.use_global_gate = use_global_gate

        if angle_list is None:
            # 默认4个方向(0°,45°,90°,135°)
            angle_list = [0, math.pi / 4, math.pi / 2, 3 * math.pi / 4]
        self.angle_list = angle_list
        self.num_directions = len(angle_list)

        # 基础卷积(不带方向)
        self.base_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # 分支: 为每个angle建立一个 "方向卷积核"
        # 这里先用普通 conv, 后面在 forward 初始化/旋转 kernel
        self.dir_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
            for _ in range(self.num_directions)
        ])

        # gating => (out_channels) or (out_channels * num_directions)
        # 这里选global gating, 每个方向都有一个 out_channels 的权重
        if self.use_global_gate:
            # 形状 (num_directions, out_channels)
            self.gate_params = nn.Parameter(torch.ones(self.num_directions, out_channels))
        else:
            # pixel-wise gating => 需要1x1 conv => shape(num_directions*out_ch -> out_ch???)
            # 示例就不给实现, 原理类似
            raise NotImplementedError("Pixel-wise gating not implemented in this demo")

        # 可选: 初始化方向核(steerable). 这里演示固定Gaussian+旋转
        self._init_direction_kernels()

    def _init_direction_kernels(self):
        """用简单的 2D Gaussian + rotate 方式构造方向核(可继续改进为Gabor等)"""
        center = self.kernel_size // 2
        coords = torch.arange(self.kernel_size) - center
        yy, xx = torch.meshgrid(coords, coords)
        # 先生成一个高斯
        sigma = self.kernel_size / 2.0
        gauss = torch.exp(-0.5 * (xx ** 2 + yy ** 2) / sigma ** 2)
        gauss = gauss / gauss.sum()  # 归一化
        # shape => (1,1,k,k)
        base_filter = gauss.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            for i, angle in enumerate(self.angle_list):
                # naive approach: 这里不做真正旋转, 仅复制 base_filter.
                # 如果要精确旋转, 可用rotate函数(插值).
                # 这里仅演示 => 保留gauss shape,
                # 你也可改成 Gabor(orientation=angle)
                k = base_filter.repeat(self.out_channels, self.in_channels, 1, 1)
                self.dir_convs[i].weight.copy_(k)

    def forward(self, x):
        # 1) 基础conv
        base_feat = self.base_conv(x)  # (B,out_ch,H,W)

        # 2) 方向分支
        dir_feats = []
        for i in range(self.num_directions):
            f = self.dir_convs[i](x)  # (B,out_ch,H,W)
            dir_feats.append(f)

        # 3) gating融合
        # shape(B,out_ch,H,W) for each
        # gating => self.gate_params => shape(num_directions, out_ch)
        # expand to => (1, out_ch, 1, 1) broadcast
        out = base_feat
        for i in range(self.num_directions):
            # gate shape =>(out_ch,) => unsqueeze => (1,out_ch,1,1)
            gate_i = self.gate_params[i].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            out += dir_feats[i] * gate_i

        return F.relu(out)

