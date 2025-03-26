import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectionalPriorConv(nn.Module):
    """
    方向先验卷积层，专门加强直线特征提取，内置方向卷积核
    """

    def __init__(self, in_channels, out_channels, directions=['horizontal', 'vertical', 'diag1', 'diag2']):
        super(DirectionalPriorConv, self).__init__()
        self.directions = directions
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 普通卷积
        self.base_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # 方向卷积核（固定）
        self.direction_kernels = nn.ModuleDict({
            'horizontal': nn.Conv2d(in_channels, out_channels, (1, 7), padding=(0, 3), bias=False),
            'vertical': nn.Conv2d(in_channels, out_channels, (7, 1), padding=(3, 0), bias=False),
            'diag1': nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            'diag2': nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        })

        # 初始化为检测方向（斜线用 Sobel）
        with torch.no_grad():
            self._init_diag_kernels()

    def _init_diag_kernels(self):
        # Sobel-like diagonals
        diag1_kernel = torch.tensor([[[
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]]], dtype=torch.float32)
        diag2_kernel = torch.tensor([[[
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0]
        ]]], dtype=torch.float32)
        for name, kernel in zip(['diag1', 'diag2'], [diag1_kernel, diag2_kernel]):
            k = kernel.repeat(self.out_channels, self.in_channels, 1, 1)
            self.direction_kernels[name].weight.data.copy_(k)

    def forward(self, x):
        out = self.base_conv(x)
        for d in self.directions:
            out += self.direction_kernels[d](x)
        return F.relu(out)
