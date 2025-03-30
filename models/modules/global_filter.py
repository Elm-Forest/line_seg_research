import torch
import torch.fft
import torch.nn as nn


class GlobalFilter(nn.Module):

    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)

    def forward(self, x):
        B, H, W, C = x.shape

        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')

        weight = torch.view_as_complex(self.complex_weight)

        x = x * weight

        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')

        return x
