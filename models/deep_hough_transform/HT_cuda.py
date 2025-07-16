import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage

from models.deep_hough_transform.dht_module.dht import DHT
from models.deep_hough_transform.idht_module.idht import IDHT


def make_conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=False):
    layers = []
    layers += [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)]
    ###  no batchnorm layers
    # layers += [nn.BatchNorm2d(out_channels)]
    layers += [nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class HTIHT_Cuda(nn.Module):
    def __init__(self, inplanes, outplanes, H, W, theta_res=1.0, rho_res=1.0, mid_ch = 256):
        super(HTIHT_Cuda, self).__init__()
        mid_ch = inplanes
        self.conv1 = nn.Sequential(
            *make_conv_block(inplanes, mid_ch, kernel_size=(9, 1), padding=(4, 0), bias=True, groups=inplanes))
        self.conv2 = nn.Sequential(
            *make_conv_block(mid_ch, mid_ch, kernel_size=(9, 1), padding=(4, 0), bias=True))
        self.conv3 = nn.Sequential(
            *make_conv_block(mid_ch, outplanes, kernel_size=(9, 1), padding=(4, 0), bias=True))

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.ht = DHT(H, W, theta_res, rho_res)
        self.iht = IDHT(H, W, theta_res, rho_res)

        filtersize = 4
        x = np.zeros(shape=((2 * filtersize + 1)))
        x[filtersize] = 1
        z = []
        for _ in range(0, inplanes):
            sigma = np.random.uniform(low=1, high=2.5, size=(1))
            y = ndimage.filters.gaussian_filter(x, sigma=sigma, order=2)
            y = -y / np.sum(np.abs(y))
            z.append(y)
        z = np.stack(z)
        self.conv1[0].weight.data.copy_(torch.from_numpy(z).unsqueeze(1).unsqueeze(3))
        nn.init.kaiming_normal_(self.conv2[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3[0].weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, **kwargs):
        out = self.ht(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.iht(out)
        return out


class CAT_HTIHT_Cuda(nn.Module):
    def __init__(self, inplanes, outplanes, H, W, theta_res=1.0, rho_res=1.0):
        super(CAT_HTIHT_Cuda, self).__init__()
        self.htiht = HTIHT_Cuda(inplanes, outplanes, H, W, theta_res, rho_res)
        self.bn = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv_cat = nn.Conv2d(inplanes + outplanes, inplanes, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        y = self.htiht(x)
        out = self.conv_cat(torch.cat([x, y], dim=1))
        return out
