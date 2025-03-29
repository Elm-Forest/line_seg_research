import torch.nn as nn

from models.deep_hough_transform.dht_module.dht_func import C_dht
from models.direction_mask.dm import DirectionalMaskModule


class DHT_Layer(nn.Module):
    def __init__(self, input_dim, dim, numAngle, numRho):
        super(DHT_Layer, self).__init__()
        self.fist_conv = nn.Sequential(
            nn.Conv2d(input_dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.dht = DHT_Module(numAngle=numAngle, numRho=numRho)
        self.mask_gen = DirectionalMaskModule(numAngle, numRho)

        self.convs = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.fist_conv(x)
        x1 = self.dht(x1)
        x1 = self.mask_gen(x1, (x.shape(-2), x.shape(-1)))
        x = self.convs(x)
        return x


class DHT_Module(nn.Module):
    def __init__(self, numAngle, numRho):
        super(DHT_Module, self).__init__()
        self.line_agg = C_dht(numAngle, numRho)

    def forward(self, x):
        accum = self.line_agg(x)
        return accum
