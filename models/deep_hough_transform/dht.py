import deep_hough_plus as dh
import torch
import torch.nn as nn

from models.direction_mask.dm import DirectionalMaskModule


class C_dht_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feat, numangle, numrho):
        N, C, _, _ = feat.size()
        out = torch.zeros(N, C, numangle, numrho).type_as(feat).cuda()
        out = dh.forward(feat, out, numangle, numrho)
        outputs = out[0]
        ctx.save_for_backward(feat)
        ctx.numangle = numangle
        ctx.numrho = numrho
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        feat = ctx.saved_tensors[0]
        numangle = ctx.numangle
        numrho = ctx.numrho
        out = torch.zeros_like(feat).type_as(feat).cuda()
        out = dh.backward(grad_output.contiguous(), out, feat, numangle, numrho)
        grad_in = out[0]
        return grad_in, None, None


class C_dht(torch.nn.Module):
    def __init__(self, numAngle, numRho):
        super(C_dht, self).__init__()
        self.numAngle = numAngle
        self.numRho = numRho

    def forward(self, feat):
        return C_dht_Function.apply(feat, self.numAngle, self.numRho)


class DHT_Layer(nn.Module):
    def __init__(self, input_dim, dim, numAngle, numRho):
        super(DHT_Layer, self).__init__()
        self.fist_conv = nn.Sequential(
            nn.Conv2d(input_dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.dht = DHT(numAngle=numAngle, numRho=numRho)
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


class DHT(nn.Module):
    def __init__(self, numAngle, numRho):
        super(DHT, self).__init__()
        self.line_agg = C_dht(numAngle, numRho)

    def forward(self, x):
        accum = self.line_agg(x)
        return accum
