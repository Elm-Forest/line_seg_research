import os
import random


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


##############################################################################
# 2) 积分式 Inverse Hough Transform (IHT) ——从 Hough 域反投影到图像域
#    它不会补线，只是将原有 Hough 累加能量还回去
##############################################################################
class C_iht_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hough_feat, out_shape, numangle, numrho):
        """
        hough_feat: [N, C, numangle, numrho]
        out_shape:  (N, C, H, W) ——需要知道还原的图像大小
        return:     [N, C, H, W]
        """
        out_img = torch.zeros(out_shape, dtype=hough_feat.dtype, device=hough_feat.device)
        out_img = dh.inverse(hough_feat, out_img, numangle, numrho)
        outputs = out_img[0]

        # 保存上下文
        ctx.save_for_backward(hough_feat)
        ctx.out_shape = out_shape
        ctx.numangle = numangle
        ctx.numrho = numrho

        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        """
        若你需要让 inverse 也可反向传播梯度到 hough_feat，可在这实现:
        hough_feat -> ???  (示例中返回None)
        """
        # (hough_feat,) = ctx.saved_tensors
        # grad_in_hough = ...
        # return grad_in_hough, None, None, None

        return None, None, None, None


class C_iht(torch.nn.Module):
    def __init__(self, out_shape, numAngle, numRho):
        """
        out_shape: (N, C, H, W) 反投影回来的图像大小
        """
        super().__init__()
        self.out_shape = out_shape
        self.numAngle = numAngle
        self.numRho = numRho

    def forward(self, hough_feat):
        return C_iht_Function.apply(hough_feat, self.out_shape, self.numAngle, self.numRho)


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


def create_complex_line_image():
    # 创建一个空白图像 [1, 1, 180, 180]，1个batch，1个channel，180x180像素
    img = torch.zeros((1, 1, 180, 180), dtype=torch.float32)

    # 第一条线段：从(20, 20)到(100, 100)
    for x in range(20, 100):
        img[0, 0, x, x] = 1.0  # 线段1

    # 第二条线段：从(60, 40)到(120, 100)
    for x in range(60, 120):
        img[0, 0, x, x - 20] = 1.0  # 线段2

    # 第三条线段：从(20, 100)到(120, 100)
    for x in range(20, 120):
        img[0, 0, 100, x] = 1.0  # 线段3

    # 第四条线段：从(40, 30)到(100, 90) (断裂)
    for x in range(40, 100):
        img[0, 0, x, x + 50] = 1.0  # 线段4 (断裂)

    # 添加少量随机噪点（比如 20 个）
    num_noise_points = 1000
    for _ in range(num_noise_points):
        noise_x = random.randint(0, 179)
        noise_y = random.randint(0, 179)
        img[0, 0, noise_y, noise_x] = 1.0  # 亮点噪声

    return img


if __name__ == '__main__':
    dht = C_dht(180, 200).cuda()
    img = create_complex_line_image().cuda()
    hm = dht(img)
    img = img.cuda().requires_grad_(True)
    test = torch.autograd.gradcheck(
        dht,
        img,
        eps=1e-6,
        atol=1e-4,
        check_undefined_grad=False
    )

    print("Grad Check Result: ", test)
