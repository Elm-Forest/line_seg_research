import math

import torch
import torch.nn as nn

from models.deep_hough_transform.idht_module.idht_func import C_idht


def compute_hough_parameters(H, W, theta_res=1.0, rho_res=1.0):
    """
    根据图像大小 (H, W) 以及角度分辨率 theta_res、距离分辨率 rho_res
    动态计算 (numangle, numrho).
    """
    # 1) 角度范围 [0, 180) (或根据需要设定)
    numangle = int(math.ceil(180.0 / theta_res))

    # 2) 图像对角线
    D = math.sqrt((H - 1) ** 2 + (W - 1) ** 2)
    # 3) rho 范围: [-D, D], 步长 = rho_res => 2D / rho_res + 1
    numrho = int(math.ceil((2.0 * D) / rho_res) + 1)

    return numangle, numrho


class IDHT(nn.Module):
    """
       根据 (H, W, theta_res, rho_res) 动态计算 (numangle, numrho)
       然后执行 逆Hough -> [N, C, H, W].
       """

    def __init__(self, H, W, theta_res=1.0, rho_res=1.0):
        super().__init__()
        # 用与前面相同的公式:
        self.numangle, self.numrho = compute_hough_parameters(H, W, theta_res, rho_res)
        self.out_H = H
        self.out_W = W

        print(f"[C_idht_res] => numangle={self.numangle}, numrho={self.numrho}, (H,W)=({H},{W})")

        # 内部封装:
        self.idht = C_idht(self.numangle, self.numrho, H, W)

    def forward(self, hough_map):
        # hough: [N, C, numangle, numrho]
        # 这里要求 hough 的 numangle==self.numangle, numrho==self.numrho
        return self.idht(hough_map)


if __name__ == "__main__":
    # 假设我们想要逆变换到图像大小 32x32, 并使用 1°、1像素 的分辨率
    H, W = 32, 32
    model = IDHT(H, W, theta_res=1.0, rho_res=1.0).cuda()

    # 生成一个 [N=1, C=1, numangle, numrho] 的随机Hough特征
    #   其中 numangle, numrho 由 model内部 compute_hough_parameters 得到
    N, C = 1, 1
    hough_in = torch.randn(
        (N, C, model.numangle, model.numrho),
        dtype=torch.double,  # 用 double 方便 gradcheck
        device='cuda'
    )
    hough_in.requires_grad_(True)

    # 前向 => [1, 1, 32, 32]
    out_image = model(hough_in)

    # 假设 loss = out_image.sum()
    loss = out_image.sum()
    loss.backward()

    print("Backward done!")
    print("hough_in.grad shape:", hough_in.grad.shape)

