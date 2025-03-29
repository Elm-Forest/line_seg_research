import math

import torch
import torch.nn as nn

from models.deep_hough_transform.dht_module.dht_func import C_dht


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


class DHT(nn.Module):
    """
    这个类将 "分辨率" 和 "图像大小" 作为初始化参数。
    初始化时, 动态计算 numangle, numrho, 再构造内部的 C_dht 模块.
    """

    def __init__(self, H, W, theta_res=1.0, rho_res=1.0):
        super(DHT, self).__init__()
        # 计算 (numangle, numrho)
        numangle, numrho = compute_hough_parameters(H, W, theta_res, rho_res)
        print(f"=> [C_dht_res] computed numangle={numangle}, numrho={numrho} from (H={H},W={W})")

        self.numangle = numangle
        self.numrho = numrho
        # 内部使用 C_dht
        self.hough = C_dht(numangle, numrho)

    def forward(self, feat):
        """
        feat: [N, C, H, W] - 假设与 (H, W) 相符合
        """
        return self.hough(feat)


if __name__ == "__main__":
    # ====== 示例用法 ======
    # 1) 假设图像大小 64x64, 角度分辨率 1°, 距离分辨率 1.0
    model = DHT(64, 64, theta_res=1.0, rho_res=1.0)

    # 2) 生成随机输入
    x = torch.randn(1, 1, 64, 64, device='cuda', dtype=torch.float)
    x.requires_grad_(True)

    # 3) 前向
    y = model(x)
    print("Hough Output shape:", y.shape)  # [1,1, numangle, numrho]

    # 4) 反向
    #  (a) 假设损失就是 output.sum()
    loss = y.sum()
    loss.backward()
    print("x.grad shape:", x.grad.shape)
