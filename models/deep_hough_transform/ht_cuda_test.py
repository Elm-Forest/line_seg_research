import random

import matplotlib.pyplot as plt
import torch

from models.deep_hough_transform.dht_module.dht import DHT
from models.deep_hough_transform.idht_module.idht import IDHT

img_size = 128


# img = np.zeros((img_size, img_size), dtype=np.float32)


# 1) 创建包含多条线段和断裂的虚拟图像
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


img = create_complex_line_image()

# 多条直线

plt.imshow(img[0, 0].cpu(), cmap='gray')
plt.title('Test Image: Multiple Lines')
plt.show()

# 转 Tensor
tensor_img = img.cuda()

# Hough 参数
num_angle, num_rho = 180, 400
theta_res = 1.0
rho_res = 2.0
# dht = C_dht(num_angle, num_rho).cuda()
dht = DHT(img_size, img_size, theta_res, rho_res).cuda()
hough_space = dht(tensor_img)

plt.imshow(hough_space[0, 0].detach().cpu().numpy(), cmap='hot')
plt.title('Hough Space')
plt.xlabel('Rho')
plt.ylabel('Theta')
plt.colorbar()
plt.show()

# 简单阈值滤波
hough_space[hough_space < max(hough_space.mean(), hough_space.max() * 0.5)] = 0

plt.imshow(hough_space[0, 0].detach().cpu().numpy(), cmap='hot')
plt.title('Hough Space (Filter Noise)')
plt.xlabel('Rho')
plt.ylabel('Theta')
plt.colorbar()
plt.show()

# 逆 Hough
# ciht = C_idht(num_angle, num_rho, out_H=tensor_img.size(-2), out_W=tensor_img.size(-1)).cuda()
idht = IDHT(img_size, img_size, theta_res, rho_res).cuda()
reconstructed_img = idht(hough_space)

plt.imshow(reconstructed_img[0, 0].detach().cpu().numpy(), cmap='gray')
plt.title('Line Prior Mask')
plt.show()
