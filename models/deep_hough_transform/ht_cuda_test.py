import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.deep_hough_transform.dht_module.dht import DHT
from models.deep_hough_transform.idht_module.idht import IDHT

img_size = 128
img = np.zeros((img_size, img_size), dtype=np.float32)

# 多条直线
cv2.line(img, (10, 20), (110, 20), 1, 2)  # 水平直线
cv2.line(img, (20, 10), (20, 110), 1, 2)  # 垂直直线
cv2.line(img, (30, 30), (100, 100), 1, 2)  # 斜对角
cv2.line(img, (100, 30), (30, 100), 1, 2)  # 另一条对角

plt.imshow(img, cmap='gray')
plt.title('Test Image: Multiple Lines')
plt.show()

# 转 Tensor
tensor_img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).cuda()

# Hough 参数
num_angle, num_rho = 180, 400
theta_res = 3.0
rho_res = 1.0
# dht = C_dht(num_angle, num_rho).cuda()
dht = DHT(img_size, img_size, theta_res, rho_res).cuda()
hough_space = dht(tensor_img)

# 简单阈值滤波
hough_space[hough_space < max(hough_space.mean(), hough_space.max() * 0.5)] = 0

plt.imshow(hough_space[0, 0].detach().cpu().numpy(), cmap='hot', aspect='auto')
plt.title('Hough Space (Multiple Lines)')
plt.xlabel('Rho')
plt.ylabel('Theta (Angle)')
plt.colorbar()
plt.show()

# 逆 Hough
# ciht = C_idht(num_angle, num_rho, out_H=tensor_img.size(-2), out_W=tensor_img.size(-1)).cuda()
idht = IDHT(img_size, img_size, theta_res, rho_res).cuda()
reconstructed_img = idht(hough_space)

plt.imshow(reconstructed_img[0, 0].detach().cpu().numpy(), cmap='gray')
plt.title('Reconstructed Image (Multiple Lines)')
plt.show()
