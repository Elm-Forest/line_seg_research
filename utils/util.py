import random

import torch
from matplotlib import pyplot as plt


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


def create_simple_line_image():
    # 创建一个空白图像 [1, 1, 180, 180]，1个batch，1个channel，180x180像素
    img = torch.zeros((1, 1, 50, 50), dtype=torch.float32)

    # 第一条线段：从(20, 20)到(100, 100)
    for x in range(10, 40):
        img[0, 0, x, x] = 1.0  # 线段1

    # 第二条线段：从(60, 40)到(120, 100)
    for x in range(10, 40):
        img[0, 0, x, x - 10] = 1.0  # 线段2

    return img


def visualize_results(img, hough_map, direction_mask):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(img[0, 0].cpu().numpy(), cmap='gray')
    axs[0].set_title("Original Image")

    axs[1].imshow(hough_map[0, 0].cpu().numpy(), cmap='hot')
    axs[1].set_title("Hough Domain")

    axs[2].imshow(direction_mask[0, 0].cpu().numpy(), cmap='hot')
    axs[2].set_title("Directional Mask")

    plt.show()
