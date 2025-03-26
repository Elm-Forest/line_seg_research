import numpy as np
from PIL import Image
import os


def calculate_mean_std(image_paths):
    means = []
    stds = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        img = np.array(img) / 255.0
        means.append(img.mean(axis=(0, 1)))
        stds.append(img.std(axis=(0, 1)))

    means = np.mean(means, axis=0)
    stds = np.mean(stds, axis=0)
    return means, stds


# 获取文件夹中的所有图片路径
image_folder = "K://dataset//power line dataset//images"
image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if
               fname.endswith(('jpg', 'png', 'jpeg'))]

# 计算均值和标准差
mean, std = calculate_mean_std(image_paths)
print(f"Mean: {mean}, Std: {std}")
