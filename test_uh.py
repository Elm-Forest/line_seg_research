import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models.Unet.unet_hough import UNetHough


# 加载模型
def load_model(model_path, device):
    # 初始化模型（假设您使用的是 Segformer）
    model = UNetHough(n_channels=3, n_classes=2, bilinear=True).to(device)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()  # 切换到评估模式
    return model


# 图像预处理
def preprocess_image(image_path, img_size=512):
    img = Image.open(image_path).convert('RGB')  # 打开图像并转为RGB格式
    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = preprocess(img).unsqueeze(0)  # 增加批次维度
    return img, img_tensor  # 返回原图和预处理后的图像


# 推理函数
def infer(model, img_tensor, device):
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        output, o1, o2, o3 = model(img_tensor)  # 模型推理
        pred = torch.argmax(output, dim=1).squeeze(0)  # 获取预测结果，选择最大值作为类别
        # p1 = torch.argmax(o1, dim=0).squeeze(0)  # 获取预测结果，选择最大值作为类别
        # p2 = torch.argmax(o2, dim=0).squeeze(0)  # 获取预测结果，选择最大值作为类别
        # p3 = torch.argmax(o2, dim=0).squeeze(0)  # 获取预测结果，选择最大值作为类别
        p1 = o1.squeeze(0).squeeze(0)  # 获取预测结果，选择最大值作为类别
        p2 = o2.squeeze(0).squeeze(0)  # 获取预测结果，选择最大值作为类别
        p3 = o3.squeeze(0).squeeze(0)  # 获取预测结果，选择最大值作为类别
    return pred.cpu().numpy(), p1.cpu().numpy(), p2.cpu().numpy(), p3.cpu().numpy()  # 返回 numpy 数组，便于后续处理


# 可视化推理结果
def visualize_result(image, pred_mask, img_size, file_name):
    # 将预测掩码恢复到原始图像的尺寸
    pred_mask_resized = Image.fromarray(pred_mask.astype(np.uint8)).resize(image.size, Image.NEAREST)

    # 显示原图和预测结果
    plt.imshow(image)
    plt.imshow(pred_mask_resized, alpha=0.5)  # 以透明度0.5叠加预测掩码
    plt.axis('off')
    plt.savefig(f'result{file_name}.png')
    plt.show()


# 主程序
def main(model_path, image_path, device='cuda'):
    # 加载模型
    model = load_model(model_path, device)

    # 预处理图像
    image, img_tensor = preprocess_image(image_path)

    # 推理
    pred_mask, p1, p2, p3 = infer(model, img_tensor, device)

    # 可视化结果
    visualize_result(image, pred_mask, img_size=512, file_name="pred.png")  # 保持输入的resize尺寸
    visualize_result(image, p1, img_size=512, file_name="p1.png")  # 保持输入的resize尺寸
    visualize_result(image, p2, img_size=512, file_name="p2.png")  # 保持输入的resize尺寸
    visualize_result(image, p3, img_size=512, file_name="p3.png")  # 保持输入的resize尺寸


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Power Line UNet Test")
    parser.add_argument('--img_path', type=str, default=r'K:\\dataset\\power line dataset\\images\\313.png')
    parser.add_argument('--weight_path', type=str, default=r'checkpoints/unet_powerline.pth')
    args = parser.parse_args()
    model_path = args.weight_path  # 模型路径
    # image_path = "K:\\dataset\\coco_powerline_1\\train\\25_00694.jpg"

    image_path = args.img_path  # 输入图像路径
    main(model_path, image_path, device='cuda')  # 可以选择 'cuda' 或 'cpu'
