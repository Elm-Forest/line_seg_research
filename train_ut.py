import argparse
import glob
import os
import warnings

import numpy as np

from losses.dice import DiceLoss, FocalLoss
from models.Unet.unet_ht import UNetHT
from utils.metrics import get_metrics

warnings.filterwarnings("ignore")

import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from datasets.line_dataset import PowerLineDataset


def train(args):
    # 数据路径
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_paths = sorted(glob.glob(os.path.join(args.image_dir, '*.png')))
    mask_paths = sorted(glob.glob(os.path.join(args.mask_dir, '*.png')))
    assert len(image_paths) == len(mask_paths), "图像和Mask数量不一致！"

    # 数据划分
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=args.val_split, random_state=42)

    train_dataset = PowerLineDataset(train_imgs, train_masks, size=args.img_size, train=True,
                                     num_classes=args.num_classes)
    val_dataset = PowerLineDataset(val_imgs, val_masks, size=args.img_size, train=False, num_classes=args.num_classes)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 模型
    model = UNetHT(n_channels=3, n_classes=args.num_classes,
                   bilinear=True,
                   angle_res=args.angle_res,
                   rho_res=args.rho_res,
                   img_size=args.img_size).to(device)
    if args.pretrained is not None and args.pretrained != "":
        model.load_state_dict(torch.load(args.pretrained, map_location=torch.device("cpu")), strict=False)
        model.to(device)
        print("load pretrained model!")
    # Mask2Former
    bce_criterion = FocalLoss(logits=True).to(device)
    dice_criterion = DiceLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print_step = 10
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0

        for step, (images, masks) in enumerate(train_loader, 1):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            # 直接不使用 autocast，进行标准精度训练
            preds = model(images)
            bce_loss = bce_criterion(preds, masks)
            dice_loss = dice_criterion(preds, masks)
            loss = bce_loss + dice_loss

            # 反向传播
            loss.backward()

            # 优化器更新
            optimizer.step()

            # 获取指标
            metrics = get_metrics(preds, masks, 0.5)
            epoch_loss += bce_loss.item()

            # 控制打印的频率，每 print_step 步打印一次
            if step % print_step == 0:
                print(f"[Epoch {epoch}] Step {step}/{len(train_loader)}: Loss = {bce_loss.item():.4f}, "
                      f"Metrics = {metrics}")

        # 验证阶段
        model.eval()
        total_metrics = {"AUC": 0, "F1": 0, "Acc": 0, "Sen": 0, "Spe": 0, "pre": 0, "IOU": 0}
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                preds = model(images)
                metrics = get_metrics(preds, masks, 0.5)

                # 累加各个指标
                for key in total_metrics:
                    total_metrics[key] += metrics[key]

        # 计算平均损失和验证指标
        avg_loss = epoch_loss / len(train_loader)
        avg_metrics = {key: total_metrics[key] / len(val_loader) for key in total_metrics}

        # 打印训练损失和验证指标
        print(f"\n[Epoch {epoch}] Avg Loss: {avg_loss:.4f}")
        print(f"Validation Metrics: AUC: {avg_metrics['AUC']:.4f}, F1: {avg_metrics['F1']:.4f}, "
              f"Acc: {avg_metrics['Acc']:.4f}, Sen: {avg_metrics['Sen']:.4f}, Spe: {avg_metrics['Spe']:.4f}, "
              f"Pre: {avg_metrics['pre']:.4f}, IOU: {avg_metrics['IOU']:.4f}")

        # 保存模型检查点
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save(model.state_dict(),
                   os.path.join(args.save_dir, f'unet_powerline_ep_{epoch}_{avg_metrics["IOU"]:.4f}.pth'))
        print("Save checkpoint!")

    # 保存模型
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'unet_powerline.pth'))
    print("✅ 训练完成，模型已保存！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Power Line UNet Training")
    parser.add_argument('--image_dir', type=str, default=r'K:\dataset\power line dataset\images')
    parser.add_argument('--mask_dir', type=str, default=r'K:\dataset\power line dataset\labels')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--pretrained', type=str, default="")  # ./checkpoints/unet_powerline.pth
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--angle_res', type=int, default=3)
    parser.add_argument('--rho_res', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--num_classes', type=int, default=1)
    args = parser.parse_args()
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    train(args)
