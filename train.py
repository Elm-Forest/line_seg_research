import argparse
import glob
import os
import warnings

import numpy as np

from losses.dice import DiceLoss, FocalLoss
from models.Unet.unet_std import UNet
from utils.metrics import mIoU

warnings.filterwarnings("ignore")

import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.line_dataset import PowerLineDataset


def train(args):
    # 数据路径
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        print("CUDA或MPS不可用, 使用CPU进行训练。") 
        device = torch.device("cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    # model = UNet_Line(n_channels=3, n_classes=args.num_classes, bilinear=True).to(device)
    # model = UNet(n_channels=3, n_classes=args.num_classes, bilinear=True).to(device)
    model = UNet(n_channels=3, n_classes=args.num_classes, bilinear=True).to(device)
    # Mask2Former
    bce_criterion = FocalLoss(logits=True).to(device)
    # dice_criterion = DiceLoss().to(device)
    dice_criterion = DiceLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # metric = IoU2()

    scaler = GradScaler()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{args.epochs}", unit='batch') as pbar:
            for images, masks in train_loader:
                images = images.to(device)
                masks = masks.to(device)

                optimizer.zero_grad()

                with autocast():
                    preds = model(images)
                    bce_loss = bce_criterion(preds, masks)
                    dice_loss = dice_criterion(preds, masks)
                    loss = bce_loss + dice_loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                # miou = mean_iou(preds, masks, args.num_classes)
                miou = mIoU(preds, masks).item()
                # miou = metric(preds, masks).item()
                epoch_loss += bce_loss.item()

                pbar.set_postfix(loss=bce_loss.item(), miou=round(miou, 4))
                pbar.update(1)

        # 验证阶段
        model.eval()
        total_miou = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                preds = model(images)
                miou = mIoU(preds, masks).item()
                total_miou += miou

        avg_loss = epoch_loss / len(train_loader)
        avg_miou = total_miou / len(val_loader)

        print(f"\n[Epoch {epoch}] Avg Loss: {avg_loss:.4f} | Val mIoU: {avg_miou:.4f}\n")

    # 保存模型
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'unet_powerline_unet.pth'))
    print("✅ 训练完成，模型已保存！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Power Line UNet Training")
    parser.add_argument('--image_dir', type=str, default=r'K:\dataset\power line dataset\images')
    parser.add_argument('--mask_dir', type=str, default=r'K:\dataset\power line dataset\labels')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--num_classes', type=int, default=2)
    args = parser.parse_args()
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    train(args)
