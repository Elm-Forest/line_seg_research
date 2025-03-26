import torch
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.2, gamma=2, logits=True, sampling='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.sampling = sampling

    def forward(self, y_pred, y_true):
        alpha = self.alpha
        alpha_ = (1 - self.alpha)
        if self.logits:
            y_pred = torch.sigmoid(y_pred)

        pt_positive = torch.where(y_true == 1, y_pred, torch.ones_like(y_pred))
        pt_negative = torch.where(y_true == 0, y_pred, torch.zeros_like(y_pred))
        pt_positive = torch.clamp(pt_positive, 1e-3, .999)
        pt_negative = torch.clamp(pt_negative, 1e-3, .999)
        pos_ = (1 - pt_positive) ** self.gamma
        neg_ = pt_negative ** self.gamma

        pos_loss = -alpha * pos_ * torch.log(pt_positive)
        neg_loss = -alpha_ * neg_ * torch.log(1 - pt_negative)
        loss = pos_loss + neg_loss

        if self.sampling == "mean":
            return loss.mean()
        elif self.sampling == "sum":
            return loss.sum()
        elif self.sampling == None:
            return loss


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, logits, targets, smooth=1e-6):
        """
        计算Dice Loss。

        参数：
        logits (Tensor): 模型的输出，形状为 [batch_size, 1, H, W] 或 [batch_size, 2, H, W]。
        targets (Tensor): 目标标签，形状为 [batch_size, 1, H, W]。
        smooth (float): 避免除零错误的小常数。

        返回：
        Tensor: Dice损失值。
        """
        # 确保目标是浮动类型，并将其展平
        targets = targets.float()

        # 对logits应用Sigmoid激活，生成预测的概率
        logits = torch.sigmoid(logits)

        # 将logits和targets展平成1D向量
        logits = logits.reshape(logits.size(0), -1)
        targets = targets.reshape(targets.size(0), -1)

        # 计算交集部分
        intersection = (logits * targets).sum(dim=1)

        # 计算Dice损失
        dice_score = (2. * intersection + smooth) / (logits.sum(dim=1) + targets.sum(dim=1) + smooth)

        # 计算损失，即1减去Dice系数
        loss = 1 - dice_score.mean()

        return loss


class FocalDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(FocalDiceLoss, self).__init__()
        self.focal = FocalLoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        focal_loss = self.focal(pred, target)
        dice_loss = self.dice(pred, target)
        loss = self.wd * dice_loss + self.wb * focal_loss
        return loss
