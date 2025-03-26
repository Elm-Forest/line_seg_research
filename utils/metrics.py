import torch
import torch.nn.functional as F


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def iou(pr, gt, eps=1e-7, threshold=None):
    """Calculate Intersection over Union between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """
    pr = _threshold(pr, threshold=threshold)
    intersection = torch.sum((gt * pr).float())
    union = torch.sum(gt) + torch.sum(pr) - intersection + eps
    return (intersection + eps) / union


def mIoU(pr, gt, eps=1e-7, n_classes=2):
    """
    Calculate mean Intersection over Union (mIoU) for multi-class segmentation.
    Args:
        pr (torch.Tensor): predicted tensor of shape [batch_size, num_classes, H, W]
        gt (torch.Tensor): ground truth tensor of shape [batch_size, num_classes, H, W]
        eps (float): epsilon to avoid zero division
        n_classes (int): number of classes (background + object, so n_classes=2 for binary classification)
    Returns:
        float: mIoU score (average of IoU for each class)
    """
    pr = F.softmax(pr, dim=1)  # Apply softmax across the class dimension
    pr = torch.argmax(pr, dim=1).squeeze(1)  # Convert to class labels (0 for background, 1 for object)
    gt = torch.argmax(gt, dim=1).squeeze(1)  # Ground truth class labels

    iou_per_class = []

    # Iterate through each class
    sem_class = 1
    pr_inds = (pr == sem_class)
    gt_inds = (gt == sem_class)

    if gt_inds.long().sum().item() == 0:
        iou_per_class.append(torch.tensor(float('nan')))
    else:
        intersect = torch.logical_and(pr_inds, gt_inds).sum().float().item()
        union = torch.logical_or(pr_inds, gt_inds).sum().float().item()
        iou = (intersect + eps) / (union + eps)
        iou_per_class.append(iou)

    # Return mean IoU, ignoring NaNs (if any class does not appear in ground truth)
    return torch.nanmean(torch.tensor(iou_per_class))
