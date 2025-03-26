import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class DirectionalMaskFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hough_map, image_size, num_angle, num_rho, mask_width=3.0):
        N, C, num_angle, num_rho = hough_map.shape
        H, W = image_size
        device = hough_map.device

        mask = torch.zeros((N, C, H, W), device=device)
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        x_centered = x_coords - (W - 1) / 2.0
        y_centered = y_coords - (H - 1) / 2.0

        max_rho = np.sqrt((W / 2) ** 2 + (H / 2) ** 2)
        delta_rho = (2 * max_rho) / (num_rho - 1)
        peak_mask_all = torch.zeros_like(hough_map, dtype=torch.bool)

        for b in range(N):
            for c in range(C):
                hough_slice = hough_map[b, c]
                hough_4d = hough_slice.unsqueeze(0).unsqueeze(0)
                pooled = F.max_pool2d(hough_4d, kernel_size=3, stride=1, padding=1).squeeze()
                peak_mask = (hough_slice == pooled) & (hough_slice > 0.5 * hough_slice.max())
                peak_mask_all[b, c] = peak_mask

                angle_indices, rho_indices = torch.where(peak_mask)
                for a_idx, r_idx in zip(angle_indices, rho_indices):
                    theta = a_idx.item() * np.pi / num_angle
                    cos_theta = np.cos(theta)
                    sin_theta = np.sin(theta)
                    r_phys = (r_idx.item() - (num_rho - 1) / 2) * delta_rho

                    r_est = x_centered * cos_theta + y_centered * sin_theta
                    distance = torch.abs(r_est - r_phys)
                    line_mask = (distance < mask_width).float()
                    mask[b, c] = torch.max(mask[b, c], line_mask)

        ctx.save_for_backward(x_centered, y_centered, hough_map, peak_mask_all,
                              torch.tensor([mask_width], device=device))
        ctx.num_angle = num_angle
        ctx.num_rho = num_rho
        ctx.image_size = image_size
        return mask

    @staticmethod
    def backward(ctx, grad_output):
        x_centered, y_centered, hough_map, peak_mask_all, mask_width_tensor = ctx.saved_tensors
        mask_width = mask_width_tensor.item()
        num_angle = ctx.num_angle
        num_rho = ctx.num_rho
        H, W = ctx.image_size
        max_rho = np.sqrt((W / 2) ** 2 + (H / 2) ** 2)
        delta_rho = (2 * max_rho) / (num_rho - 1)
        grad_hough_map = torch.zeros_like(hough_map)

        for b in range(grad_output.shape[0]):
            for c in range(grad_output.shape[1]):
                current_peak_mask = peak_mask_all[b, c]
                for a_idx in range(num_angle):
                    for r_idx in range(num_rho):
                        if current_peak_mask[a_idx, r_idx]:
                            theta = a_idx * np.pi / num_angle
                            cos_theta = np.cos(theta)
                            sin_theta = np.sin(theta)
                            r_phys = (r_idx - (num_rho - 1) / 2) * delta_rho

                            r_est = x_centered * cos_theta + y_centered * sin_theta
                            distance = torch.abs(r_est - r_phys)
                            line_mask = (distance < mask_width).float()

                            grad_contribution = (grad_output[b, c] * line_mask).sum()
                            grad_hough_map[b, c, a_idx, r_idx] = grad_contribution

        return grad_hough_map, None, None, None, None


class DirectionalMaskModule(nn.Module):
    def __init__(self, num_angle, num_rho):
        super().__init__()
        self.num_angle = num_angle
        self.num_rho = num_rho
        self.mask_width = nn.Parameter(torch.tensor(3.0))  # 可学习参数

    def forward(self, hough_map, image_size):
        return DirectionalMaskFunction.apply(
            hough_map, image_size,
            self.num_angle, self.num_rho,
            self.mask_width
        )
