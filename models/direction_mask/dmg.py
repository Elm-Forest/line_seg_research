import torch
import torch.nn as nn
import torch.nn.functional as F


class PeakDetection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hough_map):
        pooled = F.max_pool2d(hough_map, 3, 1, 1)
        peak_mask = (hough_map == pooled) & (hough_map > 0.5 * hough_map.amax(dim=(-2, -1), keepdim=True))
        ctx.save_for_backward(peak_mask)
        return peak_mask.float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.saved_tensors[0].float()


class DirectionalMaskGenerator(nn.Module):
    def __init__(self, image_size, num_angle, num_rho, mask_width=3.0):
        super().__init__()
        H, W = image_size

        # 预计算坐标系统（内存优化版）
        y, x = torch.meshgrid(
            torch.arange(H, dtype=torch.float32) - (H - 1) / 2,
            torch.arange(W, dtype=torch.float32) - (W - 1) / 2,
            indexing='ij'
        )
        self.register_buffer('xy_grid', torch.stack([x, y], 0))  # [2, H, W]

        # 计算参数
        max_rho = torch.sqrt(torch.tensor((W / 2) ** 2 + (H / 2) ** 2))
        self.register_buffer('delta_rho', (2 * max_rho) / (num_rho - 1))
        self.params = (num_angle, num_rho, mask_width)

        # 自适应批处理参数
        self.register_buffer('_dummy', torch.tensor(0), persistent=False)  # 用于获取设备信息

    def forward(self, hough_map):
        num_angle, num_rho, mask_width = self.params
        N, C, A, R = hough_map.shape
        H, W = self.xy_grid.shape[1:]
        device = hough_map.device

        # 峰值检测
        peak_mask = PeakDetection.apply(hough_map)
        n_idx, c_idx, a_idx, r_idx = torch.where(peak_mask)
        P = n_idx.shape[0]

        if P == 0:
            return torch.zeros((N, C, H, W), device=device)

        # 预计算所有直线参数（显存友好）
        theta = a_idx.float() * (torch.pi / num_angle)  # [P]
        rho = (r_idx.float() - num_rho / 2) * self.delta_rho  # [P]
        nc_idx = n_idx * C + c_idx  # [P]

        # 动态批处理策略
        mask = torch.zeros(N * C, H * W, device=device)  # 线性布局
        xy = self.xy_grid.view(2, -1).to(device)  # [2, H*W]

        # 根据显存自动调整批次大小
        total_elements = self._dummy.element_size() * 3 * H * W  # 3个中间张量
        free_mem, _ = torch.cuda.mem_get_info(device)
        batch_size = min(1024, int(free_mem / (total_elements * 1.2)))  # 保留20%余量
        batch_size = max(64, batch_size)  # 最低批次

        for i in range(0, P, batch_size):
            # 当前批次参数
            b_slice = slice(i, min(i + batch_size, P))
            theta_b = theta[b_slice]  # [B]
            rho_b = rho[b_slice]  # [B]
            nc_b = nc_idx[b_slice]  # [B]
            B = theta_b.shape[0]

            # 向量化距离计算（内存优化）
            cos_t = torch.cos(theta_b)  # [B]
            sin_t = torch.sin(theta_b)  # [B]
            normal_vec = torch.stack([cos_t, sin_t], dim=1)  # [B,2]

            # 矩阵乘法计算距离（避免中间张量）
            rho_cal = normal_vec @ xy  # [B, H*W]
            distance = (rho_cal - rho_b[:, None]).abs()  # [B, H*W]
            batch_mask = (distance < mask_width).float()  # [B, H*W]

            # 快速聚合
            if B > 1:
                # 对同一nc的mask取最大值
                unique_nc, inverse_idx = torch.unique(nc_b, return_inverse=True)
                max_mask = torch.zeros(len(unique_nc), H * W, device=device)
                max_mask.scatter_reduce_(0, inverse_idx.unsqueeze(-1).expand(-1, H * W),
                                         batch_mask, reduce='amax', include_self=False)
                # 全局聚合
                mask[unique_nc] = torch.max(mask[unique_nc], max_mask)
            else:
                mask[nc_b] = torch.max(mask[nc_b], batch_mask)

        return mask.view(N, C, H, W)
