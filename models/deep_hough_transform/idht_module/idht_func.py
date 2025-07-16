import deep_inverse_hough as idh

import torch


class C_idht_Function(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, accumulator, H, W, numangle, numrho):
        """
        accumulator: [N, C, numangle, numrho]
        H, W: 输出图像大小
        numangle, numrho: Hough 空间大小信息
        返回: output: [N, C, H, W]
        """
        # 1) 创建与预期逆变换结果相同形状的张量
        N, C, _, _ = accumulator.shape  # accumulator.shape = (N, C, numangle, numrho)
        output = torch.zeros(
            (N, C, H, W),
            device=accumulator.device,
            dtype=accumulator.dtype
        )
        # 2) 调用 C++/CUDA 的 forward（inverse_accum_forward）
        #    签名: forward(hough, output, numangle, numrho) -> [output]
        output = idh.forward(accumulator, output, numangle, numrho)[0]

        # 3) 存储必要信息，用于 backward
        #    - 这里不需要存值本身, 只存 accumulator 的形状(以便反向构造 grad_accumulator)
        ctx.numangle = numangle
        ctx.numrho = numrho
        ctx.shape_accumulator = accumulator.shape  # (N, C, numangle, numrho)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: [N, C, H, W]  <-- 对 forward 返回 output 的梯度
        返回: grad w.r.t. (accumulator, H, W, numangle, numrho)
              其中只有 accumulator 有梯度, 其他都是 None
        """
        numangle = ctx.numangle
        numrho = ctx.numrho

        # 从 ctx.shape_accumulator 获得梯度张量大小
        N, C, _, _ = ctx.shape_accumulator  # (N, C, numangle, numrho)

        # 构造与 accumulator 形状相同的梯度张量
        grad_accumulator = torch.zeros(
            (N, C, numangle, numrho),
            device=grad_output.device,
            dtype=grad_output.dtype
        )

        # 调用 C++/CUDA 的 backward（inverse_accum_backward）
        #   签名: backward(grad_out, grad_in, numangle, numrho) -> [grad_in]
        grad_accumulator = idh.backward(
            grad_output.contiguous(),
            grad_accumulator,
            numangle,
            numrho
        )[0]

        # 与 forward 的 5 个输入 (accumulator, H, W, numangle, numrho) 对应
        #   accumulator 的梯度 = grad_accumulator
        #   其余 4 个输入不需要梯度 => None
        return grad_accumulator, None, None, None, None


class C_idht(torch.nn.Module):
    """
    一个简单的 PyTorch Module 封装:
    仅保存 numangle, numrho, out_H, out_W 等参数,
    在 forward 时调用上面的 C_idht_Function
    """

    def __init__(self, numangle, numrho, out_H=64, out_W=64):
        super(C_idht, self).__init__()
        self.numangle = numangle
        self.numrho = numrho
        self.out_H = out_H
        self.out_W = out_W

    def forward(self, accumulator):
        """
        accumulator: [N, C, numangle, numrho]
        returns: [N, C, out_H, out_W]
        """
        return C_idht_Function.apply(
            accumulator, self.out_H, self.out_W,
            self.numangle, self.numrho
        )
