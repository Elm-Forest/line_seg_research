import deep_hough as dh
import torch


class C_dht_Function(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, feat, numangle, numrho):
        N, C, _, _ = feat.size()
        out = torch.zeros(N, C, numangle, numrho).type_as(feat).cuda()
        out = dh.forward(feat, out, numangle, numrho)
        outputs = out[0]
        ctx.save_for_backward(feat)
        ctx.numangle = numangle
        ctx.numrho = numrho
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        feat = ctx.saved_tensors[0]
        numangle = ctx.numangle
        numrho = ctx.numrho
        out = torch.zeros_like(feat).type_as(feat).cuda()
        out = dh.backward(grad_output.contiguous(), out, feat, numangle, numrho)
        grad_in = out[0]
        return grad_in, None, None


class C_dht(torch.nn.Module):
    def __init__(self, numAngle, numRho):
        super(C_dht, self).__init__()
        self.numAngle = numAngle
        self.numRho = numRho

    def forward(self, feat):
        return C_dht_Function.apply(feat, self.numAngle, self.numRho)
