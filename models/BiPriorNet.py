import copy
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from models.deep_hough_transform.HT_cuda import HTIHT_Cuda
from models.fpn.backbone.fpn import FPN18, FPN50, FPN101, ResNext50_FPN
from models.fpn.backbone.res2net import res2net50_FPN
from models.fpn.backbone.vgg_fpn import VGG_FPN


# from models.deep_hough_transform.dht import DHT_Layer


class CrossAttention(nn.Module):
    def __init__(self, num_queries, hidden_dim):
        super(CrossAttention, self).__init__()
        self.num_queries = num_queries

    def forward(self, attention_weight, num_queries):
        pass


class QueryBlock(nn.Module):
    def __init__(self, num_queries, d_model=256, nhead=8, dim_feedforward=2048, dropout=0.1, attn_weight=False):
        super(QueryBlock, self).__init__()
        self.cross_attention = TransformerDecoderLayer(d_model, nhead,
                                                       dim_feedforward=dim_feedforward,
                                                       dropout=dropout,
                                                       attn_weight=attn_weight
                                                       )

    def forward(self, src, tgt, query_embed):
        query_object = self.cross_attention(tgt, src, query_pos=query_embed)
        return query_object


class QueryModule(nn.Module):
    def __init__(self, num_queries, num_blocks=3, d_model=128, nhead=8, dim_feedforward=2048,
                 dropout=0.1):
        super(QueryModule, self).__init__()
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.query_blocks_list = _get_clones(
            QueryBlock(num_queries, d_model, nhead, dim_feedforward, dropout),
            num_blocks)
        self.cross_attn = QueryBlock(num_queries, d_model, nhead, dim_feedforward, dropout, attn_weight=True)

    def forward(self, src):
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros_like(query_embed)
        for query_block in self.query_blocks_list:
            tgt = query_block(src, tgt, query_embed)
        tgt = self.cross_attn(src, tgt, query_embed)  # bs * num_queries * h*w
        tgt = tgt.permute(1, 0, 2).view(bs, self.num_queries, h, w)
        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    import torch.nn.functional as F
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, attn_weight=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.attn_weight = attn_weight

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tout = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt2, attn_weight = tout[0], tout[1]
        if self.attn_weight:
            return attn_weight
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class Net(nn.Module):
    def __init__(self,backbone, angle_res=3, rho_res=1, img_size=512, num_queries=10):
        super(Net, self).__init__()
        if backbone == 'resnet18':
            self.backbone = FPN18(pretrained=True, output_stride=32)
            output_stride = 32
        if backbone == 'resnet50':
            self.backbone = FPN50(pretrained=True, output_stride=16)
            output_stride = 16
        if backbone == 'resnet101':
            self.backbone = FPN101(output_stride=16)
            output_stride = 16
        if backbone == 'resnext50':
            self.backbone = ResNext50_FPN(output_stride=16)
            output_stride = 16
        if backbone == 'vgg16':
            self.backbone = VGG_FPN()
            output_stride = 16
        if backbone == 'res2net50':
            self.backbone = res2net50_FPN()
            output_stride = 32

        self.query_module1 = QueryModule(num_queries, d_model=256)
        self.query_module2 = QueryModule(num_queries, d_model=256)
        self.query_module3 = QueryModule(num_queries, d_model=256)
        self.query_module4 = QueryModule(num_queries, d_model=256)

        self.ht1 = HTIHT_Cuda(num_queries, num_queries, img_size // 4, img_size // 4, angle_res, rho_res)
        self.ht2 = HTIHT_Cuda(num_queries, num_queries, img_size // 8, img_size // 8, angle_res, rho_res)
        self.ht3 = HTIHT_Cuda(num_queries, num_queries, img_size // 16, img_size // 16, angle_res, rho_res)
        self.ht4 = HTIHT_Cuda(num_queries, num_queries, img_size // 16, img_size // 16, angle_res, rho_res)

    def forward(self, x):
        p1, p2, p3, p4 = self.backbone(x)

        p1_query = self.query_module1(p1)
        p1 = self.ht1(p1_query)
        p2_query = self.query_module2(p2)
        p2 = self.ht2(p2_query)
        p3_query = self.query_module3(p3)
        p3 = self.ht3(p3_query)
        p4_query = self.query_module4(p4)
        p4 = self.ht4(p4_query)
        # cat = self.upsample_cat(p1, p2, p3, p4)
        # logist = self.last_conv(cat)

        logist = None
        return logist


if __name__ == '__main__':
    net = Net("resnet50").cuda()
    x = torch.randn(1, 3, 512, 512).cuda()
    x = net(x)
    params = list(net.parameters())
    num_params = 0
    for param in params:
        curr_num_params = 1
        for size_count in param.size():
            curr_num_params *= size_count
        num_params += curr_num_params
    print("total number of parameters: " + str(num_params))
#
# if __name__ == '__main__':
#     qm = QueryModule(num_queries=10).cuda()
#     x = torch.randn(1, 128, 32, 32).cuda()
#     x = qm(x)
#     print(x.shape)
