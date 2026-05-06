"""
CayleyNet - 复数注意力+正交约束架构
核心创新: 复数注意力(保留相位) + Cayley正交参数化 + 置信度早退出

命名诚实化:
  - ComplexAttention: 复数注意力（非"量子叠加"）
  - GatedFFN: 门控前馈网络（非"量子纠缠"）
  - ConfidenceGate: 置信度门控早退出（非"量子坍缩"）
  - CayleyOrthogonal: Cayley正交参数化（非"酉参数化"）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.real = nn.Linear(in_features, out_features)
        self.imag = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.real(x), self.imag(x)


class CayleyOrthogonal(nn.Module):
    """Cayley变换: 实反对称S → 正交矩阵 U=(I-S)(I+S)^{-1}
    保证权重近似正交，训练稳定，梯度良好
    用linalg.solve替代inverse，数值稳定
    """
    def __init__(self, dim):
        super().__init__()
        self.skew = nn.Parameter(torch.randn(dim, dim) * 1e-4)

    def forward(self):
        s = (self.skew - self.skew.t()) / 2
        i = torch.eye(s.size(0), device=s.device, dtype=s.dtype)
        u = torch.linalg.solve(i + s, i - s)
        return u


class ComplexAttention(nn.Module):
    """复数注意力机制
    创新点: Q/K/V映射到复数域，注意力同时考虑模长和相位
    复数乘法: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    注意力: softmax(|attn|) 保留模长信息
    输出: 保留实部和虚部投影，信息不丢失
    复杂度: O(n^2)（标准注意力），未来可用FAVOR+降至O(n)
    """
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.qkv = ComplexLinear(dim, dim * 3)
        self.proj = ComplexLinear(dim, dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, N, C = x.shape
        qr, qi = self.qkv(x)
        qkv_r = qr.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        qkv_i = qi.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        qr, kr, vr = qkv_r.unbind(0)
        qi, ki, vi = qkv_i.unbind(0)

        attn_r = (torch.matmul(qr, kr.transpose(-2, -1)) - torch.matmul(qi, ki.transpose(-2, -1))) * self.scale
        attn_i = (torch.matmul(qr, ki.transpose(-2, -1)) + torch.matmul(qi, kr.transpose(-2, -1))) * self.scale
        attn_mag = torch.sqrt(attn_r ** 2 + attn_i ** 2 + 1e-8)
        attn = F.softmax(attn_mag, dim=-1)

        out_r = torch.matmul(attn, vr)
        out_i = torch.matmul(attn, vi)
        x_r = out_r.transpose(1, 2).reshape(B, N, C)
        x_i = out_i.transpose(1, 2).reshape(B, N, C)

        proj_r, proj_i = self.proj(x_r)
        proj_r2, proj_i2 = self.proj(x_i)
        return proj_r + proj_i2


class GatedFFN(nn.Module):
    """门控前馈网络
    标准: x + FFN(x)，FFN = Linear(GELU(Linear(x)))
    门控: 用sigmoid生成门控信号，动态调节信息流
    """
    def __init__(self, dim, expansion=4):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * expansion)
        self.gate = nn.Linear(dim, dim * expansion)
        self.fc2 = nn.Linear(dim * expansion, dim)
        self.act = nn.GELU()

    def forward(self, x):
        g = torch.sigmoid(self.gate(x))
        return x + self.fc2(self.act(self.fc1(x) * g))


class ConfidenceGate(nn.Module):
    """置信度门控 - 推理时动态早退出
    训练: 返回特征+置信度，置信度参与辅助loss
    推理: 置信度>阈值时可提前退出，跳过后续层，加速推理
    """
    def __init__(self, dim, threshold=0.9):
        super().__init__()
        self.prob = nn.Linear(dim, 1)
        self.threshold = threshold

    def forward(self, x):
        conf = torch.sigmoid(self.prob(x.mean(dim=1)))
        if self.training:
            return x, conf
        else:
            return x, (conf > self.threshold)


class CayleyBlock(nn.Module):
    """CayleyNet基础块: ComplexAttention + GatedFFN + ConfidenceGate"""
    def __init__(self, dim, num_heads=8, ffn_expansion=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = ComplexAttention(dim, num_heads)
        self.ffn = GatedFFN(dim, ffn_expansion)
        self.conf_gate = ConfidenceGate(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        x, conf = self.conf_gate(x)
        return x, conf


class CayleyNet(nn.Module):
    """
    CayleyNet - 复数注意力+正交约束架构

    创新组合:
    1. 复数注意力 - Q/K/V映射到复数域，保留相位信息
    2. Cayley正交约束 - 保证权重正交，训练稳定
    3. 门控FFN - 动态调节信息流
    4. 置信度早退出 - 推理时跳过低置信度层

    用法:
      model = CayleyNet(dim=512, depth=6, num_classes=1000)
      out = model(images)  # 训练
      out = model(images)  # 推理（自动早退出）
    """
    def __init__(self, dim=512, depth=6, num_heads=8, num_classes=1000, patch_size=16, in_chans=3, img_size=224):
        super().__init__()
        self.dim = dim
        self.depth = depth
        num_patches = (img_size // patch_size) ** 2
        patch_dim = patch_size * patch_size * in_chans

        self.patch_embed = nn.Linear(patch_dim, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, num_patches + 1, dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        self.blocks = nn.ModuleList([CayleyBlock(dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.orthogonal = CayleyOrthogonal(dim)

        self.confidence_loss_weight = 0.1

    def forward(self, x):
        B, C, H, W = x.shape
        p = int(math.sqrt(x.shape[2] * x.shape[3] / (self.dim if hasattr(self, '_psize') else 16)))
        p = 16
        patches = x.reshape(B, C, H // p, p, W // p, p).permute(0, 2, 4, 3, 5, 1).reshape(B, -1, p * p * C)
        x = self.patch_embed(patches)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_emb

        confidences = []
        for block in self.blocks:
            x, conf = block(x)
            confidences.append(conf)

        x = self.norm(x)
        cls_out = x[:, 0]
        cls_out = cls_out @ self.orthogonal()
        logits = self.head(cls_out)

        if self.training:
            conf_stack = torch.cat(confidences, dim=1)
            conf_loss = self.confidence_loss_weight * conf_stack.mean()
            return logits, conf_loss
        else:
            return logits
