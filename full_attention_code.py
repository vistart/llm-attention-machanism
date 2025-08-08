"""
全注意力（Full/Dense Attention）实现
标准的Transformer注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力
    实现公式: Attention(Q,K,V) = softmax(QK^T/√d_k)V
    """

    def __init__(self, d_k: int, dropout: float = 0.1):
        """
        Args:
            d_k: 键的维度
            dropout: dropout概率
        """
        super().__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            Q: torch.Tensor,
            K: torch.Tensor,
            V: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            Q: 查询张量 [batch_size, n_heads, seq_len, d_k]
            K: 键张量 [batch_size, n_heads, seq_len, d_k]
            V: 值张量 [batch_size, n_heads, seq_len, d_v]
            mask: 注意力掩码 [batch_size, 1, seq_len, seq_len]

        Returns:
            output: 注意力输出 [batch_size, n_heads, seq_len, d_v]
            attn_weights: 注意力权重 [batch_size, n_heads, seq_len, seq_len]
        """
        batch_size, n_heads, seq_len, d_k = Q.size()

        # 1. 计算注意力分数: QK^T / √d_k
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        # 2. 应用mask（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 3. 应用softmax获得注意力权重
        attn_weights = F.softmax(scores, dim=-1)

        # 4. 应用dropout
        attn_weights = self.dropout(attn_weights)

        # 5. 计算输出: 注意力权重 × V
        output = torch.matmul(attn_weights, V)

        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    将注意力计算并行化为h个头，每个头学习不同的表示子空间
    """

    def __init__(
            self,
            d_model: int,
            n_heads: int,
            dropout: float = 0.1,
            bias: bool = True
    ):
        """
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            dropout: dropout概率
            bias: 是否使用偏置
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度

        # 线性变换层
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)

        # 注意力计算
        self.attention = ScaledDotProductAttention(self.d_k, dropout)

        # 最终的dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            return_attention: bool = False
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            query: 查询张量 [batch_size, seq_len, d_model]
            key: 键张量 [batch_size, seq_len, d_model]
            value: 值张量 [batch_size, seq_len, d_model]
            mask: 注意力掩码 [batch_size, seq_len, seq_len]
            return_attention: 是否返回注意力权重

        Returns:
            output: 输出张量 [batch_size, seq_len, d_model]
            attn_weights (可选): 注意力权重
        """
        batch_size, seq_len, _ = query.size()

        # 1. 线性变换并分头
        # [batch_size, seq_len, d_model] -> [batch_size, n_heads, seq_len, d_k]
        Q = self.W_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 2. 调整mask形状（如果有）
        if mask is not None:
            mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]

        # 3. 应用缩放点积注意力
        attn_output, attn_weights = self.attention(Q, K, V, mask)

        # 4. 合并多头
        # [batch_size, n_heads, seq_len, d_k] -> [batch_size, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        # 5. 最终线性变换
        output = self.W_o(attn_output)
        output = self.dropout(output)

        if return_attention:
            return output, attn_weights
        return output


class SelfAttention(MultiHeadAttention):
    """
    自注意力层（query, key, value来自同一输入）
    """

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            return_attention: bool = False
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            mask: 注意力掩码
            return_attention: 是否返回注意力权重
        """
        return super().forward(x, x, x, mask, return_attention)


class CrossAttention(MultiHeadAttention):
    """
    交叉注意力层（query来自一个输入，key和value来自另一个输入）
    常用于编码器-解码器架构
    """

    def forward(
            self,
            query: torch.Tensor,
            context: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            return_attention: bool = False
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            query: 查询张量（来自解码器）
            context: 上下文张量（来自编码器）
            mask: 注意力掩码
            return_attention: 是否返回注意力权重
        """
        return super().forward(query, context, context, mask, return_attention)


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    创建因果注意力掩码（用于自回归生成）

    Args:
        seq_len: 序列长度
        device: 设备

    Returns:
        mask: 下三角掩码 [seq_len, seq_len]
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask


def create_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    创建padding掩码

    Args:
        lengths: 每个序列的实际长度 [batch_size]
        max_len: 最大序列长度

    Returns:
        mask: padding掩码 [batch_size, max_len]
    """
    batch_size = lengths.size(0)
    mask = torch.arange(max_len, device=lengths.device).expand(
        batch_size, max_len
    ) < lengths.unsqueeze(1)
    return mask


# 使用示例
if __name__ == "__main__":
    # 设置参数
    batch_size = 2
    seq_len = 10
    d_model = 512
    n_heads = 8

    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)

    # 1. 自注意力
    self_attn = SelfAttention(d_model, n_heads)
    output = self_attn(x)
    print(f"自注意力输出形状: {output.shape}")

    # 2. 带因果掩码的自注意力（用于GPT类模型）
    causal_mask = create_causal_mask(seq_len, x.device)
    output = self_attn(x, mask=causal_mask)
    print(f"因果自注意力输出形状: {output.shape}")

    # 3. 交叉注意力
    encoder_output = torch.randn(batch_size, seq_len, d_model)
    decoder_input = torch.randn(batch_size, seq_len // 2, d_model)
    cross_attn = CrossAttention(d_model, n_heads)
    output = cross_attn(decoder_input, encoder_output)
    print(f"交叉注意力输出形状: {output.shape}")

    # 4. 获取注意力权重用于可视化
    output, attn_weights = self_attn(x, return_attention=True)
    print(f"注意力权重形状: {attn_weights.shape}")