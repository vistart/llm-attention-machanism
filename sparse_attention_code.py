"""
稀疏注意力（Sparse Attention）实现
包括局部窗口注意力、块稀疏注意力、跨步注意力等变体
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class LocalAttention(nn.Module):
    """
    局部窗口注意力（Sliding Window Attention）
    每个位置只关注其周围固定窗口内的位置
    复杂度: O(n * window_size * d)
    """

    def __init__(
            self,
            d_model: int,
            n_heads: int,
            window_size: int,
            dropout: float = 0.1
    ):
        """
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            window_size: 窗口大小（每个位置关注的范围）
            dropout: dropout概率
        """
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size
        self.d_k = d_model // n_heads

        # 线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def create_local_mask(
            self,
            seq_len: int,
            window_size: int,
            device: torch.device
    ) -> torch.Tensor:
        """
        创建局部窗口mask

        Args:
            seq_len: 序列长度
            window_size: 窗口大小
            device: 设备

        Returns:
            mask: [seq_len, seq_len]的mask矩阵
        """
        mask = torch.zeros(seq_len, seq_len, device=device)
        for i in range(seq_len):
            # 每个位置关注[i - window_size//2, i + window_size//2]范围
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            mask[i, start:end] = 1
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, d_model]

        Returns:
            output: 输出张量 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.size()

        # 线性变换并分头
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 创建局部注意力mask
        mask = self.create_local_mask(seq_len, self.window_size, x.device)
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

        # 计算局部注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)

        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)

        return output


class BlockSparseAttention(nn.Module):
    """
    块稀疏注意力
    将序列分成固定大小的块，只在块内计算注意力
    复杂度: O(n * block_size * d)
    """

    def __init__(
            self,
            d_model: int,
            n_heads: int,
            block_size: int,
            dropout: float = 0.1
    ):
        """
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            block_size: 块大小
            dropout: dropout概率
        """
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.block_size = block_size
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, d_model]

        Returns:
            output: 输出张量 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.size()

        # 确保序列长度能被块大小整除
        assert seq_len % self.block_size == 0, f"序列长度{seq_len}必须能被块大小{self.block_size}整除"
        n_blocks = seq_len // self.block_size

        # 线性变换并分头
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 重塑为块
        # [batch_size, n_heads, n_blocks, block_size, d_k]
        Q = Q.view(batch_size, self.n_heads, n_blocks, self.block_size, self.d_k)
        K = K.view(batch_size, self.n_heads, n_blocks, self.block_size, self.d_k)
        V = V.view(batch_size, self.n_heads, n_blocks, self.block_size, self.d_k)

        # 块内注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)

        # 重塑回原始形状
        output = output.view(batch_size, self.n_heads, seq_len, self.d_k)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)

        return output


class StridedAttention(nn.Module):
    """
    跨步/稀疏采样注意力（Strided/Dilated Attention）
    每个位置只关注固定间隔的位置
    """

    def __init__(
            self,
            d_model: int,
            n_heads: int,
            stride: int,
            dropout: float = 0.1
    ):
        """
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            stride: 采样步长
            dropout: dropout概率
        """
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.stride = stride
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def create_strided_mask(
            self,
            seq_len: int,
            stride: int,
            device: torch.device
    ) -> torch.Tensor:
        """
        创建跨步采样mask
        """
        mask = torch.zeros(seq_len, seq_len, device=device)
        for i in range(seq_len):
            # 每个位置关注：自己 + 每隔stride的位置
            mask[i, i] = 1  # 自己
            for j in range(i % stride, seq_len, stride):
                mask[i, j] = 1
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        # 线性变换并分头
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 创建跨步mask
        mask = self.create_strided_mask(seq_len, self.stride, x.device)
        mask = mask.unsqueeze(0).unsqueeze(0)

        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)

        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)

        return output


class LongformerAttention(nn.Module):
    """
    Longformer注意力
    结合局部窗口注意力和全局注意力
    """

    def __init__(
            self,
            d_model: int,
            n_heads: int,
            window_size: int,
            global_attention_indices: Optional[list] = None,
            dropout: float = 0.1
    ):
        """
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            window_size: 局部窗口大小
            global_attention_indices: 全局注意力位置索引
            dropout: dropout概率
        """
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size
        self.global_attention_indices = global_attention_indices or []
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # 全局注意力的额外参数
        self.W_q_global = nn.Linear(d_model, d_model)
        self.W_k_global = nn.Linear(d_model, d_model)
        self.W_v_global = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def create_longformer_mask(
            self,
            seq_len: int,
            window_size: int,
            global_indices: list,
            device: torch.device
    ) -> torch.Tensor:
        """
        创建Longformer的混合mask
        """
        # 先创建局部窗口mask
        mask = torch.zeros(seq_len, seq_len, device=device)

        for i in range(seq_len):
            # 局部窗口
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            mask[i, start:end] = 1

            # 全局注意力
            if i in global_indices:
                mask[i, :] = 1  # 全局位置关注所有位置
                mask[:, i] = 1  # 所有位置关注全局位置

        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        # 线性变换并分头
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 创建混合mask
        mask = self.create_longformer_mask(
            seq_len, self.window_size, self.global_attention_indices, x.device
        )
        mask = mask.unsqueeze(0).unsqueeze(0)

        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)

        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)

        return output


class BigBirdAttention(nn.Module):
    """
    BigBird注意力
    结合随机注意力、局部窗口注意力和全局注意力
    """

    def __init__(
            self,
            d_model: int,
            n_heads: int,
            window_size: int,
            num_random_blocks: int = 3,
            dropout: float = 0.1
    ):
        """
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            window_size: 局部窗口大小
            num_random_blocks: 随机块数量
            dropout: dropout概率
        """
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size
        self.num_random_blocks = num_random_blocks
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def create_bigbird_mask(
            self,
            seq_len: int,
            window_size: int,
            num_random_blocks: int,
            device: torch.device
    ) -> torch.Tensor:
        """
        创建BigBird的混合mask
        包括：局部窗口 + 全局tokens + 随机注意力
        """
        mask = torch.zeros(seq_len, seq_len, device=device)

        # 1. 局部窗口注意力
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            mask[i, start:end] = 1

        # 2. 全局注意力（首尾tokens）
        mask[0, :] = 1  # 第一个token关注所有
        mask[-1, :] = 1  # 最后一个token关注所有
        mask[:, 0] = 1  # 所有关注第一个token
        mask[:, -1] = 1  # 所有关注最后一个token

        # 3. 随机注意力
        for i in range(1, seq_len - 1):  # 除了首尾
            # 随机选择一些位置进行关注
            random_indices = torch.randperm(seq_len)[:num_random_blocks]
            mask[i, random_indices] = 1

        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        # 线性变换并分头
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 创建BigBird mask
        mask = self.create_bigbird_mask(
            seq_len, self.window_size, self.num_random_blocks, x.device
        )
        mask = mask.unsqueeze(0).unsqueeze(0)

        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)

        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)

        return output


# 使用示例
if __name__ == "__main__":
    batch_size = 2
    seq_len = 128
    d_model = 512
    n_heads = 8

    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)

    # 1. 局部窗口注意力
    local_attn = LocalAttention(d_model, n_heads, window_size=16)
    output = local_attn(x)
    print(f"局部注意力输出形状: {output.shape}")

    # 2. 块稀疏注意力
    block_attn = BlockSparseAttention(d_model, n_heads, block_size=16)
    output = block_attn(x)
    print(f"块稀疏注意力输出形状: {output.shape}")

    # 3. 跨步注意力
    strided_attn = StridedAttention(d_model, n_heads, stride=4)
    output = strided_attn(x)
    print(f"跨步注意力输出形状: {output.shape}")

    # 4. Longformer注意力
    longformer_attn = LongformerAttention(
        d_model, n_heads, window_size=16,
        global_attention_indices=[0, seq_len - 1]  # 首尾使用全局注意力
    )
    output = longformer_attn(x)
    print(f"Longformer注意力输出形状: {output.shape}")

    # 5. BigBird注意力
    bigbird_attn = BigBirdAttention(d_model, n_heads, window_size=16, num_random_blocks=3)
    output = bigbird_attn(x)
    print(f"BigBird注意力输出形状: {output.shape}")