"""
分组查询注意力（GQA）和多查询注意力（MQA）实现
用于优化推理时的KV缓存内存占用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class KVCache:
    """KV缓存数据结构"""
    k_cache: torch.Tensor  # [batch_size, n_kv_heads, max_seq_len, d_k]
    v_cache: torch.Tensor  # [batch_size, n_kv_heads, max_seq_len, d_k]
    seq_len: int  # 当前缓存的序列长度


class GroupedQueryAttention(nn.Module):
    """
    分组查询注意力（GQA）
    将查询头分成G组，每组共享一个键值对
    在MHA和MQA之间的折中方案
    """

    def __init__(
            self,
            d_model: int,
            n_heads: int,
            n_kv_heads: int,
            dropout: float = 0.1,
            max_seq_len: int = 2048,
            use_cache: bool = False
    ):
        """
        Args:
            d_model: 模型维度
            n_heads: 查询头的数量
            n_kv_heads: 键值头的数量（必须能整除n_heads）
            dropout: dropout概率
            max_seq_len: 最大序列长度（用于缓存）
            use_cache: 是否使用KV缓存
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"
        assert n_heads % n_kv_heads == 0, "n_heads必须能被n_kv_heads整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads  # 每个KV头服务的Q头数量
        self.d_k = d_model // n_heads
        self.max_seq_len = max_seq_len
        self.use_cache = use_cache

        # Q使用全部头，K和V使用较少的头
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

        # 可选的RoPE（旋转位置编码）
        self.rope_theta = 10000.0
        self._precompute_rope_cache()

    def _precompute_rope_cache(self):
        """预计算RoPE的频率"""
        dim = self.d_k
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def apply_rope(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """
        应用旋转位置编码

        Args:
            x: 输入张量 [batch, heads, seq_len, d_k]
            position_ids: 位置索引 [batch, seq_len]
        """
        batch_size, n_heads, seq_len, d_k = x.shape

        # 计算旋转角度
        sincos = torch.einsum('bi,j->bij', position_ids.float(), self.inv_freq)
        sin, cos = sincos.sin(), sincos.cos()

        # 扩展到正确的形状
        sin = sin.unsqueeze(1).repeat(1, n_heads, 1, 1)
        cos = cos.unsqueeze(1).repeat(1, n_heads, 1, 1)

        # 应用旋转
        x_rot = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).flatten(-2)
        x_out = x * cos + x_rot * sin

        return x_out

    def forward(
            self,
            x: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            past_kv_cache: Optional[KVCache] = None,
            use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            attention_mask: 注意力掩码 [batch_size, seq_len, seq_len]
            position_ids: 位置索引 [batch_size, seq_len]
            past_kv_cache: 之前的KV缓存
            use_cache: 是否返回更新的KV缓存

        Returns:
            output: 输出张量 [batch_size, seq_len, d_model]
            new_cache: 更新的KV缓存（如果use_cache=True）
        """
        batch_size, seq_len, _ = x.size()

        # 1. 计算Q（全部头）
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 2. 计算K和V（较少的头）
        K = self.W_k(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)

        # 3. 应用位置编码（如果提供）
        if position_ids is not None:
            Q = self.apply_rope(Q, position_ids)
            K = self.apply_rope(K, position_ids)

        # 4. 处理KV缓存
        if past_kv_cache is not None:
            # 使用缓存的K和V
            past_seq_len = past_kv_cache.seq_len
            K_cache = past_kv_cache.k_cache[:, :, :past_seq_len, :]
            V_cache = past_kv_cache.v_cache[:, :, :past_seq_len, :]

            # 拼接新旧KV
            K = torch.cat([K_cache, K], dim=2)
            V = torch.cat([V_cache, V], dim=2)

            # 更新缓存
            if use_cache:
                new_cache = KVCache(
                    k_cache=K.clone(),
                    v_cache=V.clone(),
                    seq_len=past_seq_len + seq_len
                )
        else:
            new_cache = None
            if use_cache:
                new_cache = KVCache(
                    k_cache=K.clone(),
                    v_cache=V.clone(),
                    seq_len=seq_len
                )

        # 5. 重复K和V以匹配Q的头数
        # 每个KV头服务n_groups个Q头
        K = K.repeat_interleave(self.n_groups, dim=1)
        V = V.repeat_interleave(self.n_groups, dim=1)

        # 6. 标准注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 应用掩码
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)

        # 7. 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)

        return output, new_cache


class MultiQueryAttention(nn.Module):
    """
    多查询注意力（MQA）
    所有查询头共享单一的键值对
    最大程度减少KV缓存
    """

    def __init__(
            self,
            d_model: int,
            n_heads: int,
            dropout: float = 0.1,
            max_seq_len: int = 2048,
            use_cache: bool = False
    ):
        """
        Args:
            d_model: 模型维度
            n_heads: 查询头的数量
            dropout: dropout概率
            max_seq_len: 最大序列长度
            use_cache: 是否使用KV缓存
        """
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.max_seq_len = max_seq_len
        self.use_cache = use_cache

        # Q使用多头，K和V使用单头
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, self.d_k, bias=False)  # 只输出d_k维
        self.W_v = nn.Linear(d_model, self.d_k, bias=False)  # 只输出d_k维
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            x: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            past_kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            attention_mask: 注意力掩码
            past_kv_cache: (k_cache, v_cache)元组
            use_cache: 是否返回KV缓存

        Returns:
            output: 输出张量
            new_cache: 更新的KV缓存
        """
        batch_size, seq_len, _ = x.size()

        # 1. 计算Q（多头）
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 2. 计算K和V（单头）
        K = self.W_k(x).view(batch_size, seq_len, 1, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, 1, self.d_k).transpose(1, 2)

        # 3. 处理KV缓存
        if past_kv_cache is not None:
            past_k, past_v = past_kv_cache
            K = torch.cat([past_k, K], dim=2)
            V = torch.cat([past_v, V], dim=2)

        new_cache = None
        if use_cache:
            new_cache = (K.clone(), V.clone())

        # 4. 扩展K和V以匹配Q的头数
        K = K.expand(-1, self.n_heads, -1, -1)
        V = V.expand(-1, self.n_heads, -1, -1)

        # 5. 标准注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)

        # 6. 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)

        return output, new_cache


class AdaptiveAttention(nn.Module):
    """
    自适应注意力
    根据输入动态选择MHA、GQA或MQA
    """

    def __init__(
            self,
            d_model: int,
            n_heads: int,
            n_kv_heads_options: list = [8, 4, 1],  # MHA, GQA, MQA
            dropout: float = 0.1
    ):
        """
        Args:
            d_model: 模型维度
            n_heads: 查询头数量
            n_kv_heads_options: KV头数量选项列表
            dropout: dropout概率
        """
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads_options = n_kv_heads_options

        # 创建不同配置的注意力层
        self.attention_layers = nn.ModuleList([
            GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout)
            for n_kv_heads in n_kv_heads_options
        ])

        # 路由网络（决定使用哪种注意力）
        self.router = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, len(n_kv_heads_options))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        # 使用输入的平均池化作为路由信号
        routing_input = x.mean(dim=1)  # [batch_size, d_model]
        routing_logits = self.router(routing_input)  # [batch_size, num_options]
        routing_probs = F.softmax(routing_logits, dim=-1)

        # 选择最可能的注意力配置
        selected_idx = routing_probs.argmax(dim=-1)

        # 动态批处理：将相同选择的样本分组
        outputs = torch.zeros_like(x)
        for idx in range(len(self.n_kv_heads_options)):
            mask = (selected_idx == idx)
            if mask.any():
                batch_input = x[mask]
                batch_output, _ = self.attention_layers[idx](batch_input)
                outputs[mask] = batch_output

        return outputs


def compare_memory_usage():
    """比较不同注意力机制的内存使用"""
    batch_size = 1
    seq_len = 2048
    d_model = 512
    n_heads = 8

    print("KV缓存内存占用比较:")
    print("-" * 50)

    # MHA（标准多头注意力）
    mha_kv_size = 2 * batch_size * n_heads * seq_len * (d_model // n_heads) * 4  # float32
    print(f"MHA (8个KV头): {mha_kv_size / (1024 ** 2):.2f} MB")

    # GQA with 4 KV heads
    gqa4_kv_size = 2 * batch_size * 4 * seq_len * (d_model // n_heads) * 4
    print(f"GQA (4个KV头): {gqa4_kv_size / (1024 ** 2):.2f} MB")
    print(f"  节省: {(1 - gqa4_kv_size / mha_kv_size) * 100:.1f}%")

    # GQA with 2 KV heads
    gqa2_kv_size = 2 * batch_size * 2 * seq_len * (d_model // n_heads) * 4
    print(f"GQA (2个KV头): {gqa2_kv_size / (1024 ** 2):.2f} MB")
    print(f"  节省: {(1 - gqa2_kv_size / mha_kv_size) * 100:.1f}%")

    # MQA
    mqa_kv_size = 2 * batch_size * 1 * seq_len * (d_model // n_heads) * 4
    print(f"MQA (1个KV头): {mqa_kv_size / (1024 ** 2):.2f} MB")
    print(f"  节省: {(1 - mqa_kv_size / mha_kv_size) * 100:.1f}%")


def benchmark_inference_speed():
    """基准测试推理速度"""
    import time

    batch_size = 4
    seq_len = 512
    d_model = 512
    n_heads = 8
    num_iterations = 100

    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)

    print("\n推理速度基准测试:")
    print("-" * 50)

    # MHA (GQA with n_kv_heads = n_heads)
    mha = GroupedQueryAttention(d_model, n_heads, n_kv_heads=n_heads)
    mha.eval()

    start = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _, _ = mha(x)
    mha_time = time.time() - start
    print(f"MHA: {mha_time:.3f}秒 (基准)")

    # GQA with 4 KV heads
    gqa4 = GroupedQueryAttention(d_model, n_heads, n_kv_heads=4)
    gqa4.eval()

    start = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _, _ = gqa4(x)
    gqa4_time = time.time() - start
    print(f"GQA (4 KV heads): {gqa4_time:.3f}秒 ({(mha_time / gqa4_time - 1) * 100:.1f}%加速)")

    # MQA
    mqa = MultiQueryAttention(d_model, n_heads)
    mqa.eval()

    start = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _, _ = mqa(x)
    mqa_time = time.time() - start
    print(f"MQA: {mqa_time:.3f}秒 ({(mha_time / mqa_time - 1) * 100:.1f}%加速)")


# 使用示例
if __name__ == "__main__":
    batch_size = 2
    seq_len = 128
    d_model = 512
    n_heads = 8

    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)

    print("功能测试:")
    print("-" * 50)

    # 1. GQA测试
    gqa = GroupedQueryAttention(d_model, n_heads, n_kv_heads=2)
    output, cache = gqa(x, use_cache=True)
    print(f"GQA输出形状: {output.shape}")
    if cache:
        print(f"GQA KV缓存形状: K={cache.k_cache.shape}, V={cache.v_cache.shape}")

    # 2. MQA测试
    mqa = MultiQueryAttention(d_model, n_heads)
    output, cache = mqa(x, use_cache=True)
    print(f"MQA输出形状: {output.shape}")
    if cache:
        print(f"MQA KV缓存形状: K={cache[0].shape}, V={cache[1].shape}")

    # 3. 自适应注意力测试
    adaptive = AdaptiveAttention(d_model, n_heads, n_kv_heads_options=[8, 4, 1])
    output = adaptive(x)
    print(f"自适应注意力输出形状: {output.shape}")

    # 4. 内存使用比较
    print("\n" + "=" * 50)
    compare_memory_usage()

    # 5. 速度基准测试
    print("\n" + "=" * 50)
    benchmark_inference_speed()