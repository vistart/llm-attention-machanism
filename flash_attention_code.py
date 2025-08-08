"""
Flash Attention实现
通过分块计算和IO优化实现高效的注意力计算
注意：这是一个教学版本的实现，真正的Flash Attention需要CUDA kernel级别的优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import warnings


class FlashAttention(nn.Module):
    """
    Flash Attention的PyTorch实现
    核心思想：
    1. 分块计算以减少HBM访问
    2. 在线softmax以保持数值稳定性
    3. 重计算而非存储中间结果
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        block_size_q: int = 64,
        block_size_kv: int = 64,
        dropout: float = 0.0,
        use_triton: bool = False
    ):
        """
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            block_size_q: Query的块大小
            block_size_kv: Key/Value的块大小
            dropout: dropout概率
            use_triton: 是否使用Triton kernel（需要安装triton）
        """
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.block_size_q = block_size_q
        self.block_size_kv = block_size_kv
        self.dropout_p = dropout
        self.use_triton = use_triton

        self.scale = 1.0 / math.sqrt(self.d_k)

        # 线性变换层
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

    def _flash_attention_forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False
    ) -> torch.Tensor:
        """
        Flash Attention的前向传播（分块计算版本）

        Args:
            Q: [batch_size, n_heads, seq_len_q, d_k]
            K: [batch_size, n_heads, seq_len_k, d_k]
            V: [batch_size, n_heads, seq_len_v, d_k]
            mask: 可选的注意力掩码
            is_causal: 是否使用因果掩码

        Returns:
            output: [batch_size, n_heads, seq_len_q, d_k]
        """
        batch_size, n_heads, seq_len_q, d_k = Q.shape
        _, _, seq_len_kv, _ = K.shape

        # 块大小
        Br = min(self.block_size_q, seq_len_q)
        Bc = min(self.block_size_kv, seq_len_kv)

        # 初始化输出和统计量
        O = torch.zeros_like(Q)
        L = torch.zeros((batch_size, n_heads, seq_len_q), device=Q.device)

        # 外层循环：遍历Q的块
        for i in range(0, seq_len_q, Br):
            i_end = min(i + Br, seq_len_q)
            Qi = Q[:, :, i:i_end, :]  # [batch, heads, Br, d_k]

            # 初始化块的输出和统计量
            Oi = torch.zeros_like(Qi)
            Li = torch.zeros((batch_size, n_heads, i_end - i), device=Q.device)
            Mi = torch.full((batch_size, n_heads, i_end - i), float('-inf'), device=Q.device)

            # 内层循环：遍历K,V的块
            for j in range(0, seq_len_kv, Bc):
                j_end = min(j + Bc, seq_len_kv)

                # 因果掩码检查
                if is_causal and j > i_end:
                    break

                Kj = K[:, :, j:j_end, :]  # [batch, heads, Bc, d_k]
                Vj = V[:, :, j:j_end, :]  # [batch, heads, Bc, d_k]

                # 计算注意力分数
                Sij = torch.matmul(Qi, Kj.transpose(-2, -1)) * self.scale  # [batch, heads, Br, Bc]

                # 应用因果掩码
                if is_causal:
                    causal_mask = self._create_causal_mask_block(i, i_end, j, j_end, device=Q.device)
                    Sij = Sij.masked_fill(causal_mask, float('-inf'))

                # 应用自定义掩码
                if mask is not None:
                    mask_block = mask[:, :, i:i_end, j:j_end]
                    Sij = Sij.masked_fill(mask_block == 0, float('-inf'))

                # 在线softmax更新
                Mi_new = torch.max(Mi, Sij.max(dim=-1)[0])  # 新的最大值

                # 计算校正因子（不同用途需要不同的形状）
                correction = torch.exp(Mi - Mi_new)  # [batch, heads, Br]

                # 更新统计量
                Pij = torch.exp(Sij - Mi_new.unsqueeze(-1))  # [batch, heads, Br, Bc]
                Li_new = correction * Li + Pij.sum(dim=-1)  # [batch, heads, Br]

                # 更新输出（这里需要unsqueeze）
                Oi = correction.unsqueeze(-1) * Oi + torch.matmul(Pij, Vj)

                # 更新统计量
                Mi = Mi_new
                Li = Li_new

            # 归一化输出块
            Oi = Oi / Li.unsqueeze(-1).clamp(min=1e-6)

            # 应用dropout（如果有）
            if self.dropout_p > 0 and self.training:
                Oi = self.dropout(Oi)

            # 写回输出
            O[:, :, i:i_end, :] = Oi
            L[:, :, i:i_end] = Li

        return O

    def _create_causal_mask_block(
        self,
        i_start: int,
        i_end: int,
        j_start: int,
        j_end: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        创建因果掩码块

        Returns:
            mask: [1, 1, Br, Bc]的布尔掩码
        """
        Br = i_end - i_start
        Bc = j_end - j_start

        # 创建位置索引
        i_indices = torch.arange(i_start, i_end, device=device).view(Br, 1)
        j_indices = torch.arange(j_start, j_end, device=device).view(1, Bc)

        # 因果掩码：j > i的位置为True（需要被mask）
        mask = j_indices > i_indices

        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            attention_mask: 可选的注意力掩码
            is_causal: 是否使用因果掩码
            return_attention: 是否返回注意力权重（Flash Attention不支持）

        Returns:
            output: 输出张量 [batch_size, seq_len, d_model]
        """
        if return_attention:
            warnings.warn("Flash Attention不支持返回注意力权重，将忽略return_attention参数")

        batch_size, seq_len, _ = x.size()

        # 线性变换并分头
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Flash Attention前向传播
        output = self._flash_attention_forward(Q, K, V, attention_mask, is_causal)

        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)

        return output


class FlashAttentionV2(nn.Module):
    """
    Flash Attention v2
    改进版本，包含更多优化：
    1. 更好的并行化策略
    2. 减少共享内存的使用
    3. 支持不同的注意力偏置
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: Optional[int] = None,
        alibi_slopes: Optional[torch.Tensor] = None
    ):
        """
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            window_size: 局部注意力窗口大小
            alibi_slopes: ALiBi位置偏置的斜率
        """
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.window_size = window_size

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # ALiBi位置偏置
        if alibi_slopes is not None:
            self.register_buffer('alibi_slopes', alibi_slopes)
        else:
            self.alibi_slopes = None

    def _compute_alibi_bias(
        self,
        seq_len_q: int,
        seq_len_k: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        计算ALiBi位置偏置

        Returns:
            bias: [n_heads, seq_len_q, seq_len_k]
        """
        if self.alibi_slopes is None:
            return None

        # 创建位置差矩阵
        q_pos = torch.arange(seq_len_q, device=device).unsqueeze(1)
        k_pos = torch.arange(seq_len_k, device=device).unsqueeze(0)
        relative_pos = q_pos - k_pos  # [seq_len_q, seq_len_k]

        # 应用斜率
        alibi_bias = self.alibi_slopes.unsqueeze(-1).unsqueeze(-1) * relative_pos.unsqueeze(0)

        return alibi_bias

    def forward(self, x: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        # 线性变换并分头
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 这里应该调用优化的CUDA kernel
        # 为了演示，使用标准实现
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 添加ALiBi偏置
        if self.alibi_slopes is not None:
            alibi_bias = self._compute_alibi_bias(seq_len, seq_len, x.device)
            scores = scores + alibi_bias.unsqueeze(0)

        # 局部窗口掩码
        if self.window_size is not None:
            window_mask = self._create_window_mask(seq_len, self.window_size, x.device)
            scores = scores.masked_fill(~window_mask, float('-inf'))

        # 因果掩码
        if is_causal:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
            scores = scores.masked_fill(~causal_mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)

        return output

    def _create_window_mask(
        self,
        seq_len: int,
        window_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        创建局部窗口掩码
        """
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            mask[i, start:end] = True
        return mask


class OptimizedFlashAttention(nn.Module):
    """
    优化的Flash Attention实现
    包含多种内存和计算优化技术
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        use_rope: bool = True,
        use_alibi: bool = False,
        use_xpos: bool = False
    ):
        """
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            use_rope: 是否使用RoPE位置编码
            use_alibi: 是否使用ALiBi位置编码
            use_xpos: 是否使用xPos位置编码
        """
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # 位置编码选项
        self.use_rope = use_rope
        self.use_alibi = use_alibi
        self.use_xpos = use_xpos

        if use_rope:
            self._init_rope()
        if use_alibi:
            self._init_alibi()
        if use_xpos:
            self._init_xpos()

    def _init_rope(self):
        """初始化RoPE"""
        dim = self.d_k
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('rope_inv_freq', inv_freq)

    def _init_alibi(self):
        """初始化ALiBi"""
        slopes = torch.tensor([
            2 ** (-8 * i / self.n_heads) for i in range(self.n_heads)
        ])
        self.register_buffer('alibi_slopes', slopes)

    def _init_xpos(self):
        """初始化xPos"""
        # xPos参数
        self.xpos_scale_base = 512
        self.xpos_scale_factor = 1.0

    def apply_rope(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor
    ) -> torch.Tensor:
        """应用RoPE位置编码"""
        seq_len = position_ids.shape[1]

        # 计算旋转角度
        sincos = torch.einsum('bi,j->bij', position_ids.float(), self.rope_inv_freq)
        sin, cos = sincos.sin(), sincos.cos()

        # 应用旋转
        x_rot = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).flatten(-2)
        x = x * cos.unsqueeze(1) + x_rot * sin.unsqueeze(1)

        return x

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: bool = False
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        # 生成位置ID
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)

        # 线性变换并分头
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 应用位置编码
        if self.use_rope:
            Q = self.apply_rope(Q, position_ids)
            K = self.apply_rope(K, position_ids)

        # 计算注意力（这里使用标准实现，实际应使用优化kernel）
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 应用ALiBi
        if self.use_alibi:
            # 简化的ALiBi实现
            position_bias = self._compute_alibi_bias(seq_len, seq_len, x.device)
            scores = scores + position_bias.unsqueeze(0)

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)

        return output

    def _compute_alibi_bias(
        self,
        seq_len_q: int,
        seq_len_k: int,
        device: torch.device
    ) -> torch.Tensor:
        """计算ALiBi偏置"""
        q_pos = torch.arange(seq_len_q, device=device).unsqueeze(1)
        k_pos = torch.arange(seq_len_k, device=device).unsqueeze(0)
        relative_pos = -(q_pos - k_pos).abs()  # 使用绝对距离

        alibi_bias = self.alibi_slopes.unsqueeze(-1).unsqueeze(-1) * relative_pos.unsqueeze(0)
        return alibi_bias


def benchmark_flash_attention():
    """基准测试Flash Attention性能"""
    import time
    import torch.utils.benchmark as benchmark

    batch_size = 4
    seq_lengths = [512, 1024, 2048, 4096]
    d_model = 512
    n_heads = 8

    print("Flash Attention性能基准测试")
    print("=" * 60)
    print(f"批次大小: {batch_size}, 模型维度: {d_model}, 头数: {n_heads}")
    print("-" * 60)

    for seq_len in seq_lengths:
        print(f"\n序列长度: {seq_len}")
        print("-" * 40)

        x = torch.randn(batch_size, seq_len, d_model).cuda()

        # 标准注意力
        from full_attention_code import MultiHeadAttention
        standard_attn = MultiHeadAttention(d_model, n_heads).cuda()
        standard_attn.eval()

        # Flash Attention
        flash_attn = FlashAttention(d_model, n_heads).cuda()
        flash_attn.eval()

        # 测试标准注意力
        with torch.no_grad():
            start = time.time()
            for _ in range(10):
                _ = standard_attn(x, x, x)
            torch.cuda.synchronize()
            standard_time = time.time() - start

        # 测试Flash Attention
        with torch.no_grad():
            start = time.time()
            for _ in range(10):
                _ = flash_attn(x)
            torch.cuda.synchronize()
            flash_time = time.time() - start

        print(f"标准注意力: {standard_time:.3f}秒")
        print(f"Flash注意力: {flash_time:.3f}秒")
        print(f"加速比: {standard_time/flash_time:.2f}x")

        # 内存使用对比
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = standard_attn(x, x, x)
        standard_memory = torch.cuda.max_memory_allocated() / 1024**2

        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = flash_attn(x)
        flash_memory = torch.cuda.max_memory_allocated() / 1024**2

        print(f"标准注意力内存: {standard_memory:.1f} MB")
        print(f"Flash注意力内存: {flash_memory:.1f} MB")
        print(f"内存节省: {(1 - flash_memory/standard_memory)*100:.1f}%")


# 使用示例
if __name__ == "__main__":
    batch_size = 2
    seq_len = 256
    d_model = 512
    n_heads = 8

    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)

    print("Flash Attention功能测试")
    print("=" * 50)

    # 1. 基础Flash Attention
    flash_attn = FlashAttention(d_model, n_heads, block_size_q=32, block_size_kv=32)
    output = flash_attn(x)
    print(f"Flash Attention输出形状: {output.shape}")

    # 2. 带因果掩码的Flash Attention
    output_causal = flash_attn(x, is_causal=True)
    print(f"因果Flash Attention输出形状: {output_causal.shape}")

    # 3. Flash Attention v2
    flash_v2 = FlashAttentionV2(d_model, n_heads, window_size=64)
    output_v2 = flash_v2(x)
    print(f"Flash Attention v2输出形状: {output_v2.shape}")

    # 4. 优化的Flash Attention（带RoPE）
    optimized_flash = OptimizedFlashAttention(d_model, n_heads, use_rope=True)
    output_opt = optimized_flash(x)
    print(f"优化Flash Attention输出形状: {output_opt.shape}")

    # 5. 性能基准测试（如果有GPU）
    if torch.cuda.is_available():
        print("\n" + "=" * 50)
        benchmark_flash_attention()
    else:
        print("\n注意：GPU不可用，跳过性能基准测试")