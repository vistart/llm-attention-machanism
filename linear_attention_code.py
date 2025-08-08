"""
线性注意力（Linear Attention）实现
包括Performer、Linformer、Linear Transformer等变体
通过核技巧或低秩分解实现O(n)复杂度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Callable
import numpy as np


class LinearAttention(nn.Module):
    """
    基础线性注意力
    使用特征映射将注意力计算从O(n²d)降到O(nd²)
    """

    def __init__(
            self,
            d_model: int,
            n_heads: int,
            feature_map: str = 'elu',
            eps: float = 1e-6
    ):
        """
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            feature_map: 特征映射类型 ('elu', 'relu', 'identity')
            eps: 数值稳定性参数
        """
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.eps = eps

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.feature_map = self._get_feature_map(feature_map)

    def _get_feature_map(self, feature_map: str) -> Callable:
        """选择特征映射函数"""
        if feature_map == 'elu':
            return lambda x: F.elu(x) + 1
        elif feature_map == 'relu':
            return F.relu
        elif feature_map == 'identity':
            return lambda x: x
        else:
            raise ValueError(f"未知的特征映射: {feature_map}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        核心思想：改变计算顺序，先计算K^T V，再乘以Q
        """
        batch_size, seq_len, _ = x.size()

        # 线性变换并分头
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 应用特征映射（使注意力非负）
        Q = self.feature_map(Q)
        K = self.feature_map(K)

        # 线性注意力计算
        # 1. 先计算 K^T V: [batch, heads, d_k, d_k]
        KV = torch.matmul(K.transpose(-2, -1), V)

        # 2. 计算归一化因子: [batch, heads, seq_len, 1]
        Z = 1 / (torch.einsum('bhnd,bhd->bhn', Q, K.sum(dim=2)).unsqueeze(-1) + self.eps)

        # 3. 计算输出: Q(K^T V)
        output = torch.matmul(Q, KV) * Z

        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)

        return output


class PerformerAttention(nn.Module):
    """
    Performer注意力
    使用随机特征（FAVOR+）近似softmax核
    论文: Rethinking Attention with Performers
    """

    def __init__(
            self,
            d_model: int,
            n_heads: int,
            nb_features: Optional[int] = None,
            ortho_scaling: float = 0.0,
            eps: float = 1e-6
    ):
        """
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            nb_features: 随机特征数量（默认为d_k）
            ortho_scaling: 正交化缩放因子
            eps: 数值稳定性参数
        """
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.nb_features = nb_features or self.d_k
        self.ortho_scaling = ortho_scaling
        self.eps = eps

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # 创建随机投影矩阵
        self.create_projection_matrix()

    def create_projection_matrix(self):
        """创建随机高斯投影矩阵"""
        # 使用正交化的随机矩阵以获得更好的近似
        projection = torch.randn(self.nb_features, self.d_k)

        if self.ortho_scaling > 0.0:
            # QR分解获得正交矩阵
            q, _ = torch.qr(projection.T)
            projection = q.T * math.sqrt(self.nb_features)

        self.register_buffer('projection_matrix', projection)

    def softmax_kernel_transformation(
            self,
            data: torch.Tensor,
            is_query: bool,
            projection_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        FAVOR+算法的核心：随机特征近似
        将softmax核近似为随机特征的内积
        """
        # data: [batch, heads, seq_len, d_k]
        # projection: [nb_features, d_k]

        # 计算数据的范数用于缩放
        data_normalizer = 1.0 / math.sqrt(math.sqrt(self.d_k))
        data = data * data_normalizer

        # 计算随机特征: [batch, heads, seq_len, nb_features]
        data_proj = torch.matmul(data, projection_matrix.T)

        # 应用非线性（ReLU用于正定性）
        if is_query:
            # 查询使用不同的非线性以保持方差
            data_proj = F.relu(data_proj) + 1e-6
        else:
            data_proj = F.relu(data_proj) + 1e-6

        return data_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        # 线性变换并分头
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 应用FAVOR+核变换
        Q = self.softmax_kernel_transformation(Q, True, self.projection_matrix)
        K = self.softmax_kernel_transformation(K, False, self.projection_matrix)

        # 线性注意力计算
        # K: [batch, heads, seq_len, nb_features]
        # V: [batch, heads, seq_len, d_k]
        # KV: [batch, heads, nb_features, d_k]
        KV = torch.matmul(K.transpose(-2, -1), V)

        # Q: [batch, heads, seq_len, nb_features]
        # QKV: [batch, heads, seq_len, d_k]
        QKV = torch.matmul(Q, KV)

        # 归一化
        Z = 1.0 / (torch.einsum('bhnf,bhf->bhn', Q, K.sum(dim=2)).unsqueeze(-1) + self.eps)
        output = QKV * Z

        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)

        return output


class LinformerAttention(nn.Module):
    """
    Linformer注意力
    使用低秩投影将键值序列长度从n降到k
    论文: Linformer: Self-Attention with Linear Complexity
    """

    def __init__(
            self,
            d_model: int,
            n_heads: int,
            seq_len: int,
            k: int,
            share_kv_projection: bool = False
    ):
        """
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            seq_len: 最大序列长度
            k: 投影维度（k << seq_len）
            share_kv_projection: 是否共享K和V的投影矩阵
        """
        super().__init__()
        assert d_model % n_heads == 0
        assert k < seq_len, "投影维度k必须小于序列长度"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.seq_len = seq_len
        self.k = k

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # 低秩投影矩阵
        if share_kv_projection:
            self.E = nn.Parameter(torch.randn(seq_len, k))
            self.F = self.E  # 共享投影
        else:
            self.E = nn.Parameter(torch.randn(seq_len, k))  # K的投影
            self.F = nn.Parameter(torch.randn(seq_len, k))  # V的投影

        # 初始化投影矩阵
        nn.init.xavier_uniform_(self.E)
        if not share_kv_projection:
            nn.init.xavier_uniform_(self.F)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        # 处理变长序列
        if seq_len != self.seq_len:
            # 动态调整投影矩阵大小
            E = F.adaptive_avg_pool1d(
                self.E.unsqueeze(0).transpose(1, 2),
                seq_len
            ).transpose(1, 2).squeeze(0)
            F_proj = F.adaptive_avg_pool1d(
                self.F.unsqueeze(0).transpose(1, 2),
                seq_len
            ).transpose(1, 2).squeeze(0) if self.F is not self.E else E
        else:
            E = self.E
            F_proj = self.F

        # 线性变换并分头
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 低秩投影：将K和V的序列维度从n降到k
        # K: [batch, heads, seq_len, d_k] -> [batch, heads, k, d_k]
        K = torch.matmul(E.T.unsqueeze(0).unsqueeze(0), K)
        # V: [batch, heads, seq_len, d_k] -> [batch, heads, k, d_k]
        V = torch.matmul(F_proj.T.unsqueeze(0).unsqueeze(0), V)

        # 标准注意力计算（但K和V的维度已降低）
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)

        return output


class NystromAttention(nn.Module):
    """
    Nyström注意力
    使用Nyström方法近似注意力矩阵
    论文: Nyströmformer: A Nyström-based Algorithm for Approximating Self-Attention
    """

    def __init__(
            self,
            d_model: int,
            n_heads: int,
            num_landmarks: int,
            eps: float = 1e-6
    ):
        """
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            num_landmarks: 地标点数量（采样点）
            eps: 数值稳定性参数
        """
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.num_landmarks = num_landmarks
        self.eps = eps

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        # 线性变换并分头
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 选择地标点（可以是随机或学习的）
        if seq_len <= self.num_landmarks:
            # 序列太短，使用标准注意力
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            attn_weights = F.softmax(scores, dim=-1)
            output = torch.matmul(attn_weights, V)
        else:
            # 使用Nyström近似
            # 1. 选择地标点索引（这里使用均匀采样）
            landmark_indices = torch.linspace(
                0, seq_len - 1, self.num_landmarks, dtype=torch.long, device=x.device
            )

            # 2. 获取地标点的Q, K, V
            Q_landmarks = Q[:, :, landmark_indices]  # [batch, heads, m, d_k]
            K_landmarks = K[:, :, landmark_indices]  # [batch, heads, m, d_k]

            # 3. 计算三个关键矩阵
            # A: Q与地标K的注意力 [batch, heads, n, m]
            A = torch.matmul(Q, K_landmarks.transpose(-2, -1)) / math.sqrt(self.d_k)
            A = F.softmax(A, dim=-1)

            # B: 地标Q与地标K的注意力 [batch, heads, m, m]
            B = torch.matmul(Q_landmarks, K_landmarks.transpose(-2, -1)) / math.sqrt(self.d_k)
            B = F.softmax(B, dim=-1)

            # 4. 计算伪逆
            B_inv = torch.pinverse(B + self.eps * torch.eye(
                self.num_landmarks, device=B.device
            ).unsqueeze(0).unsqueeze(0))

            # 5. Nyström近似：A @ B^(-1) @ A^T @ V
            # 先计算 A @ B^(-1) @ A^T
            attention_matrix_approx = torch.matmul(torch.matmul(A, B_inv), A.transpose(-2, -1))

            # 归一化
            attention_matrix_approx = attention_matrix_approx / (
                    attention_matrix_approx.sum(dim=-1, keepdim=True) + self.eps
            )

            # 应用到V
            output = torch.matmul(attention_matrix_approx, V)

        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)

        return output


class CosformerAttention(nn.Module):
    """
    Cosformer注意力
    使用余弦重加权实现线性注意力
    论文: COSFORMER: Rethinking Softmax in Attention
    """

    def __init__(
            self,
            d_model: int,
            n_heads: int,
            eps: float = 1e-6
    ):
        """
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            eps: 数值稳定性参数
        """
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.eps = eps

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def cos_sim(self, x: torch.Tensor) -> torch.Tensor:
        """计算余弦相似度核"""
        # 使用ReLU作为特征映射
        return F.relu(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        # 线性变换并分头
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 应用余弦核
        Q = self.cos_sim(Q)
        K = self.cos_sim(K)

        # 添加位置编码的重加权
        # 生成位置编码
        position_indices = torch.arange(seq_len, device=x.device).float()
        position_indices = position_indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1)

        # 应用指数衰减
        decay = torch.exp(-position_indices / seq_len)
        Q = Q * decay
        K = K * decay

        # 线性注意力计算
        KV = torch.matmul(K.transpose(-2, -1), V)
        Z = 1.0 / (torch.einsum('bhnd,bhd->bhn', Q, K.sum(dim=2)).unsqueeze(-1) + self.eps)
        output = torch.matmul(Q, KV) * Z

        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)

        return output


# 使用示例
if __name__ == "__main__":
    batch_size = 2
    seq_len = 512
    d_model = 512
    n_heads = 8

    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)

    # 1. 基础线性注意力
    linear_attn = LinearAttention(d_model, n_heads, feature_map='elu')
    output = linear_attn(x)
    print(f"线性注意力输出形状: {output.shape}")

    # 2. Performer注意力
    performer = PerformerAttention(d_model, n_heads, nb_features=256)
    output = performer(x)
    print(f"Performer输出形状: {output.shape}")

    # 3. Linformer注意力
    linformer = LinformerAttention(d_model, n_heads, seq_len=seq_len, k=64)
    output = linformer(x)
    print(f"Linformer输出形状: {output.shape}")

    # 4. Nyström注意力
    nystrom = NystromAttention(d_model, n_heads, num_landmarks=32)
    output = nystrom(x)
    print(f"Nyström输出形状: {output.shape}")

    # 5. Cosformer注意力
    cosformer = CosformerAttention(d_model, n_heads)
    output = cosformer(x)
    print(f"Cosformer输出形状: {output.shape}")

    # 性能对比
    import time

    print("\n性能对比（序列长度=512）:")

    # 标准注意力（作为基准）
    from full_attention_code import MultiHeadAttention

    standard_attn = MultiHeadAttention(d_model, n_heads)

    start = time.time()
    for _ in range(10):
        _ = standard_attn(x, x, x)
    print(f"标准注意力: {time.time() - start:.3f}秒")

    start = time.time()
    for _ in range(10):
        _ = linear_attn(x)
    print(f"线性注意力: {time.time() - start:.3f}秒")

    start = time.time()
    for _ in range(10):
        _ = performer(x)
    print(f"Performer: {time.time() - start:.3f}秒")