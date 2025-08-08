"""
注意力机制综合使用示例
展示如何在实际场景中选择和使用不同的注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any
from dataclasses import dataclass
import time


# ================== 配置类 ==================

@dataclass
class AttentionConfig:
    """注意力机制配置"""
    attention_type: str = "full"  # full, sparse, linear, gqa, mqa, flash
    d_model: int = 512
    n_heads: int = 8
    n_kv_heads: Optional[int] = None  # for GQA
    window_size: Optional[int] = None  # for local attention
    block_size: Optional[int] = None  # for block sparse
    projection_dim: Optional[int] = None  # for Linformer
    num_landmarks: Optional[int] = None  # for Nystrom
    dropout: float = 0.1
    max_seq_len: int = 2048


# ================== 统一的注意力工厂 ==================

class AttentionFactory:
    """注意力机制工厂类"""

    @staticmethod
    def create_attention(config: AttentionConfig) -> nn.Module:
        """根据配置创建相应的注意力模块"""

        if config.attention_type == "full":
            from full_attention_code import MultiHeadAttention
            return MultiHeadAttention(
                config.d_model,
                config.n_heads,
                config.dropout
            )

        elif config.attention_type == "local":
            from sparse_attention_code import LocalAttention
            return LocalAttention(
                config.d_model,
                config.n_heads,
                config.window_size or 256,
                config.dropout
            )

        elif config.attention_type == "block_sparse":
            from sparse_attention_code import BlockSparseAttention
            return BlockSparseAttention(
                config.d_model,
                config.n_heads,
                config.block_size or 64,
                config.dropout
            )

        elif config.attention_type == "linear":
            from linear_attention_code import LinearAttention
            return LinearAttention(
                config.d_model,
                config.n_heads,
                feature_map='elu'
            )

        elif config.attention_type == "performer":
            from linear_attention_code import PerformerAttention
            return PerformerAttention(
                config.d_model,
                config.n_heads,
                nb_features=256
            )

        elif config.attention_type == "linformer":
            from linear_attention_code import LinformerAttention
            return LinformerAttention(
                config.d_model,
                config.n_heads,
                config.max_seq_len,
                config.projection_dim or 256
            )

        elif config.attention_type == "gqa":
            from gqa_mqa_code import GroupedQueryAttention
            return GroupedQueryAttention(
                config.d_model,
                config.n_heads,
                config.n_kv_heads or config.n_heads // 2,
                config.dropout
            )

        elif config.attention_type == "mqa":
            from gqa_mqa_code import MultiQueryAttention
            return MultiQueryAttention(
                config.d_model,
                config.n_heads,
                config.dropout
            )

        elif config.attention_type == "flash":
            from flash_attention_code import FlashAttention
            return FlashAttention(
                config.d_model,
                config.n_heads,
                block_size_q=64,
                block_size_kv=64,
                dropout=config.dropout
            )

        else:
            raise ValueError(f"未知的注意力类型: {config.attention_type}")


# ================== Transformer层实现 ==================

class TransformerBlock(nn.Module):
    """使用可配置注意力的Transformer块"""

    def __init__(self, config: AttentionConfig):
        super().__init__()

        # 注意力层
        self.attention = AttentionFactory.create_attention(config)

        # 层归一化
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4),
            nn.GELU(),
            nn.Linear(config.d_model * 4, config.d_model),
            nn.Dropout(config.dropout)
        )

        self.config = config

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 注意力子层
        if self.config.attention_type in ["gqa", "mqa"]:
            attn_output, _ = self.attention(x)
        elif self.config.attention_type == "flash":
            attn_output = self.attention(x, is_causal=mask is not None)
        elif self.config.attention_type in ["local", "block_sparse", "linear", "performer", "linformer"]:
            attn_output = self.attention(x)
        else:
            attn_output = self.attention(x, x, x, mask)

        x = x + attn_output
        x = self.ln1(x)

        # FFN子层
        ffn_output = self.ffn(x)
        x = x + ffn_output
        x = self.ln2(x)

        return x


# ================== 模型示例 ==================

class GPTModel(nn.Module):
    """使用可配置注意力的GPT模型"""

    def __init__(
            self,
            vocab_size: int,
            config: AttentionConfig,
            n_layers: int = 12
    ):
        super().__init__()

        self.config = config
        self.n_layers = n_layers

        # Token嵌入
        self.token_embedding = nn.Embedding(vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        # Transformer层
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(n_layers)
        ])

        # 输出层
        self.ln_final = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, vocab_size, bias=False)

        # 权重共享
        self.token_embedding.weight = self.lm_head.weight

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # 位置索引
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        # 嵌入
        x = self.token_embedding(input_ids) + self.position_embedding(position_ids)

        # 通过Transformer层
        for layer in self.layers:
            x = layer(x, attention_mask)

        # 输出
        x = self.ln_final(x)
        logits = self.lm_head(x)

        return logits


# ================== 场景化选择器 ==================

class AttentionSelector:
    """根据场景自动选择最佳注意力机制"""

    @staticmethod
    def select_for_scenario(
            scenario: str,
            seq_len: int,
            memory_constraint: Optional[int] = None,  # MB
            latency_constraint: Optional[float] = None  # ms
    ) -> AttentionConfig:
        """
        根据场景选择注意力配置

        Args:
            scenario: 场景类型 (training, inference, long_context, memory_limited)
            seq_len: 序列长度
            memory_constraint: 内存限制（MB）
            latency_constraint: 延迟限制（ms）
        """

        config = AttentionConfig()

        if scenario == "training":
            # 训练场景：优先考虑性能和梯度稳定性
            if seq_len <= 512:
                config.attention_type = "flash"
            elif seq_len <= 2048:
                config.attention_type = "flash"
                config.window_size = 256
            else:
                config.attention_type = "local"
                config.window_size = 256

        elif scenario == "inference":
            # 推理场景：优先考虑内存效率
            if memory_constraint and memory_constraint < 1000:
                config.attention_type = "mqa"
            elif memory_constraint and memory_constraint < 2000:
                config.attention_type = "gqa"
                config.n_kv_heads = 2
            else:
                config.attention_type = "flash"

        elif scenario == "long_context":
            # 长文本场景：必须使用线性或稀疏注意力
            if seq_len > 8192:
                config.attention_type = "linear"
            elif seq_len > 4096:
                config.attention_type = "local"
                config.window_size = 512
            else:
                config.attention_type = "block_sparse"
                config.block_size = 128

        elif scenario == "memory_limited":
            # 内存受限场景：最小化KV缓存
            config.attention_type = "mqa"

        elif scenario == "edge_device":
            # 边缘设备：极限优化
            config.attention_type = "linear"
            config.d_model = 256
            config.n_heads = 4

        return config


# ================== 性能分析器 ==================

class AttentionProfiler:
    """注意力机制性能分析器"""

    @staticmethod
    def profile_attention(
            attention_type: str,
            batch_size: int = 4,
            seq_len: int = 512,
            d_model: int = 512,
            n_heads: int = 8,
            num_iterations: int = 100
    ) -> Dict[str, Any]:
        """
        分析注意力机制的性能

        Returns:
            包含延迟、吞吐量、内存使用等指标的字典
        """
        config = AttentionConfig(
            attention_type=attention_type,
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=seq_len
        )

        # 创建模型
        model = TransformerBlock(config)
        model.eval()

        # 如果有GPU，使用GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # 创建输入
        x = torch.randn(batch_size, seq_len, d_model, device=device)

        # 预热
        for _ in range(10):
            with torch.no_grad():
                _ = model(x)

        # 测量延迟
        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.time()
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()

        total_time = time.time() - start_time
        avg_latency = (total_time / num_iterations) * 1000  # ms

        # 计算吞吐量
        throughput = (batch_size * seq_len * num_iterations) / total_time

        # 测量内存（如果在GPU上）
        memory_mb = 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = model(x)
            memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        # 计算FLOPs（近似）
        flops = 4 * batch_size * seq_len * seq_len * d_model  # 简化计算

        return {
            "attention_type": attention_type,
            "avg_latency_ms": avg_latency,
            "throughput_tokens_per_sec": throughput,
            "memory_mb": memory_mb,
            "flops": flops,
            "device": str(device)
        }


# ================== 实际使用示例 ==================

def example_training_setup():
    """训练场景的设置示例"""
    print("=" * 60)
    print("训练场景设置示例")
    print("=" * 60)

    # 根据序列长度自动选择注意力
    seq_lengths = [512, 2048, 8192]

    for seq_len in seq_lengths:
        config = AttentionSelector.select_for_scenario(
            scenario="training",
            seq_len=seq_len
        )

        print(f"\n序列长度: {seq_len}")
        print(f"推荐注意力类型: {config.attention_type}")

        # 创建模型
        model = GPTModel(vocab_size=50000, config=config, n_layers=6)

        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数量: {total_params / 1e6:.1f}M")

        # 创建示例输入
        batch_size = 2
        input_ids = torch.randint(0, 50000, (batch_size, min(seq_len, 128)))

        # 前向传播
        logits = model(input_ids)
        print(f"输出形状: {logits.shape}")


def example_inference_optimization():
    """推理优化示例"""
    print("\n" + "=" * 60)
    print("推理优化示例")
    print("=" * 60)

    configs = [
        ("标准注意力(MHA)", AttentionConfig(attention_type="full")),
        ("GQA(4头)", AttentionConfig(attention_type="gqa", n_kv_heads=4)),
        ("MQA(1头)", AttentionConfig(attention_type="mqa")),
    ]

    batch_size = 1
    seq_len = 512

    for name, config in configs:
        print(f"\n{name}:")

        # 创建模型
        model = TransformerBlock(config)
        model.eval()

        # 计算KV缓存大小
        if config.attention_type == "gqa":
            kv_heads = config.n_kv_heads
        elif config.attention_type == "mqa":
            kv_heads = 1
        else:
            kv_heads = config.n_heads

        kv_cache_size = 2 * batch_size * kv_heads * seq_len * (config.d_model // config.n_heads) * 4
        print(f"  KV缓存大小: {kv_cache_size / (1024 * 1024):.2f} MB")

        # 测试推理速度
        x = torch.randn(batch_size, seq_len, config.d_model)

        with torch.no_grad():
            start = time.time()
            for _ in range(10):
                _ = model(x)
            inference_time = time.time() - start

        print(f"  推理时间(10次): {inference_time:.3f}秒")


def example_adaptive_attention():
    """自适应注意力示例"""
    print("\n" + "=" * 60)
    print("自适应注意力示例")
    print("=" * 60)

    scenarios = [
        ("移动设备", "edge_device", 512),
        ("云端训练", "training", 2048),
        ("长文档处理", "long_context", 16384),
        ("实时推理", "inference", 1024),
    ]

    for scenario_name, scenario_type, seq_len in scenarios:
        print(f"\n场景: {scenario_name} (序列长度={seq_len})")

        config = AttentionSelector.select_for_scenario(
            scenario=scenario_type,
            seq_len=seq_len,
            memory_constraint=2000 if scenario_type == "edge_device" else None
        )

        print(f"  选择的注意力: {config.attention_type}")
        print(f"  模型维度: {config.d_model}")
        print(f"  头数: {config.n_heads}")

        if config.window_size:
            print(f"  窗口大小: {config.window_size}")
        if config.n_kv_heads:
            print(f"  KV头数: {config.n_kv_heads}")


def example_performance_comparison():
    """性能对比示例"""
    print("\n" + "=" * 60)
    print("性能对比示例")
    print("=" * 60)

    attention_types = ["full", "local", "linear", "gqa", "mqa"]

    if not torch.cuda.is_available():
        print("GPU不可用，使用CPU进行测试（结果可能不够准确）")

    results = []
    for attention_type in attention_types:
        try:
            profile = AttentionProfiler.profile_attention(
                attention_type=attention_type,
                batch_size=4,
                seq_len=512,
                num_iterations=20
            )
            results.append(profile)
        except Exception as e:
            print(f"测试{attention_type}时出错: {e}")

    # 打印结果表格
    print("\n性能对比结果:")
    print("-" * 80)
    print(f"{'类型':<15} {'延迟(ms)':<12} {'吞吐量':<15} {'内存(MB)':<12}")
    print("-" * 80)

    for result in results:
        print(f"{result['attention_type']:<15} "
              f"{result['avg_latency_ms']:<12.2f} "
              f"{result['throughput_tokens_per_sec']:<15.0f} "
              f"{result['memory_mb']:<12.2f}")


def example_custom_model():
    """自定义模型示例"""
    print("\n" + "=" * 60)
    print("自定义混合注意力模型示例")
    print("=" * 60)

    class HybridAttentionModel(nn.Module):
        """混合使用不同注意力的模型"""

        def __init__(self, vocab_size: int, d_model: int = 512):
            super().__init__()

            self.embedding = nn.Embedding(vocab_size, d_model)

            # 底层使用局部注意力（处理局部依赖）
            self.local_layers = nn.ModuleList([
                TransformerBlock(AttentionConfig(
                    attention_type="local",
                    d_model=d_model,
                    window_size=128
                )) for _ in range(4)
            ])

            # 中层使用稀疏注意力（捕捉中距离依赖）
            self.sparse_layers = nn.ModuleList([
                TransformerBlock(AttentionConfig(
                    attention_type="block_sparse",
                    d_model=d_model,
                    block_size=64
                )) for _ in range(2)
            ])

            # 顶层使用全注意力（整合全局信息）
            self.global_layers = nn.ModuleList([
                TransformerBlock(AttentionConfig(
                    attention_type="full",
                    d_model=d_model
                )) for _ in range(2)
            ])

            self.output = nn.Linear(d_model, vocab_size)

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            x = self.embedding(input_ids)

            # 逐层处理
            for layer in self.local_layers:
                x = layer(x)

            for layer in self.sparse_layers:
                x = layer(x)

            for layer in self.global_layers:
                x = layer(x)

            return self.output(x)

    # 创建模型
    model = HybridAttentionModel(vocab_size=10000)

    # 统计不同层的参数
    local_params = sum(p.numel() for layer in model.local_layers for p in layer.parameters())
    sparse_params = sum(p.numel() for layer in model.sparse_layers for p in layer.parameters())
    global_params = sum(p.numel() for layer in model.global_layers for p in layer.parameters())

    print(f"局部注意力层参数: {local_params / 1e6:.2f}M")
    print(f"稀疏注意力层参数: {sparse_params / 1e6:.2f}M")
    print(f"全局注意力层参数: {global_params / 1e6:.2f}M")

    # 测试前向传播
    input_ids = torch.randint(0, 10000, (2, 256))
    output = model(input_ids)
    print(f"输出形状: {output.shape}")


# ================== 主函数 ==================

if __name__ == "__main__":
    print("注意力机制综合使用示例")
    print("=" * 80)

    # 1. 训练场景设置
    example_training_setup()

    # 2. 推理优化
    example_inference_optimization()

    # 3. 自适应选择
    example_adaptive_attention()

    # 4. 性能对比
    example_performance_comparison()

    # 5. 自定义模型
    example_custom_model()

    print("\n" + "=" * 80)
    print("所有示例完成！")
    print("\n关键要点总结:")
    print("1. 短序列(<1K): 使用Flash Attention或标准注意力")
    print("2. 中等序列(1K-8K): 使用GQA/MQA或局部注意力")
    print("3. 长序列(>8K): 使用线性注意力或稀疏注意力")
    print("4. 推理优化: 优先使用MQA/GQA减少KV缓存")
    print("5. 训练优化: 使用Flash Attention减少内存带宽")
    print("6. 可以混合使用不同注意力机制以平衡效果和效率")