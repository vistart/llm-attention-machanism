# Attention Mechanisms Library

[大语言模型注意力机制详解](https://vistart.github.io/examples/llm/attention_mechanism_comparison.html#gqa)

[English](#english) | [中文](#中文)

## English

A comprehensive PyTorch implementation of various attention mechanisms used in modern Transformer architectures. This library provides educational and practical implementations of state-of-the-art attention variants, from standard multi-head attention to efficient alternatives like Flash Attention, Linear Attention, and Sparse Attention.

### 🎯 Features

- **Complete Implementations**: 15+ different attention mechanisms with detailed comments
- **Performance Optimized**: Memory-efficient implementations with benchmarking tools
- **Easy to Use**: Unified interface and factory pattern for easy switching between attention types
- **Educational**: Well-documented code with theoretical explanations
- **Production Ready**: Includes caching, position encodings, and optimization techniques

### 📋 Requirements

```bash
torch>=1.10.0
numpy>=1.19.0
```

### 🚀 Quick Start

```python
from main import AttentionFactory, AttentionConfig

# Create a configuration
config = AttentionConfig(
    attention_type="flash",  # Options: full, flash, linear, sparse, gqa, mqa
    d_model=512,
    n_heads=8,
    dropout=0.1
)

# Create attention module
attention = AttentionFactory.create_attention(config)

# Use in your model
import torch
x = torch.randn(2, 128, 512)  # [batch_size, seq_len, d_model]
output = attention(x)
```

### 📁 Project Structure

```
attention-mechanisms/
├── full_attention_code.py      # Standard multi-head attention
├── flash_attention_code.py     # Flash Attention & Flash Attention v2
├── linear_attention_code.py    # Linear complexity attention variants
├── sparse_attention_code.py    # Sparse attention patterns
├── gqa_mqa_code.py             # Grouped Query & Multi-Query Attention
└── main.py                     # Main interface and examples
```

### 🔧 Attention Mechanisms

#### 1. **Standard Attention** (`full_attention_code.py`)
- Multi-Head Attention (MHA)
- Self-Attention
- Cross-Attention
- Scaled Dot-Product Attention

#### 2. **Flash Attention** (`flash_attention_code.py`)
- Flash Attention v1 & v2
- Block-wise computation for memory efficiency
- Support for causal masks and ALiBi position encoding
- O(N²) time, O(N) memory complexity

#### 3. **Linear Attention** (`linear_attention_code.py`)
- **Linear Attention**: Basic linear attention with feature maps
- **Performer**: Random feature approximation of softmax kernel
- **Linformer**: Low-rank projection of keys and values
- **Nyström**: Nyström approximation of attention matrix
- **Cosformer**: Cosine-based reweighting
- O(N) complexity for long sequences

#### 4. **Sparse Attention** (`sparse_attention_code.py`)
- **Local/Sliding Window**: Fixed window attention
- **Block Sparse**: Block-diagonal patterns
- **Strided/Dilated**: Strided attention patterns
- **Longformer**: Combination of local and global attention
- **BigBird**: Random, local, and global attention

#### 5. **Memory-Efficient Attention** (`gqa_mqa_code.py`)
- **Grouped Query Attention (GQA)**: Groups of queries share key-value pairs
- **Multi-Query Attention (MQA)**: All queries share single key-value pair
- Significantly reduces KV-cache memory during inference

### 💡 Usage Examples

#### Scenario-Based Selection

```python
from main import AttentionSelector

# Automatically select best attention for your scenario
config = AttentionSelector.select_for_scenario(
    scenario="long_context",  # Options: training, inference, long_context, memory_limited
    seq_len=8192,
    memory_constraint=2000  # MB
)
```

#### Custom Model with Mixed Attention

```python
from main import TransformerBlock, AttentionConfig

# Create a model with different attention types per layer
configs = [
    AttentionConfig(attention_type="local", window_size=128),     # Bottom layers
    AttentionConfig(attention_type="block_sparse", block_size=64), # Middle layers  
    AttentionConfig(attention_type="full")                         # Top layers
]

layers = [TransformerBlock(config) for config in configs]
```

#### Performance Benchmarking

```python
from main import AttentionProfiler

# Profile different attention mechanisms
profile = AttentionProfiler.profile_attention(
    attention_type="flash",
    batch_size=4,
    seq_len=2048,
    num_iterations=100
)
print(f"Latency: {profile['avg_latency_ms']:.2f}ms")
print(f"Memory: {profile['memory_mb']:.2f}MB")
```

### 📊 Performance Comparison

| Attention Type | Time Complexity | Memory Complexity | Best For |
|---|---|---|---|
| Standard MHA | O(N²d) | O(N²) | Short sequences (<1K) |
| Flash Attention | O(N²d) | O(N) | Training with long sequences |
| Linear Attention | O(Nd²) | O(Nd) | Very long sequences (>8K) |
| Local Attention | O(Nwd) | O(Nw) | Long sequences with local patterns |
| GQA/MQA | O(N²d) | O(N²/g) | Inference optimization |

### 🎓 Educational Resources

Each implementation includes:
- Detailed docstrings explaining the algorithm
- Time and space complexity analysis
- References to original papers
- Practical usage examples

### 🔍 Advanced Features

- **Position Encodings**: RoPE, ALiBi, xPos support
- **KV Caching**: For efficient autoregressive generation
- **Mixed Precision**: Compatible with fp16/bf16 training
- **Custom Masks**: Support for arbitrary attention patterns

### 📝 License

[MIT License](LICENSE)

---

## 中文

一个全面的PyTorch注意力机制实现库，包含现代Transformer架构中使用的各种注意力变体。本库提供了从标准多头注意力到Flash Attention、线性注意力和稀疏注意力等高效替代方案的教育性和实用性实现。

### 🎯 主要特性

- **完整实现**：15+种不同的注意力机制，带详细注释
- **性能优化**：内存高效的实现，包含基准测试工具
- **易于使用**：统一接口和工厂模式，便于在不同注意力类型间切换
- **教育性强**：代码文档完善，包含理论解释
- **生产就绪**：包含缓存、位置编码和优化技术

### 📋 环境要求

```bash
torch>=1.10.0
numpy>=1.19.0
```

### 🚀 快速开始

```python
from main import AttentionFactory, AttentionConfig

# 创建配置
config = AttentionConfig(
    attention_type="flash",  # 选项：full, flash, linear, sparse, gqa, mqa
    d_model=512,
    n_heads=8,
    dropout=0.1
)

# 创建注意力模块
attention = AttentionFactory.create_attention(config)

# 在模型中使用
import torch
x = torch.randn(2, 128, 512)  # [批次大小, 序列长度, 模型维度]
output = attention(x)
```

### 📁 项目结构

```
attention-mechanisms/
├── full_attention_code.py      # 标准多头注意力
├── flash_attention_code.py     # Flash Attention及v2版本
├── linear_attention_code.py    # 线性复杂度注意力变体
├── sparse_attention_code.py    # 稀疏注意力模式
├── gqa_mqa_code.py             # 分组查询和多查询注意力
└── main.py                     # 主接口和示例
```

### 🔧 注意力机制详解

#### 1. **标准注意力** (`full_attention_code.py`)
- 多头注意力（MHA）
- 自注意力
- 交叉注意力
- 缩放点积注意力

#### 2. **Flash注意力** (`flash_attention_code.py`)
- Flash Attention v1和v2
- 分块计算以提高内存效率
- 支持因果掩码和ALiBi位置编码
- O(N²)时间复杂度，O(N)内存复杂度

#### 3. **线性注意力** (`linear_attention_code.py`)
- **线性注意力**：使用特征映射的基础线性注意力
- **Performer**：软最大核的随机特征近似
- **Linformer**：键值的低秩投影
- **Nyström**：注意力矩阵的Nyström近似
- **Cosformer**：基于余弦的重加权
- 长序列的O(N)复杂度

#### 4. **稀疏注意力** (`sparse_attention_code.py`)
- **局部/滑动窗口**：固定窗口注意力
- **块稀疏**：块对角模式
- **跨步/稀释**：跨步注意力模式
- **Longformer**：局部和全局注意力的组合
- **BigBird**：随机、局部和全局注意力

#### 5. **内存高效注意力** (`gqa_mqa_code.py`)
- **分组查询注意力（GQA）**：查询组共享键值对
- **多查询注意力（MQA）**：所有查询共享单个键值对
- 显著减少推理时的KV缓存内存

### 💡 使用示例

#### 基于场景的自动选择

```python
from main import AttentionSelector

# 根据场景自动选择最佳注意力
config = AttentionSelector.select_for_scenario(
    scenario="long_context",  # 选项：training, inference, long_context, memory_limited
    seq_len=8192,
    memory_constraint=2000  # MB
)
```

#### 混合注意力的自定义模型

```python
from main import TransformerBlock, AttentionConfig

# 创建每层使用不同注意力类型的模型
configs = [
    AttentionConfig(attention_type="local", window_size=128),     # 底层
    AttentionConfig(attention_type="block_sparse", block_size=64), # 中层
    AttentionConfig(attention_type="full")                         # 顶层
]

layers = [TransformerBlock(config) for config in configs]
```

#### 性能基准测试

```python
from main import AttentionProfiler

# 分析不同注意力机制的性能
profile = AttentionProfiler.profile_attention(
    attention_type="flash",
    batch_size=4,
    seq_len=2048,
    num_iterations=100
)
print(f"延迟: {profile['avg_latency_ms']:.2f}ms")
print(f"内存: {profile['memory_mb']:.2f}MB")
```

### 📊 性能对比

| 注意力类型 | 时间复杂度 | 内存复杂度 | 最适用于 |
|---|---|---|---|
| 标准MHA | O(N²d) | O(N²) | 短序列（<1K） |
| Flash注意力 | O(N²d) | O(N) | 长序列训练 |
| 线性注意力 | O(Nd²) | O(Nd) | 超长序列（>8K） |
| 局部注意力 | O(Nwd) | O(Nw) | 具有局部模式的长序列 |
| GQA/MQA | O(N²d) | O(N²/g) | 推理优化 |

### 🎓 教育资源

每个实现都包含：
- 解释算法的详细文档字符串
- 时间和空间复杂度分析
- 原始论文引用
- 实际使用示例

### 🔍 高级功能

- **位置编码**：支持RoPE、ALiBi、xPos
- **KV缓存**：用于高效的自回归生成
- **混合精度**：兼容fp16/bf16训练
- **自定义掩码**：支持任意注意力模式

### 🤝 贡献

欢迎提交问题和拉取请求！

### 📝 许可证

[MIT License](LICENSE)