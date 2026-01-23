import einops
import torch
import torch.nn as nn


class Linear(nn.Module):
    """
    线性层
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Given the weights of a Linear layer, compute the transformation of a batched input.

        Args:
            in_features: 输入的最终维度
            out_features: 输出的最终维度
            device: 存储参数的设备
            dtype: 参数的数据类型
        """
        super().__init__()

        # 创建权重参数 w
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))

        # 初始化线性权重
        std = (2.0 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, std=std, mean=0.0, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入的张量

        Returns:
            输出的张量
        """
        # x: (..., in_features), self.w: (out_features, in_features) -> 输出: (..., out_features)
        return einops.einsum(self.weight, x, "out_features in_features, ... in_features -> ... out_features")


class Embedding(nn.Module):
    """
    嵌入层
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Given the weights of an Embedding layer, compute the transformation of a batched input.

        Args:
            num_embeddings: 词典大小
            embedding_dim: 嵌入维度
            device: 存储参数的设备
            dtype: 参数的数据类型
        """
        super().__init__()

        # 创建嵌入矩阵
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))

        # 初始化嵌入矩阵
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: 形状任意的整数张量

        Returns:
            给定词元 ID 的嵌入向量，形状为(*token_ids.shape, embedding_dim)
        """
        # 直接用索引获取对应的行
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """
    RMS 归一化层
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Given the weights of an RMSNorm layer, compute the transformation of a batched input.

        Args:
            d_model: 模型维度
            eps: 小常数，防止除以 0
            device: 存储参数的设备
            dtype: 参数的数据类型
        """
        super().__init__()
        self.eps = eps

        # γ (gain) 初始化为全 1
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 形状任意的张量

        Returns:
            归一化后的张量
        """
        # 保存原始 dtype
        in_dtype = x.dtype

        # 上转到 float32 防止溢出
        x = x.to(torch.float32)

        # 计算 RMS: sqrt(mean(x^2) + eps)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)

        # 归一化并缩放
        result = (x / rms) * self.weight

        # 转回原始 dtype
        return result.to(in_dtype)


class SwiGLU(nn.Module):
    """
    SwiGLU 前馈网络

    SwiGLU(x) = W2 * (SiLU(W1 * x) ⊙ W3 * x)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Args:
            d_model: 模型维度
            d_ff: 前馈网络内部维度
            device: 存储参数的设备
            dtype: 参数的数据类型
        """
        super().__init__()

        # Linear(in_dim, out_dim) -> 权重 (out_dim, in_dim)
        # W1: d_model -> d_ff，权重 (d_ff, d_model)
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)

        # W2: d_ff -> d_model，权重 (d_model, d_ff)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

        # W3: d_model -> d_ff，权重 (d_ff, d_model)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入的张量

        Returns:
            输出的张量
        """
        # SiLU(W1 @ x) ⊙ (W3 @ x)，然后 W2 @ result
        # SiLU(z) = z * sigmoid(z)

        # (..., d_ff)
        w1_out = self.w1(x)

        # SiLU
        silu_out = w1_out * torch.sigmoid(w1_out)

        # (..., d_ff)
        gate_out = self.w3(x)

        # 逐元素乘
        hidden = silu_out * gate_out

        # (..., d_model)
        return self.w2(hidden)


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE)
    """

    def __init__(self, theta: float, d_k: int, max_seq_length: int, device: torch.device | None = None):
        """
        Args:
            theta: RoPE 的 Theta 值
            d_k: 查询和键向量的维度
            max_seq_length: 被输入的最大序列长度
            device: 存储参数的设备
        """
        super().__init__()

        # 计算 theta
        k = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)  # [0, 2, 4, ..., d-2]
        freqs = 1.0 / (theta ** (k / d_k))  # shape: (d_k/2,)

        # 位置: [0, 1, ..., max_seq_len-1]
        positions = torch.arange(max_seq_length, device=device, dtype=torch.float32)

        # 角度：外积
        angles = einops.einsum(positions, freqs, "pos, freq -> pos freq")  # shape: (max_seq_length, d_k/2)

        # 预计算 cos 和 sin
        self.register_buffer("cos_cache", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cache", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        处理形状为 (..., seq_len, d_k) 的输入张量并返回形状相同的张量
        Args:
            x: (..., seq_len, d_k)
            token_positions: (..., seq_len) 整数张量
        Returns:
            (..., seq_len, d_k)
        """

        # 计算 cos/sin
        cos = self.cos_cache[token_positions]  # (..., seq_len, d_k/2)
        sin = self.sin_cache[token_positions]  # (..., seq_len, d_k/2)

        x_pairs = einops.rearrange(
            x, "... seq_len (half_d two) -> ... seq_len half_d two", two=2
        )  # x_pairs[..., 0] 是偶数索引，x_pairs[..., 1] 是奇数索引

        x1 = x_pairs[..., 0]
        x2 = x_pairs[..., 1]

        # 应用旋转
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos

        return einops.rearrange([y1, y2], "two ... half_d -> ... (half_d two)")  # (..., seq_len, d_k)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    沿着指定维度应用 softmax

    Args:
        x: 输入张量
        dim: 要应用 softmax 的维度

    Returns:
        与输入形状相同的张量，指定维度上是归一化的概率分布
    """
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_shifted = x - x_max
    exp_x = torch.exp(x_shifted)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp


def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    缩放点积注意力

    Args:
        Q: (..., queries, d_k)
        K: (..., keys, d_k)
        V: (..., keys, d_v)
        mask: (..., queries, keys) 布尔张量，True=保留，False=屏蔽

    Returns:
        (..., queries, d_v)
    """
    # 获取 d_k 维度
    d_k = einops.parse_shape(q, "... quries d_k")["d_k"]

    # Q  @ K^T / sqrt(d_k)
    scores = einops.einsum(q, k, "... quries d_k, ... keys d_k -> ... quries keys") / (d_k**0.5)

    # 应用 mask
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    # softmax
    attention_weights = softmax(scores, dim=-1)  # dim = -1 是最后一个维度

    # attention_weights @ V
    return einops.einsum(attention_weights, v, "... quries keys, ... keys d_v -> ... quries d_v")


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        number_heads: int,
        max_seq_len: int = 0,
        rope_theta: float = 10000.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Args:
            d_model: 模型维度
            number_heads: 头数
            max_seq_len: 最大序列长度
            rope_theta: RoPE 的 Theta 值
            device: 存储参数的设备
            dtype: 存储参数的数据类型
        """
        super().__init__()

        self.d_model = d_model
        self.number_heads = number_heads
        self.d_k = d_model // number_heads

        # QKV 投影
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        # 输出投影
        self.out_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        # RoPE (可选)
        self.rope = None
        if max_seq_len > 0:
            self.rope = RotaryPositionalEmbedding(
                theta=rope_theta,
                d_k=self.d_k,
                max_seq_length=max_seq_len,
                device=device,
            )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (..., seq_len, d_model)
            mask: (..., queries, keys) 布尔张量
            token_positions: (..., seq_len) RoPE 需要的位置索引
        Returns:
            (..., seq_len, d_model)
        """
        seq_len = einops.parse_shape(x, "... seq_len d_model")["seq_len"]

        # QKV 投影
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 分割多头
        q = einops.rearrange(q, "... seq_len (heads d_k) -> ... heads seq_len d_k", heads=self.number_heads)
        k = einops.rearrange(k, "... seq_len (heads d_k) -> ... heads seq_len d_k", heads=self.number_heads)
        v = einops.rearrange(v, "... seq_len (heads d_k) -> ... heads seq_len d_k", heads=self.number_heads)

        # 应用 RoPE
        if self.rope is not None and token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # 因果掩码，下三角为1 (True=保留)
        if mask is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))

        # SDPA
        attn_out = scaled_dot_product_attention(q, k, v, mask)

        # 合并多头
        output = einops.rearrange(attn_out, "... heads seq_len d_k -> ... seq_len (heads d_k)", heads=self.number_heads)

        # 输出投影
        return self.out_proj(output)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int = 0,
        rope_theta: float = 10000.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Args:
            d_model: 模型维度
            num_heads: 头数
            d_ff: 前馈网络内部维度
            max_seq_len: 最大序列长度
            rope_theta: RoPE 的 Theta 值
            device: 存储参数的设备
            dtype: 参数的数据类型
        """
        super().__init__()

        # RMSNorm
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)

        # 多头自注意力
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, rope_theta, device, dtype)

        # FFN (SwiGLU)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: 输入的张量 (..., seq_len, d_model)
            token_positions: 位置索引 (..., seq_len)

        Returns:
            输出的张量 (..., seq_len, d_model)
        """
        # 自动生成位置索引
        if token_positions is None:
            seq_len = einops.parse_shape(x, "... seq_len d_model")["seq_len"]
            token_positions = torch.arange(seq_len, device=x.device)

        # Pre-norm + MHA + 残差
        h = x + self.attn(self.norm1(x), token_positions=token_positions)

        # Pre-norm + FFN + 残差
        output = h + self.ffn(self.norm2(h))

        return output


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Args:
            vocab_size: 词表大小
            context_length: 上下文长度
            d_model: 模型维度
            num_layers: 层数
            num_heads: 头数
            d_ff: 前馈网络内部维度
            rope_theta: RoPE 的 Theta 值
            device: 存储参数的设备
            dtype: 参数的数据类型
        """
        super().__init__()

        # Token Embedding
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        # Transformer Blocks
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device, dtype)
                for _ in range(num_layers)
            ]
        )

        # Final LayerNorm
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)

        # LM Head (输出投影)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (batch_size, seq_len)
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        # Embedding
        x = self.token_embeddings(token_ids)  # (batch, seq, d_model)

        # 生成位置索引
        seq_len = einops.parse_shape(x, "... seq_len d_model")["seq_len"]
        token_positions = torch.arange(seq_len, device=x.device)

        # 通过所有 Transformer Block
        for layer in self.layers:
            x = layer(x, token_positions)

        # Final Norm
        x = self.ln_final(x)

        # LM Head → Logits
        logits = self.lm_head(x)  # (batch, seq, vocab_size)

        return logits
