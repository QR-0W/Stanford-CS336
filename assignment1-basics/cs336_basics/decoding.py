import torch
from cs336_basics.transformer import TransformerLM


def generate(
    model: TransformerLM,
    prompt_tokens: list[int],
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
) -> list[int]:
    """
    自回归生成文本

    Args:
        model: 语言模型
        prompt_tokens: 输入的 token IDs
        max_new_tokens: 最大生成 token 数
        temperature: 温度 (>1 更随机, <1 更确定)
        top_p: Nucleus sampling 阈值 (1.0 = 禁用)
        eos_token_id: 结束 token ID (None = 不提前终止)

    Returns:
        完整序列 (prompt + generated)
    """
    model.eval()
    tokens = list(prompt_tokens)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 1. 准备输入 (可能需要截断到 context_length)
            # 从模型获取 context_length 和 device
            context_length = model.context_length
            device = next(model.parameters()).device
            input_ids = torch.tensor([tokens[-context_length:]], device=device)

            # 2. 前向传播，获取最后一个位置的 logits
            logits = model(input_ids)  # (1, seq, vocab)
            next_logits = logits[0, -1, :]  # (vocab,)

            # 3. 温度缩放
            if temperature != 1.0:
                next_logits = next_logits / temperature

            # 4. Top-p (Nucleus) 采样
            if top_p < 1.0:
                next_logits = top_p_filtering(next_logits, top_p)

            # 5. Softmax → 概率分布
            probs = torch.softmax(next_logits, dim=-1)

            # 6. 采样下一个 token
            next_token = torch.multinomial(probs, num_samples=1).item()

            # 7. 添加到序列
            tokens.append(next_token)

            # 8. 检查终止条件
            if eos_token_id is not None and next_token == eos_token_id:
                break

    return tokens


def top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    保留累积概率达到 top_p 的最小 token 集合，其余设为 -inf
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(probs, dim=-1)

    # 找到第一个累积概率超过 top_p 的位置
    sorted_mask = cumulative_probs > top_p
    # 保留第一个超过阈值的 token（shift right）
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False

    # 将被过滤的 logits 设为 -inf
    sorted_logits[sorted_mask] = float("-inf")

    # 恢复原始顺序
    original_logits = torch.zeros_like(logits)
    original_logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

    return original_logits
