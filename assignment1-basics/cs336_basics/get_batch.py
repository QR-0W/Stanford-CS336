import torch
import einops
import numpy as np


def get_batch(
    dataset: np.ndarray, batch_size: int, context_length: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        dataset (np.ndarray): 数据集
        batch_size (int): 批处理大小
        context_length (int): 上文长度
        device (torch.device): 设备
    Returns:
        tuple[torch.Tensor, torch.Tensor]: 采样输入序列和相应的下一个标记序列
    """
    n = len(dataset)
    # 1. 计算有效起始位置的范围需要 context_length 个 input tokens + 1 个 target token，所以最大起始位置是 n - context_length - 1
    max_start = n - context_length - 1

    # 2. 随机采样 batch_size 个起始位置
    start_indices = np.random.randint(0, max_start + 1, size=batch_size)

    # 3. 构造输入序列和目标序列
    inputs = np.stack([dataset[i : i + context_length] for i in start_indices])
    targets = np.stack([dataset[i + 1 : i + 1 + context_length] for i in start_indices])

    # 4. 转换为 PyTorch 张量并移动到指定设备（需要 long 类型用于 embedding indexing）
    inputs = torch.from_numpy(inputs).long().to(device)
    targets = torch.from_numpy(targets).long().to(device)

    return inputs, targets
