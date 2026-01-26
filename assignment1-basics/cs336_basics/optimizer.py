from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class SGD(torch.optim.Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer With few subtleties
    """

    def __init__(self, params, lr=1e-3):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None):
        """
        Args:
            closure: 可选的闭包函数
        Returns:
            loss: 如果提供了闭包函数，返回闭包函数的返回值
        """
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-0,
        weight_decay: float = 0.0,
    ):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate")

        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(
        self,
        closure: Optional[Callable[[], float]] = None,
    ):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # 初始化状态
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                # 更新迭代计数
                state["t"] += 1
                t = state["t"]
                m, v = state["m"], state["v"]

                # 更新动量 (不需要 no_grad，m/v 不是 Parameter)
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).add_(grad * grad, alpha=1 - beta2)

                # 偏置矫正后的学习率
                bias_correction1 = 1 - beta1**t
                bias_correction2 = 1 - beta2**t
                alpha_t = lr * math.sqrt(bias_correction2) / bias_correction1

                # 更新参数 (需要 no_grad，p 是 Parameter)
                with torch.no_grad():
                    p.addcdiv_(m, v.sqrt() + eps, value=-alpha_t)
                    # 权重衰减 (decoupled)
                    p.add_(p, alpha=-lr * weight_decay)

        return loss


def get_lr_cosine_schedule(it: int, max_lr: float, min_lr: float, warmup_steps: int, cosine_cycle_iters: int) -> float:
    """
    带预热的余弦退火学习率调度
    Args:
        it: 当前迭代次数
        max_lr: 最大学习率
        min_lr: 最小学习率
        warmup_steps: 热身步数
        cosine_cycle_iters: 余弦周期步数
    Returns:
        学习率
    """
    if it < warmup_steps:
        # 线性预热
        return max_lr * it / warmup_steps
    elif it < cosine_cycle_iters:
        # 余弦退火
        progress = (it - warmup_steps) / (cosine_cycle_iters - warmup_steps)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
    else:
        return min_lr

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    梯度裁剪（原地修改）
    Args:
        parameters: 参数迭代器
        max_l2_norm: 最大 L2 范数
    """
    # 转为列表（可能需要多次迭代）
    params = [p for p in parameters if p.grad is not None]

    if len(params) == 0:
        return

    # 计算全局 L2 范数
    total_norm_sq = sum(p.grad.pow(2).sum() for p in params)
    total_norm = total_norm_sq.sqrt()

    # 如果超过最大范数，缩放所有梯度
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + 1e-6)
        for p in params:
            p.grad.mul_(clip_coef)