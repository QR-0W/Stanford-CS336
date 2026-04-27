from __future__ import annotations

import math
from typing import Callable, Tuple

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - allows CPU-only test environments
    triton = None
    tl = None


_DEFAULT_Q_TILE = 64
_DEFAULT_K_TILE = 64
_BACKWARD_FN_CACHE: dict[bool, Callable[..., Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}


def _validate_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("Q, K, V must be rank-3 tensors shaped [B, T, D].")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("Q, K, V must have matching batch size.")
    if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
        raise ValueError("Q, K, V must have matching hidden size D.")
    if q.device != k.device or q.device != v.device:
        raise ValueError("Q, K, V must be on the same device.")


def _flash_attention_forward_tiled_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
    q_tile_size: int = _DEFAULT_Q_TILE,
    k_tile_size: int = _DEFAULT_K_TILE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _validate_inputs(q, k, v)

    bsz, n_queries, d = q.shape
    n_keys = k.shape[1]
    scale = 1.0 / math.sqrt(d)

    out = torch.empty((bsz, n_queries, d), device=q.device, dtype=q.dtype)
    lse = torch.empty((bsz, n_queries), device=q.device, dtype=torch.float32)

    for q_start in range(0, n_queries, q_tile_size):
        q_end = min(q_start + q_tile_size, n_queries)
        bq = q_end - q_start

        q_tile = q[:, q_start:q_end, :].to(torch.float32)
        m = torch.full((bsz, bq), float("-inf"), device=q.device, dtype=torch.float32)
        l = torch.zeros((bsz, bq), device=q.device, dtype=torch.float32)
        o_acc = torch.zeros((bsz, bq, d), device=q.device, dtype=torch.float32)

        if is_causal:
            q_idx = torch.arange(q_start, q_end, device=q.device)

        for k_start in range(0, n_keys, k_tile_size):
            k_end = min(k_start + k_tile_size, n_keys)

            k_tile = k[:, k_start:k_end, :].to(torch.float32)
            v_tile = v[:, k_start:k_end, :].to(torch.float32)

            s = torch.matmul(q_tile, k_tile.transpose(-1, -2)) * scale
            if is_causal:
                k_idx = torch.arange(k_start, k_end, device=q.device)
                causal = q_idx[:, None] >= k_idx[None, :]
                s = s.masked_fill(~causal.unsqueeze(0), -1e6)

            m_new = torch.maximum(m, torch.max(s, dim=-1).values)
            p_tilde = torch.exp(s - m_new.unsqueeze(-1))
            alpha = torch.exp(m - m_new)

            l = alpha * l + torch.sum(p_tilde, dim=-1)
            o_acc = o_acc * alpha.unsqueeze(-1) + torch.matmul(p_tilde, v_tile)
            m = m_new

        o_tile = o_acc / l.unsqueeze(-1)
        out[:, q_start:q_end, :] = o_tile.to(q.dtype)
        lse[:, q_start:q_end] = m + torch.log(l)

    return out, lse


def _flash_attention_backward_core(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
    is_causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Eq. 13-19 style backward with recomputation from Q, K, V and L.
    # Inputs are [B, T, D] (single-head simplified attention for this assignment section).
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)

    s = torch.matmul(q.to(torch.float32), k.to(torch.float32).transpose(-1, -2)) * scale
    if is_causal:
        t_q = q.shape[1]
        t_k = k.shape[1]
        q_idx = torch.arange(t_q, device=q.device)
        k_idx = torch.arange(t_k, device=q.device)
        causal = q_idx[:, None] >= k_idx[None, :]
        s = s.masked_fill(~causal.unsqueeze(0), -1e6)

    p = torch.exp(s - lse.to(torch.float32).unsqueeze(-1))
    if is_causal:
        p = p.masked_fill(~causal.unsqueeze(0), 0.0)

    dO = do.to(torch.float32)
    O = o.to(torch.float32)
    V = v.to(torch.float32)
    Q = q.to(torch.float32)
    K = k.to(torch.float32)

    # D = rowsum(dO * O), shape [B, T]
    D_vec = torch.sum(dO * O, dim=-1)

    dV = torch.matmul(p.transpose(-1, -2), dO)
    dP = torch.matmul(dO, V.transpose(-1, -2))
    dS = p * (dP - D_vec.unsqueeze(-1))
    if is_causal:
        dS = dS.masked_fill(~causal.unsqueeze(0), 0.0)

    dQ = torch.matmul(dS, K) * scale
    dK = torch.matmul(dS.transpose(-1, -2), Q) * scale

    return dQ.to(q.dtype), dK.to(k.dtype), dV.to(v.dtype)


def _get_backward_fn(is_causal: bool):
    cached = _BACKWARD_FN_CACHE.get(bool(is_causal))
    if cached is not None:
        return cached

    def eager_fn(q, k, v, o, do, lse):
        return _flash_attention_backward_core(q=q, k=k, v=v, o=o, do=do, lse=lse, is_causal=is_causal)

    if not hasattr(torch, "compile"):
        _BACKWARD_FN_CACHE[bool(is_causal)] = eager_fn
        return eager_fn

    try:
        compiled_fn = torch.compile(eager_fn, fullgraph=False)
        _BACKWARD_FN_CACHE[bool(is_causal)] = compiled_fn
        return compiled_fn
    except Exception:
        _BACKWARD_FN_CACHE[bool(is_causal)] = eager_fn
        return eager_fn


def _flash_attention_backward_recompute(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
    is_causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    backward_fn = _get_backward_fn(is_causal=bool(is_causal))
    return backward_fn(q, k, v, o, do, lse)


class FlashAttention2PyTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        out, lse = _flash_attention_forward_tiled_pytorch(q, k, v, is_causal=is_causal)
        ctx.save_for_backward(lse, q, k, v, out)
        ctx.is_causal = bool(is_causal)
        return out

    @staticmethod
    def backward(ctx, do: torch.Tensor):
        lse, q, k, v, out = ctx.saved_tensors
        dq, dk, dv = _flash_attention_backward_recompute(
            q=q,
            k=k,
            v=v,
            o=out,
            do=do,
            lse=lse,
            is_causal=ctx.is_causal,
        )
        return dq, dk, dv, None


if triton is not None:

    @triton.jit
    def _flash_fwd_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        l_ptr,
        stride_qb,
        stride_qq,
        stride_qd,
        stride_kb,
        stride_kk,
        stride_kd,
        stride_vb,
        stride_vk,
        stride_vd,
        stride_ob,
        stride_oq,
        stride_od,
        stride_lb,
        stride_lq,
        scale,
        N_QUERIES: tl.constexpr,
        N_KEYS: tl.constexpr,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        q_tile_idx = tl.program_id(0)
        batch_idx = tl.program_id(1)

        offs_q = tl.arange(0, Q_TILE_SIZE)
        offs_k = tl.arange(0, K_TILE_SIZE)
        offs_d = tl.arange(0, D)

        q_start = q_tile_idx * Q_TILE_SIZE
        q_idx = q_start + offs_q
        q_mask = q_idx < N_QUERIES

        q_ptrs = q_ptr + batch_idx * stride_qb + q_idx[:, None] * stride_qq + offs_d[None, :] * stride_qd
        q_tile = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)

        m_i = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)
        l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
        o_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

        for k_start in range(0, N_KEYS, K_TILE_SIZE):
            k_idx = k_start + offs_k
            k_mask = k_idx < N_KEYS

            k_ptrs = k_ptr + batch_idx * stride_kb + k_idx[:, None] * stride_kk + offs_d[None, :] * stride_kd
            v_ptrs = v_ptr + batch_idx * stride_vb + k_idx[:, None] * stride_vk + offs_d[None, :] * stride_vd

            k_tile = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0).to(tl.float32)
            v_tile = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)

            s = tl.dot(q_tile, tl.trans(k_tile), allow_tf32=False) * scale
            s = tl.where(k_mask[None, :], s, -float("inf"))

            if IS_CAUSAL:
                causal_mask = q_idx[:, None] >= k_idx[None, :]
                s = tl.where(causal_mask, s, -1.0e6)

            m_ij = tl.maximum(m_i, tl.max(s, axis=1))
            p_tilde = tl.exp(s - m_ij[:, None])
            alpha = tl.exp(m_i - m_ij)

            l_i = alpha * l_i + tl.sum(p_tilde, axis=1)
            o_i = o_i * alpha[:, None] + tl.dot(p_tilde.to(v_tile.dtype), v_tile, allow_tf32=False)
            m_i = m_ij

        o_i = o_i / l_i[:, None]
        lse = m_i + tl.log(l_i)

        o_ptrs = o_ptr + batch_idx * stride_ob + q_idx[:, None] * stride_oq + offs_d[None, :] * stride_od
        tl.store(o_ptrs, o_i, mask=q_mask[:, None])

        l_ptrs = l_ptr + batch_idx * stride_lb + q_idx * stride_lq
        tl.store(l_ptrs, lse, mask=q_mask)


class FlashAttention2Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        _validate_inputs(q, k, v)
        if triton is None or tl is None:
            raise RuntimeError("Triton is not available in this environment.")
        if not q.is_cuda:
            raise RuntimeError("FlashAttention2Triton expects CUDA tensors.")

        q_contig = q.contiguous()
        k_contig = k.contiguous()
        v_contig = v.contiguous()

        bsz, n_queries, d = q_contig.shape
        n_keys = k_contig.shape[1]

        out = torch.empty_like(q_contig)
        lse = torch.empty((bsz, n_queries), device=q_contig.device, dtype=torch.float32)

        q_tile = _DEFAULT_Q_TILE
        k_tile = _DEFAULT_K_TILE
        grid = (triton.cdiv(n_queries, q_tile), bsz)

        _flash_fwd_kernel[grid](
            q_contig,
            k_contig,
            v_contig,
            out,
            lse,
            q_contig.stride(0),
            q_contig.stride(1),
            q_contig.stride(2),
            k_contig.stride(0),
            k_contig.stride(1),
            k_contig.stride(2),
            v_contig.stride(0),
            v_contig.stride(1),
            v_contig.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            lse.stride(0),
            lse.stride(1),
            scale=1.0 / math.sqrt(d),
            N_QUERIES=n_queries,
            N_KEYS=n_keys,
            D=d,
            Q_TILE_SIZE=q_tile,
            K_TILE_SIZE=k_tile,
            IS_CAUSAL=bool(is_causal),
        )

        ctx.save_for_backward(lse, q_contig, k_contig, v_contig, out)
        ctx.is_causal = bool(is_causal)
        return out

    @staticmethod
    def backward(ctx, do: torch.Tensor):
        lse, q, k, v, out = ctx.saved_tensors
        dq, dk, dv = _flash_attention_backward_recompute(
            q=q,
            k=k,
            v=v,
            o=out,
            do=do,
            lse=lse,
            is_causal=ctx.is_causal,
        )
        return dq, dk, dv, None
