"""Pure-math GPTQ "OBS rounding" primitive, factored out from prismaquant.

Reference: `_gptq_obs_rounding_nvfp4` in
`prismaquant/export_native_compressed.py:1238-1369`. We keep the
Hessian-build / Cholesky-inverse / per-block RTN + error-propagation
skeleton, but drop the NVFP4-specific codebook so the result is a
quant-format-agnostic float32 weight.

Input:  W           [out, in]   float32 weight
        activations [*, in]     cached pre-Linear activations (the act-cache
                                stores `{'inputs': tensor}` per Linear)
        bits        int         simulator grid: 2^bits distinct values per
                                row. We pick the integer floor of the target
                                format's bpw (Q4_K_M ≈ 4.8 → 4, Q5_K_S ≈ 5.5
                                → 5, Q6_K ≈ 6.5 → 6, Q8_0 ≈ 8.5 → 8).
        block_size  int         columns processed per OBS block (default 128,
                                independent of llama-quantize's k-quant
                                group_size — see preconditioning-handoff.md
                                gotcha 2 option (a)).
        damping     float       λ added to H as `λ · mean(diag(H))`. 0.01 is
                                prismaquant's reference default.

Output: W_out       [out, in]   float32 weight after OBS error propagation.
                                Each row's columns end up on a uniform-RTN
                                grid at the chosen `bits`; the columns
                                downstream of each block have been shifted to
                                pre-compensate for the rounding error the
                                preceding block incurred.

Cholesky-fail behavior: retry once with `damping × 10` (capped at 0.1) per
preconditioning-handoff.md gotcha 1. If the retry also fails, return the
unrounded `W` so the caller can fall back to vanilla llama-quantize on the
original BF16. The return-value's `cholesky_fallback` field on the stats
dict signals which branch ran.

This module deliberately does NOT model llama-quantize's per-format codebook
(k-quant nested scales, IQ-quant lattice, etc.). The GPTQ-rounded weight is
written back as BF16 and re-quantized by Stage H. The expectation is that
the activation-aware redistribution of per-row magnitudes improves the
final post-quant error in expectation across reasonable quantizers — not
that our simulator exactly matches the target's codebook. See the design
doc § "Phase plan / P3" and § "Open questions / Format-bpw threshold".
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class GptqStats:
    """Per-tensor diagnostics for the precondition manifest."""
    bits: int
    block_size: int
    lambda_used: float
    n_blocks: int
    n_dead_channels: int
    cholesky_fallback: bool          # True iff both damping attempts failed
    mse_before: float                # ||W - RTN(W)||² (uniform-RTN at `bits`)
    mse_after: float                 # ||W - W_out||² (post-OBS, same bits)
    mse_ratio: float                 # mse_after / mse_before (≤1 is good)


def _block_rtn_dequant(block: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-row symmetric integer RTN.

    For each row r:
        s_r = max(|block[r, :]|) / q_max
        q_r = clip(round(block[r, :] / s_r), -q_max, q_max)
        out[r, :] = q_r · s_r

    where `q_max = 2^(bits-1) - 1`. Returns the dequantized block in float32.
    `s_r = 0` (all-zero row in this block) is replaced by `1` so the round
    step doesn't 0/0; the block stays zero through round-and-rescale anyway.
    """
    q_max = float(2 ** (bits - 1) - 1)
    block_max = block.abs().amax(dim=-1, keepdim=True)               # [rows, 1]
    s = block_max / q_max
    s = torch.where(s > 0, s, torch.ones_like(s))
    q = torch.round(block / s).clamp(-q_max, q_max)
    return q * s


def _rtn_baseline(W: torch.Tensor, bits: int, block_size: int) -> torch.Tensor:
    """RTN-only dequant for the same row+block+bits scheme. Used for the
    `mse_before` baseline that lets the orchestrator decide whether GPTQ
    helped this tensor (a sanity gate — most tensors should improve)."""
    out = torch.empty_like(W)
    rows, cols = W.shape
    for s in range(0, cols, block_size):
        e = min(s + block_size, cols)
        out[:, s:e] = _block_rtn_dequant(W[:, s:e], bits)
    return out


def _build_hessian(activations: torch.Tensor, in_features: int,
                   ) -> torch.Tensor:
    """Build `H = X.T @ X / N` from cached activations.

    The act-cache stores activations as either `[N, in]` (post-flatten from
    the probe's hooks) or `[*, in]` (raw forward-pass shape). We flatten
    everything but the last axis and demand the last axis equals
    `in_features`. Float32 for numerical conditioning.

    Normalizing by N keeps damping comparable across Linears with very
    different sample counts — the reference's `H.diagonal().add_(damp *
    diag_mean)` is N-invariant after this normalization.
    """
    X = activations.detach().to(torch.float32)
    if X.dim() > 2:
        X = X.reshape(-1, X.shape[-1])
    if X.shape[-1] != in_features:
        raise ValueError(
            f"GPTQ act-shape mismatch: acts last-axis={X.shape[-1]} ≠ "
            f"weight in_features={in_features}")
    N = max(X.shape[0], 1)
    H = (X.t() @ X) / N
    return H


def _cholesky_inverse_upper(H: torch.Tensor, damping: float
                              ) -> tuple[Optional[torch.Tensor], int]:
    """Damp + Cholesky + invert + upper-triangular re-factor.

    Returns `(U, n_dead)` where `U` is upper-triangular with `U.T @ U = H⁻¹`
    (the form GPTQ uses for the column-update step), or `(None, n_dead)`
    if the Cholesky factorization failed even with the requested damping.

    `n_dead` is the count of input channels whose un-damped diagonal was
    ≤ 0 — those rows/cols are set to the identity row before factoring
    (and the caller zeros the corresponding weight columns, matching
    reference line 1290-1293).
    """
    H_work = H.clone()
    diag_mean = torch.diagonal(H_work).mean().clamp_min(1e-12)
    H_work.diagonal().add_(damping * diag_mean)

    dead = torch.diagonal(H_work) <= 0
    n_dead = int(dead.sum().item())
    if n_dead:
        H_work[dead, dead] = 1.0

    try:
        L = torch.linalg.cholesky(H_work)
        Hinv = torch.cholesky_inverse(L)
        U = torch.linalg.cholesky(Hinv, upper=True)
        return U, n_dead
    except Exception:
        return None, n_dead


def gptq_apply(weight: torch.Tensor,
               activations: torch.Tensor,
               bits: int,
               block_size: int = 128,
               damping: float = 0.01,
               damping_retry: float = 0.1,
               ) -> tuple[torch.Tensor, GptqStats]:
    """Run GPTQ OBS rounding on one Linear's weight.

    Args:
      weight:      float32 [out, in] weight (pass the AWQ-rescaled weight
                   from P2's output for AWQ+GPTQ; pass the raw BF16 weight
                   for the GPTQ-only path on non-AWQ-folded tensors).
      activations: float32 [*, in] activations for this Linear. Caller is
                   responsible for the AWQ `a/s` divide when applicable.
      bits:        target simulator bits (4/5/6/8).
      block_size:  OBS block columns. 128 matches the design's "decoupled
                   from quant group_size" decision.
      damping:     initial λ. Reference default 0.01.
      damping_retry: stronger λ used on a single Cholesky retry. Reference
                   gotcha-1 recipe: `max(λ * 10, 0.1)`. The caller passes
                   0.1 directly here so the retry is deterministic.

    Returns:
      (W_out, stats)
      W_out is float32 [out, in]. On Cholesky-fail both passes, W_out is
      `weight` unchanged and `stats.cholesky_fallback=True` (caller treats
      this entry as `applied:awq` only / `skip:gptq_cholesky`).
    """
    if bits < 2 or bits > 8:
        raise ValueError(f"gptq_apply: bits must be in [2,8], got {bits}")
    if block_size < 1:
        raise ValueError(f"gptq_apply: block_size must be ≥1, got {block_size}")

    W = weight.detach().to(torch.float32).clone()
    rows, cols = W.shape

    # Baseline MSE (uniform-RTN at the same bits/block_size, no propagation)
    # — gives the orchestrator a per-tensor "did GPTQ help" gauge in the
    # manifest.
    with torch.no_grad():
        W_rtn = _rtn_baseline(W, bits, block_size)
        mse_before = float((W - W_rtn).pow(2).sum().item())

    # Build H + damped upper-tri inverse factor.
    H = _build_hessian(activations, in_features=cols)
    U, n_dead = _cholesky_inverse_upper(H, damping)
    lambda_used = damping
    if U is None:
        # Retry once with stronger damping.
        U, n_dead = _cholesky_inverse_upper(H, damping_retry)
        lambda_used = damping_retry
    if U is None:
        # Both passes failed — fall back to the unrounded weight. Caller
        # decides whether to keep AWQ-only fold or revert entirely.
        return weight.detach().to(torch.float32), GptqStats(
            bits=bits, block_size=block_size, lambda_used=lambda_used,
            n_blocks=0, n_dead_channels=n_dead, cholesky_fallback=True,
            mse_before=mse_before, mse_after=mse_before, mse_ratio=1.0,
        )

    # Zero dead-channel weight columns (matches reference line 1293).
    if n_dead:
        dead = torch.diagonal(H) <= 0
        W[:, dead] = 0.0

    # Block-wise OBS loop.
    n_blocks = 0
    for block_start in range(0, cols, block_size):
        block_end = min(block_start + block_size, cols)
        block = W[:, block_start:block_end]

        block_dq = _block_rtn_dequant(block, bits)
        block_err = block - block_dq

        if block_end < cols:
            U_block_diag = torch.diagonal(U)[block_start:block_end].clamp_min(1e-12)
            U_offdiag = U[block_start:block_end, block_end:]
            # err_block / diag(U_block) @ U_offdiag  →  [rows, rest]
            prop = (block_err / U_block_diag.unsqueeze(0)) @ U_offdiag
            W[:, block_end:] = W[:, block_end:] - prop

        W[:, block_start:block_end] = block_dq
        n_blocks += 1

    mse_after = float((weight.detach().to(torch.float32) - W).pow(2).sum().item())
    mse_ratio = mse_after / mse_before if mse_before > 0 else 1.0

    return W, GptqStats(
        bits=bits, block_size=block_size, lambda_used=lambda_used,
        n_blocks=n_blocks, n_dead_channels=n_dead, cholesky_fallback=False,
        mse_before=mse_before, mse_after=mse_after, mse_ratio=mse_ratio,
    )


def bits_for_fmt(fmt: str, bpw: Optional[float]) -> Optional[int]:
    """Map a GGUF format name (or its bpw) to a simulator-bits integer.

    Heuristic: floor of bpw, clamped to [2, 8]. Returns None for formats
    we can't classify (caller treats as skip:gptq_unknown_bits).

    Per the design doc, GPTQ runs only at ≥4 bits; the orchestrator gates
    on `cfg.precondition_bpw_floor` before calling this. We still return
    a value for the 2-3 bit range so the function is testable in isolation.
    """
    if bpw is None:
        return None
    b = int(bpw)
    if b < 2:
        return 2
    if b > 8:
        return 8
    return b
