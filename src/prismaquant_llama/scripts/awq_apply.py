"""Pure-math AWQ "proper-fold" primitives, factored out from prismaquant.

Three helpers (all torch float32, no external state):

  - `awq_joint_channel_scale(activations_list)` — per-input-channel scale
    `s[c] = mean|a[:, c]|^0.5` with geomean normalization + hard clamp.
  - `awq_apply_to_weight(W, s)` — input-channel rescale, returns `W * s[None, :]`.
  - `awq_fold_into_norm(gamma, s)` — predecessor RMSNorm γ ← γ / s.

The "proper AWQ" invariant per fold (see `prismaquant/export_native_compressed.py`
around line 940):

    γ_new := γ / s
    For every reader M of γ:   M.W_new[:, in] := M.W[:, in] * s[in]
    then at runtime:    M(γ_new · x) = (M.W * s) · (γ/s · x) = M.W · γ · x

i.e. mathematically identical to the unfolded layer, but with weight noise
redistributed across input channels so high-activation channels get finer
quant-grid resolution after the post-fold quantizer runs.

The scale is computed only from cached activations of readers that the
caller designates as AWQ-eligible (typically the ≥4-bit ones). The fold
still applies to ALL readers of γ so the identity holds.
"""

from __future__ import annotations
from typing import Sequence

import torch


def awq_joint_channel_scale(
    activations_list: Sequence[torch.Tensor],
    eps: float = 1e-4,
    clamp_ratio: float = 10.0,
) -> torch.Tensor:
    """Compute one AWQ per-input-channel scale across a list of activation
    captures that all feed through the SAME predecessor γ.

    For Q/K/V the three captures are typically identical (they share the
    `input_layernorm` output); for gate/up the two captures are identical
    (they share `post_attention_layernorm`). The list form is defensive —
    a missing reader still leaves a valid scale from the surviving ones.

    Args:
      activations_list: list of `[*, in_features]` fp32 tensors.
      eps: floor on mean-abs (prevents zero-channel poisoning the
        normalization).
      clamp_ratio: log-symmetric clamp window `[1/clamp_ratio, clamp_ratio]`.
        Bounds bf16 cancellation error in `(W * s) · (γ / s)` at runtime —
        bf16 mantissa is 8 bits so per-product error ~0.4 %; capping
        max(s) / min(s) at clamp_ratio² keeps the accumulated cancellation
        error in check on layers whose raw activation-mean ratio is
        ~1e4 (some Qwen channels do this).

    Returns: fp32 1-D tensor `s[in_features]`.
    """
    combined = torch.cat(
        [a.detach().to(torch.float32).reshape(-1, a.shape[-1])
         for a in activations_list],
        dim=0,
    )
    mean_abs = combined.abs().mean(dim=0)                  # [in_features]
    s = mean_abs.clamp_min(eps).pow(0.5)                   # α = 0.5
    # Geomean normalization centers s around 1 in log space. See AutoAWQ
    # `quantize/quantizer.py:406` and llm-awq `auto_scale.py:130`.
    norm = (s.max() * s.min()).sqrt().clamp_min(eps)
    s = s / norm
    s = s.clamp(1.0 / clamp_ratio, clamp_ratio)
    s = torch.nan_to_num(s, nan=1.0, posinf=1.0, neginf=1.0)
    return s


def awq_apply_to_weight(weight: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """In-channel rescale: `W * s[None, :]`, where W is `[out, in]` and
    s is `[in]`. Returns float32; caller is responsible for downcasting
    back to the storage dtype.
    """
    if weight.shape[1] != s.shape[0]:
        raise ValueError(
            f"AWQ apply: weight.in={weight.shape[1]} ≠ s.len={s.shape[0]}")
    W = weight.detach().to(torch.float32)
    return W * s.to(W.device, torch.float32).unsqueeze(0)


def awq_fold_into_norm(gamma: torch.Tensor, s: torch.Tensor,
                        eps: float = 1e-12) -> torch.Tensor:
    """Predecessor γ ← γ / s. Clamps s to `eps` to dodge div-by-zero on a
    degenerate channel. Returns float32.
    """
    if gamma.shape != s.shape:
        raise ValueError(
            f"AWQ fold: gamma.shape={tuple(gamma.shape)} ≠ "
            f"s.shape={tuple(s.shape)} (must be 1-D, equal length)")
    g = gamma.detach().to(torch.float32)
    s_safe = s.to(g.device, torch.float32).clamp_min(eps)
    return g / s_safe
