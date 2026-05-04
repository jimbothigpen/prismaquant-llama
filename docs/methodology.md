# PRISMAQUANT — Bayesian per-tensor mixed quantization for GGUF

prismaquant picks a **different ggml format for every tensor** under a total-size budget, minimizing measured loss. Each tensor's allocation is the argmin of `½·H_trace·MSE + λ·size` — Fisher-trace times measured per-format MSE — solved as a multi-choice knapsack via Lagrangian relaxation.

The output is a standard GGUF that any prismaquant-aware llama.cpp build loads natively. No custom runtime, no patches.

> Built on top of [RobTand/prismaquant](https://github.com/RobTand/prismaquant), which targets vLLM/compressed-tensors. This repo adds the GGUF-targeting adapter (probe → bridge → cost → allocate → llama-quantize).

---

## The cost model

For each Linear `t`, prismaquant's surrogate Δloss is

```
Δloss(t, fmt) ≈ ½ · H_trace(t) · MSE_W(t, fmt)
```

- **H_trace(t)** — trace of the empirical Fisher diagonal at tensor t. One calibration forward pass on the HF model produces this. Captures *how much the loss actually moves* when this tensor's weights change. Computed once per model, in bf16.
- **MSE_W(t, fmt)** — per-tensor reconstruction error of `quantize → dequantize` against the original BF16 weights, with imatrix scaling applied. Computed per format the allocator can pick from. Bottlenecked by `llama-quantize-cost`.

The total is `Σ_t Δloss(t, fmt[t])`. Minimizing this under a size constraint is the multi-choice knapsack. Lagrangian relaxation: pick `fmt[t] = argmin_f (½·H_trace·MSE + λ·size)` per-tensor independently for each λ, then bisect λ to hit the budget. ~50 iterations, sub-second.

---

## Pipeline

`src/prismaquant_llama/pipeline_runner.py` orchestrates the full A→I sequence. Helper scripts (allocator, HF→GGUF bridge) live under [`src/pipeline/scripts/`](../src/pipeline/scripts/); the cost-measurement binary source under [`src/pipeline/cpp/quantize-cost/`](../src/pipeline/cpp/quantize-cost/).

```
A. Download HF safetensors
B. Convert HF → BF16 GGUF
C. Probe: HF Fisher trace (one bf16 forward pass)            → probe.pkl
D. Generate imatrix (llama-imatrix)                           → imatrix.gguf
E. Measure per-(tensor, format) MSE (llama-quantize-cost)     → costs.csv
F. Bridge HF tensor names → GGUF tensor names + Fisher        → bridge.json
G. Allocate per-tensor formats (multi-choice knapsack)        → recipe-PQ<budget>-<XYZ>.json
H. Apply recipe (llama-quantize --tensor-type-file)           → final GGUF
I. Optional: PPL eval (llama-perplexity)
```

Stages D, F, G, H are the loop you iterate on. A/B/C are one-shot per (model, calibration corpus).

### Stage E — cost measurement is the long pole

A full `quantize-cost` over a 35 B model × 12 candidate formats can take ~24 h on commodity hardware. Two reductions make it practical:

1. **Format whitelist.** Limit candidates to the formats the allocator is permitted to pick. Dropping runtime-codebook formats (e.g. `IQ4_KT`) saves a substantial fraction of wallclock since they're slow to *measure*.
2. **Representative subset + propagation.** Measure only a few exemplar layers (default `0, 3` — covers softmax-attn + linear-attn for hybrid models), then propagate same-suffix costs across peer layers. Brings Stage E down to a couple hours on a 35 B.

### Stage F — Pareto sweep

The allocator can emit a `pareto-{tag}.csv` showing format counts at multiple budgets, not just one recipe. Use this to *pick* the budget — find the inflection where adding another GB stops buying you Δloss.

---

## Pareto-front shape: where prismaquant earns its keep

Across model families, the per-tensor allocator behaves consistently:

- **Tight budgets (~20-30% of BF16)** are where prismaquant has the largest win over uniform quants. The allocator is forced to discriminate high-Fisher from low-Fisher tensors, and that discrimination captures real PPL signal that any one-format quant misses.
- **Loose budgets (≥50% of BF16)** see diminishing returns. The allocator converges toward "Q8_0 most things plus a small low-bit tail," which approaches the behavior of a well-designed uniform quant. Hand-tuned uniform formats with custom codebooks (e.g. Hadamard-rotated + Lloyd-Max) can match or beat the allocator at these budgets.
- **The "loss surrogate" over-promises at high budgets.** It drops sharply between tight and loose budgets, but actual PPL barely moves once the allocator has covered the high-Fisher tensors. The surrogate is `½·Fisher·MSE`, a first-order Taylor approximation — useful for *ranking* per-tensor formats, but a noisy *absolute* loss predictor at high bpw.

**Practical rule:** prismaquant's value lives at tight budgets. Don't pay the pipeline cost (probe + cost + allocator) if your budget is already at 50%+ of BF16.

---

## Strategies & recipes — how to pick budgets, formats, and pins

### Choosing a budget

1. **Pareto-driven (recommended).** Run the allocator with `--pareto-budgets-gb` covering a range. Look at the `loss_surrogate` column in the CSV. The "knee" — where the curve flattens — is your sweet spot.
2. **Baseline-matched.** Pick a budget equal to a known one-format size (e.g. IQ4_K's actual GB) so the comparison is direct.
3. **Hardware-fit.** Round down to fit `kv_cache + compute_buffer + budget < VRAM`.

### Choosing the format whitelist

Different whitelists optimize for different goals:

| Goal | Whitelist | Why |
|---|---|---|
| **Quality-first** (default) | `Q4_K, Q5_K, Q6_K, Q8_0, IQ4_K, IQ4_KS, IQ4_KSS, IQ4_KT, IQ4_XS, IQ3_K, IQ3_KS, IQ2_K` | Maximum allocator freedom; ~12 formats covering 2.4–8.5 bpw |
| **Speed-first (TG)** | drop runtime-codebook formats like `IQ4_KT` | Runtime-codebook formats can be slower at TG on some hardware |
| **Compat-only (mainline)** | `Q4_K, Q5_K, Q6_K, Q8_0, IQ3_XS, IQ3_S, IQ3_M, IQ4_XS` | Stock upstream formats — works on any llama.cpp build, no fork needed |
| **Aggressive low-bit** | add ternary formats | Niche; only useful at extreme budgets and for non-sensitive FFN |

Larger whitelist ≠ always better recipe — adding a format the allocator never picks just adds Stage E cost. Check `format_counts` in `recipe-*.json` after a run; if a format has count = 0, drop it next iteration.

### Pinning policy

`pinned-tensors.json` overrides the allocator with hard pins.

**Always pin (across all models):**

- `token_embd.weight: Q8_0` — input embedding; every token passes through. The allocator usually picks Q8_0 here anyway, but pinning costs nothing and avoids surprises at extreme budgets.
- `output.weight: Q6_K` — generation logits. Q6_K is the right floor; Q8_0 is wasteful since `output.weight` is often the largest single tensor.

**Pin for MoE models:**

- `blk.*.ffn_gate_inp.weight: Q8_0` — router/gate. Tiny tensor (~0.001% of size) but determines expert dispatch. The allocator may pick low-bit here; cost-of-error is high.

**Pin for hybrid SSM:**

- Small Gated-Delta-Net params (`ssm_alpha.weight`, `ssm_beta.weight`, etc.) — `Q5_K` or higher. Recipe usually allocates these well, but propagation from exemplars can mis-classify them. Worth a sanity-pin if you see PPL anomalies.

**Don't pin:**

- Attention tensors (q/k/v/output) — let the allocator decide.
- FFN expert tensors — these are where the biggest savings are. Pinning them defeats the purpose.

### Calibration data (imatrix)

| Calibration | When to use | Notes |
|---|---|---|
| **bartowski-calibration-v3** (200 chunks) | Default | Broad distribution; works for most chat/instruct models. Bundled with prismaquant-llama. |
| **wikitext103** (200+ chunks) | When PPL eval will be on wikitext | Avoids the imatrix↔eval distribution mismatch that inflates PPL on niche calibrations |
| **Domain-specific** (code, legal, etc.) | Domain inference workloads | Custom calibration; use with a matching domain probe |

Critical: **the imatrix used for `quantize-cost` (Stage E) and the imatrix used for `llama-quantize` (Stage H) MUST match.** Different imatrices → costs misalign with what quantize actually does → bad recipes. The pipeline runs both with the same `$IMATRIX` cache.

### Cost-propagation exemplars

`--exemplar-layers 0,3` is the default. The premise: layer-N tensors of the same architectural type behave similarly enough that one measurement transfers to peers. `0, 3` covers:

- Layer 0 — always present, often softmax-attn even in hybrid models
- Layer 3 — first linear-attn (Gated-Delta-Net etc.) layer in many hybrid architectures

For other architectures, adjust:

- **Pure dense (Llama-class)**: `0, 1` — all layers same type
- **Hybrid SSM (Mamba-style)**: `0, 3` (default)
- **MoE with router-on-every-layer**: `0, 1` — every layer has experts
- **Gemma-style global+local attention alternation**: `0, 1, 2, 3` to cover all 4 phases (paranoid; usually `0, 2` works)

Validate after a run: in `pipeline.log` look for `costs after propagation: N tensors`. N should be close to the GGUF tensor count. If too low, the propagator failed to classify some layer types — check `detect_layer_types` patterns in `allocator.py`.

### Validating a recipe

Before trusting a run, sanity-check:

1. **Recipe size matches budget**: `actual_size_gb` in `recipe-*.json` should be within ~1% of `budget_gb`. Bigger gap → bisection didn't converge; loosen `tol_bytes` in `bisect_lambda`.
2. **Format distribution makes sense**: count of each format. If 95%+ is one format, the allocator isn't doing its job — likely the H_trace is uniform (probe issue) or formats too narrow.
3. **`token_embd` and `output` are at safe formats** (Q8_0 / Q6_K). If the allocator picked anything below Q4_K here, your pins are wrong.
4. **Most sensitive layers are at high bit**: layer 0 attention should be Q8_0/Q6_K. If it's IQ3_K, the bridge is wrong (Fisher not landing on the right tensors).
5. **PPL beats baseline at same size**: minimum bar. If prismaquant's PPL is *worse* than the corresponding one-format quant of the same size, something is wrong — most likely cost↔Fisher unit mismatch.

---

## File-naming scheme for output GGUFs

Prismaquant GGUFs use the pattern:

```
<base-model>-PQ<budget>-<XYZ>.gguf
```

Where:

- `PQ` = prismaquant prefix (distinguishes from Bartowski/standard quants)
- `<budget>` = average target size in GB
- `<XYZ>` = 3-digit priority code: `X` = PPL weight, `Y` = PP (prompt-processing) weight, `Z` = TG (token-generation) weight, each `0`–`9`. Higher digit = higher allocator priority. Common combinations: `522` (PPL-heavy), `900` (pure PPL), `333` (balanced), `252` (PP-favoring), `225` (TG-favoring). When a single weight is dominant (e.g. `009`, `090`, `900`), the allocator may collapse to the same recipe regardless of how the remaining zero-weights split.

---

## Known limitations

- **Cost propagation is a heuristic.** Assumes layer-N peer tensors are MSE-equivalent. For very heterogeneous architectures (layered LoRA experts, attention-style shifts mid-stack), measure all layers.
- **The Fisher trace is computed in BF16** on the HF model. For models that diverge between BF16 and quantized inference (e.g. tool-use models with very tight numeric precision needs), the trace may not perfectly predict GGUF behavior.
- **MoE expert pruning** — upstream prismaquant treats `(format, dropped_expert_ids)` as a joint knapsack variable. The GGUF adapter currently only handles format selection; expert pruning is future work.

## See also

- [`docs/GETTING-STARTED.md`](GETTING-STARTED.md) — hands-on tutorial for the full pipeline
- [`src/pipeline/scripts/`](../src/pipeline/scripts/) — allocator + HF→GGUF bridge (Python)
- [`src/pipeline/cpp/quantize-cost/`](../src/pipeline/cpp/quantize-cost/) — cost-measurement binary source
- [`examples/recipes/`](../examples/recipes/) — sample allocator outputs to inspect the per-tensor format mapping
- [Upstream prismaquant](https://github.com/RobTand/prismaquant) — the original tool (vLLM/compressed-tensors target)
