# PRISMAQUANT — Bayesian per-tensor mixed quantization for GGUF

prismaquant picks a **different ggml format for every tensor** under a total-size budget, minimizing measured loss. Each tensor's allocation is the argmin of `½·H_trace·MSE + λ·size` — Fisher-trace times measured per-format MSE — solved as a multi-choice knapsack via Lagrangian relaxation.

The output is a standard GGUF that llama.cpp (this fork) loads natively. No custom runtime, no patches.

> Built on top of [RobTand/prismaquant](https://github.com/RobTand/prismaquant), which targets vLLM/compressed-tensors. This fork adds the GGUF-targeting adapter (probe → bridge → cost → allocate → llama-quantize).

---

## Headline result — Qwen3.6-35B-A3B (qwen35moe + Gated-Delta-Net hybrid SSM)

**Apples-to-apples 100-chunk PPL** on wikitext103-calibration (both prismaquant and the uniform baseline run at the same chunk count for fair comparison):

| Budget | Prismaquant PPL | Best uniform baseline at footprint | Δ |
|---:|---:|---|---:|
| 14 GB | **6.13** ± 0.03 | IQ3_KS = 6.61 ± 0.04 @ 14.2 GB | **−0.48** ✓ |
| 19 GB | **6.12** ± 0.03 | IQ4_K  = 6.12 ± 0.03 @ 19.8 GB | −0.005 (tied) |
| 21 GB | 6.09 ± 0.03 | **TQ4_1S = 6.06** ± 0.03 @ 21.9 GB | +0.035 (TQ4_1S wins) |

**Diminishing returns, fairly told.** Prismaquant wins by 0.48 PPL at the 14 GB budget — that's the headline result. At 19 GB the win shrinks to noise (mean diff −0.005, well within the ±0.03 confidence interval). At 21 GB, **TQ4_1S beats prismaquant by 0.035 PPL** — the well-tuned hand-optimized uniform quant (Hadamard + Lloyd-Max codebook) outperforms automatic per-tensor allocation when there's enough budget to "spend" without strict prioritization.

**Why diminishing returns**: see the Pareto-front analysis section below. The short version — at tight budgets, prismaquant's per-tensor sensitivity ranking captures real PPL signal that uniform quants miss. As budget grows, the allocator converges toward "Q8_0 most things" (57% Q8_0 at 19 GB, 71% at 21 GB) which approaches uniform-Q8_0 behavior; meanwhile, hand-tuned alternatives like TQ4_1S that don't use Q8_0 at all can do better on PPL at the same size.

> ⚠️ Earlier headline of "−1.21 / −0.42 / −0.47 PPL win" was based on chunks=10 prismaquant vs chunks=20 baseline numbers — apples-to-oranges. The chunks=100 apples-to-apples numbers above are the honest result.

---

## Headline result — Qwopus3.5-9B-v3.5 (qwen3_5 hybrid: linear-attention + full-attention)

Stage H sweep on 27 prismaquant recipes (9 priorities × 3 budgets) on
Llama-3.2-3B Wikitext-103 chunks=20 (matching the Qwen3.6 methodology),
ai00 gfx1150, KV cache f16:f16, default warmup, no flash-attn rotation
override.

**Cross-budget winners:**

| Budget | Best priority | PPL ± stderr | Avg size GB | pp512 tps | tg128 tps |
|---:|---|---:|---:|---:|---:|
|  4.0 GB | **252** | 8.9116 ± 0.111 | 3.84 | 264.1 | 17.49 |
|  5.25 GB | **900** | 7.4701 ± 0.090 | 5.36 | 280.1 | 12.86 |
|  6.5 GB | **522** | 7.4269 ± 0.090 | 6.75 | 306.9 | 10.51 |

**Production sweet-spot: `5.25-900` at PPL 7.4701** — captures ~99.4% of
the 6.5GB-tier quality (Δ +0.04 PPL = within stderr) at 1.4GB less
storage. The 4.0GB → 5.25GB step gives a much larger gain (Δ −1.44
PPL); 5.25GB → 6.5GB gives barely anything.

**PPL spread by budget** (max − min PPL across 9 priorities):

| Budget | Spread | Implication |
|---:|---:|---|
|  4.0 GB | 0.41 | Recipe choice dominates — pick wisely |
|  5.25 GB | 0.41 | Recipe choice still matters |
|  6.5 GB | 0.04 | Recipe choice barely matters; size dominates |

Mirrors the Qwen3.6 finding: diminishing returns above the model-specific
"sweet-spot" budget. For Qwopus3.5-9B-v3.5 that's ~5.25GB.

**Allocator collapse pattern**: priorities `009`, `090`, and `900` (one
non-zero axis weight) consistently produce identical or near-identical
PPL within a budget when their non-PPL weights are zero (e.g. 009 = 090
at 4.0GB and 6.5GB, both 7.4295). With 0 weight on PPL, the allocator
picks the same minimum-feasible recipe regardless of how the remaining
weight splits between PP and TG — at constraint-tight budgets this
flattens the priority-space output.

**"Slow-PP" recipes**: priorities **090, 252, 531** consistently land
in the top PPL tier at each budget despite having pp512 ~136-152 tps
(vs ~340 for siblings in their budget). The allocator is willing to
trade prefill throughput for quality, and these recipes win when PPL
is what's being ranked. For interactive workloads (TG-bound), the
slow-PP outliers are still acceptable picks — `5.25-090`, `5.25-252`,
`5.25-531` give very competitive PPL with TG ≈ 13.2 tps.

**4.0GB band winner — priority 252 (likely PPL=2/PP=5/TG=2)** beats
balanced 333 by Δ −0.024 (>2× stderr) — slight PP-leaning improves
quality at this constrained budget without sacrificing throughput.

Bench and PPL data is cached per `<output>/work/<run-id>/logs/`,
permitting re-evaluation on different chunks counts or KV-cache types
without re-quantizing.

### Pareto picks for Qwopus3.5-9B-v3.5

| Use case | Pick | Notes |
|---|---|---|
| Quality-first within memory limits | `6.5-522` | 7.4269 PPL, 6.75GB |
| **Production sweet-spot** | **`5.25-900`** | **7.4701 PPL, 5.36GB** — best PPL/GB |
| Maximum compression | `4.0-252` | 8.9116 PPL, 3.84GB — best 4GB recipe |
| Interactive (TG-bound) | `5.25-225` or `5.25-531` | tg ~12.7-13.2, PPL within 0.07 of `5.25-900` |
| Speed-bound prefill | `5.25-009` | pp 347.4 tps, PPL 7.5187 |

### File-naming scheme

Prismaquant GGUFs use the pattern:

```
<base-model>-PQ<budget>-<XYZ>.gguf
```

Where:
- `PQ` = prismaquant prefix (distinguishes from Bartowski/standard quants)
- `<budget>` = average target size in GB (e.g., `4.0`, `5.25`, `6.5`)
- `<XYZ>` = 3-digit priority code: X = PPL weight, Y = PP weight,
  Z = TG weight, each 0-9. Higher digit = higher priority for the
  allocator. Common priorities:
  - `522` — heavy PPL bias with mild PP/TG (production default)
  - `900` — pure PPL (best quality at the budget)
  - `252` — balanced PP-favoring (best for prefill-heavy workloads)
  - `225` / `531` — TG-favoring (interactive)
  - `009` / `090` — pure PP / pure TG (extreme cases; allocator may
    collapse to identical recipe when one weight is dominant)

Example: `Qwopus3.5-9B-v3.5-PQ5.25-900.gguf` is the prismaquant 5.25 GB
recipe with priority XYZ=900 (pure PPL bias).

> **Migration note**: pre-2026-05-02 GGUFs used `<base>-<budget>-<XYZ>.gguf`
> without the `PQ` prefix. Task #29 tracks the rename.

### Vulkan portability of current Pareto-winner GGUFs

| Recipe | Total GB | Vulkan-MISSING | Quants needed for Vulkan |
|---|---:|---:|---|
| **6.5-522 (best 6.5GB)** | 6.74 | **0.00 (0%)** ← runs on Vulkan TODAY | (pure Q6_K/Q5_K/Q8_0) |
| **5.25-900 (sweet spot)** | 5.35 | 3.02 (56%) | IQ4_KS, IQ4_K, IQ4_KSS, IQ3_KS, IQ3_K |
| **4.0-252 (extreme)** | 3.83 | 1.96 (51%) | IQ2_K + IQ3_KS, IQ4_KSS, IQ4_K, IQ4_KS |

The 6.5GB Pareto winner runs on the Vulkan backend with zero porting
work required (all weight quants are upstream-supported). The other
two require porting the IK-quant family from CUDA — see
[VULKAN-ROADMAP.md](VULKAN-ROADMAP.md) for the phased plan.

---

## The cost model

For each Linear `t`, prismaquant's surrogate Δloss is

```
Δloss(t, fmt) ≈ ½ · H_trace(t) · MSE_W(t, fmt)
```

- **H_trace(t)** — trace of the empirical Fisher diagonal at tensor t. One calibration forward pass on the HF model produces this. Captures *how much the loss actually moves* when this tensor's weights change. Computed once per model, in bf16, on whatever hardware can run inference.
- **MSE_W(t, fmt)** — per-tensor reconstruction error of `quantize→dequantize` against the original BF16 weights, with imatrix scaling applied. Computed per format the allocator can pick from. Bottlenecked by `llama-quantize-cost` (this fork's `tools/quantize-cost`).

The total is `Σ_t Δloss(t, fmt[t])`. Minimizing this under a size constraint is the multi-choice knapsack. Lagrangian relaxation: pick `fmt[t] = argmin_f (½·H_trace·MSE + λ·size)` per-tensor independently for each λ, then bisect λ to hit the budget. ~50 iterations, sub-second.

---

## Pipeline (current Qwen3.6 implementation)

Source: `src/prismaquant_llama/pipeline_runner.py` orchestrates the full A→I sequence; helper scripts (allocator, HF→GGUF bridge) live under [`src/pipeline/scripts/`](../src/pipeline/scripts/) and the cost-measurement binary source under [`src/pipeline/cpp/quantize-cost/`](../src/pipeline/cpp/quantize-cost/).

```
A. probe.pkl                          (HF Fisher trace; one bf16 forward pass)
B. wikitext sweep                     (baselines to beat)
C. imatrix.gguf                       (200 chunks, bartowski-calibration-v3)
D. costs.csv                          (llama-quantize-cost over candidate formats)
E. bridge.json                        (HF tensor names ↔ GGUF tensor names + Fisher)
F. recipe-{14G,19G,21G}.json + .txt   (allocator output, --tensor-type-file format)
G. Qwen3.6-35B-A3B-prismaquant-{N}G.gguf   (llama-quantize --tensor-type-file)
H. PPL eval                           (chunks=10 on wikitext)
I. summary report
```

Stages D, F, G, H are the loop you actually iterate on. A/B/C are one-shot per (model, calibration).

### Stage D — cost measurement is the long pole

Full `quantize-cost` on a 35 B model × 12 formats is ~24 h. Two reductions make this practical:

1. **Format whitelist** — limit to the formats the allocator is permitted to pick. We use 12 (`Q4_K, Q5_K, Q6_K, Q8_0, IQ4_K, IQ4_KS, IQ4_KSS, IQ4_KT, IQ4_XS, IQ3_K, IQ3_KS, IQ2_K`). Dropping `IQ4_KT` saves ~30% wallclock (runtime-codebook is slow to *measure*).
2. **Representative subset + propagation** — measure only a few exemplar layers (default `0,3` — covers softmax-attn + linear-attn for hybrid models), then `propagate_costs` copies same-suffix costs across peer layers. Brings Stage D to ~2 h. Validated for Qwen3.6 hybrid SSM.

### Stage F — Pareto sweep

The allocator emits not just one recipe but a `pareto-{tag}.csv` showing format counts at multiple budgets. Use this to *pick* the budget — find the inflection where adding another GB stops buying you Δloss. For Qwen3.6 the sweet spot is around 19 GB: bigger budgets don't move PPL much.

---

## Recipe gallery — what the allocator actually picked

Format distributions for Qwen3.6-35B-A3B (701 quantized tensors across 32 layers + token_embd + output):

### 21 GB recipe — quality-first
```
Q8_0:    305    (43% — most attention + sensitive FFN)
Q5_K:     62
IQ4_K:    38
IQ4_KS:   13
Q6_K:     10  (pinned: output.weight)
IQ4_XS:    4
```
Almost every attention tensor at Q8_0; large FFN_down_exps drops to Q5_K to fit.

### 19 GB recipe — balanced (recommended default)
```
Q8_0:    248    (35%)
Q6_K:     64
IQ4_K:    27
IQ4_KS:   37
IQ4_XS:   20
Q5_K:     18
IQ4_KSS:  15
IQ3_K:     3
```
Diverse mix. Allocator finds genuine per-tensor sensitivity differences and exploits them.

### 14 GB recipe — aggressive
```
Q6_K:    207    (29% — second-tier sensitivity gets Q6_K, not Q8_0)
IQ3_KS:   77    (11% — bulk FFN compression)
Q5_K:     53
Q8_0:     51    (only the most sensitive 7%)
IQ2_K:    15    (extreme aggression on least-sensitive expert tensors)
IQ4_XS:   14
IQ3_K:     9
IQ4_KSS:   3
IQ4_K:     2
IQ4_KS:    1
```
Tail with IQ2_K only at 14 GB — the allocator finds the few tensors where 2-bit doesn't hurt.

The takeaway: **uniform "everything is Q4_K_M" is leaving Δloss on the table in both directions**. Some tensors should be 8-bit, some can be 2-bit, and which-is-which is *measurable* — not guessable.

---

## Pareto-front analysis: why diminishing returns above ~14 GB

Looking at the full chunks=100 PPL data alongside the recipe format distributions reveals an important pattern:

| Budget | Prismaquant PPL | Δ between budgets | Q8_0 fraction | Loss surrogate (Σ ½·F·MSE) |
|---:|---:|---:|---:|---:|
| 14 GB | 6.13 | — | 12% (51/432) | 906.2 |
| 19 GB | 6.12 | −0.01 | **57%** (248/432) | 192.4 (−79%) |
| 21 GB | 6.09 | −0.03 | **71%** (305/432) | 103.4 (−46%) |

**Three observations**:

1. **Real PPL improvements stop at ~14 GB.** Going 14 → 21 GB buys only 0.04 PPL. The allocator's first 14 GB of bits already lands on the high-Fisher tensors; additional bits over-provision the less-sensitive ones without much PPL benefit.

2. **The "loss surrogate" over-promises at high budgets.** It drops 89% from 14 GB to 21 GB, but actual PPL barely moves. The surrogate is `½·Fisher·MSE`, a first-order Taylor approximation. For high-bpw quants (Q8_0 has tiny MSE), the surrogate becomes much smaller than the true ΔPPL — useful for *ranking* per-tensor formats, but a noisy *absolute* loss predictor.

3. **The allocator converges to "Q8_0 most things" at high budgets** — 12% → 57% → 71% Q8_0 from 14 → 19 → 21 GB. Once you're using Q8_0 for the majority of weight tensors, you've effectively become a hybrid Q8_0 + small-format-tail recipe, which is very close to what hand-tuned uniform quants can produce. That's why **TQ4_1S (which doesn't use Q8_0 at all but has Hadamard rotation + Lloyd-Max codebook) can outperform our 21 GB recipe**: at 21 GB there's enough room for a well-designed uniform format to do everything that prismaquant's automated allocation does, and TQ4_1S's hand-tuned codebook is mildly better than Q8_0 for this workload.

**Practical conclusions**:

- **Prismaquant's value lives at tight budgets** (≤ ~25% of BF16 size). Diverse multi-format mixing forces the allocator to actually distinguish high-Fisher from low-Fisher tensors, which is where the per-tensor surrogate is most informative.
- **At loose budgets, well-designed uniform quants** (TQ4_1S, IQ4_K, Q5_K_M) can match or beat the allocator. Don't pay the prismaquant pipeline cost (probe + cost + allocator) if your budget is already at 50%+ of BF16.
- **The "pure-PPL" priority (`900`) over-uses Q8_0** on hardware where Q8_0 has runtime cost (e.g. gfx1150 PP: 2.6× slower than Q5_K_M). Multi-priority allocator with `tps` weights avoids this trap by penalizing slow formats — see [`src/pipeline/scripts/allocator.py`](../src/pipeline/scripts/allocator.py) for the `--priority NNN` knob implementation.
- **Budget = 14 GB on this 35B model = ~20% of BF16**. The sweet spot for prismaquant is "20-30% of BF16," where multi-format mixing is forced by tight bits.

---

## Strategies & recipes — how to pick budgets, formats, and pins

### Choosing a budget

1. **Pareto-driven (recommended).** Run the allocator with `--pareto-budgets-gb 11.0,12.5,14.21,16.0,17.0,18.83,19.5,21.15,23.0,25.0`. Look at `loss_surrogate` column in the CSV. The "knee" — where the curve flattens — is your sweet spot.
2. **Baseline-matched.** Pick a budget equal to a known one-format size (e.g. IQ4_K's actual GB) so the comparison is direct. We did this for the 14/19/21 results.
3. **Hardware-fit.** Round down to fit `kv_cache + compute_buffer + budget < VRAM`. For ai00 (96 GB), 21 GB leaves room for a 35 B with 16k context.

### Choosing the format whitelist

Different whitelists are good for different goals:

| Goal | Whitelist | Why |
|---|---|---|
| **Quality-first** (default) | `Q4_K, Q5_K, Q6_K, Q8_0, IQ4_K, IQ4_KS, IQ4_KSS, IQ4_KT, IQ4_XS, IQ3_K, IQ3_KS, IQ2_K` | Maximum allocator freedom; 12 formats covering 2.4–8.5 bpw |
| **Speed-first (TG)** | drop `IQ4_KT` | Runtime-codebook IQ4_KT is +1–2% slower TG on gfx1150 vs Q4_K_M-class |
| **Compat-only** | `Q4_K, Q5_K, Q6_K, Q8_0, IQ3_XS, IQ3_S, IQ3_M, IQ4_XS` | Stock upstream formats — works on any llama.cpp; no fork |
| **Aggressive low-bit** | add `Q1_0_G128` | Niche; only useful at extreme budgets and for non-sensitive FFN |

Larger whitelist ≠ always better recipe — adding a format the allocator never picks just adds Stage D cost. Check `format_counts` in `recipe-*.json` after a run; if a format has count=0, drop it next iteration.

### Pinning policy

`pinned-tensors.json` overrides the allocator with hard pins. Two are mandatory; others are model-specific.

**Always pin (across all models):**
- `token_embd.weight: Q8_0` — input embedding; every token passes through. The allocator usually picks Q8_0 here anyway, but pinning costs nothing and avoids surprises at extreme budgets.
- `output.weight: Q6_K` — generation logits. Q6_K is the right floor; Q8_0 is wasteful here (this is the largest tensor in many models).

**Pin for MoE models:**
- `blk.*.ffn_gate_inp.weight: Q8_0` — router/gate. Tiny tensor (~0.001% of size) but determines expert dispatch. The allocator may pick low-bit here; cost-of-error is high.

**Pin for hybrid SSM:**
- `blk.*.ssm_alpha.weight, ssm_beta.weight: Q5_K`+ — small Gated-Delta-Net params. Recipe tends to allocate them well, but the propagation from exemplars can mis-classify these. Worth a sanity-pin if you see PPL anomalies.

**Don't pin:**
- Attention tensors (q/k/v/output) — let the allocator decide. It correctly picks Q8_0 for the most-sensitive layers (typically layer 0, 3, last) and lower-bit for the middle layers.
- FFN expert tensors — these are where the biggest savings are. Pinning them defeats the purpose.

### Calibration data (imatrix)

| Calibration | When to use | Notes |
|---|---|---|
| **bartowski-calibration-v3** (200 chunks) | Default | Broad distribution; works for most chat/instruct models |
| **wikitext103** (200+ chunks) | When PPL eval will be on wikitext | Avoids the imatrix↔eval distribution mismatch that inflates PPL on niche calibrations |
| **Unsloth-shipped imatrix** | When llama-imatrix won't run on your model (qwen35moe currently silent-exits) | Convenient but recipe-dependent — not the same calibration distribution as the probe |
| **Domain-specific** (code, legal, etc.) | Domain inference workloads | Custom calibration; use with the matching domain probe |

Critical: **the imatrix used for `quantize-cost` (Stage D) and the imatrix used for `llama-quantize` (Stage G) MUST match.** Different imatrices → costs misalign with what quantize actually does → bad recipes. Pipeline runs both with the same `$IMATRIX`.

### Cost-propagation exemplars

`--exemplar-layers 0,3` is the default. The premise: layer-N tensors of the same architectural type behave similarly enough that one measurement transfers to peers. `0,3` covers:
- Layer 0 — always present, often softmax-attn even in hybrid models
- Layer 3 — for qwen35moe, first linear-attn (Gated-Delta-Net) layer

For other architectures, adjust:
- **Pure dense (Llama-class)**: `0,1` is fine — all layers are the same type
- **Hybrid SSM (Mamba-style)**: `0,3` (default)
- **MoE with router-on-every-layer**: `0,1` — every layer has experts
- **Gemma-2/3 with global+local attention alternation**: `0,1,2,3` to cover all 4 phases (paranoid; usually `0,2` works)

Validate after a run: in `pipeline.log` look for `costs after propagation: N tensors`. N should be close to the GGUF tensor count (~700 for Qwen3.6). If too low, the propagator failed to classify some layer types — check `detect_layer_types` patterns in `allocator.py`.

### Validating a recipe

Before trusting a run, sanity-check:

1. **Recipe size matches budget**: `actual_size_gb` in `recipe-{tag}.json` should be within ~1% of `budget_gb`. Bigger gap → bisection didn't converge; loosen `tol_bytes` in `bisect_lambda`.
2. **Format distribution makes sense**: count of each format. If 95%+ is one format, the allocator isn't doing its job — likely the H_trace is uniform (probe issue) or formats too narrow.
3. **Token_embd and output are at safe formats**: Q8_0 / Q6_K. If the allocator picked anything below Q4_K here, your pins are wrong.
4. **Most sensitive layers are at high bit**: layer 0 attention should be Q8_0/Q6_K. If it's IQ3_K, your bridge is wrong (Fisher not landing on the right tensors).
5. **PPL beats baseline at same size**: minimum bar. If prismaquant's PPL is *worse* than the corresponding one-format quant of the same size, something is wrong — most likely cost↔Fisher unit mismatch.

---

## Hardware notes

- **ai00 (gfx1150)**: do NOT set `HSA_OVERRIDE_GFX_VERSION`. Native gfx1150. The override causes silent perplexity death during warmup. Pipeline script currently has the override hardcoded for Stage H — needs to be made conditional.
- **ai01 (gfx1102 / gfx1103)**: set `HSA_OVERRIDE_GFX_VERSION=11.0.2`.
- Stage C (imatrix gen) and Stage D (quantize-cost) are GPU-bound; serialize across machines or you'll stall both.

## Known limitations

- **`llama-imatrix` exits silently on qwen35moe** (qwen35moe + SSM hybrid). Workaround: use unsloth's shipped imatrix. Tracked as a separate ticket.
- **Cost propagation is a heuristic** — assumes layer-N peer tensors are MSE-equivalent. Validated for Qwen3.6 hybrid; for very heterogeneous architectures (e.g. layered LoRA experts, attention-style shifts mid-stack), measure all layers.
- **The Fisher trace is computed in BF16** on the HF model. For models that diverge between BF16 and quantized inference (e.g. tool-use models with very tight numeric precision needs), the trace may not perfectly predict GGUF behavior.
- **MoE expert pruning** — upstream prismaquant treats `(format, dropped_expert_ids)` as a joint knapsack variable. Our GGUF adapter currently only handles format selection; expert pruning is future work.

## See also

- [`docs/GETTING-STARTED.md`](GETTING-STARTED.md) — hands-on tutorial for the full pipeline
- [`src/pipeline/scripts/`](../src/pipeline/scripts/) — allocator + HF→GGUF bridge (Python)
- [`src/pipeline/cpp/quantize-cost/`](../src/pipeline/cpp/quantize-cost/) — cost-measurement binary source
- [`examples/recipes/`](../examples/recipes/) — sample allocator outputs to inspect the per-tensor format mapping
- [Upstream prismaquant](https://github.com/RobTand/prismaquant) — the original tool (vLLM/compressed-tensors target)
