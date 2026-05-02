# Gemma-4 prismaquant profile — patch series

Status: **draft / WIP** (last updated 2026-05-02)

This document captures the seven patches needed to make
[`RobTand/prismaquant`](https://github.com/RobTand/prismaquant)'s
`incremental_probe` succeed on Google's Gemma-4 model family
(`Gemma4ForConditionalGeneration` / `Gemma4ForCausalLM`). Each patch is
local to our installed prismaquant package on `ai01`; they are
catalogued here for eventual upstream PR.

The patches stack — each surfaced the next as a blocker. Tested on
`google/gemma-4-E4B-it` (4B effective / 8B with embeddings) at
nsamples=16, seqlen=512.

## Why gemma-4 is challenging

Gemma-4 introduces several architectural patterns that the streaming
probe path doesn't natively support:

| Feature | Mechanism | What breaks |
|---|---|---|
| **Multi-layer-type rope** | `cfg.rope_parameters` is a dict-of-dicts keyed by `layer_type` (`full_attention` / `sliding_attention`). Each type has its own `rope_theta` + (optionally) `partial_rotary_factor`. | `compute_default_rope_parameters(cfg, device)` raises `KeyError(None)` because the function reads `config.rope_parameters[layer_type]` and `layer_type` defaults to None |
| **Per-layer additive embedding** | Each decoder layer takes a `per_layer_input` kwarg of shape `[B, T, hidden_size_per_layer_input]`, computed model-side from a dedicated per-layer embedding table | Layer fails on `hidden_states * per_layer_input` (None × tensor) |
| **iSWA — partial rotation** | `full_attention` layers use `partial_rotary_factor=0.25` + `rope_type="proportional"` → rotates only 25% of `head_dim`. `sliding_attention` uses default rope, rotates all. **cos/sin tensors have different shape** per type | `apply_rotary_pos_emb` shape mismatch when prismaquant computes `position_embeddings` once globally and reuses across all layers |
| **Cross-layer K/V sharing** | Last `num_kv_shared_layers` layers (18 of 42 for E4B) reuse K/V from earlier layers via `shared_kv_states[kv_shared_layer_index]` — a list passed kwarg, written by non-shared layers, read by shared layers | Shared layers crash on `(k, v) = shared_kv_states[i]` because the list isn't passed (it's `None`) |
| **Orphan k_norm/v_norm/k_proj/v_proj on shared layers** | Checkpoint saves these tensors for ALL layers (legacy from training); transformers' `Gemma4TextAttention` only allocates them on **non-shared** layers (`if not self.is_kv_shared_layer`). Shared layer instances have **no `self.k_norm`/`v_norm`/`k_proj`/`v_proj` attributes** | Streaming installer fails `set_module_tensor_to_device` with "object has no attribute 'k_norm'" |

## The seven patches

Listed in surfacing order (each blocker surfaces the next). All paths
are relative to the installed prismaquant package
(`~/.local/lib/python3.13/site-packages/prismaquant/`).

### 1. Multi-layer-type rope fallback in `_init_rotary_inplace`

**File**: `streaming_model.py` (around line 115)

**Before**: Single call to `rope_init_fn(cfg, device)` raises `KeyError(None)`.

**After**: Wrap in try/except `(KeyError, TypeError)` — if the rotary's
config has a dict-of-dicts `rope_parameters`, iterate over keys and
register per-type `<layer_type>_inv_freq` buffers + a `None_inv_freq`
alias for callers that bypass per-layer dispatch.

This is the **defensive** fallback. The proper fix lives in the gemma4
profile's `init_rotaries` (patch 5 below); this patch is a backstop for
any other model with a dict-of-dicts rope structure.

### 2. `Gemma4Profile.init_rotaries` — proper per-type rope registration

**File**: `model_profiles/gemma4.py` (new method)

Mirrors `transformers.Gemma4TextRotaryEmbedding.__init__`:
- For each layer_type in `cfg.rope_parameters.keys()`:
  - Resolve `rope_init_fn` (default → `compute_default_rope_parameters`; otherwise look up in `transformers.modeling_rope_utils.ROPE_INIT_FUNCTIONS`)
  - Add `head_dim_key="global_head_dim"` kwarg for `full_attention` + `proportional` rope_type combination
  - Register `<layer_type>_inv_freq`, `<layer_type>_original_inv_freq` (clone), and `<layer_type>_attention_scaling`
- Also register `None`-keyed alias buffers + the generic `inv_freq` / `attention_scaling` fallback for compatibility with single-rope callers

Also caches in **class-level** state (not instance — see patch 4):
- `_h_per_layer_input`
- `_num_hidden_layers`
- KV shape parameters (`_head_dim`, `_global_head_dim`, `_num_kv_heads`, `_num_global_kv_heads`, `_attn_k_eq_v`, `_layer_types`)

Returns `True` when registration succeeds (signals to streaming_model's
profile dispatch that the default path should be skipped).

### 3. `Gemma4Profile.extra_layer_kwargs` — synthetic per_layer_input

**File**: `model_profiles/gemma4.py` (new method)

Returns `{"per_layer_input": torch.ones(B, T, hidden_size_per_layer_input,
dtype=bf16, device=...)}` per layer call.

**Tradeoff**: All-ones rather than the proper sliced computation
(`get_per_layer_inputs(input_ids, inputs_embeds)` →
`project_per_layer_inputs(inputs_embeds, per_layer_inputs)`) makes the
`hidden_states * per_layer_input` multiplication a no-op. Hessians for
`per_layer_input_gate.weight` and `per_layer_projection.weight` will
be slightly biased toward over-allocation (conservative). All other
weight tensors get accurate Hessians.

For **full fidelity**, the proper fix would precompute per_layer_inputs
at probe init, slice per layer index in `extra_layer_kwargs`, and keep
`embed_tokens_per_layer` + `per_layer_model_projection` +
`per_layer_projection_norm` resident on CPU. Deferred — would require
threading `layer_idx` through the call sites in `incremental_probe.py`
(currently `extra_layer_kwargs` only receives `input_ids`).

### 4. Class-level state for cross-method profile communication

**File**: `model_profiles/gemma4.py`

`profile_from_model()` returns a **fresh instance** each call (see
`model_profiles/registry.py:_resolve()`'s `inst = cls()`). Instance
state set in `init_rotaries` would be invisible to `extra_layer_kwargs`
called later. Solved by stashing on the **class** (`Gemma4Profile._h_per_layer_input`)
instead of the instance.

This is a tradeoff — class state is process-wide; running two gemma-4
probes concurrently in the same Python process would clobber. In
practice prismaquant runs one probe at a time per process, so this
is safe for now.

**Upstream fix would be**: cache profile instances in `profile_from_model`
keyed by `id(model)` so subsequent calls return the same instance.

### 5. Per-layer position_embeddings in the probe loops

**File**: `incremental_probe.py` (two call sites: phase-1 forward at
~line 1112, phase-3 backward at ~line 2054)

**Before**: `position_embeddings = _compute_position_embeddings(...)`
called **once** before the layer loop, then passed verbatim to all
layers via `_call_layer(... position_embeddings=position_embeddings)`.

**After**: Inside the loop, if `cfg.layer_types` is set (multi-layer-type
rope detected), recompute per-layer position_embeddings via a new helper
`_compute_position_embeddings_for_layer(base_model, hidden, position_ids,
layer_type)` that calls `rotary(hidden, position_ids, layer_type=...)`.
Falls back to the global value otherwise.

The helper also tries with and without the `layer_type` kwarg
(`TypeError` → fallback) so it's safe across rotary signatures.

### 6. `Gemma4Profile.extra_layer_kwargs` — synthetic shared_kv_states

**File**: `model_profiles/gemma4.py` (extending patch 3)

Synthesize `[(zeros_k, zeros_v), ...]` of length `num_hidden_layers`,
shaped per-layer based on `layer_types[i]`:

- `full_attention` (with `attention_k_eq_v=True`): use
  `(num_global_key_value_heads, global_head_dim)`
- `full_attention` (with `attention_k_eq_v=False`) or `sliding_attention`:
  use `(num_key_value_heads, head_dim)`

Shape is `[B, num_kv_heads, T, head_dim]` per slot. Non-shared layers
will overwrite their slot during forward; shared layers either read the
populated slot (if a non-shared sibling has run earlier in the same
batch — always the case in prismaquant's sequential probe) or the
zero placeholder.

**Hessian impact**: For the q_proj on kv-shared layers, the attention
output during probe is computed against zero K/V (or against earlier
non-shared layer's K/V — actually correct). Either way, downstream
hidden state is approximately correct.

### 7. Orphan-tensor skip in `_fast_install`

**File**: `layer_streaming.py` (around line 486)

Wrap the `set_module_tensor_to_device(model, model_name, device, value=t)`
fallback in `try/except AttributeError`. Some checkpoints (Gemma-4) save
weights for tensor names that the model class doesn't define — for
gemma-4 specifically, kv-shared layers (the last 18 of 42) save
`k_norm.weight`, `v_norm.weight`, `k_proj.weight`, `v_proj.weight` for
training-time reasons, but `Gemma4TextAttention` skips creating those
attrs when `is_kv_shared_layer=True`.

`transformers.from_pretrained()` ignores these orphan keys silently.
prismaquant's piecemeal installer was raising. This patch makes it
mirror transformers' behavior.

## Patch series totals

| Code lines | Files touched |
|---|---|
| ~120 lines (additive) | 4 files: `streaming_model.py`, `model_profiles/gemma4.py`, `incremental_probe.py`, `layer_streaming.py` |

## Verification

`google/gemma-4-E4B-it`, `nsamples=16`, `seqlen=512`, `device=cpu`,
`dtype=bf16`:

- Stage C phase-1 forward: **57.5 sec** for 42 layers
- Layer cache: 7.5 GB / 21.3 GB (98% hit rate)
- All 42 layers forward-passed without crashing

Phases 2-3 still in flight as of writing.

## Upstream PR plan

These patches target **two upstream repos**:

1. **`RobTand/prismaquant`**: patches 1, 5, 7 (defensive
   robustness — multi-layer-type rope fallback, per-layer
   position_embeddings, orphan-tensor skip). Generic to any
   architecture with these patterns; not gemma-4 specific.

2. **`RobTand/prismaquant` model_profiles/gemma4.py**: patches 2, 3, 4, 6
   (gemma-4-specific). Replaces the current 127-line stub
   (vLLM-allocator-metadata-only) with a full streaming-probe profile.

The profile patches require landing the defensive patches first
(specifically patch 5, per-layer position_embeddings) since the gemma4
profile depends on the call site supporting per-layer rope.

## Known limitations

1. **Synthetic per_layer_input is biased.** For full-fidelity Hessians
   on `per_layer_input_gate` and `per_layer_projection`, would need to
   compute proper per-layer inputs — see patch 3 notes.
2. **Class-level state isn't thread-safe.** Two concurrent gemma-4 probes
   in one process would clobber. See patch 4.
3. **Phase 2/3 untested.** Only phase-1 (forward) verified at time of
   writing; phase-2 (CE backward) and phase-3 (per-layer Fisher hooks)
   may surface additional gemma-4 quirks.
4. **Multimodal towers untested.** This profile only covers the text
   path. The vision/audio towers in `Gemma4ForConditionalGeneration`
   pass through as BF16 (per existing `source_passthrough_prefixes`).
