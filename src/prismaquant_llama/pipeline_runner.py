"""
End-to-end pipeline runner — orchestrates the full prismaquant build:

    A. Download HF safetensors
    B. Convert HF → BF16 GGUF
    C. Generate Hessian probe (prismaquant package)
    D. Generate imatrix (llama-imatrix)
    E. Measure per-(tensor, format) MSE (llama-quantize-cost)
    F. Bridge HF tensor names → GGUF tensor names
    G. Allocate (multi-choice knapsack)
    H. Apply recipe (llama-quantize)
    I. (Optional) PPL eval (llama-perplexity)

Each stage is idempotent — checks for output file, skips if present.
Per-stage logs go to <output>/work/<run-id>/logs/stage-<X>.log.

CLI:
    prismaquant-llama pipeline run \\
        --hf-model meta-llama/Llama-3.2-1B-Instruct \\
        --binary /path/to/llama-quantize \\
        --calibration /path/to/wikitext.txt \\
        --output ~/prismaquant-builds \\
        --priority 333

When --budget-gb is omitted, defaults to 25% of the BF16 GGUF size (slightly
tighter than mainline IQ4_XS at 27-29% real-world).

Status: scaffold — subprocess wiring complete, end-to-end run not yet
verified (waits on a free GPU for Stage C/D/E/H).
"""

from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from .paths import find_binary, WorkPaths, DEFAULT_OUTPUT_ROOT, discover_companion_binaries, sanitize_model_name
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from paths import find_binary, WorkPaths, DEFAULT_OUTPUT_ROOT, discover_companion_binaries, sanitize_model_name  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    hf_model: str                     # "meta-llama/Llama-3.2-1B-Instruct"
    hf_revision: str = "main"
    binary: Optional[Path] = None     # auto-discovered if None
    calibration: Optional[Path] = None
    output_root: Path = DEFAULT_OUTPUT_ROOT
    budget_gb: Optional[float] = None  # None → auto (0.25 × BF16 GGUF size)
    priority: str = "333"
    budget_band_gb: float = 0.25
    budget_auto_ratio: float = 0.25     # used when budget_gb is None — 25% of BF16
                                        # ≈ slightly tighter than IQ4_XS (27-29% real-world)
    formats: list[str] = field(default_factory=lambda: [
        # Mainline llama.cpp formats only — keeps prismaquant-llama compatible
        # with stock ggml-org/llama.cpp builds out of the box. To use
        # ik_llama / frankenturbo / similar fork extensions (IQ4_K, IQ4_KS,
        # IQ4_KSS, IQ3_K, IQ3_KS, etc.), pass `--formats` explicitly with the
        # extension types appended. `prismaquant-llama discover <binary>`
        # lists everything your binary supports + suggests --formats strings.
        # Coverage: 2-bit through 8-bit. Note that 2-bit i-quants are
        # "very lossy" territory; allocator may pick them for low-sensitivity
        # tensors, but if results look poor try --formats with just Q4_K and
        # up (5-format conservative).
        "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0",   # K-quants
        "IQ2_S", "IQ3_XXS", "IQ3_S",                       # i-quants 2/3-bit
        "IQ4_XS", "IQ4_NL",                                # i-quants 4-bit
    ])
    pinned: dict[str, str] = field(default_factory=lambda: {
        "output.weight": "Q6_K",
        "token_embd.weight": "Q8_0",
    })
    chunks_imatrix: int = 200
    chunks_eval: int = 100
    ctx: int = 4096
    skip_eval: bool = False
    convert_script: Optional[Path] = None  # auto-discover convert_hf_to_gguf.py
    pipeline_scripts_dir: Optional[Path] = None  # bundled bridge + allocator
    hsa_override: Optional[str] = None     # "11.0.2" for gfx1102/1103; auto-detect by hostname
    format_perf_file: Optional[Path] = None        # per-format speed proxies for allocator's
                                           # multi-objective scoring (TG/PP weights). When
                                           # None, auto-discovers via find_format_perf_file_for_binary().
    default_format_perf_override: Optional[Path] = None  # user's "my preferred default" tps file;
                                                 # used as fallback before package examples.
                                                 # Can also be set via PRISMAQUANT_DEFAULT_FORMAT_PERF env var.

    # Disk-cleanup knobs — applied after Stage H success. Defaults match
    # legacy behavior (everything retained; opt-in deletion).
    clean_shared: bool = False     # delete _shared/hf-cache/<model>/ + _shared/bf16/<model>-BF16.gguf
    clean_imatrix: bool = False    # also delete _shared/imatrix-cache/<model>-BF16.imatrix.gguf
    clean_probe: bool = False      # also delete _shared/probe/<model>-* (probe.pkl + dirs)

    # Memory strategy. Default --no-mmap is fastest for normal-size models that
    # fit in host RAM. For models where BF16 size > host RAM (e.g. 122B class
    # on 31 GB hosts), --no-mmap forces a 244 GB anon allocation that OOMs.
    # Set use_mmap=True (CLI: --mmap) to drop --no-mmap from Stage D imatrix +
    # Stage I PPL eval — kernel-managed paging keeps host memory bounded by
    # whatever the page cache allows. Slower (cephfs/disk-bound) but bounded.
    use_mmap: bool = False

    # Per-tensor minimum bpw floor. Forbids low-precision quants on tensors
    # matching given regex patterns. Two knobs:
    #   attention_floor_bpw: simple knob — sets min bpw for attention tensors
    #     (q/k/v/qkv/gate/output projection weights). Default 4.0 excludes
    #     2/3-bit quants which are known to compound errors cross-token via
    #     QK^T affinities. Set to 0 to disable.
    #   floor_bpw: advanced — additional {pattern: min_bpw} rules merged with
    #     the attention default. Highest min_bpw per tensor wins.
    # The allocator fails loudly if a rule removes all candidates for a tensor.
    attention_floor_bpw: float = 4.0
    floor_bpw: Optional[dict] = None

    def __post_init__(self):
        if self.binary is None:
            self.binary = find_binary("quantize")
        if self.calibration is not None and not Path(self.calibration).exists():
            raise FileNotFoundError(f"calibration corpus not found: {self.calibration}")
        # Auto-locate convert_hf_to_gguf.py — assumes binary is at <fork>/build/bin/
        if self.convert_script is None:
            fork_root = self.binary.resolve().parents[2]
            cand = fork_root / "convert_hf_to_gguf.py"
            if cand.exists():
                self.convert_script = cand
        # Auto-locate bundled pipeline scripts (allocator.py, bridge_probe_to_gguf.py)
        if self.pipeline_scripts_dir is None:
            self.pipeline_scripts_dir = Path(__file__).parents[1] / "pipeline" / "scripts"
        # Auto HSA override for non-native arches (gfx1102/gfx1103 → emulate gfx1100).
        # Override applied via $HSA_OVERRIDE_GFX_VERSION env var on subprocess calls.
        # Set explicitly via --hsa-override or [defaults] hsa_override in config.toml
        # if your hardware needs it; auto-detection is intentionally not hostname-based
        # since hostname conventions vary across fleets.
        # Auto-discover format-perf file: prefers binary-sha-keyed cache from a
        # `calibrate deep` run on this exact binary, falls back to the static
        # examples/format-perf-<arch>.json by hostname match. The cache provides
        # per-format pp/tg throughput so the allocator's --priority XYZ weighting
        # actually applies; without it, TG/PP terms collapse and priority
        # effectively becomes 900 (pure-PPL) regardless of XYZ.
        if self.format_perf_file is None:
            try:
                from .calibration import find_format_perf_file_for_binary
            except ImportError:
                from calibration import find_format_perf_file_for_binary  # type: ignore
            examples_dir = Path(__file__).parents[2] / "examples"
            self.format_perf_file = find_format_perf_file_for_binary(
                self.binary,
                examples_dir,
                default_format_perf_override=self.default_format_perf_override,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(paths: WorkPaths, stage: str, msg: str) -> None:
    line = f"[{_ts()}] {msg}"
    print(line)
    if paths.logs_dir.exists():
        log_path = paths.logs_dir / f"stage-{stage}.log"
        with log_path.open("a") as f:
            f.write(line + "\n")


def _run(cmd: list[str], stage_log: Path, env_extra: Optional[dict] = None,
         timeout: Optional[float] = None) -> int:
    """Run a subprocess, tee stdout+stderr to stage_log, return rc."""
    env = {**os.environ, **(env_extra or {})}
    stage_log.parent.mkdir(parents=True, exist_ok=True)
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    with stage_log.open("a") as f:
        f.write(f"\n=== {_ts()}: {' '.join(str(c) for c in cmd)} ===\n")
        f.flush()
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, text=True, bufsize=1,
        )
        try:
            for line in proc.stdout:  # type: ignore
                f.write(line)
                f.flush()
        except KeyboardInterrupt:
            proc.terminate()
            raise
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            return 124
    return proc.returncode


def _hsa_env(cfg: PipelineConfig) -> dict:
    return {"HSA_OVERRIDE_GFX_VERSION": cfg.hsa_override} if cfg.hsa_override else {}


def _auto_pick_stage_e_exemplars(bf16_path: Path) -> list[int]:
    """Read the BF16 GGUF tensor shapes, classify layers by attention shape
    signature (sliding vs full for iSWA, etc.), and pick one exemplar layer
    per type for Stage E to measure. The allocator's propagation logic uses
    matching layer-type tagging to extrapolate from these exemplars.

    For non-iSWA arches (single layer type), returns [0, 3] — the historical
    default that covers low + mid-stack as a sanity check.

    For iSWA arches (gemma-3, gemma-4): returns first layer of EACH type, so
    e.g. [0, 5] for gemma-4-E4B (sliding=0, first full=5).
    """
    try:
        # Reuse the allocator script's GGUF parser + classifier.
        sys.path.insert(0, str(Path(__file__).parents[1] / "pipeline" / "scripts"))
        from allocator import (read_gguf_tensor_meta, detect_layer_types,
                                auto_pick_exemplar_layers)
        meta = read_gguf_tensor_meta(str(bf16_path))
        layer_type = detect_layer_types(meta)
        if not layer_type:
            return [0, 3]  # fallback: legacy default
        types = set(layer_type.values())
        if len(types) <= 1:
            # Single arch type: keep the safe pair (low + mid stack)
            return [0, 3]
        # iSWA / multi-type: one exemplar per type
        return auto_pick_exemplar_layers(layer_type)
    except Exception as e:
        # Defensive: any GGUF parse failure → legacy default
        print(f"  WARN: exemplar auto-detect failed ({e}); using [0, 3] default")
        return [0, 3]


def _discover_top_level_weights(bf16_path: Path) -> list[str]:
    """Scan the BF16 GGUF for top-level weight tensors (not under `blk.`)
    that need to be in Stage E's measurement set. Without this, model-specific
    weights like gemma-4's per_layer_token_embd.weight (5.38 GB BF16) get
    skipped by Stage E → not in the recipe → fall back to the default
    quantize-fallback format → recipe-size estimate is wrong AND quality
    suffers (no allocator choice for the format).

    Returns a list of tensor *base names* (e.g. ["token_embd", "output",
    "per_layer_token_embd", "per_layer_model_proj"]) suitable for inclusion
    in the Stage E include-regex's first alternation group.
    """
    try:
        sys.path.insert(0, str(Path(__file__).parents[1] / "pipeline" / "scripts"))
        from allocator import read_gguf_tensor_meta
        meta = read_gguf_tensor_meta(str(bf16_path))
        names = set()
        for tn, shape in meta.items():
            # Top-level tensors have NO "blk." prefix and only one "." (between
            # name and ".weight"). Skip 1D tensors (norms) — they stay F32 and
            # don't need cost measurement.
            if tn.startswith("blk."):
                continue
            if not tn.endswith(".weight"):
                continue
            if len(shape) < 2:
                continue
            base = tn[:-len(".weight")]
            # base must be a plain identifier (no further dots)
            if "." in base:
                continue
            names.add(base)
        return sorted(names)
    except Exception as e:
        print(f"  WARN: top-level weight discovery failed ({e}); using "
              f"[token_embd, output] default")
        return ["token_embd", "output"]


# ─────────────────────────────────────────────────────────────────────────────
# Stage A — download HF safetensors
# ─────────────────────────────────────────────────────────────────────────────

def stage_a_download(cfg: PipelineConfig, paths: WorkPaths) -> Path:
    """Returns the local dir holding the HF model snapshot."""
    safe_name = sanitize_model_name(cfg.hf_model)
    target = paths.hf_cache / safe_name
    marker = target / ".download.complete"
    if marker.exists():
        _log(paths, "A", f"A. HF model cached at {target} (skip)")
        return target
    _log(paths, "A", f"A. downloading {cfg.hf_model}@{cfg.hf_revision} → {target}")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit(
            "ERROR: huggingface_hub not installed. Run:\n"
            "       pip install --user huggingface_hub")
    snapshot_download(
        repo_id=cfg.hf_model, revision=cfg.hf_revision,
        local_dir=str(target),
        # snapshot_download is conservative by default; let it manage cache.
    )
    marker.touch()
    _log(paths, "A", f"A. download complete → {target}")
    return target


# ─────────────────────────────────────────────────────────────────────────────
# Stage B — convert HF → BF16 GGUF
# ─────────────────────────────────────────────────────────────────────────────

def stage_b_convert(cfg: PipelineConfig, paths: WorkPaths, hf_dir: Path) -> Path:
    safe_name = sanitize_model_name(cfg.hf_model)
    bf16_path = paths.bf16_dir / f"{safe_name}-BF16.gguf"
    if bf16_path.exists():
        _log(paths, "B", f"B. BF16 GGUF cached at {bf16_path} (skip)")
        return bf16_path
    if cfg.convert_script is None or not cfg.convert_script.exists():
        raise SystemExit(
            "ERROR: convert_hf_to_gguf.py not found. Pass --convert-script "
            "or ensure it lives at <fork>/convert_hf_to_gguf.py")
    _log(paths, "B", f"B. converting {hf_dir} → {bf16_path}")
    rc = _run(
        ["python3", str(cfg.convert_script), str(hf_dir),
         "--outtype", "bf16", "--outfile", str(bf16_path)],
        paths.logs_dir / "stage-B.log",
    )
    if rc != 0 or not bf16_path.exists():
        raise SystemExit(f"FAIL: B convert_hf_to_gguf.py exit={rc}")
    _log(paths, "B", f"B. BF16 GGUF: {bf16_path.stat().st_size/1024**3:.2f} GB")
    return bf16_path


# ─────────────────────────────────────────────────────────────────────────────
# Stage C — Hessian probe (prismaquant package)
# ─────────────────────────────────────────────────────────────────────────────

def stage_c_probe(cfg: PipelineConfig, paths: WorkPaths, hf_dir: Path) -> Path:
    probe_path = paths.probe_dir / f"{sanitize_model_name(cfg.hf_model)}-probe.pkl"
    if probe_path.exists():
        _log(paths, "C", f"C. probe.pkl cached at {probe_path} (skip)")
        return probe_path
    if cfg.calibration is None:
        raise SystemExit("ERROR: stage C needs a calibration corpus (--calibration)")
    _log(paths, "C", f"C. running prismaquant.incremental_probe on {hf_dir}")
    rc = _run(
        ["python3", "-m", "prismaquant.incremental_probe",
         "--model", str(hf_dir),
         "--dataset", str(cfg.calibration),
         "--nsamples", "16", "--seqlen", "512",
         "--device", "cpu", "--dtype", "bf16",
         "--output", str(probe_path),
         "--activation-cache-dir", str(paths.probe_dir / "act-cache"),
         "--work-dir", str(paths.probe_dir / "work")],
        paths.logs_dir / "stage-C.log",
    )
    if rc != 0 or not probe_path.exists():
        raise SystemExit(
            f"FAIL: C prismaquant.incremental_probe exit={rc}\n"
            f"  Likely cause: prismaquant package not installed.\n"
            f"  Install our fork (carries Gemma-4 + NemotronH patches):\n"
            f"    pip install git+https://github.com/jimbothigpen/prismaquant.git\n"
            f"  Upstream (simpler architectures only): "
            f"https://github.com/RobTand/prismaquant")
    _log(paths, "C", f"C. probe.pkl: {probe_path.stat().st_size/1024**2:.1f} MB")
    return probe_path


# ─────────────────────────────────────────────────────────────────────────────
# Stage D — imatrix (llama-imatrix)
# ─────────────────────────────────────────────────────────────────────────────

def stage_d_imatrix(cfg: PipelineConfig, paths: WorkPaths, bf16_path: Path) -> Path:
    cache_path = paths.imatrix_cache / f"{bf16_path.stem}.imatrix.gguf"
    work_path = paths.imatrix_dir / cache_path.name
    if cache_path.exists():
        _log(paths, "D", f"D. imatrix cached at {cache_path} (skip)")
        if not work_path.exists():
            paths.link_imatrix(cache_path)
        return cache_path
    companions = discover_companion_binaries(cfg.binary)
    imatrix_bin = companions.get("imatrix")
    if not imatrix_bin or not imatrix_bin.exists():
        raise SystemExit("ERROR: llama-imatrix not found alongside llama-quantize")
    _log(paths, "D", f"D. generating imatrix → {cache_path}")
    mmap_flag = [] if cfg.use_mmap else ["--no-mmap"]
    rc = _run(
        [str(imatrix_bin), "-m", str(bf16_path), "-f", str(cfg.calibration),
         "-o", str(cache_path),
         "-c", str(cfg.ctx), "-ngl", "99", *mmap_flag,
         "--chunks", str(cfg.chunks_imatrix),
         # Force f16 KV cache for imatrix calibration. Some forks (e.g.
         # frankenturbo2's experiment/buun-tcq-port branch) build with
         # turbo3_tcq as the default SWA cache type, which crashes on
         # gfx1102 with iSWA arches like gemma-3 ("ROCm error: unspecified
         # launch failure" inside ggml_backend_cuda_synchronize). f16 is
         # the right neutral choice for calibration regardless of inference
         # default — we want the reference activation distribution, not a
         # quant-perturbed one.
         "-ctk", "f16", "-ctv", "f16"],
        paths.logs_dir / "stage-D.log", env_extra=_hsa_env(cfg),
    )
    if rc != 0 or not cache_path.exists():
        raise SystemExit(f"FAIL: D llama-imatrix exit={rc}")
    paths.link_imatrix(cache_path)
    _log(paths, "D", f"D. imatrix: {cache_path.stat().st_size/1024**2:.1f} MB")
    return cache_path


# ─────────────────────────────────────────────────────────────────────────────
# Stage E — quantize-cost (per-tensor MSE per format)
# ─────────────────────────────────────────────────────────────────────────────

def stage_e_costs(cfg: PipelineConfig, paths: WorkPaths,
                  bf16_path: Path, imatrix_path: Path) -> Path:
    costs_path = paths.costs_dir / "costs.csv"
    if costs_path.exists():
        _log(paths, "E", f"E. costs.csv cached at {costs_path} (skip)")
        return costs_path
    companions = discover_companion_binaries(cfg.binary)
    cost_bin = cfg.binary.parent / "llama-quantize-cost"
    if not cost_bin.exists():
        raise SystemExit(
            f"ERROR: llama-quantize-cost not found at {cost_bin}.\n"
            f"  Build llama.cpp with -DGGML_BUILD_TOOLS=ON.")
    # Auto-detect exemplar layers from BF16 GGUF tensor shapes so iSWA arches
    # (gemma-3, gemma-4 with full_attention vs sliding_attention head_dim diff)
    # get one exemplar per layer-type signature. Falls back to safe defaults
    # for non-iSWA arches.
    exemplar_layers = _auto_pick_stage_e_exemplars(bf16_path)
    layer_alt = "|".join(str(L) for L in exemplar_layers)
    # Auto-detect ALL top-level weight tensors (not just token_embd/output).
    # This catches gemma-4's per_layer_token_embd.weight (5.38 GB BF16),
    # NemotronH's embeddings.weight, etc. Without this, the allocator's
    # recipe-size estimate is missing these tensors → wrong by GB on some
    # archs.
    top_level = _discover_top_level_weights(bf16_path)
    top_level_alt = "|".join(top_level)
    include_regex = rf"^({top_level_alt}|blk\.({layer_alt}))\."
    _log(paths, "E", f"E. measuring per-(tensor, format) MSE → {costs_path}")
    _log(paths, "E", f"E. top-level weights: {top_level}")
    _log(paths, "E", f"E. exemplar layers: {exemplar_layers}")
    _log(paths, "E", f"E. include-regex: {include_regex}")
    rc = _run(
        [str(cost_bin),
         "--model", str(bf16_path),
         "--types", ",".join(cfg.formats),
         "--imatrix", str(imatrix_path),
         "--include-regex", include_regex,
         "--output", str(costs_path)],
        paths.logs_dir / "stage-E.log",
    )
    if rc != 0 or not costs_path.exists():
        raise SystemExit(f"FAIL: E llama-quantize-cost exit={rc}")
    _log(paths, "E", f"E. costs.csv rows: {sum(1 for _ in costs_path.open())}")
    return costs_path


# ─────────────────────────────────────────────────────────────────────────────
# Stage F — bridge probe.pkl → bridge.json (HF → GGUF tensor names)
# ─────────────────────────────────────────────────────────────────────────────

def stage_f_bridge(cfg: PipelineConfig, paths: WorkPaths, probe_path: Path) -> Path:
    bridge_path = paths.work / "bridge.json"
    if bridge_path.exists():
        _log(paths, "F", f"F. bridge.json cached at {bridge_path} (skip)")
        return bridge_path
    bridge_script = cfg.pipeline_scripts_dir / "bridge_probe_to_gguf.py"
    if not bridge_script.exists():
        raise SystemExit(f"ERROR: bridge_probe_to_gguf.py not found at {bridge_script}")
    _log(paths, "F", f"F. bridging probe → {bridge_path}")
    rc = _run(
        ["python3", str(bridge_script),
         "--probe", str(probe_path),
         "--output", str(bridge_path),
         "--aggregate", "sum",
         "--unmapped-out", str(paths.work / "bridge-unmapped.json")],
        paths.logs_dir / "stage-F.log",
    )
    if rc != 0 or not bridge_path.exists():
        raise SystemExit(f"FAIL: F bridge exit={rc}")
    return bridge_path


# ─────────────────────────────────────────────────────────────────────────────
# Stage G — allocator (multi-choice knapsack at target budget + priority)
# ─────────────────────────────────────────────────────────────────────────────

def stage_g_allocate(cfg: PipelineConfig, paths: WorkPaths,
                     bridge_path: Path, costs_path: Path,
                     bf16_path: Path) -> Path:
    recipe_path = paths.recipes_dir / f"recipe-PQ{cfg.budget_gb}-{cfg.priority}.json"
    if recipe_path.exists():
        _log(paths, "G", f"G. recipe cached at {recipe_path} (skip)")
        return recipe_path
    allocator_script = cfg.pipeline_scripts_dir / "allocator.py"
    if not allocator_script.exists():
        raise SystemExit(f"ERROR: allocator.py not found at {allocator_script}")
    pinned_path = paths.work / "pinned.json"
    pinned_path.write_text(json.dumps(cfg.pinned, indent=2))
    _log(paths, "G", f"G. allocating @ {cfg.budget_gb} GB priority {cfg.priority}")
    cmd = [
        "python3", str(allocator_script),
        "--bridge", str(bridge_path),
        "--costs", str(costs_path),
        "--budget-gb", str(cfg.budget_gb),
        "--budget-band-gb", str(cfg.budget_band_gb),
        "--pinned", str(pinned_path),
        "--priority", str(cfg.priority),
        "--recipe-out", str(recipe_path),
        "--allow-types", ",".join(cfg.formats),
        "--gguf", str(bf16_path),
        "--propagate-from-exemplars",
        "--exemplar-layers", "0,3",
    ]
    if cfg.format_perf_file is not None and cfg.format_perf_file.exists():
        cmd += ["--tps", str(cfg.format_perf_file)]
        _log(paths, "G", f"G. multi-objective TPS data: {cfg.format_perf_file}")
    # Build per-tensor floor-bpw rules: attention default + user advanced rules.
    floor_rules = dict(cfg.floor_bpw or {})
    if cfg.attention_floor_bpw and cfg.attention_floor_bpw > 0:
        floor_rules.setdefault(
            r"^blk\..*\.attn_(q|k|v|qkv|gate|output)\.weight$",
            cfg.attention_floor_bpw,
        )
    if floor_rules:
        cmd += ["--floor-bpw", json.dumps(floor_rules)]
        _log(paths, "G", f"G. floor-bpw rules: {floor_rules}")
    rc = _run(cmd, paths.logs_dir / "stage-G.log")
    if rc != 0 or not recipe_path.exists():
        raise SystemExit(f"FAIL: G allocator exit={rc}")
    _log(paths, "G", f"G. recipe: {recipe_path}")
    return recipe_path


# ─────────────────────────────────────────────────────────────────────────────
# Stage H — apply recipe (llama-quantize)
# ─────────────────────────────────────────────────────────────────────────────

def stage_h_quantize(cfg: PipelineConfig, paths: WorkPaths,
                     bf16_path: Path, recipe_path: Path,
                     imatrix_path: Path) -> Path:
    safe_name = sanitize_model_name(cfg.hf_model)
    out_gguf = paths.gguf_output_path(safe_name, cfg.budget_gb, cfg.priority)
    if out_gguf.exists():
        _log(paths, "H", f"H. GGUF cached at {out_gguf} (skip)")
        return out_gguf
    # The allocator outputs JSON; llama-quantize wants a tensor-type-file.
    recipe_txt = recipe_path.with_suffix(".txt")
    if not recipe_txt.exists():
        recipe_data = json.loads(recipe_path.read_text())
        type_assignments = recipe_data.get("assignments") or recipe_data
        with recipe_txt.open("w") as f:
            for tensor, fmt in type_assignments.items():
                if isinstance(fmt, dict):
                    fmt = fmt.get("type") or fmt.get("format")
                f.write(f"{tensor}={fmt}\n")
    _log(paths, "H", f"H. applying recipe → {out_gguf}")
    rc = _run(
        [str(cfg.binary),
         "--imatrix", str(imatrix_path),
         "--tensor-type-file", str(recipe_txt),
         str(bf16_path), str(out_gguf), "IQ4_KS"],
        paths.logs_dir / "stage-H.log", env_extra=_hsa_env(cfg),
    )
    if rc != 0 or not out_gguf.exists():
        raise SystemExit(f"FAIL: H llama-quantize exit={rc}")
    size_gb = out_gguf.stat().st_size / 1024**3
    _log(paths, "H", f"H. final GGUF: {out_gguf} ({size_gb:.2f} GB)")
    return out_gguf


# ─────────────────────────────────────────────────────────────────────────────
# Stage I — PPL eval (optional)
# ─────────────────────────────────────────────────────────────────────────────

def stage_i_eval(cfg: PipelineConfig, paths: WorkPaths,
                 final_gguf: Path) -> Optional[float]:
    if cfg.skip_eval:
        _log(paths, "I", "I. eval skipped (--skip-eval)")
        return None
    companions = discover_companion_binaries(cfg.binary)
    perp_bin = companions.get("perplexity")
    if not perp_bin:
        _log(paths, "I", "I. WARN: llama-perplexity not found, skipping eval")
        return None
    eval_log = paths.logs_dir / "stage-I.log"
    _log(paths, "I", f"I. PPL eval @ chunks={cfg.chunks_eval}")
    mmap_flag_eval = [] if cfg.use_mmap else ["--no-mmap"]
    rc = _run(
        [str(perp_bin), "-m", str(final_gguf), "-f", str(cfg.calibration),
         "-c", str(cfg.ctx), "-b", "2048",
         "-ctk", "f16", "-ctv", "f16", "-fa", "on",
         "-ngl", "99", "--chunks", str(cfg.chunks_eval), *mmap_flag_eval],
        eval_log, env_extra=_hsa_env(cfg),
    )
    # Parse PPL from log
    import re
    text = eval_log.read_text()
    m = re.search(r"Final estimate:\s*PPL\s*=\s*([\d.]+)", text)
    if m:
        ppl = float(m.group(1))
        _log(paths, "I", f"I. PPL = {ppl:.4f}")
        return ppl
    _log(paths, "I", f"I. WARN: no Final estimate in eval log (rc={rc})")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(cfg: PipelineConfig) -> Path:
    """Run all 9 stages. Returns path to final GGUF."""
    paths = WorkPaths.for_run(
        root=cfg.output_root,
        model_name=sanitize_model_name(cfg.hf_model),
    )
    paths.make()

    print("=" * 70)
    print(f"prismaquant-llama pipeline — run {paths.run_label}")
    print("=" * 70)
    for line in paths.summary_lines():
        print(line)
    print()

    # A → B → C → D → E → F → G → H → I
    hf_dir       = stage_a_download(cfg, paths)
    bf16_path    = stage_b_convert(cfg, paths, hf_dir)
    # Auto-budget: derive after Stage B once BF16 size is known.
    if cfg.budget_gb is None:
        bf16_gb = bf16_path.stat().st_size / 1024**3
        cfg.budget_gb = round(bf16_gb * cfg.budget_auto_ratio, 2)
        _log(paths, "B",
             f"B. auto-budget: BF16={bf16_gb:.2f} GB × {cfg.budget_auto_ratio:.0%} "
             f"= {cfg.budget_gb} GB target (override with --budget-gb)")
    probe_path   = stage_c_probe(cfg, paths, hf_dir)
    imatrix_path = stage_d_imatrix(cfg, paths, bf16_path)
    costs_path   = stage_e_costs(cfg, paths, bf16_path, imatrix_path)
    bridge_path  = stage_f_bridge(cfg, paths, probe_path)
    recipe_path  = stage_g_allocate(cfg, paths, bridge_path, costs_path, bf16_path)
    final_gguf   = stage_h_quantize(cfg, paths, bf16_path, recipe_path, imatrix_path)
    ppl          = stage_i_eval(cfg, paths, final_gguf)

    # Optional cleanup of heavy intermediates (--clean-shared / --clean-imatrix / --clean-probe).
    if cfg.clean_shared or cfg.clean_imatrix or cfg.clean_probe:
        safe_name = sanitize_model_name(cfg.hf_model)
        freed = paths.cleanup_shared(
            safe_name,
            hf_cache=cfg.clean_shared,
            bf16=cfg.clean_shared,
            imatrix=cfg.clean_imatrix,
            probe=cfg.clean_probe,
        )
        if freed:
            total_gb = sum(freed.values()) / 1024**3
            parts = [f"{k}={v/1024**3:.2f}GB" for k, v in freed.items()]
            _log(paths, "Z", f"Z. cleanup: freed {total_gb:.2f} GB ({', '.join(parts)})")

    print()
    print("=" * 70)
    print(f"  ✓ DONE: {final_gguf}")
    print(f"    size: {final_gguf.stat().st_size/1024**3:.2f} GB")
    if ppl is not None:
        print(f"    PPL:  {ppl:.4f}")
    print(f"    logs: {paths.logs_dir}")
    print("=" * 70)
    return final_gguf


# ─────────────────────────────────────────────────────────────────────────────
# Dry-run: estimate disk footprint without touching anything
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_disk(cfg: PipelineConfig) -> dict:
    """Return {category: bytes} estimate. Queries HF for safetensors total."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise SystemExit("ERROR: huggingface_hub not installed.")
    api = HfApi()
    info = api.model_info(repo_id=cfg.hf_model, revision=cfg.hf_revision,
                          files_metadata=True)
    weight_exts = (".safetensors", ".bin", ".gguf", ".pt", ".pth", ".npz", ".h5")
    weight_bytes = sum(
        (s.size or 0) for s in (info.siblings or [])
        if s.rfilename.endswith(weight_exts) and (s.size or 0) > 0
    )
    tokenizer_other = sum(
        (s.size or 0) for s in (info.siblings or [])
        if not s.rfilename.endswith(weight_exts) and (s.size or 0) > 0
    )
    safetensors_total = weight_bytes + tokenizer_other

    # BF16 GGUF roughly equals weight_bytes when source is BF16/FP16 (most modern HF
    # repos). FP32 sources halve to BF16; quantized sources (4-bit, etc.) blow up
    # back to BF16. We can't know without inspecting config.json, so use weight_bytes
    # as the base and overcount slightly to be safe.
    bf16_gguf = max(int(weight_bytes * 1.05), weight_bytes)

    if cfg.budget_gb is not None:
        target = int(cfg.budget_gb * 1024**3)
    else:
        target = int(bf16_gguf * cfg.budget_auto_ratio)

    return {
        "safetensors": safetensors_total,
        "bf16_gguf":   bf16_gguf,
        "imatrix":     5 * 1024**2,    # ~5 MB rule of thumb
        "probe":       200 * 1024,     # ~200 KB rule of thumb
        "target_gguf": target,
    }


def _fmt_bytes(b: int) -> str:
    if b >= 1024**3:
        return f"{b/1024**3:.2f} GB"
    if b >= 1024**2:
        return f"{b/1024**2:.2f} MB"
    if b >= 1024:
        return f"{b/1024:.2f} KB"
    return f"{b} B"


def dry_run(cfg: PipelineConfig) -> int:
    """Print disk-usage estimate and exit. No download, no conversion."""
    import shutil as _shutil
    print("=" * 70)
    print("prismaquant-llama pipeline — DRY RUN (no download / no conversion)")
    print("=" * 70)
    print(f"  HF model: {cfg.hf_model}@{cfg.hf_revision}")
    print(f"  Output:   {cfg.output_root}")
    print(f"  Budget:   {'auto = '+str(cfg.budget_auto_ratio*100)+'% of BF16' if cfg.budget_gb is None else f'{cfg.budget_gb} GB'}")
    print(f"  Cleanup:  shared={cfg.clean_shared} imatrix={cfg.clean_imatrix} probe={cfg.clean_probe}")
    print()

    try:
        est = _estimate_disk(cfg)
    except Exception as e:
        print(f"ERROR: failed to query HF model size: {e}", file=sys.stderr)
        return 1

    print("Estimated disk usage per stage artifact:")
    print(f"  Safetensors download (Stage A):    {_fmt_bytes(est['safetensors'])}")
    print(f"  BF16 GGUF intermediate (Stage B):  {_fmt_bytes(est['bf16_gguf'])}")
    print(f"  Imatrix cache (Stage D):           {_fmt_bytes(est['imatrix'])}")
    print(f"  Probe artifacts (Stage C):         {_fmt_bytes(est['probe'])}")
    print(f"  Target PQ GGUF (Stage H):          {_fmt_bytes(est['target_gguf'])}")
    print()

    peak = sum(est.values())
    final_default = peak  # nothing cleaned
    final_clean_shared = est["imatrix"] + est["probe"] + est["target_gguf"]
    final_clean_all = est["target_gguf"]

    print(f"Peak disk usage during run:           {_fmt_bytes(peak)}")
    print(f"Final retained (default, no clean):   {_fmt_bytes(final_default)}")
    print(f"Final retained with --clean-shared:   {_fmt_bytes(final_clean_shared)}")
    print(f"Final retained with --clean-{{shared,imatrix,probe}}: {_fmt_bytes(final_clean_all)}")
    print()

    out = cfg.output_root.expanduser().resolve()
    try:
        out.mkdir(parents=True, exist_ok=True)
        free = _shutil.disk_usage(out).free
        margin = free - peak
        print(f"Output filesystem free space:         {_fmt_bytes(free)}")
        if margin < 0:
            print(f"  ⚠ INSUFFICIENT — short by {_fmt_bytes(-margin)} at peak.")
            print(f"     With --clean-shared in the same run, peak is unchanged "
                  f"(cleanup runs AFTER Stage H), so increase free space first.")
        elif margin < 5 * 1024**3:
            print(f"  ⚠ TIGHT — only {_fmt_bytes(margin)} margin at peak.")
        else:
            print(f"  ✓ ample margin ({_fmt_bytes(margin)})")
    except Exception as e:
        print(f"  (could not stat output filesystem: {e})")

    print()
    print("=" * 70)
    print("Re-run without --dry-run to execute.")
    print("=" * 70)
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# CLI: `prismaquant-llama pipeline run|status`
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="prismaquant-llama pipeline",
        description="Direct pipeline exec without the wizard TUI")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="run the full A→I pipeline")

    # User-config-aware defaults: ~/.prismaquant-llama/config/config.toml overrides
    # the hardcoded fallbacks below. CLI args still override the config file.
    from .paths import (get_user_config_value, get_user_default_path,
                        get_packaged_calibration_corpus)
    _user_output_root = get_user_default_path("output_root", DEFAULT_OUTPUT_ROOT)
    _user_budget_auto_ratio = get_user_config_value("defaults", "budget_auto_ratio", 0.25)
    _user_budget_band_gb = get_user_config_value("defaults", "budget_band_gb", 0.25)
    _user_priority = get_user_config_value("defaults", "priority", "333")
    _user_chunks_imatrix = get_user_config_value("defaults", "chunks_imatrix", 200)
    _user_chunks_eval = get_user_config_value("defaults", "chunks_eval", 100)
    _user_ctx = get_user_config_value("defaults", "ctx", 4096)
    _user_no_mmap_default = get_user_config_value("defaults", "no_mmap", False)
    _user_calibration = get_user_config_value("defaults", "calibration_corpus", None)
    # Calibration fall-through: user CLI > user config > package-shipped default
    _calibration_default = (
        Path(_user_calibration).expanduser() if _user_calibration
        else get_packaged_calibration_corpus()
    )

    pr.add_argument("--hf-model", required=True,
                    help="HuggingFace model ID (e.g., google/gemma-4-E4B-it)")
    pr.add_argument("--hf-revision", default="main")
    pr.add_argument("--binary", type=Path, default=None,
                    help="path to llama-quantize (default: auto-discover, then "
                         "[binaries.<default_set>] from user config.toml)")
    pr.add_argument("--calibration", type=Path,
                    default=_calibration_default,
                    required=(_calibration_default is None),
                    help="calibration corpus text file. Resolution order: this "
                         "flag > [defaults] calibration_corpus in config.toml > "
                         "package-shipped default (~280 KB compiled from public "
                         "text). For best results on production runs, supply a "
                         "corpus closer to your target deployment domain.")
    pr.add_argument("--output", "-o", type=Path, default=_user_output_root)
    pr.add_argument("--budget-gb", type=float, default=None,
                    help="target GGUF size in GB. Default: auto = budget_auto_ratio "
                         "of BF16 GGUF size (computed after Stage B).")
    pr.add_argument("--budget-auto-ratio", type=float, default=_user_budget_auto_ratio,
                    help="when --budget-gb is unset, target this fraction of the "
                         "BF16 GGUF size (default from user config or 0.25)")
    pr.add_argument("--budget-band-gb", type=float, default=_user_budget_band_gb,
                    help="allocator's wiggle room around --budget-gb (default: ±0.25 GB)")
    pr.add_argument("--priority", default=_user_priority,
                    help="3-digit XYZ — X=PPL, Y=TG, Z=PP (default from user config or 333)")
    pr.add_argument("--formats",
                    help="comma-separated format whitelist (default: 11-format prismaquant default; "
                         "user override via ~/.prismaquant-llama/config/default-formats.txt)")
    pr.add_argument("--chunks-imatrix", type=int, default=_user_chunks_imatrix)
    pr.add_argument("--chunks-eval", type=int, default=_user_chunks_eval)
    pr.add_argument("--ctx", type=int, default=_user_ctx)
    pr.add_argument("--skip-eval", action="store_true",
                    help="skip stage I (PPL eval)")
    pr.add_argument("--convert-script", type=Path,
                    help="path to convert_hf_to_gguf.py (default: auto-discover)")
    pr.add_argument("--format-perf", "--tps-file", type=Path, default=None,
                    dest="format_perf",
                    help="per-format performance characteristics JSON (currently "
                         "pp/tg throughput; reserved keys for latency, mem footprint, "
                         "etc.). Consumed by the allocator's multi-objective scoring. "
                         "Default: auto-discover via find_format_perf_file_for_binary "
                         "(binary-sha cache → --default-format-perf → package examples). "
                         "Without this, TG/PP weights in --priority XYZ have no data "
                         "and the allocator collapses to pure-PPL. "
                         "Legacy alias: --tps-file (deprecated).")
    pr.add_argument("--default-format-perf", "--default-tps", type=Path, default=None,
                    dest="default_format_perf",
                    help="user's preferred default format-perf file (overrides the "
                         "package-shipped examples/format-perf-<arch>.json but "
                         "is overridden by the per-binary cache and explicit "
                         "--format-perf). Can also be set via PRISMAQUANT_DEFAULT_FORMAT_PERF "
                         "env var. Legacy alias: --default-tps (deprecated).")
    pr.add_argument("--clean-shared", action="store_true",
                    help="after Stage H success, delete _shared/hf-cache/<model>/ and "
                         "_shared/bf16/<model>-BF16.gguf. The target GGUF in ggufs/ "
                         "is preserved. Re-running this model later will re-download "
                         "+ re-convert.")
    pr.add_argument("--clean-imatrix", action="store_true",
                    help="also delete _shared/imatrix-cache/<model>-BF16.imatrix.gguf "
                         "after Stage H (small ~MB file; default keep).")
    pr.add_argument("--clean-probe", action="store_true",
                    help="also delete _shared/probe/<model>-* after Stage H "
                         "(probe.pkl + per-model dirs; default keep).")
    pr.add_argument("--mmap", action="store_true", dest="use_mmap",
                    help="Drop --no-mmap from Stage D imatrix and Stage I PPL eval. "
                         "Use when BF16 model size > host RAM (e.g. 122B-class "
                         "models on 31 GB hosts) — without this, llama-imatrix "
                         "hits OOM during model load. Trade-off: slower I/O-bound "
                         "stages (cephfs/disk paging) but bounded host RAM.")
    pr.add_argument("--attention-floor-bpw", type=float, default=4.0,
                    help="Minimum bits-per-weight for attention tensors "
                         "(q/k/v/qkv/gate/output projections). Default 4.0 "
                         "forbids 2/3-bit quants on attention because attention "
                         "errors compound cross-token via QK^T affinities and "
                         "have outsized PPL impact. Mirrors established K-quant "
                         "practice (Q4_K_M keeps attn_v at Q6_K, etc.). Set to "
                         "0 to disable and let the allocator pick freely.")
    pr.add_argument("--floor-bpw", default=None,
                    help="Advanced: JSON dict of additional {regex_pattern: "
                         "min_bpw_float} rules, merged with the attention "
                         "default. Multiple rules may match a tensor; highest "
                         "min_bpw wins. Example: '{\"^blk\\\\.\\\\d+\\\\.ffn_down_exps\\\\.weight$\": 3.0}' "
                         "to enforce min 3 bpw on FFN down-experts.")
    pr.add_argument("--clean-all", action="store_true",
                    help="shorthand for --clean-shared --clean-imatrix --clean-probe. "
                         "After Stage H success, removes every per-model intermediate; "
                         "only the target GGUF in ggufs/ + per-run logs in work/ remain.")
    pr.add_argument("--dry-run", action="store_true",
                    help="query HF for safetensors size, estimate intermediate + final "
                         "disk usage, compare against free space on the output filesystem, "
                         "and exit. No download or conversion.")

    args = p.parse_args(argv)

    if args.cmd == "run":
        # Resolve binary: CLI → user-config binary set → auto-discover (handled later in pipeline)
        resolved_binary = args.binary
        if resolved_binary is None:
            from .paths import get_user_default_binary_set
            binset = get_user_default_binary_set()
            if binset and binset.get("quantize"):
                resolved_binary = Path(str(binset["quantize"])).expanduser()
                print(f"[pipeline] using binary from user config: {resolved_binary}")

        # Resolve use_mmap: CLI --mmap (store_true) wins; otherwise consult user config's
        # `[defaults] no_mmap`. If no_mmap=true → use_mmap=False; else dataclass default.
        resolved_use_mmap = args.use_mmap
        if not resolved_use_mmap and _user_no_mmap_default:
            resolved_use_mmap = False  # explicit user preference for --no-mmap behavior

        cfg_kwargs = dict(
            hf_model=args.hf_model, hf_revision=args.hf_revision,
            binary=resolved_binary, calibration=args.calibration,
            output_root=args.output,
            budget_gb=args.budget_gb, budget_band_gb=args.budget_band_gb,
            budget_auto_ratio=args.budget_auto_ratio,
            priority=args.priority,
            chunks_imatrix=args.chunks_imatrix,
            chunks_eval=args.chunks_eval,
            ctx=args.ctx, skip_eval=args.skip_eval,
            convert_script=args.convert_script,
            format_perf_file=args.format_perf,
            default_format_perf_override=args.default_format_perf,
            clean_shared=args.clean_shared or args.clean_all,
            clean_imatrix=args.clean_imatrix or args.clean_all,
            clean_probe=args.clean_probe or args.clean_all,
            use_mmap=resolved_use_mmap,
            attention_floor_bpw=args.attention_floor_bpw,
            floor_bpw=json.loads(args.floor_bpw) if args.floor_bpw else None,
        )
        if args.formats:
            cfg_kwargs["formats"] = [f.strip() for f in args.formats.split(",")]
        else:
            # No CLI override → check for user's default-formats.txt
            from .paths import load_user_default_formats, DEFAULT_USER_FORMATS_PATH
            user_formats = load_user_default_formats()
            if user_formats:
                print(f"[pipeline] using formats from {DEFAULT_USER_FORMATS_PATH}: {','.join(user_formats)}")
                cfg_kwargs["formats"] = user_formats
        cfg = PipelineConfig(**cfg_kwargs)
        if args.dry_run:
            return dry_run(cfg)
        try:
            run_full_pipeline(cfg)
            return 0
        except SystemExit as e:
            print(f"\nFAIL: {e}", file=sys.stderr)
            return 1

    return 1


if __name__ == "__main__":
    sys.exit(main())
