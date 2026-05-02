# Pipeline scripts — parameterization TODO

These scripts were copied as-is from frankenturbo2 (sister repo at the
time of extraction, 2026-05-02). They have hard-coded paths that assume
a specific filesystem layout. **Before publishing the wizard for
external use**, replace the constants below with env-var overrides.

## Hard-coded paths to replace

Run `grep -nE '/usr/src|/mnt/cephfs|frankenturbo|gfx1150|/home/builduser'`
across `run-pipeline.sh` + `comparison-sweep/*.sh` + `scripts/*.sh` to
find every occurrence.

### Variables to introduce

| Variable | Default | Purpose |
|---|---|---|
| `PRISMAQUANT_BIN_DIR` | auto-discover (PATH → `<cwd>/build/bin`) | dir holding llama-quantize, llama-perplexity, llama-bench, llama-imatrix |
| `PRISMAQUANT_MODELS_DIR` | `~/prismaquant-builds/_shared/hf-cache` | base dir for cached HF safetensors + BF16 GGUFs |
| `PRISMAQUANT_CALIBRATION` | `~/prismaquant-builds/_shared/calibration/wikitext.txt` | calibration corpus path |
| `PRISMAQUANT_WORK_ROOT` | `~/prismaquant-builds/work/<run-id>` | per-run scratch directory |
| `PRISMAQUANT_OUTPUT_DIR` | `~/prismaquant-builds/ggufs` | final GGUF output dir |
| `HOST` | `$(hostname)` | identifies the running machine for HSA_OVERRIDE / arch-specific behavior |

### Substitution pattern

Replace lines like:
```bash
REPO=/path/to/llama-fork
QWOPUS=/mnt/cephfs/0/Container/models/Jackrong/Qwopus3.5-9B-v3.5-GGUF
WIKI=/mnt/cephfs/0/Container/models/wikitext103-calibration.txt
```

With:
```bash
REPO=${REPO:-$(cd "$(dirname "$0")"/../.. && pwd)}
QWOPUS=${QWOPUS:-${PRISMAQUANT_MODELS_DIR}/Jackrong/Qwopus3.5-9B-v3.5-GGUF}
WIKI=${WIKI:-${PRISMAQUANT_CALIBRATION}}
```

### Expected effort

~2-3 hours for a thorough pass across all scripts in this directory.

### Already-known fork-specific bits to remove or generalize

- `if [[ "$HOST" == "ai00" ]]; then ... fi` blocks (gfx1150-specific
  HSA_OVERRIDE handling) — generalize to "if running native arch, no
  override; else use HSA_OVERRIDE_GFX_VERSION".
- `gfx1150` mentions in comments — replace with "native arch".
- `frankenturbo` references in comments — replace with "this fork" or
  remove.
- `ai00` / `ai01` as machine names — replace with `$(hostname)` or
  remove from comments.

### Why we copied as-is

The user directive at extraction time was "move wholesale, ensure no
disruption to ongoing pipelines, parameterize later." The scripts work
correctly for the original frankenturbo2 use case; they just fail
silently on other systems because of hard-coded paths.
