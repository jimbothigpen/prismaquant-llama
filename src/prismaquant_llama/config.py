"""
Config loader for prismaquant-llama.

The single config file lives at ~/.prismaquant-llama/config.toml. On first
invocation, the shipped default (data/config.toml.default) is auto-copied
there and a one-line notice is printed. After that, the user edits the file
by hand. The `--config PATH` flag overrides the location for one run.

Schema is a single flat section:
    [prismaquant-llama]
    base, path, quants, budget, priority,
    ppl_corpus, imatrix_corpus, ppl_chunks, imatrix_chunks
"""

from __future__ import annotations
import os
import shutil
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Standard install location. Independent of `base`, since `base` lives
# inside this file (chicken-and-egg).
DEFAULT_CONFIG_PATH = Path.home() / ".prismaquant-llama" / "config.toml"

# Package-data resources.
_DATA_DIR = Path(__file__).parent / "data"
SHIPPED_CONFIG = _DATA_DIR / "config.toml.default"
BUNDLED_PPL_CORPUS = _DATA_DIR / "wikitext-2-raw-test.txt"
BUNDLED_IMATRIX_CORPUS = _DATA_DIR / "bartowski-imatrix-v5-semantic.txt"

# llama.cpp tool names.
LLAMA_TOOLS = ("llama-quantize", "llama-quantize-cost",
               "llama-perplexity", "llama-bench", "llama-imatrix")


@dataclass
class Config:
    base: Path
    path: Optional[Path]            # None ⇒ resolve via $PATH
    quants: list[str]
    budget: int                     # percentage of BF16 size
    priority: str                   # 3-digit ratio
    ppl_corpus: str                 # "" / on-disk path / URL
    imatrix_corpus: str             # "" / on-disk path / URL
    ppl_chunks: int
    imatrix_chunks: int
    convert_script: Optional[Path]  # path to convert_hf_to_gguf.py; None ⇒ search
    libs: Optional[Path]            # extra dir prepended to LD_LIBRARY_PATH; None ⇒ no override
    mtp_format: str = "BF16"        # format to pin MTP/NEXTN tensors to.
                                    # Applied by the allocator when the model has
                                    # MTP layers (num_nextn_predict_layers > 0).
                                    # BF16 is the production default until
                                    # speculative-decode acceptance is validated
                                    # for quantized MTP weights.
    reference_format: str = "bf16"  # "bf16" (default) or "f16"; controls Stage B
                                    # outtype + the calibration's reference for
                                    # Δppl / pp/tg ratios. Use "f16" on backends
                                    # whose hipblas lacks BF16 GEMM kernels (e.g.
                                    # gfx1102/1103 ROCm). Output JSON field names
                                    # (ppl_delta_vs_f16, pp_ratio_vs_bf16, etc.)
                                    # retain their legacy names for backward
                                    # compatibility regardless of this setting.

    config_path: Path = field(default_factory=lambda: DEFAULT_CONFIG_PATH)


def install_default_config(target: Path = DEFAULT_CONFIG_PATH) -> bool:
    """Copy the shipped default config to `target` if it doesn't exist.
    Returns True if a new file was written."""
    if target.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SHIPPED_CONFIG, target)
    return True


def load_config(config_path: Optional[Path] = None,
                libs: Optional[Path] = None) -> Config:
    """Load and validate a config.toml.

    If no path given, uses ~/.prismaquant-llama/config.toml and auto-installs
    the shipped default there on first run.

    Args:
        config_path: explicit --config override. Must exist if given.
        libs: --libs flag value, prepended to LD_LIBRARY_PATH for subprocesses.

    Returns: a fully-resolved Config.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
        if install_default_config(config_path):
            print(f"[prismaquant-llama] wrote starter config → {config_path}",
                  file=sys.stderr)
            print(f"[prismaquant-llama] edit it to set defaults; CLI flags "
                  f"override per run.", file=sys.stderr)
    else:
        config_path = Path(config_path).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"--config: file not found: {config_path}")

    with config_path.open("rb") as f:
        data = tomllib.load(f)

    section = data.get("prismaquant-llama", {})
    if not section:
        raise ValueError(
            f"config at {config_path} is missing the [prismaquant-llama] "
            f"section. Compare with the shipped default at {SHIPPED_CONFIG}.")

    base = _expand_path(section.get("base") or "~/.prismaquant-llama/")

    raw_path = section.get("path") or ""
    path = _expand_path(raw_path) if raw_path else None

    quants = section.get("quants") or []
    if not isinstance(quants, list) or not all(isinstance(q, str) for q in quants):
        raise ValueError(f"config 'quants' must be a list of strings; got {quants!r}")
    quants = [q.strip().upper() for q in quants if q.strip()]
    if not quants:
        raise ValueError(
            "config 'quants' is empty. Set it to a non-empty list, e.g. "
            "['Q4_K', 'Q5_K', 'Q6_K', 'Q8_0', 'IQ4_XS'].")

    budget = int(section.get("budget", 25))
    if not (1 <= budget <= 100):
        raise ValueError(f"config 'budget' must be in [1, 100]; got {budget}")

    priority = str(section.get("priority", "111"))
    _validate_priority(priority)

    ppl_corpus = str(section.get("ppl_corpus") or "")
    imatrix_corpus = str(section.get("imatrix_corpus") or "")

    ppl_chunks = int(section.get("ppl_chunks", 50))
    imatrix_chunks = int(section.get("imatrix_chunks", 50))
    if ppl_chunks < 1:
        raise ValueError(f"config 'ppl_chunks' must be ≥ 1; got {ppl_chunks}")
    if imatrix_chunks < 1:
        raise ValueError(f"config 'imatrix_chunks' must be ≥ 1; got {imatrix_chunks}")

    raw_convert = section.get("convert_script") or ""
    convert_script = _expand_path(raw_convert) if raw_convert else None

    reference_format = str(section.get("reference_format") or "bf16").strip().lower()
    if reference_format not in ("bf16", "f16"):
        raise ValueError(
            f"config 'reference_format' must be 'bf16' or 'f16'; got "
            f"{reference_format!r}")

    mtp_format = str(section.get("mtp_format") or "BF16").strip().upper()

    # libs: CLI --libs > [prismaquant-llama] libs > None.
    if libs is not None:
        libs_resolved = Path(libs).expanduser().resolve()
        if not libs_resolved.is_dir():
            raise FileNotFoundError(f"--libs: directory not found: {libs_resolved}")
    else:
        raw_libs = section.get("libs") or ""
        if raw_libs:
            libs_resolved = Path(str(raw_libs)).expanduser().resolve()
            if not libs_resolved.is_dir():
                raise FileNotFoundError(
                    f"config 'libs' directory not found: {libs_resolved}")
        else:
            libs_resolved = None

    return Config(
        base=base, path=path, quants=quants, budget=budget,
        priority=priority, ppl_corpus=ppl_corpus,
        imatrix_corpus=imatrix_corpus,
        ppl_chunks=ppl_chunks, imatrix_chunks=imatrix_chunks,
        convert_script=convert_script,
        libs=libs_resolved,
        reference_format=reference_format,
        mtp_format=mtp_format,
        config_path=config_path,
    )


def _expand_path(s: str) -> Path:
    return Path(s).expanduser().resolve() if s else Path()


def _validate_priority(p: str) -> None:
    if not (isinstance(p, str) and len(p) == 3 and p.isdigit()):
        raise ValueError(
            f"config 'priority' must be a 3-digit string like '111' or '522'; "
            f"got {p!r}")


def find_tool(cfg: Config, tool: str) -> Path:
    """Locate a llama.cpp tool. If cfg.path is set, look there; otherwise $PATH."""
    if cfg.path is not None:
        candidate = cfg.path / tool
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"{tool} not found at {candidate} (config 'path' = {cfg.path}). "
            f"Either fix 'path' in {cfg.config_path}, or unset it and rely "
            f"on $PATH discovery.")
    p = shutil.which(tool)
    if p:
        return Path(p).resolve()
    raise FileNotFoundError(
        f"{tool} not found on $PATH. Set 'path' in {cfg.config_path} to your "
        f"llama.cpp binary directory, or pass --path on the CLI.")


def subprocess_env(cfg: Config) -> dict:
    """Build the env dict for subprocess calls. Adds --libs to LD_LIBRARY_PATH
    if set."""
    env = dict(os.environ)
    if cfg.libs is not None:
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            f"{cfg.libs}{os.pathsep}{existing}" if existing else str(cfg.libs))
    return env


def resolve_corpus(cfg: Config, kind: str,
                   override: Optional[str] = None) -> tuple[Path, bool]:
    """Resolve a corpus to a local file path.

    Args:
        cfg: loaded config
        kind: "ppl" or "imatrix"
        override: --ppl-corpus / --imatrix-corpus flag value (if any)

    Returns: (local_path, was_downloaded)
        was_downloaded=True only if we fetched a URL this call. Used for purge.

    Resolution order:
        1. CLI override (if given)
        2. config value (cfg.ppl_corpus / cfg.imatrix_corpus)
        3. bundled default (always succeeds)
    """
    spec = override if override is not None else (
        cfg.ppl_corpus if kind == "ppl" else cfg.imatrix_corpus)

    if not spec:
        bundled = BUNDLED_PPL_CORPUS if kind == "ppl" else BUNDLED_IMATRIX_CORPUS
        if not bundled.exists():
            raise FileNotFoundError(
                f"bundled {kind} corpus missing at {bundled} — package may "
                f"be installed without data files.")
        return bundled, False

    if spec.startswith(("http://", "https://")):
        target_dir = cfg.base / (f"{kind}-corpus")
        target_dir.mkdir(parents=True, exist_ok=True)
        filename = spec.rsplit("/", 1)[-1] or f"{kind}-corpus.txt"
        local = target_dir / filename
        if local.exists():
            return local, False
        _download(spec, local)
        return local, True

    local = Path(spec).expanduser().resolve()
    if not local.exists():
        raise FileNotFoundError(f"{kind} corpus not found: {local}")
    return local, False


def _download(url: str, dst: Path) -> None:
    import urllib.request
    print(f"[prismaquant-llama] downloading {url} → {dst}", file=sys.stderr)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp, dst.open("wb") as f:
        shutil.copyfileobj(resp, f)
