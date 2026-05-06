"""
Classify and validate model inputs.

Four input forms accepted:
    1. HuggingFace safetensors id   "unsloth/Qwen3.6-35B-A3B"
    2. on-disk safetensors directory  "/full/path/to/safetensors/"
    3. on-disk f16/bf16 GGUF file     "/full/path/to/model-BF16.gguf"
    4. URL(s) to f16/bf16 GGUF        "https://...gguf"
                                       (split: "https://...-00001-of-00002.gguf,
                                                https://...-00002-of-00002.gguf")

`run` accepts only forms 1+2 (Bayesian probe needs safetensors).
`calibrate` accepts all four (no probe needed; just quantize → eval).
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Source-format suffixes stripped from sanitized model names.
_SUFFIX_RE = re.compile(r"-(?:BF16|FP16|F32|F16|FP32)$", re.IGNORECASE)
# GGUF split suffix, e.g. "-00001-of-00002".
_SPLIT_RE = re.compile(r"-\d{5}-of-\d{5}$", re.IGNORECASE)


def sanitize_model_name(raw: str) -> str:
    """Convert any model identifier into a clean filesystem-safe component.

    Examples:
        unsloth/Qwen3.6-35B-A3B            → Qwen3.6-35B-A3B
        google/gemma-4-E4B-it              → gemma-4-E4B-it
        /path/to/Qwen3.6-35B-A3B-BF16.gguf → Qwen3.6-35B-A3B
        Qwen3-30B-BF16-00001-of-00003      → Qwen3-30B
    """
    s = raw.rsplit("/", 1)[-1]
    if s.endswith(".gguf"):
        s = s[:-len(".gguf")]
    s = _SPLIT_RE.sub("", s)
    s = _SUFFIX_RE.sub("", s)
    return s.replace(" ", "_").strip("/")


@dataclass
class ResolvedInput:
    kind: str               # "hf" | "safetensors_dir" | "gguf_local" | "gguf_url"
    spec: str               # original input string
    model_name: str         # sanitized, suitable for filenames

    hf_id: Optional[str] = None
    safetensors_dir: Optional[Path] = None
    gguf_path: Optional[Path] = None              # gguf_local
    gguf_urls: Optional[list[str]] = None         # gguf_url, may be split list


def classify(spec: str) -> str:
    """Pure classification of an input string. No filesystem check beyond
    the dir/file distinction. Raises ValueError for unrecognized inputs."""
    if spec.startswith(("http://", "https://")):
        return "gguf_url"
    p = Path(spec).expanduser()
    if p.suffix.lower() == ".gguf":
        return "gguf_local"
    if p.exists() and p.is_dir():
        return "safetensors_dir"
    # HuggingFace IDs always contain a "/" and have no leading "./" or "/".
    if "/" in spec and not spec.startswith((".", "/")) and not p.exists():
        return "hf"
    raise ValueError(
        f"could not classify input {spec!r}. Expected one of:\n"
        f"  - HuggingFace id (e.g. 'unsloth/Qwen3.6-35B-A3B')\n"
        f"  - on-disk safetensors directory (must exist)\n"
        f"  - on-disk f16/bf16 GGUF file (.gguf extension)\n"
        f"  - URL to f16/bf16 GGUF (https://...) — split URLs comma-separated"
    )


def resolve(spec: str, *, allow_gguf: bool = True) -> ResolvedInput:
    """Validate and normalize an input.

    Args:
        spec: input string from the CLI
        allow_gguf: False ⇒ reject gguf_local and gguf_url forms (used by `run`,
            which requires safetensors for the probe stage).

    Raises:
        ValueError on unrecognized input or rejected GGUF when allow_gguf=False
        FileNotFoundError on missing on-disk paths
    """
    # Comma-separated URLs are split GGUFs — classify the first.
    if "," in spec and all(p.startswith(("http://", "https://"))
                            for p in (s.strip() for s in spec.split(","))
                            if p):
        kind = "gguf_url"
    else:
        kind = classify(spec)

    if not allow_gguf and kind in ("gguf_local", "gguf_url"):
        raise ValueError(
            f"`run` requires safetensors input — got {kind}. The full pipeline "
            f"needs the safetensors directory or HF id to run the Bayesian "
            f"probe stage. Use `calibrate` if you only have a GGUF.")

    if kind == "hf":
        return ResolvedInput(kind=kind, spec=spec,
                             model_name=sanitize_model_name(spec),
                             hf_id=spec)

    if kind == "safetensors_dir":
        p = Path(spec).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"safetensors directory not found: {p}")
        if not list(p.glob("*.safetensors")):
            raise ValueError(
                f"no .safetensors files found in {p}. If this is a "
                f"different model format, point at the safetensors copy.")
        return ResolvedInput(kind=kind, spec=spec,
                             model_name=sanitize_model_name(p.name),
                             safetensors_dir=p)

    if kind == "gguf_local":
        p = Path(spec).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"GGUF not found: {p}")
        return ResolvedInput(kind=kind, spec=spec,
                             model_name=sanitize_model_name(p.stem),
                             gguf_path=p)

    if kind == "gguf_url":
        urls = [u.strip() for u in spec.split(",") if u.strip()]
        if not urls:
            raise ValueError(f"no URLs found in {spec!r}")
        first_filename = urls[0].rsplit("/", 1)[-1] or "model.gguf"
        if not first_filename.endswith(".gguf"):
            raise ValueError(
                f"URL must point at a .gguf file; got {first_filename}")
        return ResolvedInput(kind=kind, spec=spec,
                             model_name=sanitize_model_name(first_filename[:-len(".gguf")]),
                             gguf_urls=urls)

    raise AssertionError(f"unreachable: kind={kind}")
