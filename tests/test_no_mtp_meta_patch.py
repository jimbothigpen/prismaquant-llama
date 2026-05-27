"""Tests for _patch_no_mtp_metadata: post-Stage-B GGUF KV metadata fix.

Covers the three cases the helper must handle:
  1. Stale metadata (nextn_predict_layers > 0) — patched; block_count decremented.
  2. Already-fixed metadata (nextn_predict_layers == 0) — no-op, returns None.
  3. Field absent entirely (non-Qwen3.5 arch) — no-op, returns None, no raise.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from gguf import GGUFReader, GGUFWriter

import prismaquant_llama.pipeline_runner as pr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gguf(path: Path, arch: str, block_count: int,
               nextn_predict_layers: int | None) -> Path:
    """Write a minimal GGUF with the given KV fields to *path*."""
    w = GGUFWriter(str(path), arch)
    w.add_uint32(f"{arch}.block_count", block_count)
    if nextn_predict_layers is not None:
        w.add_uint32(f"{arch}.nextn_predict_layers", nextn_predict_layers)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


def _read_uint32(path: Path, key: str) -> int:
    r = GGUFReader(str(path))
    field = r.fields[key]
    return int(field.parts[field.data[0]][0])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_patch_no_mtp_metadata_qwen35_patches(tmp_path):
    """Stale metadata (nextn=1, block_count=33) is corrected in place."""
    p = _make_gguf(tmp_path / "model.gguf", "qwen35", block_count=33,
                   nextn_predict_layers=1)

    result = pr._patch_no_mtp_metadata(p)

    assert result == (33, 1), f"Expected (33, 1), got {result}"
    assert _read_uint32(p, "qwen35.block_count") == 32
    assert _read_uint32(p, "qwen35.nextn_predict_layers") == 0


def test_patch_no_mtp_metadata_noop_when_already_fixed(tmp_path):
    """nextn_predict_layers already 0 — returns None and leaves metadata unchanged."""
    p = _make_gguf(tmp_path / "model.gguf", "qwen35", block_count=32,
                   nextn_predict_layers=0)

    result = pr._patch_no_mtp_metadata(p)

    assert result is None
    assert _read_uint32(p, "qwen35.block_count") == 32
    assert _read_uint32(p, "qwen35.nextn_predict_layers") == 0


def test_patch_no_mtp_metadata_noop_when_field_absent(tmp_path):
    """GGUF without nextn_predict_layers (e.g. llama arch) — returns None, no raise."""
    p = _make_gguf(tmp_path / "model.gguf", "llama", block_count=32,
                   nextn_predict_layers=None)

    result = pr._patch_no_mtp_metadata(p)

    assert result is None
    assert _read_uint32(p, "llama.block_count") == 32
