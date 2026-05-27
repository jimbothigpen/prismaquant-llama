"""Tests for Stage B --no-mtp injection for Qwen3.5/3.6 architectures.

Covers:
  - _stage_b_extra_args: pure helper that reads config.json and returns --no-mtp
    for Qwen3.5/3.6 and nothing for plain LlamaForCausalLM / missing config.
  - convert_to_bf16: integration path via monkeypatched _run confirming --no-mtp
    appears in (or is absent from) the assembled command.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import prismaquant_llama.pipeline_runner as pr
from prismaquant_llama.budget import parse_budget
from prismaquant_llama.config import Config
from prismaquant_llama.paths import Layout


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_config(directory: Path, architecture: str) -> None:
    (directory / "config.json").write_text(
        json.dumps({"architectures": [architecture], "num_hidden_layers": 28})
    )


def _make_cfg(**overrides) -> Config:
    defaults = dict(
        base=Path("/tmp/pq-test-unused"),
        path=None,
        quants=["Q4_K"],
        budget_spec=parse_budget("25"),
        priority="111",
        ppl_corpus="",
        imatrix_corpus="",
        ppl_chunks=5,
        imatrix_chunks=5,
        convert_script=None,
        libs=None,
    )
    defaults.update(overrides)
    return Config(**defaults)


def _make_layout(tmp_path: Path) -> Layout:
    layout = Layout.for_run(tmp_path, "test-model", run_timestamp="20260101-000000")
    layout.make()
    return layout


# ---------------------------------------------------------------------------
# _stage_b_extra_args — pure helper tests (no I/O beyond temp files)
# ---------------------------------------------------------------------------

def test_extra_args_qwen35(tmp_path):
    _write_config(tmp_path, "Qwen3_5ForConditionalGeneration")
    assert pr._stage_b_extra_args(tmp_path) == ["--no-mtp"]


def test_extra_args_qwen36(tmp_path):
    _write_config(tmp_path, "Qwen3_6ForCausalLM")
    assert pr._stage_b_extra_args(tmp_path) == ["--no-mtp"]


def test_extra_args_llama_no_flag(tmp_path):
    _write_config(tmp_path, "LlamaForCausalLM")
    assert pr._stage_b_extra_args(tmp_path) == []


def test_extra_args_missing_config(tmp_path):
    # No config.json present — should return empty list, not raise.
    assert pr._stage_b_extra_args(tmp_path) == []


def test_extra_args_empty_architectures(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"architectures": []}))
    assert pr._stage_b_extra_args(tmp_path) == []


# ---------------------------------------------------------------------------
# convert_to_bf16 — integration tests via monkeypatched _run
# ---------------------------------------------------------------------------

def _run_stage_b(monkeypatch, tmp_path: Path, architecture: str) -> list[str]:
    """Run convert_to_bf16 for a model with the given HF architecture string.
    Returns the captured command list passed to _run."""
    safetensors_dir = tmp_path / "model-src"
    safetensors_dir.mkdir()
    _write_config(safetensors_dir, architecture)

    cfg = _make_cfg()
    layout = _make_layout(tmp_path)

    captured: list[list[str]] = []

    def fake_run(cmd, log_path, env=None, **kwargs):
        captured.append(list(cmd))
        # Stage B checks out.exists() after _run — create it via --outfile arg.
        try:
            idx = cmd.index("--outfile")
            Path(cmd[idx + 1]).parent.mkdir(parents=True, exist_ok=True)
            Path(cmd[idx + 1]).write_bytes(b"dummy-gguf")
        except (ValueError, IndexError):
            pass
        return 0

    monkeypatch.setattr(pr, "_run", fake_run)
    monkeypatch.setattr(pr, "_find_convert_script",
                        lambda cfg: Path("/fake/convert_hf_to_gguf.py"))
    monkeypatch.setattr(pr, "subprocess_env", lambda cfg: {})
    # Stub out metadata patcher — fake_run writes a non-GGUF placeholder file;
    # the patcher is tested separately in test_no_mtp_meta_patch.py.
    monkeypatch.setattr(pr, "_patch_no_mtp_metadata", lambda _: None)

    pr.convert_to_bf16(cfg, layout, safetensors_dir, "test-model")
    assert captured, "convert_to_bf16 did not call _run"
    return captured[0]


def test_convert_to_bf16_no_mtp_present_for_qwen35(monkeypatch, tmp_path):
    cmd = _run_stage_b(monkeypatch, tmp_path, "Qwen3_5ForConditionalGeneration")
    assert "--no-mtp" in cmd, (
        f"--no-mtp must be present for Qwen3_5ForConditionalGeneration; got: {cmd}"
    )


def test_convert_to_bf16_no_mtp_absent_for_llama(monkeypatch, tmp_path):
    cmd = _run_stage_b(monkeypatch, tmp_path, "LlamaForCausalLM")
    assert "--no-mtp" not in cmd, (
        f"--no-mtp must be absent for LlamaForCausalLM; got: {cmd}"
    )
