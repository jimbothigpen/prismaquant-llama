"""Tests for --no-mmap gating via imatrix_eager_load / ppl_eager_load toggles.

Covers both config branches (True / False) for:
  - Stage D  (llama-imatrix)         via pipeline_runner.stage_d_imatrix
  - Stage I  (llama-perplexity)      via pipeline_runner.stage_i_eval
  - Calibration PPL                  via calibration._measure_perplexity
"""
from __future__ import annotations

from pathlib import Path

import pytest

import prismaquant_llama.pipeline_runner as pr
import prismaquant_llama.calibration as cal
from prismaquant_llama.budget import parse_budget
from prismaquant_llama.config import Config
from prismaquant_llama.paths import Layout


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
# Stage D — llama-imatrix
# ---------------------------------------------------------------------------

def _run_stage_d(monkeypatch, tmp_path: Path, eager: bool) -> list[str]:
    """Run stage_d_imatrix with eager_load_imatrix=eager; return captured cmd."""
    cfg = _make_cfg(imatrix_eager_load=eager)
    layout = _make_layout(tmp_path)

    bf16 = tmp_path / "model.gguf"
    bf16.write_bytes(b"dummy-bf16")
    corpus = tmp_path / "corpus.txt"
    corpus.write_bytes(b"dummy-corpus")

    captured: list[list[str]] = []

    def fake_run(cmd, log_path, env=None, **kwargs):
        captured.append(list(cmd))
        # stage_d checks cache.exists() after _run — create it via -o arg
        try:
            out_idx = cmd.index("-o")
            Path(cmd[out_idx + 1]).parent.mkdir(parents=True, exist_ok=True)
            Path(cmd[out_idx + 1]).write_bytes(b"")
        except (ValueError, IndexError):
            pass
        return 0

    monkeypatch.setattr(pr, "_run", fake_run)
    monkeypatch.setattr(pr, "find_tool", lambda cfg, tool: Path(f"/fake/{tool}"))
    monkeypatch.setattr(pr, "subprocess_env", lambda cfg: {})

    pr.stage_d_imatrix(cfg, layout, bf16, corpus)
    assert captured, "stage_d_imatrix did not call _run"
    return captured[0]


def test_stage_d_no_mmap_absent_by_default(monkeypatch, tmp_path):
    cmd = _run_stage_d(monkeypatch, tmp_path, eager=False)
    assert "--no-mmap" not in cmd, (
        f"--no-mmap must be absent when imatrix_eager_load=False; got: {cmd}"
    )


def test_stage_d_no_mmap_present_when_eager_load(monkeypatch, tmp_path):
    cmd = _run_stage_d(monkeypatch, tmp_path, eager=True)
    assert "--no-mmap" in cmd, (
        f"--no-mmap must be present when imatrix_eager_load=True; got: {cmd}"
    )


# ---------------------------------------------------------------------------
# Stage I — final llama-perplexity
# ---------------------------------------------------------------------------

def _run_stage_i(monkeypatch, tmp_path: Path, eager: bool) -> list[str]:
    cfg = _make_cfg(ppl_eager_load=eager)
    layout = _make_layout(tmp_path)

    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"dummy")
    corpus = tmp_path / "corpus.txt"
    corpus.write_bytes(b"dummy")

    captured: list[list[str]] = []

    def fake_run(cmd, log_path, env=None, **kwargs):
        captured.append(list(cmd))
        # Write a dummy PPL line so stage_i_eval doesn't WARN and returns cleanly
        Path(log_path).write_text("Final estimate: PPL = 9.9999")
        return 0

    monkeypatch.setattr(pr, "_run", fake_run)
    monkeypatch.setattr(pr, "find_tool", lambda cfg, tool: Path(f"/fake/{tool}"))
    monkeypatch.setattr(pr, "subprocess_env", lambda cfg: {})

    pr.stage_i_eval(cfg, layout, gguf, corpus)
    assert captured, "stage_i_eval did not call _run"
    return captured[0]


def test_stage_i_no_mmap_absent_by_default(monkeypatch, tmp_path):
    cmd = _run_stage_i(monkeypatch, tmp_path, eager=False)
    assert "--no-mmap" not in cmd, (
        f"--no-mmap must be absent when ppl_eager_load=False; got: {cmd}"
    )


def test_stage_i_no_mmap_present_when_eager_load(monkeypatch, tmp_path):
    cmd = _run_stage_i(monkeypatch, tmp_path, eager=True)
    assert "--no-mmap" in cmd, (
        f"--no-mmap must be present when ppl_eager_load=True; got: {cmd}"
    )


# ---------------------------------------------------------------------------
# Calibration — _measure_perplexity
# ---------------------------------------------------------------------------

def _run_calib_ppl(monkeypatch, tmp_path: Path, eager: bool) -> list[str]:
    cfg = _make_cfg(ppl_eager_load=eager)

    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"dummy")
    corpus = tmp_path / "corpus.txt"
    corpus.write_bytes(b"dummy")
    log_path = tmp_path / "logs" / "calib.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    captured: list[list[str]] = []

    def fake_run_cmd(cmd, env, log_path, **kwargs):
        captured.append(list(cmd))
        return 0, ""

    monkeypatch.setattr(cal, "_run_cmd", fake_run_cmd)
    monkeypatch.setattr(cal, "find_tool", lambda cfg, tool: Path(f"/fake/{tool}"))
    monkeypatch.setattr(cal, "subprocess_env", lambda cfg: {})

    cal._measure_perplexity(cfg, gguf, corpus, log_path)
    assert captured, "calibration._measure_perplexity did not call _run_cmd"
    return captured[0]


def test_calibration_ppl_no_mmap_absent_by_default(monkeypatch, tmp_path):
    cmd = _run_calib_ppl(monkeypatch, tmp_path, eager=False)
    assert "--no-mmap" not in cmd, (
        f"--no-mmap must be absent when ppl_eager_load=False; got: {cmd}"
    )


def test_calibration_ppl_no_mmap_present_when_eager_load(monkeypatch, tmp_path):
    cmd = _run_calib_ppl(monkeypatch, tmp_path, eager=True)
    assert "--no-mmap" in cmd, (
        f"--no-mmap must be present when ppl_eager_load=True; got: {cmd}"
    )
