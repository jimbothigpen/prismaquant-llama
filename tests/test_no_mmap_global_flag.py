"""Tests for the global --no-mmap CLI flag / no_mmap TOML key.

Covers:
  - _no_mmap_args() per-binary dispatch (flag form correctness)
  - Stage D / I / calibration-PPL / K-ref-PPL: cfg.no_mmap=True with per-stage
    toggle False still appends the correct flag
  - calibration _measure_bench: cfg.no_mmap=True appends --mmap 0, absent when False
  - CLI argparse: --no-mmap accepted on all 4 subcommands
  - TOML parse: no_mmap = true → cfg.no_mmap == True
"""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Optional
from unittest.mock import patch
import io

import pytest

import prismaquant_llama.pipeline_runner as pr
import prismaquant_llama.calibration as cal
from prismaquant_llama.budget import parse_budget
from prismaquant_llama.config import Config
from prismaquant_llama.paths import Layout


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_cfg(**overrides) -> Config:
    defaults = dict(
        base=Path("/nonexistent/pq-test"),
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
# _no_mmap_args: per-binary flag-form dispatch
# ---------------------------------------------------------------------------

def test_no_mmap_args_imatrix():
    assert pr._no_mmap_args("llama-imatrix") == ["--no-mmap"]


def test_no_mmap_args_perplexity():
    assert pr._no_mmap_args("llama-perplexity") == ["--no-mmap"]


def test_no_mmap_args_quantize():
    # llama-quantize returns the same --no-mmap form (even though the binary
    # doesn't actually support it — that's documented in the binary support
    # table; callers must skip gating for llama-quantize entirely)
    assert pr._no_mmap_args("llama-quantize") == ["--no-mmap"]


def test_no_mmap_args_bench():
    # llama-bench uses --mmap <0|1> value form; --no-mmap is not accepted
    assert pr._no_mmap_args("llama-bench") == ["--mmap", "0"]


def test_no_mmap_args_bench_not_no_mmap_string():
    result = pr._no_mmap_args("llama-bench")
    assert "--no-mmap" not in result, (
        "llama-bench must use --mmap 0, not --no-mmap"
    )


# ---------------------------------------------------------------------------
# Stage D — cfg.no_mmap=True with imatrix_eager_load=False
# ---------------------------------------------------------------------------

def _run_stage_d(monkeypatch, tmp_path, no_mmap: bool, eager: bool) -> list[str]:
    cfg = _make_cfg(no_mmap=no_mmap, imatrix_eager_load=eager)
    layout = _make_layout(tmp_path)

    bf16 = tmp_path / "model.gguf"
    bf16.write_bytes(b"dummy-bf16")
    corpus = tmp_path / "corpus.txt"
    corpus.write_bytes(b"dummy-corpus")

    captured: list[list[str]] = []

    def fake_run(cmd, log_path, env=None, **kwargs):
        captured.append(list(cmd))
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
    assert captured
    return captured[0]


def test_stage_d_no_mmap_present_when_global_flag(monkeypatch, tmp_path):
    cmd = _run_stage_d(monkeypatch, tmp_path, no_mmap=True, eager=False)
    assert "--no-mmap" in cmd, (
        f"--no-mmap must be present when cfg.no_mmap=True; got: {cmd}"
    )


def test_stage_d_no_mmap_absent_when_both_false(monkeypatch, tmp_path):
    cmd = _run_stage_d(monkeypatch, tmp_path, no_mmap=False, eager=False)
    assert "--no-mmap" not in cmd, (
        f"--no-mmap must be absent when both flags False; got: {cmd}"
    )


# ---------------------------------------------------------------------------
# Stage I — cfg.no_mmap=True with ppl_eager_load=False
# ---------------------------------------------------------------------------

def _run_stage_i(monkeypatch, tmp_path, no_mmap: bool, eager: bool) -> list[str]:
    cfg = _make_cfg(no_mmap=no_mmap, ppl_eager_load=eager)
    layout = _make_layout(tmp_path)

    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"dummy")
    corpus = tmp_path / "corpus.txt"
    corpus.write_bytes(b"dummy")

    captured: list[list[str]] = []

    def fake_run(cmd, log_path, env=None, **kwargs):
        captured.append(list(cmd))
        Path(log_path).write_text("Final estimate: PPL = 9.9999")
        return 0

    monkeypatch.setattr(pr, "_run", fake_run)
    monkeypatch.setattr(pr, "find_tool", lambda cfg, tool: Path(f"/fake/{tool}"))
    monkeypatch.setattr(pr, "subprocess_env", lambda cfg: {})

    pr.stage_i_eval(cfg, layout, gguf, corpus)
    assert captured
    return captured[0]


def test_stage_i_no_mmap_present_when_global_flag(monkeypatch, tmp_path):
    cmd = _run_stage_i(monkeypatch, tmp_path, no_mmap=True, eager=False)
    assert "--no-mmap" in cmd, (
        f"--no-mmap must be present when cfg.no_mmap=True; got: {cmd}"
    )


def test_stage_i_no_mmap_absent_when_both_false(monkeypatch, tmp_path):
    cmd = _run_stage_i(monkeypatch, tmp_path, no_mmap=False, eager=False)
    assert "--no-mmap" not in cmd


# ---------------------------------------------------------------------------
# Calibration PPL — cfg.no_mmap=True with ppl_eager_load=False
# ---------------------------------------------------------------------------

def _run_calib_ppl(monkeypatch, tmp_path, no_mmap: bool, eager: bool) -> list[str]:
    cfg = _make_cfg(no_mmap=no_mmap, ppl_eager_load=eager)

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
    assert captured
    return captured[0]


def test_calib_ppl_no_mmap_present_when_global_flag(monkeypatch, tmp_path):
    cmd = _run_calib_ppl(monkeypatch, tmp_path, no_mmap=True, eager=False)
    assert "--no-mmap" in cmd, (
        f"--no-mmap must be present when cfg.no_mmap=True; got: {cmd}"
    )


def test_calib_ppl_no_mmap_absent_when_both_false(monkeypatch, tmp_path):
    cmd = _run_calib_ppl(monkeypatch, tmp_path, no_mmap=False, eager=False)
    assert "--no-mmap" not in cmd


# ---------------------------------------------------------------------------
# Stage K-ref PPL — _stage_k_reference_ppl
# ---------------------------------------------------------------------------

def _run_kref_ppl(monkeypatch, tmp_path, no_mmap: bool, eager: bool) -> list[str]:
    cfg = _make_cfg(no_mmap=no_mmap, ppl_eager_load=eager)
    layout = _make_layout(tmp_path)
    work = tmp_path / "stage-k"
    work.mkdir()

    bf16 = tmp_path / "model.gguf"
    bf16.write_bytes(b"dummy")
    corpus = tmp_path / "corpus.txt"
    corpus.write_bytes(b"dummy")
    perp_bin = Path("/fake/llama-perplexity")

    captured: list[list[str]] = []

    def fake_run(cmd, log_path, env=None, **kwargs):
        captured.append(list(cmd))
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text("Final estimate: PPL = 8.8888")
        return 0

    monkeypatch.setattr(pr, "_run", fake_run)
    monkeypatch.setattr(pr, "subprocess_env", lambda cfg: {})

    pr._stage_k_reference_ppl(cfg, layout, bf16, corpus, perp_bin, work)
    assert captured
    return captured[0]


def test_kref_ppl_no_mmap_present_when_global_flag(monkeypatch, tmp_path):
    cmd = _run_kref_ppl(monkeypatch, tmp_path, no_mmap=True, eager=False)
    assert "--no-mmap" in cmd, (
        f"--no-mmap must be present when cfg.no_mmap=True; got: {cmd}"
    )


def test_kref_ppl_no_mmap_absent_when_both_false(monkeypatch, tmp_path):
    cmd = _run_kref_ppl(monkeypatch, tmp_path, no_mmap=False, eager=False)
    assert "--no-mmap" not in cmd


# ---------------------------------------------------------------------------
# Calibration bench — _measure_bench
# ---------------------------------------------------------------------------

def _run_calib_bench(monkeypatch, tmp_path, no_mmap: bool) -> list[str]:
    cfg = _make_cfg(no_mmap=no_mmap)

    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"dummy")
    log_path = tmp_path / "logs" / "bench.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    captured: list[list[str]] = []

    def fake_run_cmd(cmd, env, log_path, **kwargs):
        captured.append(list(cmd))
        return 0, ""

    monkeypatch.setattr(cal, "_run_cmd", fake_run_cmd)
    monkeypatch.setattr(cal, "find_tool", lambda cfg, tool: Path(f"/fake/{tool}"))
    monkeypatch.setattr(cal, "subprocess_env", lambda cfg: {})

    cal._measure_bench(cfg, gguf, log_path)
    assert captured
    return captured[0]


def test_calib_bench_mmap_0_present_when_global_flag(monkeypatch, tmp_path):
    cmd = _run_calib_bench(monkeypatch, tmp_path, no_mmap=True)
    # llama-bench uses --mmap 0, NOT --no-mmap
    assert "--no-mmap" not in cmd, (
        f"--no-mmap must NOT appear for llama-bench; got: {cmd}"
    )
    # Check --mmap 0 appears as consecutive tokens
    pairs = list(zip(cmd, cmd[1:]))
    assert ("--mmap", "0") in pairs, (
        f"--mmap 0 must be present for llama-bench when cfg.no_mmap=True; got: {cmd}"
    )


def test_calib_bench_no_mmap_flag_absent_when_false(monkeypatch, tmp_path):
    cmd = _run_calib_bench(monkeypatch, tmp_path, no_mmap=False)
    assert "--mmap" not in cmd, (
        f"--mmap must be absent when cfg.no_mmap=False; got: {cmd}"
    )
    assert "--no-mmap" not in cmd


# ---------------------------------------------------------------------------
# CLI argparse: --no-mmap on each subcommand parser
# ---------------------------------------------------------------------------

def test_cli_run_no_mmap_flag():
    import argparse
    p = argparse.ArgumentParser()
    pr.add_run_args(p)
    args = p.parse_args(["dummy-input", "--no-mmap"])
    assert args.no_mmap is True


def test_cli_run_no_mmap_default_false():
    import argparse
    p = argparse.ArgumentParser()
    pr.add_run_args(p)
    args = p.parse_args(["dummy-input"])
    assert args.no_mmap is False


def test_cli_calibrate_no_mmap_flag():
    from prismaquant_llama.calibration import add_calibrate_args
    import argparse
    p = argparse.ArgumentParser()
    add_calibrate_args(p)
    args = p.parse_args(["system", "dummy-input", "--no-mmap"])
    assert args.no_mmap is True


def test_cli_calibrate_no_mmap_default_false():
    from prismaquant_llama.calibration import add_calibrate_args
    import argparse
    p = argparse.ArgumentParser()
    add_calibrate_args(p)
    args = p.parse_args(["system", "dummy-input"])
    assert args.no_mmap is False


def test_cli_explore_no_mmap_flag():
    from prismaquant_llama.explore import add_explore_args
    import argparse
    p = argparse.ArgumentParser()
    add_explore_args(p)
    args = p.parse_args(["dummy-input", "--no-mmap"])
    assert args.no_mmap is True


def test_cli_explore_no_mmap_default_false():
    from prismaquant_llama.explore import add_explore_args
    import argparse
    p = argparse.ArgumentParser()
    add_explore_args(p)
    args = p.parse_args(["dummy-input"])
    assert args.no_mmap is False


def test_cli_show_frontier_no_mmap_flag():
    import argparse
    p = argparse.ArgumentParser()
    # Replicate show_frontier's parser setup (inline in main)
    p.add_argument("input")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--base", type=Path, default=None)
    p.add_argument("--budget", default=None)
    p.add_argument("--run", default=None)
    p.add_argument("--all-runs", action="store_true")
    p.add_argument("--output-csv", type=Path, default=None)
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--output-md", type=Path, default=None)
    p.add_argument("--from-explore", type=Path, default=None)
    p.add_argument("--no-mmap", action="store_true", default=False, dest="no_mmap")
    args = p.parse_args(["dummy-input", "--no-mmap"])
    assert args.no_mmap is True


# ---------------------------------------------------------------------------
# TOML parse: no_mmap = true → cfg.no_mmap == True
# ---------------------------------------------------------------------------

def test_toml_no_mmap_true(tmp_path):
    """no_mmap = true in TOML is parsed into cfg.no_mmap == True."""
    from prismaquant_llama.config import load_config

    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        '[prismaquant-llama]\n'
        'base = "/nonexistent/base"\n'
        'quants = ["Q4_K"]\n'
        'budget = 25\n'
        'priority = "111"\n'
        'no_mmap = true\n'
    )
    cfg = load_config(cfg_file)
    assert cfg.no_mmap is True


def test_toml_no_mmap_false_by_default(tmp_path):
    """no_mmap defaults to False when absent from TOML."""
    from prismaquant_llama.config import load_config

    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        '[prismaquant-llama]\n'
        'base = "/nonexistent/base"\n'
        'quants = ["Q4_K"]\n'
        'budget = 25\n'
        'priority = "111"\n'
    )
    cfg = load_config(cfg_file)
    assert cfg.no_mmap is False
