"""Tests for the --imatrix-ctx CLI flag / imatrix_ctx TOML key.

Covers:
  - Config.imatrix_ctx field exists with default 512
  - TOML parsing: explicit value, absent (default), value < 1 raises ValueError
  - --imatrix-ctx parses on run / calibrate / explore subcommands
  - CLI default None does not override cfg; explicit value does override
  - Layout.imatrix_cache_path includes __x{ctx} in the key
  - Two calls with same (model_sha, corpus_sha, chunks) but different ctx → different paths
  - stage_d_imatrix passes cfg.imatrix_ctx as -c to the subprocess
"""
from __future__ import annotations

from pathlib import Path

import pytest

import prismaquant_llama.pipeline_runner as pr
from prismaquant_llama.budget import parse_budget
from prismaquant_llama.config import Config, load_config
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
# 1. Config.imatrix_ctx field exists with default 512
# ---------------------------------------------------------------------------

def test_config_imatrix_ctx_default():
    cfg = _make_cfg()
    assert cfg.imatrix_ctx == 512


def test_config_imatrix_ctx_override():
    cfg = _make_cfg(imatrix_ctx=2048)
    assert cfg.imatrix_ctx == 2048


# ---------------------------------------------------------------------------
# 2. TOML parsing
# ---------------------------------------------------------------------------

def test_toml_imatrix_ctx_explicit(tmp_path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        '[prismaquant-llama]\n'
        'base = "/nonexistent/base"\n'
        'quants = ["Q4_K"]\n'
        'budget = 25\n'
        'priority = "111"\n'
        'imatrix_ctx = 1024\n'
    )
    cfg = load_config(cfg_file)
    assert cfg.imatrix_ctx == 1024


def test_toml_imatrix_ctx_absent_uses_default(tmp_path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        '[prismaquant-llama]\n'
        'base = "/nonexistent/base"\n'
        'quants = ["Q4_K"]\n'
        'budget = 25\n'
        'priority = "111"\n'
    )
    cfg = load_config(cfg_file)
    assert cfg.imatrix_ctx == 512


def test_toml_imatrix_ctx_zero_raises(tmp_path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        '[prismaquant-llama]\n'
        'base = "/nonexistent/base"\n'
        'quants = ["Q4_K"]\n'
        'budget = 25\n'
        'priority = "111"\n'
        'imatrix_ctx = 0\n'
    )
    with pytest.raises(ValueError, match="imatrix_ctx"):
        load_config(cfg_file)


def test_toml_imatrix_ctx_negative_raises(tmp_path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        '[prismaquant-llama]\n'
        'base = "/nonexistent/base"\n'
        'quants = ["Q4_K"]\n'
        'budget = 25\n'
        'priority = "111"\n'
        'imatrix_ctx = -1\n'
    )
    with pytest.raises(ValueError, match="imatrix_ctx"):
        load_config(cfg_file)


# ---------------------------------------------------------------------------
# 3. CLI --imatrix-ctx on run / calibrate / explore subcommands
# ---------------------------------------------------------------------------

def test_cli_run_imatrix_ctx_explicit():
    import argparse
    p = argparse.ArgumentParser()
    pr.add_run_args(p)
    args = p.parse_args(["dummy-input", "--imatrix-ctx", "256"])
    assert args.imatrix_ctx == 256


def test_cli_run_imatrix_ctx_default_none():
    import argparse
    p = argparse.ArgumentParser()
    pr.add_run_args(p)
    args = p.parse_args(["dummy-input"])
    assert args.imatrix_ctx is None


def test_cli_calibrate_imatrix_ctx_explicit():
    from prismaquant_llama.calibration import add_calibrate_args
    import argparse
    p = argparse.ArgumentParser()
    add_calibrate_args(p)
    args = p.parse_args(["system", "dummy-input", "--imatrix-ctx", "256"])
    assert args.imatrix_ctx == 256


def test_cli_calibrate_imatrix_ctx_default_none():
    from prismaquant_llama.calibration import add_calibrate_args
    import argparse
    p = argparse.ArgumentParser()
    add_calibrate_args(p)
    args = p.parse_args(["system", "dummy-input"])
    assert args.imatrix_ctx is None


def test_cli_explore_imatrix_ctx_explicit():
    from prismaquant_llama.explore import add_explore_args
    import argparse
    p = argparse.ArgumentParser()
    add_explore_args(p)
    args = p.parse_args(["dummy-input", "--imatrix-ctx", "256"])
    assert args.imatrix_ctx == 256


def test_cli_explore_imatrix_ctx_default_none():
    from prismaquant_llama.explore import add_explore_args
    import argparse
    p = argparse.ArgumentParser()
    add_explore_args(p)
    args = p.parse_args(["dummy-input"])
    assert args.imatrix_ctx is None


# ---------------------------------------------------------------------------
# 4. CLI default None does not override cfg; explicit value does
# ---------------------------------------------------------------------------

def test_cfg_from_args_imatrix_ctx_none_does_not_override(tmp_path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        '[prismaquant-llama]\n'
        'base = "/nonexistent/base"\n'
        'quants = ["Q4_K"]\n'
        'budget = 25\n'
        'priority = "111"\n'
        'imatrix_ctx = 2048\n'
    )
    import argparse
    p = argparse.ArgumentParser()
    pr.add_run_args(p)
    args = p.parse_args(["dummy-input", "--config", str(cfg_file)])
    cfg = pr.cfg_from_args(args)
    assert cfg.imatrix_ctx == 2048


def test_cfg_from_args_imatrix_ctx_explicit_overrides(tmp_path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        '[prismaquant-llama]\n'
        'base = "/nonexistent/base"\n'
        'quants = ["Q4_K"]\n'
        'budget = 25\n'
        'priority = "111"\n'
        'imatrix_ctx = 2048\n'
    )
    import argparse
    p = argparse.ArgumentParser()
    pr.add_run_args(p)
    args = p.parse_args(["dummy-input", "--config", str(cfg_file),
                         "--imatrix-ctx", "128"])
    cfg = pr.cfg_from_args(args)
    assert cfg.imatrix_ctx == 128


# ---------------------------------------------------------------------------
# 5. Layout.imatrix_cache_path includes __x{ctx}
# ---------------------------------------------------------------------------

def test_imatrix_cache_path_includes_ctx_segment(tmp_path):
    layout = _make_layout(tmp_path)
    path = layout.imatrix_cache_path("aabbccdd1122", "eeff00112233", 50, 512)
    assert "__x512" in path.name, f"Expected __x512 in cache key; got: {path.name}"


def test_imatrix_cache_path_different_ctx_different_paths(tmp_path):
    layout = _make_layout(tmp_path)
    p512 = layout.imatrix_cache_path("aabbccdd1122", "eeff00112233", 50, 512)
    p4096 = layout.imatrix_cache_path("aabbccdd1122", "eeff00112233", 50, 4096)
    assert p512 != p4096, "Different ctx must produce different cache paths"


def test_imatrix_cache_path_same_ctx_same_path(tmp_path):
    layout = _make_layout(tmp_path)
    p1 = layout.imatrix_cache_path("aabbccdd1122", "eeff00112233", 50, 512)
    p2 = layout.imatrix_cache_path("aabbccdd1122", "eeff00112233", 50, 512)
    assert p1 == p2


def test_imatrix_cache_path_key_format(tmp_path):
    layout = _make_layout(tmp_path)
    path = layout.imatrix_cache_path("aabbccdd1122", "eeff00112233", 100, 256)
    assert path.name == "aabbccdd1122__eeff00112233__c100__x256.imatrix.gguf"


# ---------------------------------------------------------------------------
# 6. stage_d_imatrix passes cfg.imatrix_ctx as -c in subprocess argv
# ---------------------------------------------------------------------------

def _run_stage_d_capture(monkeypatch, tmp_path, imatrix_ctx: int) -> list[str]:
    cfg = _make_cfg(imatrix_ctx=imatrix_ctx)
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
    assert captured, "stage_d_imatrix did not call _run"
    return captured[0]


def test_stage_d_imatrix_ctx_512_in_argv(monkeypatch, tmp_path):
    cmd = _run_stage_d_capture(monkeypatch, tmp_path, imatrix_ctx=512)
    assert "-c" in cmd
    c_idx = cmd.index("-c")
    assert cmd[c_idx + 1] == "512", (
        f"Expected -c 512 in argv; got: {cmd}"
    )


def test_stage_d_imatrix_ctx_256_in_argv(monkeypatch, tmp_path):
    cmd = _run_stage_d_capture(monkeypatch, tmp_path, imatrix_ctx=256)
    c_idx = cmd.index("-c")
    assert cmd[c_idx + 1] == "256", (
        f"Expected -c 256 in argv; got: {cmd}"
    )


def test_stage_d_imatrix_ctx_4096_in_argv(monkeypatch, tmp_path):
    cmd = _run_stage_d_capture(monkeypatch, tmp_path, imatrix_ctx=4096)
    c_idx = cmd.index("-c")
    assert cmd[c_idx + 1] == "4096", (
        f"Expected -c 4096 in argv; got: {cmd}"
    )
