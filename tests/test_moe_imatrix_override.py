"""Tests for MoE-aware --override-kv in Stage D llama-imatrix.

Covers:
  - Config.moe_all_experts_imatrix field exists with default True
  - TOML parsing: explicit false, explicit true, absent (default True)
  - --moe-all-experts-imatrix / --no-moe-all-experts-imatrix on run / calibrate / explore
  - CLI default None does not override cfg; explicit value does
  - _moe_override_kv_args: non-MoE, llama-MoE, qwen3moe-MoE, edge cases
  - Layout.imatrix_cache_path: moe_forced_used=None (no segment), N (segment), isolation
  - stage_d_imatrix subprocess argv: override present for MoE+on, absent for MoE+off
    and for non-MoE regardless of knob
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from gguf import GGUFWriter

import prismaquant_llama.pipeline_runner as pr
from prismaquant_llama.budget import parse_budget
from prismaquant_llama.calibration import add_calibrate_args
from prismaquant_llama.config import Config, load_config
from prismaquant_llama.explore import add_explore_args
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


def _make_moe_gguf(path: Path, arch: str, expert_count: int,
                   expert_used_count: int) -> Path:
    """Write a minimal MoE GGUF with expert_count and expert_used_count fields."""
    w = GGUFWriter(str(path), arch)
    w.add_uint32(f"{arch}.expert_count", expert_count)
    w.add_uint32(f"{arch}.expert_used_count", expert_used_count)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


def _make_non_moe_gguf(path: Path, arch: str = "llama") -> Path:
    """Write a minimal GGUF without expert_count fields."""
    w = GGUFWriter(str(path), arch)
    w.add_uint32(f"{arch}.block_count", 32)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


# ---------------------------------------------------------------------------
# 1. Config.moe_all_experts_imatrix field default
# ---------------------------------------------------------------------------

def test_config_moe_all_experts_imatrix_default():
    cfg = _make_cfg()
    assert cfg.moe_all_experts_imatrix is True


def test_config_moe_all_experts_imatrix_override_false():
    cfg = _make_cfg(moe_all_experts_imatrix=False)
    assert cfg.moe_all_experts_imatrix is False


# ---------------------------------------------------------------------------
# 2. TOML parsing
# ---------------------------------------------------------------------------

def test_toml_moe_all_experts_imatrix_explicit_false(tmp_path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        '[prismaquant-llama]\n'
        'base = "/nonexistent/base"\n'
        'quants = ["Q4_K"]\n'
        'budget = 25\n'
        'priority = "111"\n'
        'moe_all_experts_imatrix = false\n'
    )
    cfg = load_config(cfg_file)
    assert cfg.moe_all_experts_imatrix is False


def test_toml_moe_all_experts_imatrix_explicit_true(tmp_path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        '[prismaquant-llama]\n'
        'base = "/nonexistent/base"\n'
        'quants = ["Q4_K"]\n'
        'budget = 25\n'
        'priority = "111"\n'
        'moe_all_experts_imatrix = true\n'
    )
    cfg = load_config(cfg_file)
    assert cfg.moe_all_experts_imatrix is True


def test_toml_moe_all_experts_imatrix_absent_defaults_true(tmp_path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        '[prismaquant-llama]\n'
        'base = "/nonexistent/base"\n'
        'quants = ["Q4_K"]\n'
        'budget = 25\n'
        'priority = "111"\n'
    )
    cfg = load_config(cfg_file)
    assert cfg.moe_all_experts_imatrix is True


# ---------------------------------------------------------------------------
# 3. CLI flag on run / calibrate / explore subparsers
# ---------------------------------------------------------------------------

def test_cli_run_moe_flag_default_none():
    p = argparse.ArgumentParser()
    pr.add_run_args(p)
    args = p.parse_args(["dummy-input"])
    assert args.moe_all_experts_imatrix is None


def test_cli_run_moe_flag_explicit_true():
    p = argparse.ArgumentParser()
    pr.add_run_args(p)
    args = p.parse_args(["dummy-input", "--moe-all-experts-imatrix"])
    assert args.moe_all_experts_imatrix is True


def test_cli_run_moe_flag_explicit_false():
    p = argparse.ArgumentParser()
    pr.add_run_args(p)
    args = p.parse_args(["dummy-input", "--no-moe-all-experts-imatrix"])
    assert args.moe_all_experts_imatrix is False


def test_cli_calibrate_moe_flag_default_none():
    p = argparse.ArgumentParser()
    add_calibrate_args(p)
    args = p.parse_args(["system", "dummy-input"])
    assert args.moe_all_experts_imatrix is None


def test_cli_calibrate_moe_flag_explicit_true():
    p = argparse.ArgumentParser()
    add_calibrate_args(p)
    args = p.parse_args(["system", "dummy-input", "--moe-all-experts-imatrix"])
    assert args.moe_all_experts_imatrix is True


def test_cli_calibrate_moe_flag_explicit_false():
    p = argparse.ArgumentParser()
    add_calibrate_args(p)
    args = p.parse_args(["system", "dummy-input", "--no-moe-all-experts-imatrix"])
    assert args.moe_all_experts_imatrix is False


def test_cli_explore_moe_flag_default_none():
    p = argparse.ArgumentParser()
    add_explore_args(p)
    args = p.parse_args(["dummy-input"])
    assert args.moe_all_experts_imatrix is None


def test_cli_explore_moe_flag_explicit_true():
    p = argparse.ArgumentParser()
    add_explore_args(p)
    args = p.parse_args(["dummy-input", "--moe-all-experts-imatrix"])
    assert args.moe_all_experts_imatrix is True


def test_cli_explore_moe_flag_explicit_false():
    p = argparse.ArgumentParser()
    add_explore_args(p)
    args = p.parse_args(["dummy-input", "--no-moe-all-experts-imatrix"])
    assert args.moe_all_experts_imatrix is False


# ---------------------------------------------------------------------------
# 4. cfg_from_args: None does not override cfg; non-None overrides
# ---------------------------------------------------------------------------

def test_cfg_from_args_moe_none_does_not_override(tmp_path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        '[prismaquant-llama]\n'
        'base = "/nonexistent/base"\n'
        'quants = ["Q4_K"]\n'
        'budget = 25\n'
        'priority = "111"\n'
        'moe_all_experts_imatrix = false\n'
    )
    p = argparse.ArgumentParser()
    pr.add_run_args(p)
    args = p.parse_args(["dummy-input", "--config", str(cfg_file)])
    cfg = pr.cfg_from_args(args)
    assert cfg.moe_all_experts_imatrix is False


def test_cfg_from_args_moe_explicit_true_overrides(tmp_path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        '[prismaquant-llama]\n'
        'base = "/nonexistent/base"\n'
        'quants = ["Q4_K"]\n'
        'budget = 25\n'
        'priority = "111"\n'
        'moe_all_experts_imatrix = false\n'
    )
    p = argparse.ArgumentParser()
    pr.add_run_args(p)
    args = p.parse_args(["dummy-input", "--config", str(cfg_file),
                         "--moe-all-experts-imatrix"])
    cfg = pr.cfg_from_args(args)
    assert cfg.moe_all_experts_imatrix is True


def test_cfg_from_args_moe_explicit_false_overrides(tmp_path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        '[prismaquant-llama]\n'
        'base = "/nonexistent/base"\n'
        'quants = ["Q4_K"]\n'
        'budget = 25\n'
        'priority = "111"\n'
        'moe_all_experts_imatrix = true\n'
    )
    p = argparse.ArgumentParser()
    pr.add_run_args(p)
    args = p.parse_args(["dummy-input", "--config", str(cfg_file),
                         "--no-moe-all-experts-imatrix"])
    cfg = pr.cfg_from_args(args)
    assert cfg.moe_all_experts_imatrix is False


# ---------------------------------------------------------------------------
# 5. _moe_override_kv_args helper
# ---------------------------------------------------------------------------

def test_moe_override_args_non_moe_gguf(tmp_path):
    p = _make_non_moe_gguf(tmp_path / "dense.gguf")
    args, ec, arch = pr._moe_override_kv_args(p)
    assert args == []
    assert ec is None
    assert arch is None


def test_moe_override_args_llama_moe(tmp_path):
    p = _make_moe_gguf(tmp_path / "mixtral.gguf", "llama",
                       expert_count=8, expert_used_count=2)
    args, ec, arch = pr._moe_override_kv_args(p)
    assert args == ["--override-kv", "llama.expert_used_count=int:8"]
    assert ec == 8
    assert arch == "llama"


def test_moe_override_args_qwen3moe(tmp_path):
    p = _make_moe_gguf(tmp_path / "qwen3moe.gguf", "qwen3moe",
                       expert_count=128, expert_used_count=8)
    args, ec, arch = pr._moe_override_kv_args(p)
    assert args == ["--override-kv", "qwen3moe.expert_used_count=int:128"]
    assert ec == 128
    assert arch == "qwen3moe"


def test_moe_override_args_deepseek2(tmp_path):
    p = _make_moe_gguf(tmp_path / "ds2.gguf", "deepseek2",
                       expert_count=256, expert_used_count=8)
    args, ec, arch = pr._moe_override_kv_args(p)
    assert args == ["--override-kv", "deepseek2.expert_used_count=int:256"]
    assert ec == 256
    assert arch == "deepseek2"


def test_moe_override_args_edge_expert_count_equals_used(tmp_path):
    """expert_count == expert_used_count: already all-experts, override is no-op."""
    p = _make_moe_gguf(tmp_path / "dense_equiv.gguf", "llama",
                       expert_count=8, expert_used_count=8)
    args, ec, arch = pr._moe_override_kv_args(p)
    assert args == []
    assert ec is None
    assert arch is None


def test_moe_override_args_edge_expert_count_one(tmp_path):
    """expert_count == 1: not MoE; returns empty."""
    p = _make_moe_gguf(tmp_path / "single_expert.gguf", "llama",
                       expert_count=1, expert_used_count=1)
    args, ec, arch = pr._moe_override_kv_args(p)
    assert args == []
    assert ec is None
    assert arch is None


def test_moe_override_args_edge_missing_architecture(tmp_path):
    """GGUF with non-standard arch that has no architecture field triggers fallback."""
    # GGUFWriter always writes general.architecture, so we test with a valid
    # arch that lacks expert fields (same as non-MoE path).
    p = _make_non_moe_gguf(tmp_path / "no_experts.gguf", arch="llama")
    args, ec, arch = pr._moe_override_kv_args(p)
    assert args == []
    assert ec is None


# ---------------------------------------------------------------------------
# 6. Layout.imatrix_cache_path cache-key extension
# ---------------------------------------------------------------------------

def test_imatrix_cache_path_no_moe_segment_when_none(tmp_path):
    layout = _make_layout(tmp_path)
    path = layout.imatrix_cache_path("aabbccdd1122", "eeff00112233", 50, 512)
    assert "__moe" not in path.name
    assert path.name == "aabbccdd1122__eeff00112233__c50__x512.imatrix.gguf"


def test_imatrix_cache_path_moe_segment_when_set(tmp_path):
    layout = _make_layout(tmp_path)
    path = layout.imatrix_cache_path("aabbccdd1122", "eeff00112233", 50, 512,
                                     moe_forced_used=8)
    assert "__moe8" in path.name
    assert path.name == "aabbccdd1122__eeff00112233__c50__x512__moe8.imatrix.gguf"


def test_imatrix_cache_path_different_moe_different_paths(tmp_path):
    layout = _make_layout(tmp_path)
    p_none = layout.imatrix_cache_path("aabbccdd1122", "eeff00112233", 50, 512,
                                       moe_forced_used=None)
    p_moe8 = layout.imatrix_cache_path("aabbccdd1122", "eeff00112233", 50, 512,
                                       moe_forced_used=8)
    p_moe128 = layout.imatrix_cache_path("aabbccdd1122", "eeff00112233", 50, 512,
                                         moe_forced_used=128)
    assert p_none != p_moe8
    assert p_none != p_moe128
    assert p_moe8 != p_moe128


def test_imatrix_cache_path_same_moe_same_path(tmp_path):
    layout = _make_layout(tmp_path)
    p1 = layout.imatrix_cache_path("aabbccdd1122", "eeff00112233", 50, 512,
                                   moe_forced_used=8)
    p2 = layout.imatrix_cache_path("aabbccdd1122", "eeff00112233", 50, 512,
                                   moe_forced_used=8)
    assert p1 == p2


# ---------------------------------------------------------------------------
# 7. stage_d_imatrix subprocess-argv integration
# ---------------------------------------------------------------------------

def _run_stage_d_capture(monkeypatch, tmp_path, cfg: Config,
                         gguf_path: Path) -> list[str]:
    layout = _make_layout(tmp_path)
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

    pr.stage_d_imatrix(cfg, layout, gguf_path, corpus)
    assert captured, "stage_d_imatrix did not call _run"
    return captured[0]


def test_stage_d_moe_on_moe_model_has_override_kv(monkeypatch, tmp_path):
    gguf = _make_moe_gguf(tmp_path / "moe.gguf", "llama",
                          expert_count=8, expert_used_count=2)
    cfg = _make_cfg(moe_all_experts_imatrix=True)
    cmd = _run_stage_d_capture(monkeypatch, tmp_path, cfg, gguf)
    assert "--override-kv" in cmd
    ov_idx = cmd.index("--override-kv")
    assert cmd[ov_idx + 1] == "llama.expert_used_count=int:8"


def test_stage_d_moe_off_moe_model_no_override_kv(monkeypatch, tmp_path):
    gguf = _make_moe_gguf(tmp_path / "moe.gguf", "llama",
                          expert_count=8, expert_used_count=2)
    cfg = _make_cfg(moe_all_experts_imatrix=False)
    cmd = _run_stage_d_capture(monkeypatch, tmp_path, cfg, gguf)
    assert "--override-kv" not in cmd


def test_stage_d_moe_on_non_moe_model_no_override_kv(monkeypatch, tmp_path):
    gguf = _make_non_moe_gguf(tmp_path / "dense.gguf")
    cfg = _make_cfg(moe_all_experts_imatrix=True)
    cmd = _run_stage_d_capture(monkeypatch, tmp_path, cfg, gguf)
    assert "--override-kv" not in cmd


def test_stage_d_moe_off_non_moe_model_no_override_kv(monkeypatch, tmp_path):
    gguf = _make_non_moe_gguf(tmp_path / "dense.gguf")
    cfg = _make_cfg(moe_all_experts_imatrix=False)
    cmd = _run_stage_d_capture(monkeypatch, tmp_path, cfg, gguf)
    assert "--override-kv" not in cmd


def test_stage_d_moe_on_qwen3moe_correct_arch_prefix(monkeypatch, tmp_path):
    gguf = _make_moe_gguf(tmp_path / "qwen3moe.gguf", "qwen3moe",
                          expert_count=128, expert_used_count=8)
    cfg = _make_cfg(moe_all_experts_imatrix=True)
    cmd = _run_stage_d_capture(monkeypatch, tmp_path, cfg, gguf)
    assert "--override-kv" in cmd
    ov_idx = cmd.index("--override-kv")
    assert cmd[ov_idx + 1] == "qwen3moe.expert_used_count=int:128"


def test_stage_d_moe_cache_key_includes_moe_segment(monkeypatch, tmp_path):
    """Cache path for a MoE+on run includes __moe{N}; non-MoE does not."""
    cache_paths: list[Path] = []

    def fake_run(cmd, log_path, env=None, **kwargs):
        try:
            out_idx = cmd.index("-o")
            p = Path(cmd[out_idx + 1])
            cache_paths.append(p)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"")
        except (ValueError, IndexError):
            pass
        return 0

    monkeypatch.setattr(pr, "_run", fake_run)
    monkeypatch.setattr(pr, "find_tool", lambda cfg, tool: Path(f"/fake/{tool}"))
    monkeypatch.setattr(pr, "subprocess_env", lambda cfg: {})

    layout = _make_layout(tmp_path)
    corpus = tmp_path / "corpus.txt"
    corpus.write_bytes(b"dummy-corpus")

    moe_gguf = _make_moe_gguf(tmp_path / "moe.gguf", "llama",
                               expert_count=8, expert_used_count=2)
    cfg = _make_cfg(moe_all_experts_imatrix=True)
    pr.stage_d_imatrix(cfg, layout, moe_gguf, corpus)

    assert cache_paths, "stage_d_imatrix did not invoke _run"
    assert "__moe8" in cache_paths[0].name, (
        f"Expected __moe8 in cache key; got: {cache_paths[0].name}"
    )
