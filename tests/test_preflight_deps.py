"""Pre-flight check for Stage B convert subprocess deps."""
from __future__ import annotations

import importlib.util

import pytest

from prismaquant_llama import pipeline_runner


def test_preflight_passes_when_all_deps_present():
    pipeline_runner._preflight_check_run_deps()


def test_preflight_fails_when_gguf_missing(monkeypatch):
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "gguf":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    with pytest.raises(SystemExit) as excinfo:
        pipeline_runner._preflight_check_run_deps()

    msg = str(excinfo.value)
    assert "gguf" in msg
    assert "prismaquant_venv_update_procedure" in msg


def test_preflight_lists_all_missing_deps(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a, **kw: None)

    with pytest.raises(SystemExit) as excinfo:
        pipeline_runner._preflight_check_run_deps()

    msg = str(excinfo.value)
    assert "gguf" in msg
    assert "sentencepiece" in msg
    assert "protobuf" in msg
