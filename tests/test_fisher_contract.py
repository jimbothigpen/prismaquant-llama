"""Fisher upstream-symbol regression smoke for prismaquant-llama.

Checks that the symbols this fork's Fisher pipeline depends on are still
present in the installed prismaquant package.  All checks are text-based
(source-file reads) — no module import, no torch cascade.

If this test fails after an upstream sync, upstream has likely run the Fisher
excision sweep (`8146fd6`'s deferred follow-up).  Plan: vendor the affected
code into the fork as a fork-patch.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.find_spec("prismaquant")
if _spec is None:
    pytest.skip(
        "prismaquant not installed — Fisher contract smoke inactive",
        allow_module_level=True,
    )

_PKG_ROOT = Path(_spec.submodule_search_locations[0])


def _read(relative: str) -> str:
    path = _PKG_ROOT / relative
    return path.read_text(encoding="utf-8")


def test_incremental_probe_h_detail_dir_arg_present():
    src = _read("incremental_probe.py")
    assert "--h-detail-dir" in src, (
        "Symbol '--h-detail-dir' missing from prismaquant/incremental_probe.py; "
        "upstream may have excised the Fisher h-detail argparse argument."
    )


def test_measure_quant_cost_h_detail_index_class_present():
    src = _read("measure_quant_cost.py")
    assert "class HDetailIndex" in src, (
        "Symbol 'class HDetailIndex' missing from prismaquant/measure_quant_cost.py; "
        "upstream may have excised the HDetailIndex class."
    )


def test_allocator_candidates_fisher_fn_present():
    src = _read("allocator_candidates.py")
    assert "def _fisher_output_mse_allocator_enabled" in src, (
        "Symbol 'def _fisher_output_mse_allocator_enabled' missing from "
        "prismaquant/allocator_candidates.py; "
        "upstream may have excised the Fisher MSE allocator function."
    )


def test_allocator_candidates_reads_env_var():
    src = _read("allocator_candidates.py")
    assert "PRISMAQUANT_FISHER_OUTPUT_MSE_ALLOCATOR" in src, (
        "Env-var name 'PRISMAQUANT_FISHER_OUTPUT_MSE_ALLOCATOR' missing from "
        "prismaquant/allocator_candidates.py; "
        "upstream may have removed the Fisher env-var gate."
    )
