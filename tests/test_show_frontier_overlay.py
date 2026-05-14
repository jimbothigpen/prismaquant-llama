"""Regression tests for show-frontier overlay v2 (schema_version=3).

Covers the two halves shipped in S11:
- Stage-K summary may include a ``reference_ppl_f16`` field.
- show-frontier computes ``ppl_diff = measured - reference - pred_dppl``
  when both the v3 summary AND a --from-explore CSV are present.

Backward compat: v2 fixtures must still load with no ppl_diff column.
"""
from __future__ import annotations

import json
from pathlib import Path

from prismaquant_llama.show_frontier import (
    _load_explore_overlay,
    _summary_record,
)


FIXTURES = Path(__file__).parent / "fixtures" / "stage_k"
V2_SUMMARY = FIXTURES / "summary-PQ25.json"


def _write_v3_summary(path: Path, reference_ppl_f16: float = 14.0) -> None:
    """Synthetic 2-candidate v3 summary at PQ25 budget."""
    doc = {
        "schema_version": 3,
        "budget_gb": 0.5,
        "user_priority": "111",
        "winner_priority": "111",
        "winner_ppl": 15.5,
        "winner_size_gb": 0.5,
        "reference_ppl_f16": reference_ppl_f16,
        "candidates": [
            {"priority": "111", "ppl": 15.5, "size_gb": 0.5,
             "is_pareto": True, "recipe_sha": "a" * 8},
            {"priority": "900", "ppl": 16.0, "size_gb": 0.45,
             "is_pareto": True, "recipe_sha": "b" * 8},
        ],
    }
    path.write_text(json.dumps(doc, indent=2))


def _write_explore_csv(path: Path) -> None:
    """Synthetic explore CSV with one matching row per priority."""
    path.write_text(
        "budget_pct,priority,actual_GB,predicted_dppl\n"
        "25,111,0.52,1.20\n"
        "25,900,0.46,1.80\n"
    )


def test_v3_summary_loads_with_reference_ppl_and_ppl_diff(tmp_path):
    """v3 summary + --from-explore → ppl_diff computed per candidate."""
    summary = tmp_path / "summary-PQ25.json"
    _write_v3_summary(summary, reference_ppl_f16=14.0)
    explore = tmp_path / "explore.csv"
    _write_explore_csv(explore)

    explore_map = _load_explore_overlay(explore)
    rec = _summary_record(tmp_path, summary, explore_map)

    assert rec["summary_schema_version"] == 3
    assert rec["reference_ppl_f16"] == 14.0
    assert rec["has_explore_overlay"] is True
    assert rec["has_ppl_diff"] is True

    # Candidates are sorted by size_gb ascending → priority 900 first.
    by_pri = {c["priority"]: c for c in rec["candidates"]}
    # ppl_diff = measured - reference - pred_dppl
    # 111: 15.5 - 14.0 - 1.20 = 0.30
    assert by_pri["111"]["ppl_diff"] == pytest_approx(0.30)
    # 900: 16.0 - 14.0 - 1.80 = 0.20
    assert by_pri["900"]["ppl_diff"] == pytest_approx(0.20)


def test_v3_summary_without_explore_overlay_no_ppl_diff(tmp_path):
    """v3 summary alone (no --from-explore) → no ppl_diff in candidates,
    but reference_ppl_f16 still surfaces at the record level."""
    summary = tmp_path / "summary-PQ25.json"
    _write_v3_summary(summary)

    rec = _summary_record(tmp_path, summary, explore_map=None)
    assert rec["summary_schema_version"] == 3
    assert rec["reference_ppl_f16"] == 14.0
    assert rec["has_explore_overlay"] is False
    assert rec["has_ppl_diff"] is False
    for cand in rec["candidates"]:
        assert "ppl_diff" not in cand


def test_v2_summary_backcompat_no_reference_no_ppl_diff(tmp_path):
    """The S9 golden v2 fixture must keep loading with no overlay
    additions — strict backward-compat against the pre-S11 surface.
    """
    explore = tmp_path / "explore.csv"
    # Build an explore CSV keyed to PQ25 priorities present in the v2
    # fixture, so the overlay attaches but ppl_diff stays gated off.
    v2 = json.loads(V2_SUMMARY.read_text())
    rows = ["budget_pct,priority,actual_GB,predicted_dppl"]
    for c in v2["candidates"]:
        rows.append(f"25,{c['priority']},0.42,1.00")
    explore.write_text("\n".join(rows) + "\n")

    explore_map = _load_explore_overlay(explore)
    rec = _summary_record(tmp_path, V2_SUMMARY, explore_map)

    assert rec["summary_schema_version"] == 2
    assert rec["reference_ppl_f16"] is None
    assert rec["has_explore_overlay"] is True
    assert rec["has_ppl_diff"] is False
    # Overlay attaches pred_size_gb + pred_dppl + size_diff_gb but
    # leaves ppl_diff as None for every candidate.
    for cand in rec["candidates"]:
        assert cand.get("pred_dppl") == 1.00
        assert cand.get("ppl_diff") is None


def pytest_approx(expected, tol=1e-9):
    """Local epsilon-compare helper so tests don't pull in pytest.approx
    (which lives in pytest's namespace, not the std lib)."""
    import pytest
    return pytest.approx(expected, abs=tol)
