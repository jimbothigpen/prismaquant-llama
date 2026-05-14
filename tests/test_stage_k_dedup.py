"""Regression tests for Stage-K recipe-SHA dedup (S8 + S9).

Fixtures under tests/fixtures/stage_k/ are a copy of the S9 end-to-end
Stage-K artifacts on Qwen3-0.6B at PQ25 budget (10 sweep priorities,
saturated λ, 2 distinct inner recipes).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from prismaquant_llama.pipeline_runner import _compute_recipe_sha


FIXTURES = Path(__file__).parent / "fixtures" / "stage_k"
RECIPES = FIXTURES / "recipes"
SUMMARY = FIXTURES / "summary-PQ25.json"
SWEEP_PRIORITIES = [
    "009", "090", "111", "153", "225",
    "252", "333", "522", "531", "900",
]


def _recipe_path(priority: str) -> Path:
    return RECIPES / f"recipe-PQ25-{priority}.json"


def test_dedup_sha_collides_on_saturated_lambda():
    """Under saturated λ, all 9 ppl/tg/pp-balanced priorities + the
    ppl-pure 900 collapse to one inner recipe; 009 + 090 collapse to
    another. The full 10-priority sweep yields exactly 2 distinct
    recipe SHAs, matching the S9 measurement.
    """
    shas = {p: _compute_recipe_sha(_recipe_path(p)) for p in SWEEP_PRIORITIES}
    distinct = set(shas.values())
    assert len(distinct) == 2, (
        f"expected 2 distinct recipe SHAs across saturated-λ sweep, "
        f"got {len(distinct)}: {shas}"
    )

    # Cluster A: 009 + 090 (tg/pp-only weights, ignores ppl) → same recipe
    assert shas["009"] == shas["090"]
    # Cluster B: the other 8 priorities → same recipe
    cluster_b = {p for p in SWEEP_PRIORITIES if shas[p] == shas["111"]}
    assert cluster_b == {
        "111", "153", "225", "252", "333", "522", "531", "900"
    }, f"unexpected cluster-B membership: {cluster_b}"


def test_dedup_sha_distinct_for_different_assignments():
    """Two recipes whose inner `recipe` dicts differ must produce
    different SHAs. Top-level priority/weights/lambda differences are
    NOT enough — that was the S8 bug (full-file SHA missed collisions).
    """
    sha_009 = _compute_recipe_sha(_recipe_path("009"))
    sha_111 = _compute_recipe_sha(_recipe_path("111"))
    assert sha_009 != sha_111

    # And: top-level-only differences must NOT change the SHA. Construct
    # two recipes with identical inner `recipe` but different priority/
    # weights/lambda — the regression case S8 was designed to catch.
    inner = {
        "output.weight": "Q6_K",
        "token_embd.weight": "Q8_0",
        "blk.0.attn_q.weight": "IQ4_KSS",
    }
    a = {
        "priority": "111", "lambda": 1.0,
        "weights": {"ppl": 0.33, "tg": 0.33, "pp": 0.33},
        "recipe": inner,
    }
    b = {
        "priority": "900", "lambda": 5.0,
        "weights": {"ppl": 1.0, "tg": 0.0, "pp": 0.0},
        "recipe": inner,
    }
    pa = RECIPES.parent / "_synthetic_a.json"
    pb = RECIPES.parent / "_synthetic_b.json"
    try:
        pa.write_text(json.dumps(a))
        pb.write_text(json.dumps(b))
        assert _compute_recipe_sha(pa) == _compute_recipe_sha(pb)
    finally:
        pa.unlink(missing_ok=True)
        pb.unlink(missing_ok=True)


def test_summary_writes_duplicate_of():
    """Every summary entry with `duplicate_of: X` must reuse the
    candidate_gguf / recipe_sha / ppl / size_gb of the entry whose
    priority is X. Cross-validate the recorded recipe_sha against a
    fresh _compute_recipe_sha() of the on-disk recipe file.

    Uses the S9 Qwen3-0.6B PQ25 summary as the golden snapshot.
    """
    summary = json.loads(SUMMARY.read_text())
    by_priority = {c["priority"]: c for c in summary["candidates"]}
    dup_count = 0

    for cand in summary["candidates"]:
        # Computed SHA must match what the summary recorded.
        computed = _compute_recipe_sha(_recipe_path(cand["priority"]))
        assert cand["recipe_sha"] == computed, (
            f"priority={cand['priority']}: summary recipe_sha "
            f"{cand['recipe_sha']} != recomputed {computed}"
        )

        if "duplicate_of" not in cand:
            continue
        dup_count += 1
        target = by_priority[cand["duplicate_of"]]
        assert "duplicate_of" not in target, (
            f"priority={cand['priority']} points at "
            f"duplicate_of={cand['duplicate_of']} which is itself a "
            f"duplicate (chained dedup is not allowed)"
        )
        # The dedup contract: byte-identical inner recipe → reuse
        # quantize + PPL artifacts from the prior entry.
        assert cand["candidate_gguf"] == target["candidate_gguf"]
        assert cand["recipe_sha"] == target["recipe_sha"]
        assert cand["ppl"] == target["ppl"]
        assert cand["size_gb"] == target["size_gb"]

    # S9 had 8 of 10 candidates deduped. Assert at least one dup
    # exists so a regression that disables dedup entirely doesn't slip
    # by under a trivially-passing loop.
    assert dup_count >= 1, "golden summary should contain duplicates"
