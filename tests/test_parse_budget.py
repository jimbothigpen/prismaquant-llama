"""Tests for the budget input parser (budget.py).

Covers:
- parse_budget: all three forms, case-insensitivity, whitespace tolerance,
  validation errors (zero, negative, bpw>16, unknown suffix)
- budget_from_toml: int → pct, float → pct, str → parse_budget
- check_mixed_units: homogeneous passes, mixed raises
- BudgetSpec.filename_label and display_str properties
- Cache-key equivalence: two equivalent budgets (different units) that would
  produce the same budget_gb will share a recipe cache key only once budget_gb
  is resolved — confirmed by checking that filename_label differs (they ARE
  different labels, so two different budget specs produce two different filenames
  as intended; the budget_gb unification happens in the pipeline, not here).
"""
from __future__ import annotations

import pytest

from prismaquant_llama.budget import (
    BudgetSpec,
    budget_from_toml,
    check_mixed_units,
    parse_budget,
)


# ---------------------------------------------------------------------------
# parse_budget: valid inputs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec,form,value", [
    # percent — bare integer
    ("25",    "pct", 25.0),
    ("0.5",   "pct", 0.5),
    ("100",   "pct", 100.0),
    # percent — explicit %
    ("25%",   "pct", 25.0),
    ("3.14%", "pct", 3.14),
    # bpw
    ("4.5bpw",  "bpw", 4.5),
    ("3bpw",    "bpw", 3.0),
    ("2.5BPW",  "bpw", 2.5),   # case-insensitive
    ("8 bpw",   "bpw", 8.0),   # whitespace
    (" 6.0bpw", "bpw", 6.0),   # leading whitespace
    # gb
    ("16GB",   "gb", 16.0),
    ("22gb",   "gb", 22.0),     # lowercase
    ("4.75GB", "gb", 4.75),
    (" 8 GB ", "gb", 8.0),      # surrounding whitespace
])
def test_parse_budget_valid(spec, form, value):
    b = parse_budget(spec)
    assert b.form == form
    assert b.value == pytest.approx(value)
    assert b.original_input == spec


# ---------------------------------------------------------------------------
# parse_budget: validation errors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec,fragment", [
    ("0",      "non-zero"),
    ("0%",     "non-zero"),
    ("0bpw",   "non-zero"),
    ("0GB",    "non-zero"),
    ("-5",     "negative"),
    ("-1bpw",  "negative"),
    ("-2GB",   "negative"),
    ("17bpw",  "16"),          # bpw > 16
    ("16.1bpw","16"),
    ("20bpw",  "16"),
    ("abc",    "unrecognized"),
    ("25xyz",  "unrecognized"),
    ("",       "empty"),
])
def test_parse_budget_invalid(spec, fragment):
    with pytest.raises(ValueError, match=fragment):
        parse_budget(spec)


# ---------------------------------------------------------------------------
# BudgetSpec.filename_label
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec,expected_label", [
    ("25",      "PQ25"),
    ("25%",     "PQ25"),
    ("4.5bpw",  "PQ4p5bpw"),
    ("3bpw",    "PQ3bpw"),
    ("2.75bpw", "PQ2p75bpw"),
    ("16GB",    "PQ16gb"),
    ("4.75GB",  "PQ4p75gb"),
    ("8GB",     "PQ8gb"),
])
def test_filename_label(spec, expected_label):
    assert parse_budget(spec).filename_label == expected_label


# ---------------------------------------------------------------------------
# BudgetSpec.display_str
# ---------------------------------------------------------------------------

def test_display_str_pct():
    assert parse_budget("25").display_str == "25% of BF16"

def test_display_str_bpw():
    assert parse_budget("4.5bpw").display_str == "4.5 bpw"

def test_display_str_gb():
    assert parse_budget("16GB").display_str == "16 GB"


# ---------------------------------------------------------------------------
# budget_from_toml
# ---------------------------------------------------------------------------

def test_toml_int_becomes_pct():
    b = budget_from_toml(25)
    assert b.form == "pct"
    assert b.value == 25.0

def test_toml_float_becomes_pct():
    b = budget_from_toml(33.3)
    assert b.form == "pct"
    assert b.value == pytest.approx(33.3)

def test_toml_str_routed_through_parse():
    b = budget_from_toml("4.5bpw")
    assert b.form == "bpw"
    assert b.value == pytest.approx(4.5)

def test_toml_str_gb():
    b = budget_from_toml("16GB")
    assert b.form == "gb"
    assert b.value == pytest.approx(16.0)

def test_toml_invalid_str_raises():
    with pytest.raises(ValueError):
        budget_from_toml("99xyz")

def test_toml_wrong_type_raises():
    with pytest.raises(ValueError):
        budget_from_toml([25])


# ---------------------------------------------------------------------------
# check_mixed_units
# ---------------------------------------------------------------------------

def test_check_mixed_units_homogeneous_pct():
    specs = [parse_budget("25"), parse_budget("50%"), parse_budget("75")]
    check_mixed_units(specs)  # should not raise

def test_check_mixed_units_homogeneous_bpw():
    specs = [parse_budget("3bpw"), parse_budget("4.5bpw")]
    check_mixed_units(specs)

def test_check_mixed_units_homogeneous_gb():
    specs = [parse_budget("8GB"), parse_budget("16GB")]
    check_mixed_units(specs)

def test_check_mixed_units_single():
    check_mixed_units([parse_budget("25")])

def test_check_mixed_units_empty():
    check_mixed_units([])

@pytest.mark.parametrize("raw_specs", [
    ["25", "4.5bpw"],
    ["25", "16GB"],
    ["4.5bpw", "16GB"],
    ["25", "4.5bpw", "16GB"],
])
def test_check_mixed_units_raises(raw_specs):
    specs = [parse_budget(s) for s in raw_specs]
    with pytest.raises(ValueError, match="mixed"):
        check_mixed_units(specs)


# ---------------------------------------------------------------------------
# Filename label uniqueness — different units produce different labels
# ---------------------------------------------------------------------------

def test_different_units_produce_different_labels():
    """Two budget specs that happen to hit the same budget_gb in the pipeline
    still have distinct filename_labels, so they don't collide in filenames.
    The budget_gb unification is the pipeline's concern, not the parser's."""
    a = parse_budget("25")     # PQ25
    b = parse_budget("16GB")   # PQ16gb
    assert a.filename_label != b.filename_label

def test_equivalent_pct_labels_are_equal():
    """'25' and '25%' are the same spec."""
    assert parse_budget("25").filename_label == parse_budget("25%").filename_label
