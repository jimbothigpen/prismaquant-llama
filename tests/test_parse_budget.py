"""Tests for the budget input parser (budget.py).

Covers:
- parse_budget: all three forms, case-insensitivity, whitespace tolerance,
  validation errors (zero, negative, bpw>16, unknown suffix)
- budget_from_toml: int → pct, float → pct, str → parse_budget
- check_mixed_units: homogeneous passes, mixed raises
- format_bpw_label: 2-decimal-trim canonical bpw label formatter
- BudgetSpec.filename_label: v2 bpw form returns PQ<bpw>; pct/gb retain
  v1-style labels for --budget glob-filtering
- Cache-key and filename-label invariance: equivalent bpw values produce
  the same label via format_bpw_label; the same target_bpw derived from
  different input forms produces the same filename label.
"""
from __future__ import annotations

import pytest

from prismaquant_llama.budget import (
    BudgetSpec,
    budget_from_toml,
    check_mixed_units,
    format_bpw_label,
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
# format_bpw_label — canonical 2-decimal-trim bpw formatter
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bpw,expected", [
    (3.5,     "3.5"),
    (4.0,     "4"),
    (4.85,    "4.85"),
    (3.4732,  "3.47"),
    # edge cases
    (0.5,     "0.5"),
    (10.0,    "10"),
    (16.00001, "16"),
    (4.999,   "5"),    # rounds up at 3rd decimal
    (3.515,   "3.52"), # rounds up (3.505 has fp representation issues)
    (2.75,    "2.75"),
    (3.0,     "3"),
    (4.5,     "4.5"),
])
def test_format_bpw_label(bpw, expected):
    assert format_bpw_label(bpw) == expected


# ---------------------------------------------------------------------------
# BudgetSpec.filename_label — v2: bpw form uses canonical bpw label
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec,expected_label", [
    # bpw form: v2 canonical label
    ("4.5bpw",  "PQ4.5"),
    ("3bpw",    "PQ3"),
    ("2.75bpw", "PQ2.75"),
    ("4bpw",    "PQ4"),
    # pct form: v1-style label (used only for --budget glob-filter, not pipeline filenames)
    ("25",      "PQ25"),
    ("25%",     "PQ25"),
    # gb form: v1-style label (same caveat)
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
# format_bpw_label invariance — same bpw value → same label (cache-key safety)
# ---------------------------------------------------------------------------

def test_format_bpw_label_is_deterministic():
    """The same float always maps to the same label."""
    assert format_bpw_label(4.5) == format_bpw_label(4.5)
    assert format_bpw_label(4.0) == format_bpw_label(4.0)

def test_bpw_spec_filename_label_matches_format_bpw_label():
    """BudgetSpec.filename_label for bpw form is PQ + format_bpw_label."""
    for bpw_str, bpw_val in [("4.5bpw", 4.5), ("3bpw", 3.0), ("2.75bpw", 2.75)]:
        b = parse_budget(bpw_str)
        assert b.filename_label == f"PQ{format_bpw_label(bpw_val)}"

def test_equivalent_target_bpw_produces_same_label():
    """Two pipeline runs that derive the same target_bpw produce the same
    filename label, regardless of input form. Simulated here by checking
    format_bpw_label on equal floats."""
    derived_bpw_from_pct = 4.5      # hypothetical: 25% of some model = 4.5 bpw
    explicit_bpw = 4.5              # user passed --budget 4.5bpw
    assert format_bpw_label(derived_bpw_from_pct) == format_bpw_label(explicit_bpw)
    assert f"PQ{format_bpw_label(derived_bpw_from_pct)}" == parse_budget("4.5bpw").filename_label

def test_pct_form_label_unchanged():
    """pct form still returns v1-style PQ<N> label for --budget glob-filtering."""
    assert parse_budget("25").filename_label == "PQ25"
    assert parse_budget("25%").filename_label == parse_budget("25").filename_label

def test_gb_form_label_unchanged():
    """gb form retains v1-style label for --budget glob-filtering."""
    assert parse_budget("16GB").filename_label == "PQ16gb"
