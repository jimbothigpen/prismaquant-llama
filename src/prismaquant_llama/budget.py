"""Budget specification — parse and convert multi-unit budget inputs.

Users may express a quantization budget in three forms:
  - Percentage of BF16: "25" or "25%" (back-compat — bare number is %, not bpw)
  - Average bits-per-weight: "4.5bpw"
  - Absolute gigabytes: "16GB"

All three convert to the internal ``budget_gb`` representation before the
allocator runs. The ``budget_gb`` field in recipe.json remains the canonical
representation; the raw user string is preserved alongside it as
``budget_input`` for traceability.

Conversion by form:
  percent:  budget_gb = bf16_size_gb * pct / 100    (requires BF16 size)
  GB:       budget_gb = value                        (direct)
  bpw:      budget_gb = (pinned_bytes + bpw * unpinned_params / 8) / 1024³
            (requires costs.csv for allocator-domain parameter counts;
             see pipeline_runner._bpw_budget_gb for the implementation)

Bpw semantics — important nuance:
  bpw is computed over the *unpinned allocator domain* (the linear tensors
  the knapsack solve has freedom over). Pinned tensors (e.g. output.weight @
  Q6_K, token_embd.weight @ Q8_0) contribute their fixed sizes on top of the
  target. This means ``--budget 4.5bpw`` produces a recipe whose
  *unpinned-domain bytes ÷ unpinned-domain parameter-count × 8 ≈ 4.5*, and
  the final GGUF's on-disk size (which includes vocab, embeddings, and the
  pinned tensors at their high-precision formats) will be larger.
"""

from __future__ import annotations
import re
from dataclasses import dataclass


_BPW_RE = re.compile(r"^\s*([-+]?[0-9]*\.?[0-9]+)\s*bpw\s*$", re.IGNORECASE)
_GB_RE  = re.compile(r"^\s*([-+]?[0-9]*\.?[0-9]+)\s*gb\s*$",  re.IGNORECASE)
_PCT_RE = re.compile(r"^\s*([-+]?[0-9]*\.?[0-9]+)\s*%?\s*$")


@dataclass(frozen=True)
class BudgetSpec:
    """Parsed budget: form + numeric value + original user string."""
    form: str            # "pct" | "bpw" | "gb"
    value: float         # the numeric part
    original_input: str  # raw user string — written to recipe.json as budget_input

    @property
    def filename_label(self) -> str:
        """Return the ``PQ<label>`` fragment used in filenames.

        Examples:
          25%   → "PQ25"
          4.5bpw → "PQ4p5bpw"    (. → p for filesystem cleanliness)
          4bpw   → "PQ4bpw"
          16GB   → "PQ16gb"       (lowercase suffix)
          4.5GB  → "PQ4p5gb"
        """
        if self.form == "pct":
            return f"PQ{int(self.value)}"
        v = f"{self.value:g}"  # removes trailing zeros: 4.50 → '4.5', 16.0 → '16'
        v = v.replace(".", "p")
        if self.form == "bpw":
            return f"PQ{v}bpw"
        return f"PQ{v}gb"

    @property
    def display_str(self) -> str:
        """Human-readable budget description for prompts and help text."""
        if self.form == "pct":
            return f"{int(self.value)}% of BF16"
        if self.form == "bpw":
            return f"{self.value:g} bpw"
        return f"{self.value:g} GB"


def parse_budget(spec: str) -> BudgetSpec:
    """Parse a budget string into a typed :class:`BudgetSpec`.

    Grammar (case-insensitive suffix; whitespace between number and suffix ok):
      ``25`` or ``25%``  → percent of BF16 (back-compat; bare number is %, not bpw)
      ``4.5bpw``         → average bits-per-weight over unpinned allocator domain
      ``16GB``           → absolute gigabytes

    Raises:
      ValueError: empty string, negative, zero, unknown suffix, or bpw > 16.
      Note: pct > 100 is accepted (niche pseudo-upscaling experiments); callers
      should emit a warning when appropriate.
    """
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError("budget must not be empty")

    m = _BPW_RE.match(spec)
    if m:
        val = float(m.group(1))
        if val == 0:
            raise ValueError(f"budget must be non-zero; got {spec!r}")
        if val < 0:
            raise ValueError(f"budget must be positive (negative not allowed); "
                             f"got {spec!r}")
        if val > 16:
            raise ValueError(
                f"bpw must be ≤ 16 (BF16 ceiling of 16 bits/weight); "
                f"got {val:g}")
        return BudgetSpec(form="bpw", value=val, original_input=spec)

    m = _GB_RE.match(spec)
    if m:
        val = float(m.group(1))
        if val == 0:
            raise ValueError(f"budget must be non-zero; got {spec!r}")
        if val < 0:
            raise ValueError(f"budget must be positive (negative not allowed); "
                             f"got {spec!r}")
        return BudgetSpec(form="gb", value=val, original_input=spec)

    m = _PCT_RE.match(spec)
    if m:
        val = float(m.group(1))
        if val == 0:
            raise ValueError(f"budget must be non-zero; got {spec!r}")
        if val < 0:
            raise ValueError(f"budget must be positive (negative not allowed); "
                             f"got {spec!r}")
        return BudgetSpec(form="pct", value=val, original_input=spec)

    raise ValueError(
        f"unrecognized budget format: {spec!r}. "
        f"Accepted forms: '25' or '25%' (percent of BF16), "
        f"'4.5bpw' (bits per weight), '16GB' (gigabytes).")


def budget_from_toml(raw) -> BudgetSpec:
    """Convert a TOML budget value to a :class:`BudgetSpec`.

    TOML delivers ``budget`` as:
      - int/float (e.g. ``budget = 25``): treated as % (back-compat)
      - str (e.g. ``budget = "4.5bpw"``): forwarded to :func:`parse_budget`
    """
    if isinstance(raw, (int, float)):
        return parse_budget(f"{raw}%")
    if isinstance(raw, str):
        return parse_budget(raw)
    raise ValueError(
        f"config 'budget' must be a number (percentage) or a string like "
        f"'4.5bpw' or '16GB'; got {raw!r}")


def check_mixed_units(specs: list[BudgetSpec]) -> None:
    """Raise ValueError if ``specs`` contains mixed budget forms.

    Used by ``explore --budgets`` to ensure all elements use the same unit.
    """
    if len(specs) <= 1:
        return
    forms = [s.form for s in specs]
    if len(set(forms)) > 1:
        first_form = forms[0]
        offenders = [
            specs[i].original_input
            for i, f in enumerate(forms)
            if f != first_form
        ]
        raise ValueError(
            f"mixed budget units: first element uses {first_form!r} form "
            f"but element(s) {offenders!r} use a different form. "
            f"All budgets must be the same unit (all %, all bpw, or all GB).")
