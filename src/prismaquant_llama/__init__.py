"""prismaquant-llama — Bayesian per-tensor mixed-precision allocation for llama.cpp / GGUF."""
from pathlib import Path

__version__ = "0.2.0a1"


def quantize_cost_source_path() -> Path:
    """Return the path to the bundled `llama-quantize-cost` C++ source.

    Users build this against their llama.cpp tree to enable Stage E. After
    `pip install`, the source ships inside the package — copy this directory
    into your llama.cpp's `tools/quantize-cost/` and rebuild.

    Example:
        $ python -c "import prismaquant_llama; \\
                     print(prismaquant_llama.quantize_cost_source_path())"
        /usr/lib/python3.x/site-packages/prismaquant_llama/cpp/quantize-cost
    """
    return Path(__file__).parent / "cpp" / "quantize-cost"
