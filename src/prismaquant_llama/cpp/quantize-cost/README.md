# quantize-cost — per-(tensor, format) MSE measurement tool

This is a C++ build target that compiles to the `llama-quantize-cost`
binary. Stage E (cost measurement) of the prismaquant-llama pipeline
requires this binary. **You'll need to compile it against your own
llama.cpp fork** — it can't be built standalone because it links
against `ggml` and `llama.cpp`'s quantization kernels.

## Why this lives here

It's a generally useful llama.cpp tool: per-tensor MSE without writing
a GGUF, useful for any mixed-precision allocator, sensitivity analysis,
or quant-aware pruning work. We ship the source in prismaquant-llama
because the prismaquant pipeline depends on its CSV output, but the
build belongs in your llama.cpp tree.

## How to integrate into your llama.cpp fork

### Option 1: Drop-in (works for any fork — mainline, frankenturbo2, ikllama, etc.)

```bash
# Copy source files into your fork
cp /path/to/prismaquant-llama/src/pipeline/cpp/quantize-cost \
   /path/to/your/llama.cpp/tools/quantize-cost -r

# Add to your fork's tools CMakeLists.txt (typically at tools/CMakeLists.txt
# or in the root CMakeLists.txt, alongside other tool subdirectories):
echo 'add_subdirectory(quantize-cost)' >> /path/to/your/llama.cpp/tools/CMakeLists.txt

# Reconfigure + build (ensure -DGGML_BUILD_TOOLS=ON is set)
cd /path/to/your/llama.cpp/build
cmake -DGGML_BUILD_TOOLS=ON ..
cmake --build . --target llama-quantize-cost
```

The resulting binary lives at `your/llama.cpp/build/bin/llama-quantize-cost`.

### Option 2: If your fork already has it

Several prismaquant-enabled forks already ship this tool (e.g.,
[`jimbothigpen/frankenturbo2`](https://github.com/jimbothigpen/frankenturbo2)
at `tools/quantize-cost/`). If you're using one of those, you don't
need to copy anything — just rebuild that fork with
`-DGGML_BUILD_TOOLS=ON`.

## Verifying the binary works

```bash
your/llama.cpp/build/bin/llama-quantize-cost --help
# expected: usage line + flag descriptions
```

If you get "command not found", the build didn't produce the binary.
Check that `add_subdirectory(quantize-cost)` is wired in the parent
CMakeLists and that `-DGGML_BUILD_TOOLS=ON` was passed to cmake.

## Upstream status

This tool is a prime candidate for a PR to
[`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp). If/when
that lands, it'll ship in mainline llama.cpp without needing this
shim. Until then, this is the canonical source.

## Files

- `quantize-cost.cpp` — the tool source (~470 lines)
- `CMakeLists.txt` — minimal subdirectory CMake config
