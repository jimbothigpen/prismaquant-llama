# Contributing to prismaquant-llama

## Vendored C++ source

The `llama-quantize-cost` tool lives canonically in
[`jimbothigpen/frankenturbo2`](https://github.com/jimbothigpen/frankenturbo2)
at `tools/quantize-cost/`. We vendor a copy at
`src/pipeline/cpp/quantize-cost/` so users who don't have frankenturbo2 on
disk can drop the source into a vanilla llama.cpp tree and build.

### Updating the vendored copy

After committing a `tools/quantize-cost/` change in frankenturbo2:

```bash
# from the prismaquant-llama root:
./tools/sync-quantize-cost.sh
# (or pass the path explicitly: ./tools/sync-quantize-cost.sh /path/to/frankenturbo2)
```

The script copies `quantize-cost.cpp` + `CMakeLists.txt` and records the
upstream SHA in `src/pipeline/cpp/quantize-cost/VENDORED-FROM.txt`. Commit
all three files together:

```bash
git add src/pipeline/cpp/quantize-cost/quantize-cost.cpp \
        src/pipeline/cpp/quantize-cost/CMakeLists.txt \
        src/pipeline/cpp/quantize-cost/VENDORED-FROM.txt
git commit -m "vendor: sync quantize-cost from frankenturbo2 <short-sha>"
```

### Drift check

To verify the vendored copy matches a frankenturbo2 checkout:

```bash
diff -q /path/to/llama-fork/tools/quantize-cost/quantize-cost.cpp \
        src/pipeline/cpp/quantize-cost/quantize-cost.cpp
```

If they differ, either:
1. frankenturbo2 has newer commits → run `./tools/sync-quantize-cost.sh` to update,
2. someone hand-edited the vendored copy → either revert and re-sync, or
   port the change upstream to frankenturbo2 first.

The `source_sha` in `VENDORED-FROM.txt` is the canonical "this vendored copy
came from frankenturbo2 commit X" record — keep it accurate.

## Other notes

(More entries will land here as the project grows.)
