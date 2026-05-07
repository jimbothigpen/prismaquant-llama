# Contributing to prismaquant-llama

## External `llama-quantize-cost` dependency

The `llama-quantize-cost` tool is maintained as a separate repo at
[`jimbothigpen/llama-quantize-cost`](https://github.com/jimbothigpen/llama-quantize-cost).
prismaquant-llama does **not** vendor it — users clone it into their
llama.cpp tree's `tools/` directory and build the target there. See
the README of either repo for the build steps.

If you need to change `quantize-cost.cpp`, do so directly in
`jimbothigpen/llama-quantize-cost`. There's no sync workflow on the
prismaquant-llama side anymore.

## Other notes

(More entries will land here as the project grows.)
