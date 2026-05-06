"""
Directory layout for prismaquant-llama runs.

Layout under cfg.base:

    base/
    ├── _shared/                  reusable across runs
    │   ├── hf-cache/             downloaded HuggingFace safetensors
    │   ├── bf16/                 BF16 GGUFs (one per model)
    │   ├── gguf-cache/           GGUFs downloaded from URL (calibrate)
    │   ├── imatrix-cache/        imatrix files (one per model+corpus+chunks)
    │   └── probe/                prismaquant Hessian probe artifacts
    ├── ppl-corpus/               downloaded PPL corpora (purge candidates)
    ├── imatrix-corpus/           downloaded imatrix corpora (purge candidates)
    ├── calibration/
    │   ├── system.json
    │   └── models/<model>.json
    ├── ggufs/                    final prismaquant GGUFs
    └── work/<run>/               per-run scratch
        ├── costs/
        ├── recipes/
        └── logs/
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class Layout:
    base: Path
    shared: Path
    hf_cache: Path
    bf16_dir: Path
    gguf_cache: Path
    imatrix_cache: Path
    probe_dir: Path
    ppl_corpus_dir: Path
    imatrix_corpus_dir: Path
    calibration_dir: Path
    calibration_models_dir: Path
    ggufs: Path
    work: Path
    costs_dir: Path
    recipes_dir: Path
    logs_dir: Path
    run_label: str

    @classmethod
    def for_run(cls, base: Path, model_name: str,
                run_timestamp: Optional[str] = None) -> "Layout":
        base = base.expanduser().resolve()
        ts = run_timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
        run_label = f"{model_name}-{ts}"
        shared = base / "_shared"
        work = base / "work" / run_label
        return cls(
            base=base,
            shared=shared,
            hf_cache=shared / "hf-cache",
            bf16_dir=shared / "bf16",
            gguf_cache=shared / "gguf-cache",
            imatrix_cache=shared / "imatrix-cache",
            probe_dir=shared / "probe",
            ppl_corpus_dir=base / "ppl-corpus",
            imatrix_corpus_dir=base / "imatrix-corpus",
            calibration_dir=base / "calibration",
            calibration_models_dir=base / "calibration" / "models",
            ggufs=base / "ggufs",
            work=work,
            costs_dir=work / "costs",
            recipes_dir=work / "recipes",
            logs_dir=work / "logs",
            run_label=run_label,
        )

    def make(self) -> None:
        for d in (self.base, self.shared, self.hf_cache, self.bf16_dir,
                  self.gguf_cache, self.imatrix_cache, self.probe_dir,
                  self.ppl_corpus_dir, self.imatrix_corpus_dir,
                  self.calibration_dir, self.calibration_models_dir,
                  self.ggufs, self.work, self.costs_dir,
                  self.recipes_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)

    def imatrix_cache_path(self, model_sha: str, corpus_sha: str, chunks: int) -> Path:
        key = f"{model_sha[:12]}__{corpus_sha[:12]}__c{chunks}.imatrix.gguf"
        return self.imatrix_cache / key

    def gguf_output_path(self, model_name: str, budget_pct: int,
                         priority: str) -> Path:
        return self.ggufs / f"{model_name}-PQ{budget_pct}-{priority}.gguf"

    def system_calibration_path(self) -> Path:
        return self.calibration_dir / "system.json"

    def model_calibration_path(self, model_name: str) -> Path:
        return self.calibration_models_dir / f"{model_name}.json"

    def summary_lines(self) -> list[str]:
        return [
            f"  base:             {self.base}",
            f"  shared cache:     {self.shared}",
            f"  ggufs (output):   {self.ggufs}",
            f"  work scratch:     {self.work}",
        ]
