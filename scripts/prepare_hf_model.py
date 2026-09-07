#!/usr/bin/env python3
"""Copy only runtime-required files from a selected ExpertHTR checkpoint."""
from __future__ import annotations
import argparse, shutil
from pathlib import Path

REQUIRED = ("full_model_state.pt", "config.json", "tokenizer.json", "tokenizer_config.json",
            "processor_config.json", "chat_template.jinja", "checkpoint_manifest.json", "metadata.json", "metrics.json", "CHECKPOINT_COMPLETE")
def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("source", type=Path); ap.add_argument("destination", type=Path); args=ap.parse_args()
    src=args.source.expanduser().resolve(); dst=args.destination.expanduser().resolve(); dst.mkdir(parents=True, exist_ok=True)
    missing=[]
    for name in REQUIRED:
        p=src/name
        if not p.is_file(): missing.append(name); continue
        shutil.copy2(p, dst/name)
    if missing: raise SystemExit("Missing required checkpoint files: " + ", ".join(missing))
    print(f"Prepared {dst}; copied {len(REQUIRED)} runtime files. Training state/predictions were excluded.")
if __name__ == "__main__": raise SystemExit(main())
