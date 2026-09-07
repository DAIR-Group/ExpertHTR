#!/usr/bin/env python3
"""Decode one or more page records and verify the current MTHtr layout."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from PIL import Image

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_root", type=Path)
    ap.add_argument("--split", default="test", choices=("train", "valid", "test"))
    ap.add_argument("--count", type=int, default=1)
    args = ap.parse_args()
    root = args.data_root.expanduser().resolve()
    jsonl = root / "page_multitask" / f"{args.split}.jsonl"
    if not jsonl.exists(): raise SystemExit(f"missing split: {jsonl}")
    rows=[]
    for line in jsonl.open(encoding="utf-8"):
        if line.strip(): rows.append(json.loads(line))
        if len(rows) >= args.count: break
    if not rows: raise SystemExit("empty split")
    for row in rows:
        rel = Path(row["images"][0]); path = root / rel
        if not path.is_file(): raise SystemExit(f"missing image: {path}")
        with Image.open(path) as im: im.verify()
        print(f"PASS id={row.get('id')} image={rel} size={Image.open(path).size}")
    return 0
if __name__ == "__main__": raise SystemExit(main())
