#!/usr/bin/env python3
"""Assemble a trained run into a bundle the pip package can load.

`multi_accent_frontend.Frontend.from_pretrained` expects a directory (local or
a Hugging Face Hub repo) containing: config.json, src.vocab, tgt.vocab, and a
weights file. This copies those out of a training run into one clean folder,
ready to use locally or to `huggingface-cli upload`.

Usage:
    python export_bundle.py \
        --run-dir runs/frontend_edi_gam_rpx-<timestamp>-<hash> \
        --config dataset/train_config.json \
        --checkpoint best.pth.tar \
        --out bundle
"""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Sequence
from pathlib import Path


def find_checkpoint(run_dir: Path, name: str | None) -> Path:
    if name:
        p = run_dir / name
        if not p.is_file():
            raise SystemExit(f"checkpoint not found: {p}")
        return p
    # Prefer best.pth.tar, else the latest step_*.pth.tar by mtime.
    best = run_dir / "best.pth.tar"
    if best.is_file():
        return best
    candidates = sorted(run_dir.glob("*.pth.tar"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise SystemExit(f"no *.pth.tar checkpoints found in {run_dir}")
    return candidates[-1]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="Training experiment folder created under output_path.")
    ap.add_argument("--config", required=True, help="The train_config.json used for training.")
    ap.add_argument("--checkpoint", default=None, help="Checkpoint filename inside run-dir (default: best or latest).")
    ap.add_argument("--out", required=True, help="Output bundle directory to create.")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # config.json (inference reads model hyper-params + accent list from 'data')
    shutil.copyfile(args.config, out / "config.json")

    # vocab files written by train.py into <run_dir>/vocab/
    for vocab_name in ("src.vocab", "tgt.vocab"):
        src = run_dir / "vocab" / vocab_name
        if not src.is_file():
            raise SystemExit(f"missing {src}; expected train.py to have written it")
        shutil.copyfile(src, out / vocab_name)

    ckpt = find_checkpoint(run_dir, args.checkpoint)
    shutil.copyfile(ckpt, out / ckpt.name)

    print(f"bundle written to {out}/")
    print("  ", ", ".join(p.name for p in sorted(out.iterdir())))
    print("\nUse it with:")
    print("    from multi_accent_frontend import Frontend")
    print(f"    fe = Frontend.from_pretrained({str(out)!r})")
    print("Or push to the Hub:")
    print(f"    huggingface-cli upload <your-org>/<repo> {out} .")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
