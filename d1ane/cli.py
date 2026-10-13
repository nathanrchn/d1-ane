"""d1ane: d1-omni-600M on the Apple Neural Engine.

    d1ane setup     download the checkpoint, convert it and compile it for this Mac (once, about ten minutes)
    d1ane tetris    d1 plays Tetris in the browser, next to laya or a heuristic
    d1ane serve     the decision server (POST /v1/systemone) and its page

    d1ane convert | vision | audio | compile     the steps of setup, one at a time (same as python -m d1ane.<step>)

Every command takes --help.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from . import BUILD, LAYA, MODEL

CHECKPOINT, LAYA_REPO = "LiquidAI/d1-omni-600M", "FluidInference/laya-coreml"
LAYA_FILES = ["laya_multilingual_fp16_L128_options32.mlmodelc/*", "tokenizer.json"]
STEPS = {"convert": "convert", "vision": "vision", "audio": "audio", "compile": "compile", "serve": "serve", "tetris": "play"}


def setup(argv=None):
    ap = argparse.ArgumentParser(prog="d1ane setup", description="Download, convert and compile d1-omni-600M.")
    ap.add_argument("--model", type=Path, default=MODEL, help="where the checkpoint is, or goes (2.3 GB)")
    ap.add_argument("--build", type=Path, default=BUILD)
    ap.add_argument("--quant", default="int8", choices=("int8", "fp16"),
                    help="int8 (default): half the size, 1.7 times faster on short prompts; fp16: the reference precision")
    ap.add_argument("--text-only", action="store_true", help="skip the vision and audio encoders")
    ap.add_argument("--laya", action="store_true",
                    help=f"also download laya, the model d1 plays Tetris against (650 MB, into {LAYA})")
    a = ap.parse_args(argv)
    from huggingface_hub import snapshot_download

    if not (a.model / "model.safetensors").is_file():
        print(f"[1/3] downloading {CHECKPOINT} into {a.model}", flush=True)
        snapshot_download(CHECKPOINT, local_dir=a.model)
    else:
        print(f"[1/3] checkpoint found in {a.model}", flush=True)
    if a.laya and not (LAYA / "tokenizer.json").is_file():
        print(f"      downloading {LAYA_REPO} into {LAYA}", flush=True)
        snapshot_download(LAYA_REPO, local_dir=LAYA, allow_patterns=LAYA_FILES)
    print(f"[2/3] converting to {a.build} ({a.quant} weights)", flush=True)
    paths = ["--model", str(a.model), "--output", str(a.build)]
    run("convert", paths + ["--quant", a.quant])
    if not a.text_only:
        run("vision", paths)
        run("audio", paths)
    print("[3/3] compiling for this Mac's Neural Engine (a few minutes the first time)", flush=True)
    status = run("compile", ["--build", str(a.build)])
    print("\nready: d1ane tetris   or   d1ane serve", flush=True)
    return status


def run(step: str, argv: list[str]):
    return importlib.import_module(f"d1ane.{STEPS[step]}").main(argv)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    command, rest = argv[0], argv[1:]
    if command == "setup":
        return setup(rest)
    if command not in STEPS:
        print(f"unknown command {command!r}\n{__doc__}", file=sys.stderr)
        return 2
    return run(command, rest)


if __name__ == "__main__":
    sys.exit(main())
