"""Export d1-omni's trunk and decision head as Core AI packages for the Apple Neural Engine.

    d1ane convert --model models/d1-omni-600M --output build/d1-omni-600M [--widths 64,128,512]
                            [--plan all|4|0-7,8-15,head] [--silu tanh|native] [--quant fp16|int8]

Writes <output>/manifest.json, stage_*.aimodel (one entry w<T> per width over one copy of the weights),
embed_tokens_fp16.npy, type_emb.npy and model/ (config, tokenizer, prompt.py). Needs the Core AI SDK Python.
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
from pathlib import Path

import numpy as np

from .graph import Core, Stage, parse_plan

MODEL_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "prompt.py", "LICENSE")
QUANT = ("fp16", "int8")


def load_weights(model: Path, prefixes: tuple[str, ...] = ("encoder.", "head.")) -> dict[str, np.ndarray]:
    from safetensors import safe_open

    with safe_open(str(model / "model.safetensors"), "np") as f:
        return {k: f.get_tensor(k) for k in f.keys() if k.startswith(prefixes)}


def build_core(model: Path, silu: str = "tanh") -> tuple[Core, dict, dict]:
    config = json.loads((model / "config.json").read_text())
    W = load_weights(model)
    return Core(W, config, silu).eval(), config, W


def write_host_files(model: Path, out: Path, W: dict | None = None) -> None:
    """What the host side of a decision needs: prompt code, tokenizer, the embedding table, the type embeddings."""
    W = W or load_weights(model, ("encoder.embed_tokens.", "head.type_emb."))
    (out / "model").mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        if (model / name).is_file():
            shutil.copy2(model / name, out / "model" / name)
    np.save(out / "embed_tokens_fp16.npy", W["encoder.embed_tokens.weight"].astype(np.float16))
    np.save(out / "type_emb.npy", W["head.type_emb.weight"].astype(np.float32))


def compress(prog, quant: str):
    """Weight compression of a converted program: "int8" stores every weight in 8 bits, linear per output channel.
    Activations stay FP16."""
    if quant == "fp16":
        return prog
    from coreai_opt.coreai_utils import CompressionGranularity, DType, quantize_weights

    return quantize_weights(prog, DType.INT8, granularity=CompressionGranularity.PER_CHANNEL)


def save_program(entries: list[tuple[str, Stage]], out: Path, quant: str = "fp16") -> float:
    """One Core AI package holding every (entry name, module); returns its size in MB."""
    import torch
    import coreai_torch
    from coreai_opt.casting import cast_to_16_bit_precision

    conv = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    for name, mod in entries:
        ep = torch.export.export(mod, mod.example(), strict=False).run_decompositions(coreai_torch.get_decomp_table())
        cast_to_16_bit_precision(ep)
        conv.add_exported_program(ep, input_names=mod.input_names(), output_names=mod.output_names(),
                                  entrypoint_name=name)
    prog = conv.to_coreai()
    prog.optimize()
    prog = compress(prog, quant)
    shutil.rmtree(out, ignore_errors=True)
    prog.save_asset(out)
    del prog, conv
    gc.collect()
    return sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6


def export(model: Path, out: Path, widths: list[int], plan: str = "all", silu: str = "tanh", quant: str = "fp16",
           log=print) -> dict:
    import torch

    t0 = time.time()
    core, config, W = build_core(model, silu)
    write_host_files(model, out, W)
    del W
    core = core.to(torch.float16)
    man = {"version": "d1-coreai1", "kind": "d1-decision", "source": model.name, "quant": quant, "silu": silu,
           "plan": plan, "widths": sorted(widths), "hidden": core.D, "stages": []}
    if (out / "manifest.json").is_file():           # keep the media packages of a build whose trunk is re-exported
        old = json.loads((out / "manifest.json").read_text())
        man.update({k: old[k] for k in ("vision", "audio") if k in old})
    for si, part in enumerate(parse_plan(plan, len(core.layers))):
        t1 = time.time()
        tag = (f"L{part['layers'][0]:02d}-{part['layers'][-1]:02d}" if part["layers"] else "") + \
              ("_head" if part["head"] else "")
        dest = out / f"stage{si}_{tag.strip('_')}.aimodel"
        mods = [(f"w{T}", Stage(core, T, part["layers"], part["final"], part["head"]).eval().to(torch.float16))
                for T in sorted(widths)]
        mb = save_program(mods, dest, quant)
        man["stages"].append({"file": dest.name, "layers": part["layers"], "final": part["final"], "head": part["head"],
                              "inputs": mods[0][1].input_names(), "output": mods[0][1].output_names()[0],
                              "entries": {str(m.T): n for n, m in mods}, "mb": round(mb, 1),
                              "convert_s": round(time.time() - t1, 1)})
        (out / "manifest.json").write_text(json.dumps(man, indent=1))
        log(f"{dest.name}: {mb:.0f} MB, {time.time() - t1:.1f}s")
    man["convert_s"] = round(time.time() - t0, 1)
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    return man


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--widths", default="64,128,192,256,384,512,768,1024,1536,2048",
                    help="comma-separated sequence widths, one entry each")
    ap.add_argument("--plan", default="all", help='"all", a chunk size, or explicit "0-7,8-15,head"')
    ap.add_argument("--silu", default="tanh", choices=("tanh", "native"))
    ap.add_argument("--quant", default="fp16", choices=QUANT,
                    help="int8: 8-bit trunk and head weights, half the size and faster on short prompts")
    a = ap.parse_args(argv)
    man = export(a.model, a.output, [int(w) for w in a.widths.split(",")], a.plan, a.silu, a.quant)
    print(f"done in {man['convert_s']}s -> {a.output}")


if __name__ == "__main__":
    main()
