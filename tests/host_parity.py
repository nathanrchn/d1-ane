"""The rewritten graph in PyTorch against the checkpoint: fp32 checks the rewrite, fp16 the precision.

    python tests/host_parity.py --model models/d1-omni-600M --build build/host-check
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from d1ane.convert import write_host_files  # noqa: E402
from d1ane.reference import Reference  # noqa: E402
from d1ane.runtime import Decider, TorchBackend  # noqa: E402
from tests.cases import CASES  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", type=Path, required=True)
ap.add_argument("--build", type=Path, required=True)
ap.add_argument("--silu", default="tanh")
a = ap.parse_args()
if not (a.build / "embed_tokens_fp16.npy").exists():
    write_host_files(a.model, a.build)
ref = Reference(a.model)
widths = [64, 128, 256, 512, 1024]
backends = {"graph fp32 cpu": TorchBackend(a.model, widths, "float32", "cpu", a.silu),
            "graph fp16 mps": TorchBackend(a.model, widths, "float16", "mps", a.silu)}
deciders = {k: Decider(a.build, backend=b) for k, b in backends.items()}
d0 = next(iter(deciders.values()))
for state, q in CASES:                                  # tokenizer adapter against transformers
    qq = d0.P.as_question(q)
    mine = d0.P.encode(d0.tok, state, qq, 16384)
    theirs = d0.P.encode(ref.model.tokenizer, state, qq, 16384)
    assert mine == theirs, "tokenizer mismatch"
print("tokenizer: identical ids on every case")
for name, d in deciders.items():
    worst, flips = 0.0, 0
    for state, q in CASES:
        p_ref = np.array(ref.probabilities(state, q))
        p, t = d.probabilities(state, q)
        p = np.array(p)
        worst = max(worst, float(np.abs(p - p_ref).max()))
        flips += int(p.argmax() != p_ref.argmax())
        print(f"  {name}: tokens={t['tokens']:4d} width={t['width']:4d} max|dp|={np.abs(p - p_ref).max():.2e} "
              f"ref={np.round(p_ref, 3).tolist()}")
    print(f"{name}: worst max|dp| {worst:.2e}, argmax flips {flips}/{len(CASES)}")
