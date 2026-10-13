"""A build on the ANE (or gpu / cpu) against the checkpoint in PyTorch FP32, with warm latency.

    python tests/ane_smoke.py --model models/d1-omni-600M --build build/d1-omni-600M [--compute ane] [--reps 20]
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from d1ane.placement import audit  # noqa: E402
from d1ane.runtime import Decider  # noqa: E402
from tests.cases import CASES  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", type=Path)
ap.add_argument("--build", type=Path, required=True)
ap.add_argument("--compute", default="ane")
ap.add_argument("--reps", type=int, default=20)
a = ap.parse_args()
t0 = time.time()
d = Decider(a.build, a.compute)
print(f"load+compile {time.time() - t0:.1f}s widths={d.backend.widths}")
if a.compute == "ane":
    rep = audit(a.build)
    print("placement:", rep["status"], json.dumps({k: (v["status"], v["modes"], v["entries"]) for k, v in rep["stages"].items()}))
ref = None
if a.model:
    from d1ane.reference import Reference
    ref = Reference(a.model)
worst, flips, n = 0.0, 0, 0
for state, q in CASES:
    try:
        p, t = d.probabilities(state, q)
    except ValueError as e:
        print("  skip:", e)
        continue
    ms = []
    for _ in range(a.reps):
        t1 = time.perf_counter()
        d.probabilities(state, q)
        ms.append(1e3 * (time.perf_counter() - t1))
    line = f"  tokens={t['tokens']:4d} width={t['width']:4d} total={statistics.median(ms):7.2f} ms (run {t['run_ms']:.2f}, fill {t['fill_ms']:.2f}, tok {t['tokenize_ms']:.2f})"
    if ref:
        p_ref = np.array(ref.probabilities(state, q))
        dp = float(np.abs(np.array(p) - p_ref).max())
        worst, flips, n = max(worst, dp), flips + int(np.argmax(p) != p_ref.argmax()), n + 1
        line += f" max|dp|={dp:.2e}"
    print(line + f" p={np.round(p, 3).tolist()}")
if ref:
    print(f"worst max|dp| {worst:.2e}, argmax flips {flips}/{n}")
