"""FP16 against int8 weights on the Neural Engine, width by width, in alternating rounds.

    python bench/quant_latency.py --out bench/results/quant_ane.json

Both builds are loaded in one process and timed in turns (a round is `--calls` forwards of one build at one width), so
a busy machine slows them alike. One forward: embedding rows in, scores out, no tokenizer. The file keeps the median
and the 10th percentile: on a busy machine the median moves with the load, the 10th percentile much less.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from d1ane.runtime import Decider  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fp16", type=Path, default=Path("build/d1-omni-600M-fp16"))
    ap.add_argument("--int8", type=Path, default=Path("build/d1-omni-600M"))
    ap.add_argument("--widths", type=lambda s: [int(w) for w in s.split(",")], default=[64, 128, 192, 256, 384, 512])
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--calls", type=int, default=40)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    paths = {"fp16": a.fp16, "int8": a.int8}
    builds = {name: Decider(path, widths=a.widths, log=lambda *x: None) for name, path in paths.items()}
    rng = np.random.default_rng(0)
    rows = []
    for T in a.widths:
        emb = np.asarray(builds["fp16"].emb[rng.integers(100, 60000, T)])
        ms = {name: [] for name in builds}
        for _ in range(a.rounds):
            for name, d in builds.items():
                qt = d.type_emb[0]
                for _ in range(5):
                    d.backend.scores(emb, 0, qt)
                for _ in range(a.calls):
                    ms[name].append(d.backend.scores(emb, 0, qt)[1]["run_ms"])
        row = {"width": T, **{f"{n}_ms": round(statistics.median(v), 3) for n, v in ms.items()},
               **{f"{n}_p10_ms": round(float(np.percentile(v, 10)), 3) for n, v in ms.items()},
               **{f"{n}_p95_ms": round(float(np.percentile(v, 95)), 3) for n, v in ms.items()}}
        rows.append(row)
        print(f"w{T:<5d} " + "   ".join(f"{n} {row[n + '_ms']:6.2f} (p10 {row[n + '_p10_ms']:6.2f})" for n in ms), flush=True)
    size = {n: round(sum(f.stat().st_size for f in (b / json.loads((b / "manifest.json").read_text())["stages"][0]["file"]).rglob("*")
                         if f.is_file()) / 1e6) for n, b in paths.items()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"rows": rows, "package_mb": size, "rounds": a.rounds, "calls": a.calls,
                                 "load_avg": [round(v, 1) for v in os.getloadavg()],
                                 "time": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))


if __name__ == "__main__":
    main()
