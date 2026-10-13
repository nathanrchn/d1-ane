"""d1 against laya per decision, inside the Tetris game, in alternating rounds on one machine.

    python bench/tetris_speed.py --out bench/results/tetris_speed.json [--fp16 build/d1-omni-600M-fp16]

A round is one model playing `--pieces` pieces of one seed alone (harness filter on, its own wording and question),
then the other model doing the same. Rounds alternate, so a busy machine slows both alike. A decision is one landing
rated: tokenizer, host arrays, model call, readout.
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
from d1ane import BUILD, LAYA  # noqa: E402
from d1ane.laya import Laya  # noqa: E402
from d1ane.runtime import Decider  # noqa: E402
from d1ane.tetris import Game, describe, judge, shortlist  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", type=Path, default=BUILD)
    ap.add_argument("--fp16", type=Path, help="also time this fp16 d1 build")
    ap.add_argument("--laya", type=Path, default=LAYA)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--pieces", type=int, default=150)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    players = {"d1 int8": (judge(Decider(a.build, widths=[64, 128], log=lambda *x: None), "d1", "rating"), "calibrated")}
    if a.fp16:
        players["d1 fp16"] = (judge(Decider(a.fp16, widths=[64, 128], log=lambda *x: None), "d1", "rating"), "calibrated")
    players["laya"] = (judge(Laya(a.laya, 128, "ane"), "laya", "clean"), "graded")
    ms = {name: [] for name in players}                # per decision
    rate = {name: [] for name in players}              # decisions per minute of each round, game logic included
    for seed in range(1, a.rounds + 1):
        for name, (score, style) in players.items():
            game, n, t0 = Game(seed), 0, time.perf_counter()
            while game.pieces < a.pieces and (piece := game.spawn()):
                cands = shortlist(game.landings(piece))
                if not cands:
                    break
                states = [describe(c, piece, style) for c in cands]
                t1 = time.perf_counter()
                values, _ = score(states)
                ms[name] += [1e3 * (time.perf_counter() - t1) / len(cands)] * len(cands)
                n += len(cands)
                game.apply(cands[max(range(len(cands)), key=values.__getitem__)])
            rate[name].append(60 * n / (time.perf_counter() - t0))
    out = {"rows": [{"player": name, "decisions": len(v), "median_ms": round(statistics.median(v), 3),
                     "mean_ms": round(statistics.mean(v), 3), "p95_ms": round(float(np.percentile(v, 95)), 3),
                     "decisions_per_min": round(statistics.median(rate[name]))} for name, v in ms.items()],
           "rounds": a.rounds, "pieces": a.pieces, "load_avg": [round(v, 1) for v in os.getloadavg()],
           "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    for r in out["rows"]:
        print(f"{r['player']:8s} {r['median_ms']:.2f} ms per decision (mean {r['mean_ms']:.2f}, p95 {r['p95_ms']:.2f}), "
              f"{r['decisions_per_min']:,} decisions/min over {r['decisions']} decisions")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
