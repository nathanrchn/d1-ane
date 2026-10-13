"""Tetris played by decisions: every landing is one sentence, and the model rates it.

    python bench/tetris.py --policy d1 --shortlist --seeds 1-10
    python bench/tetris.py --policy laya --shortlist --seeds 1-10
    python bench/tetris.py --policy heuristic --shortlist --seeds 1-10

The harness (d1ane/tetris.py) is FluidUse's, so `--policy laya` (their published Core ML bucket, d1ane/laya.py)
and `--policy d1` play the same pieces and are filtered the same way. By default each model gets the wording and
question that were tuned to it: laya "graded" sentences and "Is this a clean placement?", d1 "calibrated" sentences
and a five-level rating. --style and --ask swap them.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from d1ane import BUILD, LAYA  # noqa: E402
from d1ane.tetris import ASKS, STYLES, Game, describe, heuristic_policy, judge, random_policy, shortlist  # noqa: E402


def seeds_of(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out += range(int(a), int(b or a) + 1)
    return out


def scorer(a):
    """states -> (a value per state, model calls made)."""
    if a.policy == "laya":
        from d1ane.laya import Laya

        return judge(Laya(a.laya, 128, "ane"), "laya", a.ask)
    from d1ane.runtime import Decider

    return judge(Decider(a.build, compute=a.compute, widths=a.widths, log=lambda *x: None), "d1", a.ask)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="d1", choices=("d1", "laya", "heuristic", "random"))
    ap.add_argument("--build", type=Path, default=BUILD)
    ap.add_argument("--laya", type=Path, default=LAYA, help="laya's Core ML bucket and tokenizer (d1ane setup --laya)")
    ap.add_argument("--compute", default="ane")
    ap.add_argument("--widths", type=lambda s: [int(w) for w in s.split(",")], help="entries to load (default: all)")
    ap.add_argument("--style", choices=STYLES, help="wording of the sentence (default: graded for laya, calibrated for d1)")
    ap.add_argument("--ask", choices=ASKS, help="the question (default: clean for laya, rating for d1)")
    ap.add_argument("--shortlist", action="store_true", help="withhold landings that bury a cell when a clean one exists")
    ap.add_argument("--seeds", default="1-10")
    ap.add_argument("--pieces", type=int, default=5000, help="piece cap per game (0: none)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    a.style = a.style or ("graded" if a.policy == "laya" else "calibrated")
    a.ask = a.ask or ("clean" if a.policy == "laya" else "rating")
    score = scorer(a) if a.policy in ("d1", "laya") else None
    games = []
    for seed in seeds_of(a.seeds):
        game, ms, calls, landings, agree, unforced = Game(seed), [], 0, 0, 0, 0
        rand = random_policy(seed)
        t0 = time.perf_counter()
        while (not a.pieces or game.pieces < a.pieces) and (piece := game.spawn()):
            cands = game.landings(piece)
            if a.shortlist:
                cands = shortlist(cands)
            if not cands:
                break
            if score is None:
                pick = heuristic_policy(game, piece, cands) if a.policy == "heuristic" else rand(game, piece, cands)
            else:
                t1 = time.perf_counter()
                probs, n = score([describe(c, piece, a.style) for c in cands])
                ms.append(1e3 * (time.perf_counter() - t1))
                calls, landings = calls + n, landings + len(cands)
                pick = cands[max(range(len(cands)), key=probs.__getitem__)]
                if len(cands) > 1:
                    unforced += 1
                    agree += pick is heuristic_policy(game, piece, cands)
            game.apply(pick)
        dt = time.perf_counter() - t0
        g = {"seed": seed, "pieces": game.pieces, "lines": game.lines, "topped_out": game.over, "seconds": round(dt, 2)}
        if ms:
            g.update(landings=landings, calls=calls, piece_ms=round(statistics.median(ms), 2),
                     landing_ms=round(sum(ms) / landings, 3), decisions_per_min=round(60 * landings / dt),
                     heuristic_agreement=round(agree / max(1, unforced), 3))
        games.append(g)
        print(json.dumps(g), flush=True)
    summary = {"policy": a.policy, "style": a.style, "ask": a.ask, "shortlist": a.shortlist, "cap": a.pieces,
               "build": str(a.build) if a.policy == "d1" else None, "games": len(games),
               "mean_pieces": round(statistics.mean(g["pieces"] for g in games), 1),
               "mean_lines": round(statistics.mean(g["lines"] for g in games), 1),
               "at_cap": sum(g["pieces"] >= a.pieces for g in games) if a.pieces else 0}
    if score:
        summary.update(landing_ms=round(sum(g["landing_ms"] * g["landings"] for g in games) / sum(g["landings"] for g in games), 3),
                       piece_ms=round(statistics.median(g["piece_ms"] for g in games), 2),
                       decisions_per_min=round(statistics.median(g["decisions_per_min"] for g in games)),
                       landings_per_piece=round(sum(g["landings"] for g in games) / sum(g["pieces"] for g in games), 2),
                       heuristic_agreement=round(statistics.mean(g["heuristic_agreement"] for g in games), 3))
    print(json.dumps(summary))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"summary": summary, "games": games}, indent=1))


if __name__ == "__main__":
    main()
