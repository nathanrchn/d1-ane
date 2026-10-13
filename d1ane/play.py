"""d1 plays Tetris, live in the browser.

    d1ane tetris [--build build/d1-omni-600M] [--laya models/laya-coreml] [--port 8798]

Every legal landing of the current piece is described in one sentence, the model rates each sentence in one
Neural Engine call, and the best-rated landing is played. The harness is FluidUse's (see tetris.py), and with --laya
the page can run their published Core ML model next to d1 on the same pieces. Each model gets the wording and the
question that were tuned to it: d1 "calibrated" sentences and a five-level rating, laya "graded" sentences and
"Is this a clean placement?".

    GET /                page
    GET /players         what can play: d1, heuristic, random, and laya when --laya is given
    GET /stream?player=d1&seed=7&filter=1&pace=0&cap=0
                         one game as server-sent events; pace is a pause in ms after each piece
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import BUILD, LAYA
from .tetris import QUESTION, RATING, Game, describe, heuristic_policy, judge, random_policy, shortlist

PAGE = Path(__file__).with_name("tetris.html")
FREE_PACE_MS = 10       # the controls have no model to wait for: at "as fast as the model" they place 100 pieces a second


class Players:
    """The models, loaded once. One model call runs at a time: two games share the Neural Engine in turns."""

    def __init__(self, build: Path | None, laya: Path | None, compute: str = "ane", log=print):
        self.lock, self.info, self.score, self.style = threading.Lock(), {}, {}, {}
        if build is not None:
            from .runtime import Decider

            d = Decider(build, compute, log=log)
            man = d.backend.man
            self.info["d1"] = (f"{Path(man['source']).name} · {man['quant']} weights · Neural Engine · asked "
                               f"“{RATING[0]}” on a five-level scale")
            self.score["d1"], self.style["d1"] = judge(d, "d1", "rating"), "calibrated"
        if laya is not None:
            from .laya import Laya

            self.info["laya"] = f"laya-multilingual · fp16 · Core ML, CPU + Neural Engine · asked “{QUESTION}”"
            self.score["laya"], self.style["laya"] = judge(Laya(laya, 128, "ane"), "laya", "clean"), "graded"
        for name, score in self.score.items():          # load every entry before the first game
            score([describe(c, "T", self.style[name]) for c in Game(1).landings("T")])
        self.info.update(heuristic="feature-weighted heuristic · no model, shown at up to 100 pieces a second",
                         random="random landing · no model, shown at up to 100 pieces a second")

    def game(self, player: str, seed: int, filtered: bool, pace: float, cap: int):
        """Events of one game: one dict per piece, then a final one."""
        game, score, rand = Game(seed), self.score.get(player), random_policy(seed)
        pace = pace or (0 if score else FREE_PACE_MS)
        yield {"type": "start", "player": player, "info": self.info[player], "seed": seed}
        model_ms = landings = calls = 0
        t_start = time.perf_counter()
        while (not cap or game.pieces < cap) and (piece := game.spawn()):
            legal = game.landings(piece)
            cands = shortlist(legal) if filtered else legal
            if not cands:
                break
            ev = {"type": "piece", "piece": piece, "legal": len(legal), "offered": len(cands)}
            if score:
                states = [describe(c, piece, self.style[player]) for c in cands]
                with self.lock:
                    t0 = time.perf_counter()
                    probs, n = score(states)
                    ms = 1e3 * (time.perf_counter() - t0)
                order = sorted(range(len(cands)), key=lambda i: -probs[i])
                pick = cands[order[0]]          # ties: the first landing, as in the original demo
                model_ms, landings, calls = model_ms + ms, landings + len(cands), calls + n
                ev.update(ms=round(ms, 2), calls=n,
                          top=[[round(probs[i], 4), states[i], cands[i].column] for i in order[:3]])
            else:
                pick = heuristic_policy(game, piece, cands) if player == "heuristic" else rand(game, piece, cands)
            game.apply(pick)
            ev.update(cells=pick.cells, cleared=pick.lines, pieces=game.pieces, lines=game.lines,
                      model_ms=round(model_ms, 1), landings=landings, model_calls=calls,
                      elapsed=round(time.perf_counter() - t_start, 3))
            yield ev
            if pace:
                time.sleep(pace / 1e3)
        yield {"type": "over", "pieces": game.pieces, "lines": game.lines, "topped_out": game.over}


def handler(players: Players):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, code: int, data: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                return self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            if url.path == "/players":
                return self.send(200, json.dumps(players.info).encode(), "application/json")
            if url.path != "/stream":
                return self.send(404, b"not found", "text/plain")
            a = {k: v[0] for k, v in parse_qs(url.query).items()}
            player = a.get("player", "d1")
            if player not in players.info:
                return self.send(422, f"unknown player {player}".encode(), "text/plain")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                for ev in players.game(player, int(a.get("seed", 7)), a.get("filter", "1") == "1",
                                       float(a.get("pace", 0)), int(a.get("cap", 0))):
                    self.wfile.write(b"data: " + json.dumps(ev, separators=(",", ":")).encode() + b"\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass                              # the page closed the stream: the game stops with it
    return H


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", type=Path, default=BUILD, help="a d1 build (d1ane setup)")
    ap.add_argument("--laya", type=Path, default=LAYA if (LAYA / "tokenizer.json").is_file() else None,
                    help=f"FluidInference/laya-coreml files, to play laya next to d1 (default: {LAYA} if d1ane setup --laya put it there)")
    ap.add_argument("--compute", default="ane", choices=("ane", "gpu", "cpu", "auto"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8798)
    a = ap.parse_args(argv)
    if not (a.build / "manifest.json").is_file():
        raise SystemExit(f"no build in {a.build}: run `d1ane setup` first")
    players = Players(a.build, a.laya, a.compute)
    srv = ThreadingHTTPServer((a.host, a.port), handler(players))
    srv.daemon_threads = True
    print(f"Tetris: http://{a.host}:{a.port}/  players: {', '.join(players.info)}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
