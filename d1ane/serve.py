"""Decision server for a d1 build on the Apple Neural Engine.

    d1ane serve [--build build/d1-omni-600M] [--model models/d1-omni-600M] [--port 8797]

    POST /v1/systemone   {"state": ..., "questions": {name: {"type", "instructions", "criteria"}},
                          "images": [base64 or data URL, ...] | "audio": base64 WAV (16 kHz mono)}
                         short form: {"state": ..., "options": [...] or {name: description}, "instructions": "..."}
    GET  /health         readiness, build, modalities, compiled widths
    GET  /v1/models
    GET  /               browser demo

Text, images and audio run on the ANE (trunk, head, vision tower, audio encoder) when the build has the packages
(d1ane.vision, d1ane.audio). Without audio.aimodel, --model runs the audio encoder on the host in PyTorch instead.
One request runs at a time.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import BUILD
from .runtime import Decider

DEMO = Path(__file__).with_name("demo.html")


class BadRequest(ValueError):
    pass


def _bytes(data: str) -> bytes:
    if data.startswith("data:"):
        data = data.split(",", 1)[1]
    try:
        return base64.b64decode(data, validate=False)
    except ValueError as e:
        raise BadRequest(f"media must be base64 or a data URL: {e}") from None


class App:
    def __init__(self, build: Path, compute: str = "ane", model: Path | None = None, log=print):
        self.build, self.compute, self.model = Path(build), compute, model
        self.man = json.loads((self.build / "manifest.json").read_text())
        self.name = Path(self.man.get("source", "d1")).name
        self.decider = Decider(self.build, compute, log=log)
        self.vision = self.audio = None
        if "vision" in self.man:
            from .vision import VisionBackend

            self.vision = VisionBackend(self.build, compute, log=log)
        if "audio" in self.man:
            from .audio import AudioBackend

            self.audio = AudioBackend(self.build, compute, log=log)
        elif model is not None:
            from .audio import HostAudio

            self.audio = HostAudio(model)
        self.lock = threading.Lock()
        self.warm()

    def warm(self) -> None:
        q = {"type": "noul", "instructions": "Is this a test?"}
        for _ in range(3):
            self.decider.system_one("warm up", {"q": q})

    def health(self) -> dict:
        b = self.decider.backend
        return {"ready": True, "model": self.name, "kind": self.man["kind"], "backend": f"coreai-{self.compute}",
                "modalities": ["text"] + (["image"] if self.vision else []) + (["audio"] if self.audio else []),
                "widths": b.widths, "max_tokens": b.widths[-1], "quant": self.man["quant"], "build": str(self.build)}

    def decide(self, body: dict) -> dict:
        if not isinstance(body, dict):
            raise BadRequest("the body must be a JSON object")
        questions = body.get("questions")
        if questions is None and "options" in body:
            opts = body["options"]
            crit = {str(o): "" for o in opts} if isinstance(opts, list) else {k: v or "" for k, v in opts.items()}
            questions = {"decision": {"type": "choice", "instructions": body.get("instructions", "Pick one."), "criteria": crit}}
        if not isinstance(questions, dict) or not questions:
            raise BadRequest("`questions` must be an object of named questions (or send `options`)")
        images, audio = body.get("images") or None, body.get("audio")
        if images is not None and audio is not None:
            raise BadRequest("a request carries images or audio, not both")
        with self.lock:
            t0 = time.perf_counter()
            prefix, kind, media = None, "text", {}
            if images is not None:
                if self.vision is None:
                    raise BadRequest("this build has no vision package (python -m d1ane.vision)")
                from PIL import Image

                pil = [Image.open(io.BytesIO(_bytes(i))) for i in ([images] if isinstance(images, str) else images)]
                prefix, media = self.vision.prefix(pil)
                kind = "image"
            elif audio is not None:
                if self.audio is None:
                    raise BadRequest("this build has no audio package (python -m d1ane.audio), and no --model for the host encoder")
                import soundfile as sf

                samples, rate = sf.read(io.BytesIO(_bytes(audio)), dtype="int16")
                if rate != 16000 or samples.ndim != 1:
                    raise BadRequest("audio must be 16 kHz mono")
                prefix, media = self.audio.prefix(samples)
                kind = "audio"
            try:
                out = self.decider.system_one(body.get("state"), questions, prefix, kind)
            except ValueError as e:
                raise BadRequest(str(e)) from None
            t = out.pop("timing")
            qs = t["questions"]
            out["model"] = self.name
            out["timings"] = {"total_ms": round(1e3 * (time.perf_counter() - t0), 3),
                              "ane_ms": round(sum(q["run_ms"] for q in qs.values()), 3),
                              "tokenize_ms": round(sum(q["tokenize_ms"] for q in qs.values()), 3),
                              **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in media.items()},
                              "questions": qs}
            return out


def handler(app: App):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def send(self, code: int, payload, ctype: str = "application/json"):
            data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/health":
                self.send(200, app.health())
            elif path == "/v1/models":
                self.send(200, {"object": "list", "data": [{"id": app.name, "object": "model"}]})
            elif path == "/":
                self.send(200, DEMO.read_bytes(), "text/html; charset=utf-8")
            else:
                self.send(404, {"error": "not found"})

        def do_POST(self):
            if self.path.split("?")[0] not in ("/v1/systemone", "/v1/decide"):
                return self.send(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                self.send(200, app.decide(body))
            except (BadRequest, json.JSONDecodeError) as e:
                self.send(422, {"error": str(e)})
            except Exception as e:  # noqa: BLE001
                self.send(500, {"error": f"{type(e).__name__}: {e}"})
    return H


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", type=Path, default=BUILD)
    ap.add_argument("--model", type=Path, help="the checkpoint directory: host audio encoder for builds without audio.aimodel")
    ap.add_argument("--compute", default="ane", choices=("ane", "gpu", "cpu", "auto"),
                    help="auto: the ANE up to 512 positions, the GPU beyond")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8797)
    a = ap.parse_args(argv)
    if not (a.build / "manifest.json").is_file():
        raise SystemExit(f"no build in {a.build}: run `d1ane setup` first")
    app = App(a.build, a.compute, a.model)
    srv = ThreadingHTTPServer((a.host, a.port), handler(app))
    print(f"d1 on {app.health()['backend']}: http://{a.host}:{a.port}/  modalities={app.health()['modalities']}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
