"""Benchmarks of a d1 decision build. One engine per process; results are JSON files that report.py tabulates.

    python bench/run.py latency --engine coreai-ane   --out bench/results/latency_coreai-ane.json
    python bench/run.py dataset --engine torch-mps-fp32 --out bench/results/dataset_torch-mps-fp32.json [--limit N]

Engines: coreai-ane | coreai-gpu (the build's packages, preferred compute unit), and torch-mps-<fp32|fp16>
(the checkpoint through transformers on the GPU, one decision per call).
Text: `latency` times one decision at fixed token counts; `dataset` runs every single-label head of
fastino/fast-decisions (development split). Images: `vision` times one question at four image sizes; `pope` runs
600 yes/no questions of lmms-lab/POPE. Audio: `audio` times one question at five clip lengths; `speech` runs the
SLURP and MINDS-14 intent clips. Dataset tasks keep the probabilities, so engines can be compared row by row.
The media files are fetched by bench/all.sh into bench/data/.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FILLER = ("Order 48213 was charged twice on the 3rd and the second payment was never authorised. The app signs me "
          "out whenever the network changes, and the invoices tab shows last year's totals. ")
QUESTION = {"type": "choice", "instructions": "Which team should handle this?",
            "criteria": {"billing": "Charges, refunds, invoices", "technical": "App or site faults",
                         "fraud": "Suspected unauthorised use", "other": "Anything else"}}
LENGTHS = [48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048]


class Engine:
    def __init__(self, name: str, model: Path, build: Path):
        from d1ane.runtime import Decider

        self.name = name
        kind, *rest = name.split("-")
        if kind == "coreai":
            self.decider, self.ref = Decider(build, rest[0], log=lambda m: print(m, file=sys.stderr)), None
        else:
            from d1ane.reference import Reference

            device, dtype = rest
            self.ref = Reference(model, device, {"fp32": "float32", "fp16": "float16"}[dtype])
            self.decider = Decider(build, backend=_NoBackend())      # prompt + tokenizer only
        self.P = self.decider.P

    def forward(self, ids, markers, q) -> np.ndarray:
        """Raw option logits of one encoded question (text only)."""
        if self.ref is None:
            return self.decider.logits(ids, markers, self.P.QTYPES[q.type])[0][:q.options]
        m, torch = self.ref.model, self.ref.torch
        with torch.no_grad():
            dev = m.device
            h = m.encoder.embed_tokens(torch.tensor([ids], device=dev))
            pad = torch.ones(1, len(ids), dtype=torch.bool, device=dev)
            h = m.encoder(h, pad, torch.zeros(1, dtype=torch.long, device=dev))
            z = m.head(h, pad, torch.tensor([markers], device=dev), torch.ones(1, len(markers), dtype=torch.bool, device=dev),
                       torch.tensor([self.P.QTYPES[q.type]], device=dev))
            return z[0].float().cpu().numpy()[:q.options]

    def probabilities(self, state, q) -> list[float]:
        if self.ref is None:
            return self.decider.probabilities(state, q)[0]
        return self.ref.probabilities(state, q)


class _NoBackend:
    widths = [16384]


def machine() -> dict:
    sh = lambda *c: subprocess.run(c, capture_output=True, text=True).stdout.strip()  # noqa: E731
    return {"chip": sh("sysctl", "-n", "machdep.cpu.brand_string"), "model": sh("sysctl", "-n", "hw.model"),
            "macos": sh("sw_vers", "-productVersion"), "build": sh("sw_vers", "-buildVersion"),
            "python": platform.python_version(), "load_avg": sh("sysctl", "-n", "vm.loadavg"),
            "time": time.strftime("%Y-%m-%d %H:%M:%S")}


def latency(eng: Engine, reps: int, warmup: int, lengths: list[int]) -> dict:
    q = eng.P.as_question(QUESTION)
    rows = []
    for L in lengths:
        ids, markers = eng.P.encode(eng.decider.tok, FILLER * 200, q, L)
        assert len(ids) == L
        try:
            for _ in range(warmup):
                eng.forward(ids, markers, q)
        except ValueError as e:                       # wider than the build
            print(f"  L={L}: {e}", file=sys.stderr)
            continue
        ms = []
        for _ in range(reps):
            t0 = time.perf_counter()
            eng.forward(ids, markers, q)
            ms.append(1e3 * (time.perf_counter() - t0))
        ms.sort()
        row = {"tokens": L, "median_ms": statistics.median(ms), "p95_ms": ms[int(0.95 * (len(ms) - 1))],
               "min_ms": ms[0], "mean_ms": statistics.fmean(ms), "reps": reps}
        if eng.ref is None:
            row["width"] = eng.decider.backend.width(L)
        rows.append(row)
        print(f"  {eng.name} L={L:5d} median {row['median_ms']:8.2f} ms  p95 {row['p95_ms']:8.2f}", file=sys.stderr)
    return {"rows": rows}


def dataset_rows(limit: int | None = None) -> list[dict]:
    """Every single-label head of fast-decisions as a d1 choice question (label names as the options)."""
    from datasets import get_dataset_config_names, load_dataset

    out = []
    for cfg in sorted(get_dataset_config_names("fastino/fast-decisions")):
        ds = load_dataset("fastino/fast-decisions", cfg, split="train")
        for i, ex in enumerate(ds):
            if limit is not None and i >= limit:
                break
            for head in ex["output"]["classifications"]:
                if head["multi_label"] or len(head["true_label"]) != 1 or len(head["labels"]) < 2:
                    continue
                out.append({"id": f"{cfg}/{i}/{head['task']}", "domain": cfg, "state": ex["input"],
                            "question": {"type": "choice", "instructions": f"Classify the {head['task'].replace('_', ' ')}.",
                                         "criteria": {label: "" for label in head["labels"]}},
                            "gold": head["true_label"][0]})
    return out


def dataset(eng: Engine, limit: int | None) -> dict:
    rows, out = dataset_rows(limit), []
    eng.probabilities(rows[0]["state"], rows[0]["question"])          # warm
    t_all = time.perf_counter()
    for i, r in enumerate(rows):
        q = eng.P.as_question(r["question"])
        try:
            ids, _ = eng.decider.encode(r["state"], q)
            t0 = time.perf_counter()
            p = eng.probabilities(r["state"], r["question"])
            ms = 1e3 * (time.perf_counter() - t0)
        except ValueError as e:
            out.append({"id": r["id"], "error": str(e)})
            continue
        names = list(r["question"]["criteria"])
        out.append({"id": r["id"], "domain": r["domain"], "tokens": len(ids), "options": len(names), "ms": ms,
                    "probs": p, "pred": names[int(np.argmax(p))], "gold": r["gold"]})
        if i % 200 == 0:
            print(f"  {eng.name} {i}/{len(rows)}", file=sys.stderr)
    return {"rows": out, "wall_s": time.perf_counter() - t_all}


# ------------------------------------------------------------------------------------------- images and audio

DATA = ROOT / "bench" / "data"
IMAGE_Q = {"type": "choice", "instructions": "How many cats are there?",
           "criteria": {"one": "One", "two": "Two", "more": "Three or more"}}
IMAGE_SIZES = {"224x224": (224, 224), "320x240": (320, 240), "640x480": (640, 480), "1600x1200": (1600, 1200)}
TOWER_GRIDS = {256: (16, 16), 512: (16, 32), 768: (24, 32), 1024: (32, 32)}        # patches: (rows, columns)
AUDIO_Q = {"type": "choice", "instructions": "What is the speaker talking about?",
           "criteria": {"food": "Food and meals", "travel": "Travel and transport", "weather": "The weather"}}
AUDIO_SECONDS = [2, 5, 10, 20, 30]


class Media:
    """Image and audio questions through one engine: the build's packages, or the checkpoint in PyTorch."""

    def __init__(self, eng: Engine, build: Path):
        self.eng, self.build, self.vis, self.aud = eng, build, None, None
        self.unit = eng.name.split("-")[1] if eng.ref is None else None
        self.log = lambda m: print(m, file=sys.stderr)

    def image(self, im, q, state=None) -> tuple[list[float], dict]:
        if self.eng.ref is not None:
            return self.eng.ref.probabilities(state, q, images=[im]), {}
        if self.vis is None:
            from d1ane.vision import VisionBackend

            self.vis = VisionBackend(self.build, self.unit, log=self.log)
        prefix, tv = self.vis.prefix(im)
        p, tt = self.eng.decider.probabilities(state, q, prefix=prefix, kind="image")
        return p, {"preprocess_ms": tv["preprocess_ms"], "encoder_ms": tv["vision_ms"], "trunk_ms": tt["run_ms"],
                   "crops": tv["crops"], "tokens": tt["tokens"], "width": tt["width"]}

    def audio(self, samples, q, state=None) -> tuple[list[float], dict]:
        if self.eng.ref is not None:
            return self.eng.ref.probabilities(state, q, audio=samples), {}
        if self.aud is None:
            from d1ane.audio import AudioBackend

            self.aud = AudioBackend(self.build, self.unit, log=self.log)
        prefix, ta = self.aud.prefix(samples)
        p, tt = self.eng.decider.probabilities(state, q, prefix=prefix, kind="audio")
        return p, {"preprocess_ms": ta["mel_ms"], "encoder_ms": ta["audio_ms"], "trunk_ms": tt["run_ms"],
                   "encoder_width": ta["audio_width"], "tokens": tt["tokens"], "width": tt["width"]}


def timed(fn, reps: int, warmup: int) -> dict:
    """Median and p95 of fn() over reps warm calls; fn returns (probabilities, parts)."""
    ms = []
    for i in range(reps + warmup):
        t0 = time.perf_counter()
        p, parts = fn()
        if i >= warmup:
            ms.append(1e3 * (time.perf_counter() - t0))
    ms.sort()
    return {"median_ms": statistics.median(ms), "p95_ms": ms[int(0.95 * (len(ms) - 1))], "probs": p, **parts}


def vision(eng: Engine, build: Path, reps: int, warmup: int) -> dict:
    """One image question end to end (preprocess, vision encoder, trunk, head) at four image sizes, and for the
    build's packages the vision encoder alone at every compiled crop size."""
    from PIL import Image

    src, media, rows, tower = Image.open(DATA / "cats.jpg"), Media(eng, build), [], []
    for name, size in IMAGE_SIZES.items():
        im = src.resize(size)
        rows.append({"image": name, **timed(lambda: media.image(im, IMAGE_Q), reps, warmup)})
        print(f"  {eng.name} {name}: {rows[-1]['median_ms']:.1f} ms", file=sys.stderr)
    if media.vis is not None:
        rng = np.random.default_rng(0)
        for n, (h, w) in TOWER_GRIDS.items():
            px = (0.5 * rng.standard_normal((n, 768))).astype(np.float16)
            tower.append({"patches": n, **timed(lambda: (media.vis.crop(px, h, w), {}), reps, warmup)})
            tower[-1].pop("probs")
            print(f"  {eng.name} tower {n} patches: {tower[-1]['median_ms']:.1f} ms", file=sys.stderr)
    return {"rows": rows, "tower": tower}


def speech_clip() -> np.ndarray:
    import soundfile as sf

    clip, rate = sf.read(DATA / "speech.flac", dtype="int16")
    assert rate == 16000 and clip.ndim == 1
    return np.tile(clip, 1 + 30 * 16000 // len(clip))


def audio(eng: Engine, build: Path, reps: int, warmup: int) -> dict:
    """One audio question end to end (log-mel, audio encoder, trunk, head) at five clip lengths."""
    clip, media, rows = speech_clip(), Media(eng, build), []
    for sec in AUDIO_SECONDS:
        c = clip[:sec * 16000]
        rows.append({"seconds": sec, **timed(lambda: media.audio(c, AUDIO_Q, "Voice note from a user."), reps, warmup)})
        print(f"  {eng.name} {sec}s: {rows[-1]['median_ms']:.1f} ms", file=sys.stderr)
    return {"rows": rows}


def pope_rows(per_category: int = 200) -> list[dict]:
    """POPE (yes/no object questions on COCO photos): rows spread evenly over each of the three categories."""
    import io

    import pyarrow.parquet as pq
    from PIL import Image

    out = []
    for cat in ("random", "popular", "adversarial"):
        t = pq.read_table(DATA / "pope" / "Full" / f"{cat}-00000-of-00001.parquet")
        for i in range(0, t.num_rows, t.num_rows // per_category):
            r = t.slice(i, 1).to_pylist()[0]
            out.append({"id": f"pope/{cat}/{r['question_id']}", "set": f"POPE {cat}", "gold": r["answer"],
                        "media": Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB"), "names": ["yes", "no"],
                        "question": {"type": "noul", "instructions": r["question"]}})
    return out


def speech_rows() -> list[dict]:
    """Spoken intents: SLURP (Dynamic-SUPERB's 200-clip subset, 5 intents) and MINDS-14 en-US (563 clips, 14 intents,
    8 kHz resampled to 16 kHz). Options are the intent names."""
    import io
    import math

    import pyarrow.parquet as pq
    import soundfile as sf
    from scipy.signal import resample_poly

    def samples(blob: bytes) -> np.ndarray:
        x, rate = sf.read(io.BytesIO(blob), dtype="float32")
        x = x.mean(1) if x.ndim > 1 else x
        if rate != 16000:
            g = math.gcd(16000, rate)
            x = resample_poly(x, 16000 // g, rate // g).astype(np.float32)
        return x

    out = []
    slurp = pq.read_table(DATA / "slurp" / "data" / "test-00000-of-00001.parquet").to_pylist()
    minds = pq.read_table(DATA / "minds14" / "en-US" / "train-00000-of-00001.parquet").to_pylist()
    for r in minds:
        r["label"] = r["path"].split("~")[1].split("/")[0].lower()
    for name, rows in (("SLURP intent", slurp), ("MINDS-14 en-US", minds)):
        names = sorted({r["label"] for r in rows})
        q = {"type": "choice", "instructions": "What does the speaker want?",
             "criteria": {n: n.replace("_", " ") for n in names}}
        for i, r in enumerate(rows):
            out.append({"id": f"{name}/{i}", "set": name, "gold": r["label"], "media": samples(r["audio"]["bytes"]),
                        "names": names, "question": q})
    return out


def media_dataset(eng: Engine, build: Path, rows: list[dict], kind: str, limit: int | None) -> dict:
    """Every row once, end to end from the decoded image or samples to the probabilities."""
    media, out = Media(eng, build), []
    ask = media.image if kind == "image" else media.audio
    rows = rows if limit is None else [r for i, r in enumerate(rows) if i % max(1, len(rows) // limit) == 0]
    for _ in range(3):
        ask(rows[0]["media"], rows[0]["question"])          # warm
    t_all = time.perf_counter()
    for i, r in enumerate(rows):
        t0 = time.perf_counter()
        p, parts = ask(r["media"], r["question"])
        ms = 1e3 * (time.perf_counter() - t0)
        out.append({"id": r["id"], "set": r["set"], "gold": r["gold"], "pred": r["names"][int(np.argmax(p))],
                    "probs": p, "ms": ms, **parts})
        if i % 200 == 0:
            print(f"  {eng.name} {i}/{len(rows)}", file=sys.stderr)
    return {"rows": out, "wall_s": time.perf_counter() - t_all}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=("latency", "dataset", "vision", "audio", "pope", "speech"))
    ap.add_argument("--engine", required=True)
    ap.add_argument("--model", type=Path, default=ROOT / "models/d1-omni-600M")
    ap.add_argument("--build", type=Path, default=ROOT / "build/d1-omni-600M")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--lengths", default=",".join(map(str, LENGTHS)))
    ap.add_argument("--limit", type=int, help="dataset: rows per domain; pope / speech: about this many rows in all")
    a = ap.parse_args(argv)
    t0 = time.time()
    eng = Engine(a.engine, a.model, a.build)
    res = {"task": a.task, "engine": a.engine, "machine": machine(), "load_s": round(time.time() - t0, 2)}
    if a.task == "latency":
        res.update(latency(eng, a.reps, a.warmup, [int(x) for x in a.lengths.split(",")]))
    elif a.task == "dataset":
        res.update(dataset(eng, a.limit))
    elif a.task in ("vision", "audio"):
        res.update((vision if a.task == "vision" else audio)(eng, a.build, a.reps, a.warmup))
    else:
        rows, kind = (pope_rows(), "image") if a.task == "pope" else (speech_rows(), "audio")
        res.update(media_dataset(eng, a.build, rows, kind, a.limit))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(f"{a.engine} {a.task}: wrote {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
