"""Run a d1 decision build: Core AI packages on the Apple Neural Engine (or GPU / CPU), or the same graph in PyTorch.

    from d1ane.runtime import Decider
    d = Decider("build/d1-omni-600M")                       # compute="ane" | "gpu" | "cpu" | "auto"
    d.system_one("I was charged twice.", {"refund": {"type": "noul", "instructions": "Refund request?"}})

One question is one fixed-shape forward at the narrowest compiled width that holds the sequence. The host looks up
embeddings, builds the masks, and reads the scores at the option markers; everything in between is the package.
`probabilities_many` asks one question about several states, one call each.
"""
from __future__ import annotations

import asyncio
import contextlib
import ctypes
import json
import time
from pathlib import Path

import numpy as np

from .graph import NEG, shapes
from .prompting import Tok, load_prompt

F16 = np.float16
SMALL = ("kmask", "mq", "tk", "pad", "keep", "hmask", "qt")


def fill(v: dict, emb: np.ndarray, n_prefix: int, qt: np.ndarray) -> None:
    """Write one sequence into a width's input arrays. emb: (L, D) prefix then text; see graph.py for the inputs."""
    L, T = emb.shape[0], v["x"].shape[-1]
    v["x"][0, :, 0, :L] = emb.T
    v["x"][0, :, 0, L:] = 0
    for name, (lo, hi, inside, outside) in {
            "kmask": (0, L, 0, NEG), "pad": (0, L, 1, 0), "mq": (0, n_prefix, 1, 0), "tk": (n_prefix, L, NEG, 0),
            "hmask": (n_prefix, L, 0, NEG), "keep": (n_prefix - 1, max(n_prefix, 0), 0, 1)}.items():
        if name in v:
            a = v[name].reshape(T)
            a[:] = outside
            a[max(lo, 0):hi] = inside
    if "qt" in v:
        v["qt"].reshape(-1)[:] = qt


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


# ---------------------------------------------------------------------------------------------- Core AI backend

def _writable(nd) -> np.ndarray:
    """Writable zero-copy view of an NDArray's storage."""
    mv = memoryview(nd._tensor)  # noqa: SLF001
    ptr = np.frombuffer(mv, dtype=np.uint8).__array_interface__["data"][0]
    return np.ctypeslib.as_array((ctypes.c_uint8 * mv.nbytes).from_address(ptr)).view(F16).reshape(nd.shape)


@contextlib.asynccontextmanager
async def _outputs(fn, inputs: dict):
    """Outputs of one call, released on exit (the binding otherwise pins one IOSurface per output per call)."""
    raw = await fn._function(inputs=inputs, state={})  # noqa: SLF001
    try:
        yield raw
    finally:
        raw.clear()


UNITS = ("ane", "gpu", "cpu")
GPU_ABOVE = 512     # compute="auto": widths above this run on the GPU (measured crossover on an M4 Max, see RESULTS.md)


def compute_units(compute: str, widths: list[int], gpu_above: int = GPU_ABOVE) -> dict[int, str]:
    """The compute unit of each width. "auto": the ANE up to gpu_above positions, the GPU beyond."""
    if compute == "auto":
        return {T: "ane" if T <= gpu_above else "gpu" for T in widths}
    if compute not in UNITS:
        raise ValueError(f"compute must be one of {', '.join(UNITS)} or auto, got {compute!r}")
    return {T: compute for T in widths}


def load_package(path: Path, unit: str):
    """(AIModel specialized for a preferred compute unit, the event loop that loaded it is the caller's)."""
    from coreai.runtime import AIModel, ComputeUnitKind, SpecializationOptions

    kind = {"ane": ComputeUnitKind.neural_engine, "gpu": ComputeUnitKind.gpu, "cpu": ComputeUnitKind.cpu}[unit]()
    return AIModel.load(path, specialization_options=SpecializationOptions.from_preferred_compute_unit_kind(kind))


class CoreAIBackend:
    """The compiled packages. compute: "ane", "gpu", "cpu" (the preferred unit; placement.py reports what the ANE
    build was given), or "auto" (ANE for short widths, GPU for long ones; loads both specializations)."""

    def __init__(self, build: Path, compute: str = "ane", widths: list[int] | None = None, log=print,
                 gpu_above: int = GPU_ABOVE):
        from coreai.runtime import NDArray
        from coreai.runtime._ndarray import StorageKind

        self.build, self.compute, self._NDArray = Path(build), compute, NDArray
        self.man = json.loads((self.build / "manifest.json").read_text())
        self.D = int(self.man["hidden"])
        self.widths = sorted(widths or self.man["widths"])
        self.unit = compute_units(compute, self.widths, gpu_above)
        self.loop = asyncio.new_event_loop()
        self.stages, self.load_s = [], {}
        for st in self.man["stages"]:
            fns = {}
            for unit in sorted(set(self.unit.values())):
                t0 = time.time()
                pkg = self.loop.run_until_complete(load_package(self.build / st["file"], unit))
                fns.update({T: pkg.load_function(st["entries"][str(T)]) for T in self.widths if self.unit[T] == unit})
                self.load_s[f"{st['file']}[{unit}]"] = round(time.time() - t0, 2)
                log(f"loaded {st['file']} [{unit}] in {time.time() - t0:.2f}s")
            self.stages.append({**st, "fns": fns})
        self.buf = {}
        for T in self.widths:
            s = shapes(T, self.D)
            s["xb"] = s["x"]
            nd = {n: NDArray(np.zeros(shp, F16), StorageKind.IO_SURFACE) for n, shp in s.items()}
            self.buf[T] = ({n: a._tensor for n, a in nd.items()}, {n: _writable(a) for n, a in nd.items()}, nd)  # noqa: SLF001

    def width(self, L: int) -> int:
        for T in self.widths:
            if T >= L:
                return T
        raise ValueError(f"{L} positions exceed the widest compiled entry ({self.widths[-1]})")

    def scores(self, emb: np.ndarray, n_prefix: int, qt: np.ndarray) -> tuple[np.ndarray, dict]:
        """(scores at the L positions, timing). One call per stage."""
        L = emb.shape[0]
        T = self.width(L)
        tensors, views, _ = self.buf[T]
        t0 = time.perf_counter()
        fill(views, emb, n_prefix, qt)
        t1 = time.perf_counter()

        async def run():
            x = tensors["x"]
            for st in self.stages:
                ins = {n: (x if n == "x" else tensors[n]) for n in st["inputs"]}
                async with _outputs(st["fns"][T], ins) as out:
                    y = self._NDArray._wrap(out[st["output"]]).numpy()  # noqa: SLF001
                    if st["head"]:
                        return np.array(y, np.float32).reshape(-1)[:L]
                    views["xb"][:] = y
                    x = tensors["xb"]
        s = self.loop.run_until_complete(run())
        t2 = time.perf_counter()
        return s, {"width": T, "unit": self.unit[T], "fill_ms": 1e3 * (t1 - t0), "run_ms": 1e3 * (t2 - t1)}


# ---------------------------------------------------------------------------------------------- PyTorch backend

class TorchBackend:
    """The exported graph run by PyTorch (parity checks: fp32 isolates the rewrite, fp16 the precision)."""

    def __init__(self, model: Path, widths: list[int], dtype: str = "float32", device: str = "cpu",
                 silu: str = "tanh"):
        import torch

        from .convert import build_core
        from .graph import Stage

        self.torch, self.device, self.dtype = torch, device, getattr(torch, dtype)
        core, _, _ = build_core(Path(model), silu)
        self.core = core.to(self.dtype).to(device)
        self.D, self.widths = core.D, sorted(widths)
        n = len(core.layers)
        self.stage = {T: Stage(self.core, T, list(range(n)), True, True).eval().to(self.dtype).to(device)
                      for T in self.widths}

    width = CoreAIBackend.width

    def scores(self, emb: np.ndarray, n_prefix: int, qt: np.ndarray) -> tuple[np.ndarray, dict]:
        torch, L = self.torch, emb.shape[0]
        T = self.width(L)
        t0 = time.perf_counter()
        v = {n: np.zeros(s, np.float32) for n, s in shapes(T, self.D).items()}
        fill(v, emb.astype(np.float32), n_prefix, qt)
        st = self.stage[T]
        with torch.no_grad():
            s = st(*(torch.from_numpy(v[n]).to(self.dtype).to(self.device) for n in st.input_names()))
            s = s.float().cpu().numpy().reshape(-1)[:L]
        return s, {"width": T, "unit": f"torch-{self.device}", "fill_ms": 0.0, "run_ms": 1e3 * (time.perf_counter() - t0)}


# ---------------------------------------------------------------------------------------------- decisions

class Decider:
    """`system_one(state, questions)` over a build, with the checkpoint's prompt, temperatures and answer format."""

    def __init__(self, build: Path, compute: str = "ane", backend=None, widths: list[int] | None = None, log=print):
        self.build = Path(build)
        model = self.build / "model"
        self.config = json.loads((model / "config.json").read_text())
        self.P = load_prompt(model)
        self.tok = Tok(model, int(self.config.get("bos_token_id", 1)))
        self.emb = np.load(self.build / "embed_tokens_fp16.npy", mmap_mode="r")
        self.type_emb = np.load(self.build / "type_emb.npy")
        self.backend = backend or CoreAIBackend(self.build, compute, widths, log)
        self.max_width = self.backend.widths[-1]

    def encode(self, state, q, n_prefix: int = 0, kind: str = "text") -> tuple[list[int], list[int]]:
        c = self.config
        limit = {"text": c["max_length"], "image": c["image_text_length"], "audio": c["audio_text_length"]}[kind]
        noul = None if kind == "text" else {"false": "no", "true": "yes"}
        if kind == "audio" and state is None:
            state = {}
        room = min(limit, c["max_length"] - n_prefix, self.max_width - n_prefix)
        if room < 64:
            raise ValueError(f"the media take {n_prefix} of the {self.max_width} compiled positions")
        return self.P.encode(self.tok, "" if state is None else state, q, room, noul, kind == "audio")

    def logits(self, ids: list[int], markers: list[int], qtype: int, prefix: np.ndarray | None = None):
        """Raw option logits of one encoded question and the timing of its forward."""
        emb = np.asarray(self.emb[np.asarray(ids)])
        p = 0 if prefix is None else prefix.shape[0]
        if p:
            emb = np.concatenate([prefix.astype(F16), emb])
        s, t = self.backend.scores(emb, p, self.type_emb[qtype])
        return s[p + np.asarray(markers)], t

    def probabilities(self, state, q, prefix: np.ndarray | None = None, kind: str = "text"):
        q = self.P.as_question(q)
        t0 = time.perf_counter()
        ids, markers = self.encode(state, q, 0 if prefix is None else prefix.shape[0], kind)
        t1 = time.perf_counter()
        z, t = self.logits(ids, markers, self.P.QTYPES[q.type], prefix)
        z = z[:q.options].astype(np.float64)
        if kind == "text":
            temps = self.config["temperatures"]
            z = z / temps.get(self.P.temperature_key(q), temps.get(q.type, 1.0))
        p = softmax(z)
        t.update(tokens=len(ids) + (0 if prefix is None else prefix.shape[0]), tokenize_ms=1e3 * (t1 - t0))
        return (p[::-1] if q.type == "noul" else p).tolist(), t

    def probabilities_many(self, states: list, q) -> tuple[list[list[float]], dict]:
        """One question about several text states: `probabilities` of each (one call per state), and the timing
        of the whole."""
        q = self.P.as_question(q)
        t0 = time.perf_counter()
        out = [self.probabilities(state, q)[0] for state in states]
        return out, {"calls": len(states), "total_ms": 1e3 * (time.perf_counter() - t0)}

    def system_one(self, state, questions: dict, prefix: np.ndarray | None = None, kind: str = "text") -> dict:
        t0 = time.perf_counter()
        answers, timing, read = {}, {}, 0
        for name, q in questions.items():
            q = self.P.as_question(q)
            probs, t = self.probabilities(state, q, prefix, kind)
            answers[name] = self.P.answer(q, probs)
            timing[name] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in t.items()}
            read += t["tokens"]
        return {"answers": answers, "usage": {"input_tokens": read, "output_tokens": 0},
                "timing": {"total_ms": round(1e3 * (time.perf_counter() - t0), 3), "questions": timing}}
