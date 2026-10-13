"""d1-omni's vision encoder on the Apple Neural Engine: the SigLIP2 NaFlex tower and the pixel-unshuffle projector.

    python -m d1ane.vision --model models/d1-omni-600M --output build/d1-omni-600M

Adds vision.aimodel to a build: tower entries n<N> (N patches per crop) and projector entries p<P> (P = N / 4
prefix embeddings), one copy of the weights. The host tiles and patchifies the image with the checkpoint's own
preprocessing, resizes the position embeddings to each crop's grid, and applies the 2x2 unshuffle between the two
calls; an image becomes a (P, 1024) prefix for the trunk (runtime.Decider, kind="image").

Tower inputs (FP16): pix (1, 768, 1, N) patch pixels, pos (1, 768, 1, N) position embeddings, kmask (1, 1, 1, N)
0 on real patches and NEG on padding. Output y (1, 768, 1, N). Projector: x (1, 3072, 1, P) -> y (1, 1024, 1, P).
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import NEG, chan, conv1x1, gelu, gelu_tanh, layer_norm

F16 = np.float16
PRE = "vision.tower.vision_model."
TOWER_WIDTHS, MODEL_FILES = (256, 512, 768, 1024), ("vision.py",)
GPU_ABOVE = 512     # compute="auto": crops with more patches run on the GPU


class VitLayer(nn.Module):
    def __init__(self, W: dict, i: int, cfg: dict):
        super().__init__()
        p = f"{PRE}encoder.layers.{i}."
        self.D, self.H, self.eps = cfg["hidden_size"], cfg["num_attention_heads"], cfg["layer_norm_eps"]
        self.q, self.k, self.v, self.o = (conv1x1(W[f"{p}self_attn.{n}.weight"], W[f"{p}self_attn.{n}.bias"])
                                          for n in ("q_proj", "k_proj", "v_proj", "out_proj"))
        self.fc1 = conv1x1(W[p + "mlp.fc1.weight"], W[p + "mlp.fc1.bias"])
        self.fc2 = conv1x1(W[p + "mlp.fc2.weight"], W[p + "mlp.fc2.bias"])
        for n in ("layer_norm1", "layer_norm2"):
            self.register_buffer(n + "w", chan(W[f"{p}{n}.weight"]))
            self.register_buffer(n + "b", chan(W[f"{p}{n}.bias"]))

    def forward(self, h, kmask):
        N, hd = h.shape[-1], self.D // self.H
        x = layer_norm(h, self.layer_norm1w, self.layer_norm1b, self.eps)
        q, k, v = (m(x).reshape(self.H, hd, N) for m in (self.q, self.k, self.v))
        p = torch.softmax((q.transpose(1, 2) @ k) * hd ** -0.5 + kmask.reshape(1, 1, N), -1)
        o = (p @ v.transpose(1, 2)).permute(1, 0, 2).reshape(N, self.D)
        h = h + self.o(o.transpose(0, 1).reshape(1, self.D, 1, N))
        x = layer_norm(h, self.layer_norm2w, self.layer_norm2b, self.eps)
        return h + self.fc2(gelu_tanh(self.fc1(x)))


class VisionCore(nn.Module):
    def __init__(self, W: dict, config: dict):
        super().__init__()
        cfg = self.cfg = config["vision_config"]
        self.D, self.out = cfg["hidden_size"], config["text_config"]["hidden_size"]
        self.patch = conv1x1(W[PRE + "embeddings.patch_embedding.weight"], W[PRE + "embeddings.patch_embedding.bias"])
        self.layers = nn.ModuleList(VitLayer(W, i, cfg) for i in range(cfg["num_hidden_layers"]))
        self.register_buffer("postw", chan(W[PRE + "post_layernorm.weight"]))
        self.register_buffer("postb", chan(W[PRE + "post_layernorm.bias"]))
        self.l1 = conv1x1(W["vision.projector.linear_1.weight"], W["vision.projector.linear_1.bias"])
        self.l2 = conv1x1(W["vision.projector.linear_2.weight"], W["vision.projector.linear_2.bias"])


class Tower(nn.Module):
    def __init__(self, core: VisionCore, N: int):
        super().__init__()
        self.core, self.T = core, N

    def input_names(self):
        return ["pix", "pos", "kmask"]

    def output_names(self):
        return ["y"]

    def example(self, dtype=torch.float16):
        D, N = self.core.D, self.T
        return tuple(torch.zeros(s, dtype=dtype) for s in ((1, D, 1, N), (1, D, 1, N), (1, 1, 1, N)))

    def forward(self, pix, pos, kmask):
        c = self.core
        h = c.patch(pix) + pos
        for layer in c.layers:
            h = layer(h, kmask)
        return layer_norm(h, c.postw, c.postb, c.cfg["layer_norm_eps"])


class Projector(nn.Module):
    def __init__(self, core: VisionCore, P: int):
        super().__init__()
        self.core, self.T = core, P

    def input_names(self):
        return ["x"]

    def output_names(self):
        return ["y"]

    def example(self, dtype=torch.float16):
        return (torch.zeros(1, 4 * self.core.D, 1, self.T, dtype=dtype),)

    def forward(self, x):
        return self.core.l2(gelu(self.core.l1(x)))


def unshuffle(x: np.ndarray, h: int, w: int) -> np.ndarray:
    """The projector's 2x2 pixel unshuffle: (h * w, C) patch states -> (h/2 * w/2, 4C), as the checkpoint orders it."""
    c = x.shape[-1]
    x = x.reshape(1, h, w // 2, c * 2).transpose(0, 2, 1, 3)
    x = x.reshape(1, w // 2, h // 2, c * 4).transpose(0, 2, 1, 3)
    return x.reshape(-1, c * 4)


def export(model: Path, out: Path, widths=TOWER_WIDTHS, log=print) -> dict:
    from .convert import load_weights, save_program

    t0 = time.time()
    config = json.loads((model / "config.json").read_text())
    W = load_weights(model, ("vision.",))
    core = VisionCore(W, config).eval().to(torch.float16)
    np.save(out / "vision_pos.npy", W[PRE + "embeddings.position_embedding.weight"].astype(np.float32))
    for name in MODEL_FILES:
        (out / "model" / name).write_bytes((model / name).read_bytes())
    entries = [(f"n{N}", Tower(core, N).eval()) for N in widths] + [(f"p{N // 4}", Projector(core, N // 4).eval()) for N in widths]
    mb = save_program(entries, out / "vision.aimodel")
    man = json.loads((out / "manifest.json").read_text())
    man["vision"] = {"file": "vision.aimodel", "widths": list(widths), "mb": round(mb, 1), "hidden": core.D,
                     "entries": [n for n, _ in entries], "convert_s": round(time.time() - t0, 1)}
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    log(f"vision.aimodel: {mb:.0f} MB, {time.time() - t0:.1f}s")
    return man["vision"]


class VisionBackend:
    """Images -> prefix embeddings through vision.aimodel. One tower call and one projector call per crop."""

    def __init__(self, build: Path, compute: str = "ane", log=print):
        from coreai.runtime import NDArray
        from coreai.runtime._ndarray import StorageKind

        from .runtime import _writable, load_package

        self.build, self._NDArray = Path(build), NDArray
        man = json.loads((self.build / "manifest.json").read_text())["vision"]
        spec = importlib.util.spec_from_file_location("d1_vision", self.build / "model" / "vision.py")
        self.V = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.V)
        self.widths, self.D = sorted(man["widths"]), int(man["hidden"])
        self.pos = torch.from_numpy(np.load(self.build / "vision_pos.npy"))
        side = int(self.pos.shape[0] ** 0.5)
        self.pos = self.pos.reshape(side, side, -1).permute(2, 0, 1).unsqueeze(0)
        self._pos_cache: dict[tuple[int, int], np.ndarray] = {}
        # auto: the GPU runs the 768- and 1,024-patch crops faster (RESULTS.md); the ANE is ahead below them
        self.unit = {N: ("gpu" if N > GPU_ABOVE else "ane") if compute == "auto" else compute for N in self.widths}
        self.loop, self.tower, self.proj, self.load_s = asyncio.new_event_loop(), {}, {}, 0.0
        for unit in sorted(set(self.unit.values())):
            t0 = time.time()
            pkg = self.loop.run_until_complete(load_package(self.build / man["file"], unit))
            for N in (n for n in self.widths if self.unit[n] == unit):
                self.tower[N], self.proj[N] = pkg.load_function(f"n{N}"), pkg.load_function(f"p{N // 4}")
            self.load_s += round(time.time() - t0, 2)
            log(f"loaded {man['file']} [{unit}] in {time.time() - t0:.2f}s")
        self.buf = {}
        for N in self.widths:
            nd = {n: NDArray(np.zeros(s, F16), StorageKind.IO_SURFACE) for n, s in (
                ("pix", (1, self.D, 1, N)), ("pos", (1, self.D, 1, N)), ("kmask", (1, 1, 1, N)),
                ("x", (1, 4 * self.D, 1, N // 4)))}
            self.buf[N] = ({n: a._tensor for n, a in nd.items()}, {n: _writable(a) for n, a in nd.items()}, nd)  # noqa: SLF001

    def positions(self, h: int, w: int) -> np.ndarray:
        """(h * w, D) position embeddings for one grid, as SigLIP2 NaFlex resizes them."""
        if (h, w) not in self._pos_cache:
            r = F.interpolate(self.pos, size=(h, w), mode="bilinear", align_corners=False, antialias=True)
            self._pos_cache[(h, w)] = r.reshape(self.D, h * w).numpy().astype(F16)
        return self._pos_cache[(h, w)]

    def crop(self, pixels: np.ndarray, h: int, w: int) -> np.ndarray:
        """(h * w, 768) patch pixels of one crop -> (h * w / 4, 1024) prefix embeddings."""
        n = h * w
        N = next(x for x in self.widths if x >= n)
        tensors, v, _ = self.buf[N]
        v["pix"][0, :, 0, :n], v["pix"][0, :, 0, n:] = pixels.T, 0
        v["pos"][0, :, 0, :n], v["pos"][0, :, 0, n:] = self.positions(h, w), 0
        v["kmask"].reshape(N)[:n], v["kmask"].reshape(N)[n:] = 0, NEG

        async def run():
            raw = await self.tower[N]._function(inputs={k: tensors[k] for k in ("pix", "pos", "kmask")}, state={})  # noqa: SLF001
            hidden = np.array(self._NDArray._wrap(raw["y"]).numpy()[0, :, 0, :n].T)  # noqa: SLF001
            raw.clear()
            v["x"][0, :, 0, :n // 4], v["x"][0, :, 0, n // 4:] = unshuffle(hidden, h, w).T, 0
            raw = await self.proj[N]._function(inputs={"x": tensors["x"]}, state={})  # noqa: SLF001
            y = np.array(self._NDArray._wrap(raw["y"]).numpy()[0, :, 0, :n // 4].T)  # noqa: SLF001
            raw.clear()
            return y
        return self.loop.run_until_complete(run())

    def prefix(self, images) -> tuple[np.ndarray, dict]:
        """PIL image(s) -> ((P, 1024) FP16 prefix, timing): every image's tiles then its thumbnail, in order."""
        images = [images] if hasattr(images, "convert") else list(images)
        t0 = time.perf_counter()
        out, crops, pre = [], 0, 0.0
        for image in images:
            t1 = time.perf_counter()
            inputs = self.V.preprocess(image)
            pre += time.perf_counter() - t1
            px = inputs["pixel_values"].numpy()
            for i, (h, w) in enumerate(inputs["spatial_shapes"].tolist()):
                out.append(self.crop(px[i, :h * w], h, w))
                crops += 1
        total = time.perf_counter() - t0
        return np.concatenate(out), {"crops": crops, "preprocess_ms": 1e3 * pre, "vision_ms": 1e3 * (total - pre)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True, help="an existing build (d1ane.convert) to add vision to")
    ap.add_argument("--widths", default=",".join(map(str, TOWER_WIDTHS)))
    a = ap.parse_args(argv)
    export(a.model, a.output, tuple(int(w) for w in a.widths.split(",")))


if __name__ == "__main__":
    sys.exit(main())
