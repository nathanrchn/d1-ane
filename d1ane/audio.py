"""d1-omni's audio encoder: the FastConformer, adapter and residual on the Apple Neural Engine.

    python -m d1ane.audio --model models/d1-omni-600M --output build/d1-omni-600M

Adds audio.aimodel to a build: one entry a<T> per width (T prefix positions, 80 ms each, 8 T mel frames). The host
runs the checkpoint's own log-mel front end (FP32) and builds the length masks; a clip becomes a (P, 1024) prefix
for the trunk (runtime.Decider, kind="audio"). HostAudio is the same encoder in PyTorch, for builds without the
package and for parity checks.

Entry inputs (FP16): mel (1, 1, 8T, 128) features, zeros past the clip; m0..m3 (1, 1, 8T | 4T | 2T | T, 1) with 1
on valid frames at each subsampling stage; kmask (1, 1, 1, T) 0 on valid positions and NEG on padding; valid
(1, 1, 1, T) the same as 1 / 0. Output y (1, 1024, 1, T).
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import NEG, chan, conv1x1, gelu, layer_norm, silu

F16 = np.float16
AUDIO_WIDTHS = (64, 128, 256, 376)      # 5.1, 10.2, 20.5 and 30 s
GPU_ABOVE = 256                         # compute="auto": wider entries run on the GPU (RESULTS.md)
ENC = "audio.encoder."


def checkpoint_audio(model: Path):
    spec = importlib.util.spec_from_file_location("d1_audio", Path(model) / "audio.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def conv2d(w: np.ndarray, b: np.ndarray, stride: int = 1, groups: int = 1) -> nn.Conv2d:
    c = nn.Conv2d(w.shape[1] * groups, w.shape[0], w.shape[2:], stride, padding=w.shape[2] // 2, groups=groups)
    c.weight = nn.Parameter(torch.from_numpy(np.ascontiguousarray(w, np.float32)), requires_grad=False)
    c.bias = nn.Parameter(torch.from_numpy(np.ascontiguousarray(b, np.float32)), requires_grad=False)
    return c


class Norm(nn.Module):
    def __init__(self, W: dict, p: str):
        super().__init__()
        self.register_buffer("w", chan(W[p + ".weight"]))
        self.register_buffer("b", chan(W[p + ".bias"]))

    def forward(self, x):
        return layer_norm(x, self.w, self.b, 1e-5)


class ConformerLayer(nn.Module):
    def __init__(self, W: dict, i: int, d: int, heads: int):
        super().__init__()
        p = f"{ENC}layers.{i}."
        lin = lambda n: conv1x1(W[p + n + ".weight"].reshape(W[p + n + ".weight"].shape[:2]), W.get(p + n + ".bias"))  # noqa: E731
        self.D, self.H = d, heads
        self.n1, self.na, self.nc, self.n2, self.no = (Norm(W, p + n) for n in (
            "norm_feed_forward1", "norm_self_att", "norm_conv", "norm_feed_forward2", "norm_out"))
        self.f1a, self.f1b = lin("feed_forward1.linear1"), lin("feed_forward1.linear2")
        self.f2a, self.f2b = lin("feed_forward2.linear1"), lin("feed_forward2.linear2")
        self.q, self.k, self.v, self.o = (lin(f"self_attn.linear_{n}") for n in ("q", "k", "v", "out"))
        self.register_buffer("u", torch.from_numpy(W[p + "self_attn.pos_bias_u"].astype(np.float32))[:, None, :])
        self.register_buffer("vb", torch.from_numpy(W[p + "self_attn.pos_bias_v"].astype(np.float32))[:, None, :])
        self.pos_w = torch.from_numpy(W[p + "self_attn.linear_pos.weight"].astype(np.float32))
        self.pw1, self.pw2 = lin("conv.pointwise_conv1"), lin("conv.pointwise_conv2")
        # depthwise conv with the eval-mode batch norm folded in: bn(conv(x)) = conv(x) * g + (beta - mean * g)
        g = W[p + "conv.batch_norm.weight"] / np.sqrt(W[p + "conv.batch_norm.running_var"] + 1e-5)
        dw, db = W[p + "conv.depthwise_conv.weight"], W[p + "conv.depthwise_conv.bias"]
        k = dw.shape[-1]
        self.dw = nn.Conv2d(d, d, (1, k), padding=(0, (k - 1) // 2), groups=d)
        self.dw.weight = nn.Parameter(torch.from_numpy((dw * g[:, None, None]).astype(np.float32)).reshape(d, 1, 1, k),
                                      requires_grad=False)
        self.dw.bias = nn.Parameter(torch.from_numpy(
            ((db - W[p + "conv.batch_norm.running_mean"]) * g + W[p + "conv.batch_norm.bias"]).astype(np.float32)),
            requires_grad=False)

    def positions(self, T: int) -> torch.Tensor:
        """linear_pos of the sinusoids for relative positions T-1 .. -(T-1): (H, d_k, 2T-1), a constant of the width."""
        pos = torch.arange(T - 1, -T, -1, dtype=torch.float32)[:, None]
        div = torch.exp(torch.arange(0, self.D, 2, dtype=torch.float32) * -(math.log(10000.0) / self.D))
        pe = torch.zeros(len(pos), self.D)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        return (pe @ self.pos_w.T).reshape(2 * T - 1, self.H, self.D // self.H).permute(1, 2, 0).contiguous()

    def attention(self, x, P, kmask):
        T, H, dk = x.shape[-1], self.H, self.D // self.H
        q, k, v = (m(x).reshape(H, dk, T) for m in (self.q, self.k, self.v))
        q = q.transpose(1, 2)                                         # (H, T, dk)
        ac = (q + self.u) @ k                                         # (H, T, T)
        bd = (q + self.vb) @ P                                        # (H, T, 2T-1): column c is relative position T-1-c
        bd = F.pad(bd, (1, 0)).reshape(H, 2 * T, T)[:, 1:].reshape(H, T, 2 * T - 1)[:, :, :T]   # -> column j is i - j
        p = torch.softmax((ac + bd) * dk ** -0.5 + kmask.reshape(1, 1, T), -1)
        o = (p @ v.transpose(1, 2)).permute(1, 0, 2).reshape(T, self.D)
        return self.o(o.transpose(0, 1).reshape(1, self.D, 1, T))

    def forward(self, x, P, kmask, valid):
        x = x + self.f1b(silu(self.f1a(self.n1(x)), "tanh")) * 0.5
        x = x + self.attention(self.na(x), P, kmask)
        a, b = self.pw1(self.nc(x)).chunk(2, dim=1)
        x = x + self.pw2(silu(self.dw(a * torch.sigmoid(b) * valid), "tanh"))
        x = x + self.f2b(silu(self.f2a(self.n2(x)), "tanh")) * 0.5
        return self.no(x)


class AudioCore(nn.Module):
    def __init__(self, W: dict, config: dict):
        super().__init__()
        cfg = config["audio_config"]
        d = self.D = cfg["d_model"]
        self.out_dim = config["text_config"]["hidden_size"]
        pre = ENC + "pre_encode."
        self.c0 = conv2d(W[pre + "conv.0.weight"], W[pre + "conv.0.bias"], 2)
        self.d1 = conv2d(W[pre + "conv.2.weight"], W[pre + "conv.2.bias"], 2, cfg["subsampling_conv_channels"])
        self.p1 = conv2d(W[pre + "conv.3.weight"], W[pre + "conv.3.bias"])
        self.d2 = conv2d(W[pre + "conv.5.weight"], W[pre + "conv.5.bias"], 2, cfg["subsampling_conv_channels"])
        self.p2 = conv2d(W[pre + "conv.6.weight"], W[pre + "conv.6.bias"])
        self.lin = conv1x1(W[pre + "out.weight"], W[pre + "out.bias"])
        self.layers = nn.ModuleList(ConformerLayer(W, i, d, cfg["n_heads"]) for i in range(cfg["n_layers"]))
        self.an = Norm(W, "audio.adapter.norm")
        self.a1 = conv1x1(W["audio.adapter.linear_1.weight"], W["audio.adapter.linear_1.bias"])
        self.a2 = conv1x1(W["audio.adapter.linear_2.weight"], W["audio.adapter.linear_2.bias"])
        self.rn = Norm(W, "audio.residual.ln")
        self.rd = conv1x1(W["audio.residual.down.weight"], W["audio.residual.down.bias"])
        self.ru = conv1x1(W["audio.residual.up.weight"], W["audio.residual.up.bias"])


class AudioEntry(nn.Module):
    def __init__(self, core: AudioCore, T: int):
        super().__init__()
        self.core, self.T = core, T
        for i, layer in enumerate(core.layers):
            self.register_buffer(f"P{i}", layer.positions(T))

    def input_names(self):
        return ["mel", "m0", "m1", "m2", "m3", "kmask", "valid"]

    def output_names(self):
        return ["y"]

    def example(self, dtype=torch.float16):
        return tuple(torch.zeros(s, dtype=dtype) for s in audio_shapes(self.T).values())

    def forward(self, mel, m0, m1, m2, m3, kmask, valid):
        c, T = self.core, self.T
        x = F.relu(c.c0(mel * m0) * m1)
        x = F.relu(c.p1(c.d1(x * m1) * m2) * m2)
        x = F.relu(c.p2(c.d2(x * m2) * m3) * m3) * m3                    # (1, C, T, F)
        x = c.lin(x.permute(0, 1, 3, 2).reshape(1, -1, 1, T))            # channel = c * F + f, as the checkpoint flattens
        for i, layer in enumerate(c.layers):
            x = layer(x, getattr(self, f"P{i}"), kmask, valid)
        x = c.a2(gelu(c.a1(c.an(x))))
        return x + c.ru(gelu(c.rd(c.rn(x))))


def audio_shapes(T: int) -> dict:
    return {"mel": (1, 1, 8 * T, 128), "m0": (1, 1, 8 * T, 1), "m1": (1, 1, 4 * T, 1), "m2": (1, 1, 2 * T, 1),
            "m3": (1, 1, T, 1), "kmask": (1, 1, 1, T), "valid": (1, 1, 1, T)}


def lengths(frames: int) -> list[int]:
    """Valid frames after each of the three stride-2 convolutions."""
    out = [frames]
    for _ in range(3):
        out.append((out[-1] + 2 - 3) // 2 + 1)
    return out


def fill_audio(v: dict, mel: np.ndarray, frames: int) -> int:
    """mel: (columns, 128) features of one clip. Writes an entry's inputs, returns the number of valid positions."""
    T = v["kmask"].shape[-1]
    n = min(mel.shape[0], 8 * T)
    v["mel"][0, 0, :n], v["mel"][0, 0, n:] = mel[:n], 0
    L = lengths(frames)
    for name, k in zip(("m0", "m1", "m2", "m3"), L):
        a = v[name].reshape(-1)
        a[:k], a[k:] = 1, 0
    v["kmask"].reshape(T)[:L[3]], v["kmask"].reshape(T)[L[3]:] = 0, NEG
    v["valid"].reshape(T)[:L[3]], v["valid"].reshape(T)[L[3]:] = 1, 0
    return L[3]


def export(model: Path, out: Path, widths=AUDIO_WIDTHS, log=print) -> dict:
    from .convert import load_weights, save_program

    t0 = time.time()
    config = json.loads((model / "config.json").read_text())
    core = AudioCore(load_weights(model, ("audio.",)), config).eval()
    entries = [(f"a{T}", AudioEntry(core, T).eval().to(torch.float16)) for T in widths]
    (out / "model" / "audio.py").write_bytes((model / "audio.py").read_bytes())
    mb = save_program(entries, out / "audio.aimodel")
    man = json.loads((out / "manifest.json").read_text())
    man["audio"] = {"file": "audio.aimodel", "widths": list(widths), "mb": round(mb, 1),
                    "entries": [n for n, _ in entries], "convert_s": round(time.time() - t0, 1)}
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    log(f"audio.aimodel: {mb:.0f} MB, {time.time() - t0:.1f}s")
    return man["audio"]


class AudioFrontend:
    """The checkpoint's waveform() and MelFrontend (PyTorch, FP32, CPU)."""

    def __init__(self, model: Path):
        self.A = checkpoint_audio(model)
        self.front = self.A.MelFrontend(128)

    def __call__(self, audio) -> tuple[np.ndarray, int]:
        mel, frames = self.front(self.A.waveform(audio))
        return mel[0].T.contiguous().numpy(), int(frames[0])                 # (columns, 128)


class AudioBackend:
    """A clip -> prefix embeddings through audio.aimodel (one call)."""

    def __init__(self, build: Path, compute: str = "ane", log=print):
        from coreai.runtime import NDArray
        from coreai.runtime._ndarray import StorageKind

        from .runtime import _writable, load_package

        self.build, self._NDArray = Path(build), NDArray
        man = json.loads((self.build / "manifest.json").read_text())["audio"]
        self.front = AudioFrontend(self.build / "model")
        self.widths = sorted(man["widths"])
        # auto: the GPU runs the 30 s entry in about half the ANE's time; the ANE is ahead or level below it
        self.unit = {T: ("gpu" if T > GPU_ABOVE else "ane") if compute == "auto" else compute for T in self.widths}
        self.loop, self.fns = asyncio.new_event_loop(), {}
        for unit in sorted(set(self.unit.values())):
            t0 = time.time()
            pkg = self.loop.run_until_complete(load_package(self.build / man["file"], unit))
            self.fns.update({T: pkg.load_function(f"a{T}") for T in self.widths if self.unit[T] == unit})
            log(f"loaded {man['file']} [{unit}] in {time.time() - t0:.2f}s")
        self.buf = {}
        for T in self.widths:
            nd = {n: NDArray(np.zeros(s, F16), StorageKind.IO_SURFACE) for n, s in audio_shapes(T).items()}
            self.buf[T] = ({n: a._tensor for n, a in nd.items()}, {n: _writable(a) for n, a in nd.items()}, nd)  # noqa: SLF001

    def prefix(self, audio) -> tuple[np.ndarray, dict]:
        """One 16 kHz mono clip (int16 or float samples) -> ((P, 1024) FP16 prefix, timing)."""
        t0 = time.perf_counter()
        mel, frames = self.front(audio)
        t1 = time.perf_counter()
        T = next(w for w in self.widths if 8 * w >= mel.shape[0] or w == self.widths[-1])
        tensors, v, _ = self.buf[T]
        n = fill_audio(v, mel, min(frames, 8 * T))

        async def run():
            raw = await self.fns[T]._function(inputs=tensors, state={})  # noqa: SLF001
            y = np.array(self._NDArray._wrap(raw["y"]).numpy()[0, :, 0, :n].T)  # noqa: SLF001
            raw.clear()
            return y
        y = self.loop.run_until_complete(run())
        return y, {"mel_ms": 1e3 * (t1 - t0), "audio_ms": 1e3 * (time.perf_counter() - t1), "audio_width": T,
                   "audio_backend": f"coreai-{self.unit[T]}"}


class HostAudio:
    """The checkpoint's own audio encoder in PyTorch (FP32)."""

    def __init__(self, model: Path, device: str = "cpu"):
        from safetensors import safe_open

        config = json.loads((Path(model) / "config.json").read_text())
        self.device = device
        self.net = checkpoint_audio(model).Audio(config["audio_config"], config["text_config"]["hidden_size"])
        with safe_open(str(Path(model) / "model.safetensors"), "pt") as f:
            self.net.load_state_dict({k[len("audio."):]: f.get_tensor(k) for k in f.keys() if k.startswith("audio.")})
        self.net = self.net.eval().to(device)

    def prefix(self, audio) -> tuple[np.ndarray, dict]:
        t0 = time.perf_counter()
        with torch.no_grad():
            p = self.net(audio)[0].float().cpu().numpy()
        return p.astype(F16), {"audio_ms": 1e3 * (time.perf_counter() - t0), "audio_backend": f"torch-{self.device}"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True, help="an existing build (d1ane.convert) to add audio to")
    ap.add_argument("--widths", default=",".join(map(str, AUDIO_WIDTHS)))
    a = ap.parse_args(argv)
    export(a.model, a.output, tuple(int(w) for w in a.widths.split(",")))


if __name__ == "__main__":
    sys.exit(main())
