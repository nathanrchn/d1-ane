"""ANE-shaped graphs of d1-omni's trunk and decision head.

The checkpoint's modules (encoder.py in the model repo) work on (B, L, D) with nn.Linear. Here every activation is
(1, C, 1, T) in FP16 with 1x1 convolutions, the layout the Apple Neural Engine runs natively, at a fixed width T.
d1 is a bidirectional encoder, so a decision is one fixed-shape forward: no KV cache, no chained calls, and the RoPE
tables are constants of each width.

Inputs of a stage (all FP16; the host builds them, see runtime.py):
    x      (1, D, 1, T)  embeddings: media prefix (if any), then text, zeros on the padding
    kmask  (1, 1, 1, T)  0 on real positions, NEG on padding                       (trunk attention keys)
    mq     (1, 1, T, 1)  1 on media positions, else 0                              (media queries ...)
    tk     (1, 1, 1, T)  NEG on text positions, else 0                             (... never read text keys)
    pad    (1, 1, 1, T)  1 on real positions, else 0                               (convolution input)
    keep   (1, 1, 1, T)  0 on the last media position, else 1                      (its right tap reads no text)
    hmask  (1, 1, 1, T)  0 on text positions, NEG on media and padding             (head attention keys)
    qt     (1, D, 1, 1)  the question-type embedding
Output of the last stage: scores (1, T), the scorer applied at every position; the host reads the option markers.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NEG = -1e4          # additive mask value that fits FP16 (the checkpoint uses finfo.min)
NORM_FLOOR = 2.0 ** -6


def conv1x1(w: np.ndarray, b: np.ndarray | None = None) -> nn.Conv2d:
    c = nn.Conv2d(w.shape[1], w.shape[0], 1, bias=b is not None)
    c.weight = nn.Parameter(torch.from_numpy(np.ascontiguousarray(w, np.float32)).reshape(*w.shape, 1, 1),
                            requires_grad=False)
    if b is not None:
        c.bias = nn.Parameter(torch.from_numpy(np.ascontiguousarray(b, np.float32)), requires_grad=False)
    return c


def rms(x, w, eps: float, dim: int = 1):
    """RMSNorm without squaring raw activations: xs = x / max|x| keeps the squares in [0, 1] (FP16 overflows the plain
    form above |x| = 255). Exact for any scale m: xs * rsqrt(mean(xs^2) + eps / m^2) = x * rsqrt(mean(x^2) + eps)."""
    m = x.abs().amax(dim, keepdim=True).clamp_min(NORM_FLOOR)
    xs = x / m
    return xs * torch.rsqrt((xs * xs).mean(dim, keepdim=True) + eps / (m * m)) * w


def layer_norm(x, w, b, eps: float):
    """nn.LayerNorm over the channel axis, in the same scale-free form."""
    xc = x - x.mean(1, keepdim=True)
    m = xc.abs().amax(1, keepdim=True).clamp_min(NORM_FLOOR)
    xs = xc / m
    return xs * torch.rsqrt((xs * xs).mean(1, keepdim=True) + eps / (m * m)) * w + b


def silu(x, mode: str):
    if mode == "native":
        return F.silu(x)
    half = x * 0.5          # x * sigmoid(x) = x/2 * (1 + tanh(x/2)); the ANE's own silu is coarse near 0
    return half * (1 + torch.tanh(half))


def gelu(x):
    """GELU written out. The ANE's own gelu is coarse: with it the audio prefix was 4.3% off PyTorch FP32 (relative
    L2), with this form 0.4% (FP16 in PyTorch: 0.2%)."""
    return 0.5 * x * (1 + torch.erf(x * 0.7071067811865476))


def gelu_tanh(x):
    """PyTorch's gelu(approximate="tanh"), written out for the same reason. x^2 may overflow FP16; tanh saturates."""
    return 0.5 * x * (1 + torch.tanh(0.7978845608028654 * x * (1 + 0.044715 * x * x)))


def shapes(T: int, D: int) -> dict[str, tuple]:
    """The inputs of a stage at width T."""
    return {"x": (1, D, 1, T), "kmask": (1, 1, 1, T), "mq": (1, 1, T, 1), "tk": (1, 1, 1, T), "pad": (1, 1, 1, T),
            "keep": (1, 1, 1, T), "hmask": (1, 1, 1, T), "qt": (1, D, 1, 1)}


def chan(v: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(v, np.float32)).reshape(1, -1, 1, 1)


class Attention(nn.Module):
    """LFM2 grouped-query attention with RoPE and per-head q/k RMSNorm."""

    def __init__(self, W: dict, p: str, cfg: dict):
        super().__init__()
        self.H, self.KV = cfg["num_attention_heads"], cfg["num_key_value_heads"]
        self.D, self.eps = cfg["hidden_size"], cfg["norm_eps"]
        self.hd, self.G = self.D // self.H, self.H // self.KV
        self.q, self.k, self.v, self.o = (conv1x1(W[f"{p}{n}.weight"]) for n in ("q_proj", "k_proj", "v_proj", "out_proj"))
        self.register_buffer("qn", torch.from_numpy(W[p + "q_layernorm.weight"].astype(np.float32)))
        self.register_buffer("kn", torch.from_numpy(W[p + "k_layernorm.weight"].astype(np.float32)))

    def forward(self, x, cos, sin, mask):
        T, hd, half = x.shape[-1], self.hd, self.hd // 2

        def rope(t):
            rot = torch.cat((-t[..., half:], t[..., :half]), -1)
            return t * cos + rot * sin

        q = rope(rms(self.q(x).reshape(self.H, hd, T).transpose(1, 2), self.qn, self.eps, -1))
        k = rope(rms(self.k(x).reshape(self.KV, hd, T).transpose(1, 2), self.kn, self.eps, -1))
        v = self.v(x).reshape(self.KV, hd, T).transpose(1, 2)
        s = (q.reshape(self.KV, self.G * T, hd) @ k.transpose(1, 2)) * hd ** -0.5
        p = torch.softmax(s.reshape(self.KV, self.G, T, T) + mask, -1).reshape(self.KV, self.G * T, T)
        o = (p @ v).reshape(self.KV, self.G, T, hd).permute(2, 0, 1, 3).reshape(T, self.D)
        return self.o(o.transpose(0, 1).reshape(1, self.D, 1, T))


class ShortConv(nn.Module):
    """out(C * conv(B * x)): centred 3-tap depthwise convolution. The right tap is a second convolution so `keep`
    can switch it off at the last media position."""

    def __init__(self, W: dict, p: str, cfg: dict):
        super().__init__()
        d = cfg["hidden_size"]
        self.inp, self.out = conv1x1(W[p + "in_proj.weight"]), conv1x1(W[p + "out_proj.weight"])
        w = W[p + "conv.weight"][:, 0, :].astype(np.float32)             # (D, 3): taps on t-1, t, t+1
        left, right = w.copy(), np.zeros_like(w)
        left[:, 2], right[:, 2] = 0, w[:, 2]
        self.left, self.right = (nn.Conv2d(d, d, (1, 3), padding=(0, 1), groups=d, bias=False) for _ in range(2))
        for conv, taps in ((self.left, left), (self.right, right)):
            conv.weight = nn.Parameter(torch.from_numpy(taps).reshape(d, 1, 1, 3), requires_grad=False)

    def forward(self, x, pad, keep):
        b, c, u = self.inp(x * pad).chunk(3, dim=1)
        bx = b * u
        return self.out(c * (self.left(bx) + self.right(bx) * keep))


class Layer(nn.Module):
    def __init__(self, W: dict, i: int, cfg: dict, silu_mode: str):
        super().__init__()
        p = f"encoder.layers.{i}."
        self.attention, self.eps, self.silu_mode = cfg["layer_types"][i] == "full_attention", cfg["norm_eps"], silu_mode
        self.op = Attention(W, p + "self_attn.", cfg) if self.attention else ShortConv(W, p + "conv.", cfg)
        self.w1, self.w2, self.w3 = (conv1x1(W[f"{p}feed_forward.w{n}.weight"]) for n in (1, 2, 3))
        self.register_buffer("on", chan(W[p + "operator_norm.weight"]))
        self.register_buffer("fn", chan(W[p + "ffn_norm.weight"]))

    def forward(self, h, cos, sin, mask, pad, keep):
        x = rms(h, self.on, self.eps)
        h = h + (self.op(x, cos, sin, mask) if self.attention else self.op(x, pad, keep))
        x = rms(h, self.fn, self.eps)
        return h + self.w2(silu(self.w1(x), self.silu_mode) * self.w3(x))


class HeadLayer(nn.Module):
    """nn.TransformerEncoderLayer(norm_first=True, activation=relu), as the checkpoint's decision head builds it."""

    def __init__(self, W: dict, i: int, d: int):
        super().__init__()
        p = f"head.head.layers.{i}."
        self.H, self.D = d // 64, d
        w, b = W[p + "self_attn.in_proj_weight"], W[p + "self_attn.in_proj_bias"]
        self.q, self.k, self.v = (conv1x1(w[j * d:(j + 1) * d], b[j * d:(j + 1) * d]) for j in range(3))
        self.o = conv1x1(W[p + "self_attn.out_proj.weight"], W[p + "self_attn.out_proj.bias"])
        self.l1 = conv1x1(W[p + "linear1.weight"], W[p + "linear1.bias"])
        self.l2 = conv1x1(W[p + "linear2.weight"], W[p + "linear2.bias"])
        for n in ("norm1", "norm2"):
            self.register_buffer(n + "w", chan(W[f"{p}{n}.weight"]))
            self.register_buffer(n + "b", chan(W[f"{p}{n}.bias"]))

    def forward(self, h, hmask):
        T, hd = h.shape[-1], self.D // self.H
        x = layer_norm(h, self.norm1w, self.norm1b, 1e-5)
        q, k, v = (m(x).reshape(self.H, hd, T) for m in (self.q, self.k, self.v))
        p = torch.softmax((q.transpose(1, 2) @ k) * hd ** -0.5 + hmask.reshape(1, 1, T), -1)
        o = (p @ v.transpose(1, 2)).permute(1, 0, 2).reshape(T, self.D)
        h = h + self.o(o.transpose(0, 1).reshape(1, self.D, 1, T))
        x = layer_norm(h, self.norm2w, self.norm2b, 1e-5)
        return h + self.l2(F.relu(self.l1(x)))


class Core(nn.Module):
    """The weights, shared by every stage and width of one build."""

    def __init__(self, W: dict, config: dict, silu_mode: str = "tanh"):
        super().__init__()
        cfg = self.cfg = config["text_config"]
        d = self.D = cfg["hidden_size"]
        self.eps = cfg["norm_eps"]
        self.layers = nn.ModuleList(Layer(W, i, cfg, silu_mode) for i in range(cfg["num_hidden_layers"]))
        self.register_buffer("final", chan(W["encoder.embedding_norm.weight"]))
        self.head = nn.ModuleList(HeadLayer(W, i, d) for i in range(config["head_layers"]))
        self.register_buffer("snw", chan(W["head.scorer.0.weight"]))
        self.register_buffer("snb", chan(W["head.scorer.0.bias"]))
        self.s1 = conv1x1(W["head.scorer.1.weight"], W["head.scorer.1.bias"])
        self.s3 = conv1x1(W["head.scorer.3.weight"], W["head.scorer.3.bias"])

    def rope(self, T: int) -> tuple[torch.Tensor, torch.Tensor]:
        hd = self.D // self.cfg["num_attention_heads"]
        inv = 1.0 / self.cfg["rope_theta"] ** (np.arange(0, hd, 2, dtype=np.float64) / hd)
        ang = np.outer(np.arange(T, dtype=np.float64), inv)
        ang = np.concatenate([ang, ang], -1).astype(np.float32)
        return torch.from_numpy(np.cos(ang)), torch.from_numpy(np.sin(ang))


class Stage(nn.Module):
    """Trunk layers `layers` at width T, then (optionally) the final norm and the decision head."""

    def __init__(self, core: Core, T: int, layers: list[int], final: bool, head: bool):
        super().__init__()
        self.core, self.T, self.idx, self.final, self.with_head = core, T, list(layers), final, head
        cos, sin = core.rope(T)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        kinds = [core.layers[i].attention for i in self.idx]
        self.names = ["x"]
        self.names += ["kmask", "mq", "tk"] if any(kinds) else []
        self.names += ["pad", "keep"] if not all(kinds) else []
        self.names += ["hmask", "qt"] if head else []

    def input_names(self) -> list[str]:
        return list(self.names)

    def output_names(self) -> list[str]:
        return ["scores" if self.with_head else "y"]

    def shapes(self) -> dict[str, tuple]:
        return shapes(self.T, self.core.D)

    def example(self, dtype=torch.float16) -> tuple:
        s = self.shapes()
        return tuple(torch.zeros(s[n], dtype=dtype) for n in self.names)

    def forward(self, *args):
        a = dict(zip(self.names, args))
        h, c = a["x"], self.core
        mask = a["kmask"] + a["mq"] * a["tk"] if "kmask" in a else None
        for i in self.idx:
            h = c.layers[i](h, self.cos, self.sin, mask, a.get("pad"), a.get("keep"))
        if self.final:
            h = rms(h, c.final, c.eps)
        if not self.with_head:
            return h
        h = h + a["qt"]
        for layer in c.head:
            h = layer(h, a["hmask"])
        s = c.s3(gelu(c.s1(layer_norm(h, c.snw, c.snb, 1e-5))))
        return s.reshape(1, self.T)


def parse_plan(plan: str, n_layers: int) -> list[dict]:
    """"all" -> one package; "4" -> 4-layer chunks, head with the last; "0-7,8-15,head" -> as written."""
    if plan == "all":
        return [{"layers": list(range(n_layers)), "final": True, "head": True}]
    if plan.isdigit():
        step = int(plan)
        parts = [list(range(a, min(a + step, n_layers))) for a in range(0, n_layers, step)]
        return [{"layers": p, "final": p[-1] == n_layers - 1, "head": p[-1] == n_layers - 1} for p in parts]
    out = []
    for part in plan.split(","):
        if part == "head":
            out.append({"layers": [], "final": False, "head": True})
        else:
            a, b = (int(v) for v in part.split("-"))
            out.append({"layers": list(range(a, b + 1)), "final": b == n_layers - 1, "head": False})
    return out
