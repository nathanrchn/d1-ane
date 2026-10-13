"""The ANE-shaped graph against the checkpoint's own modules, on tiny random weights (CPU, no Core AI, no weights).

Needs the checkpoint's code (encoder.py, vision.py) in models/d1-omni-600M; skipped without it.
"""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from d1ane.graph import Core, Stage, parse_plan  # noqa: E402
from d1ane.graph import shapes  # noqa: E402
from d1ane.runtime import fill  # noqa: E402
from d1ane.vision import unshuffle  # noqa: E402

MODEL = ROOT / "models" / "d1-omni-600M"
CFG = {"vocab_size": 50, "hidden_size": 128, "intermediate_size": 192, "num_hidden_layers": 4, "num_attention_heads": 4,
       "num_key_value_heads": 2, "layer_types": ["conv", "full_attention", "conv", "full_attention"], "norm_eps": 1e-5,
       "conv_L_cache": 3, "block_ffn_dim_multiplier": 1.0, "block_multiple_of": 32, "rope_theta": 1000000.0}


def checkpoint_module(name: str):
    spec = importlib.util.spec_from_file_location(f"d1_{name}", MODEL / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@unittest.skipUnless((MODEL / "encoder.py").is_file(), "checkpoint code not downloaded")
class GraphParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        E = checkpoint_module("encoder")
        cls.trunk, cls.head = E.Trunk(CFG).eval(), E.DecisionHead(CFG["hidden_size"], 2).eval()
        for p in list(cls.trunk.parameters()) + list(cls.head.parameters()):
            torch.nn.init.normal_(p, std=0.2 if p.ndim > 1 else 1.0)
        W = {f"encoder.{k}": v.numpy() for k, v in cls.trunk.state_dict().items()}
        W.update({f"head.{k}": v.numpy() for k, v in cls.head.state_dict().items()})
        cls.core = Core(W, {"text_config": CFG, "head_layers": 2}, silu_mode="native").eval()
        cls.type_emb = W["head.type_emb.weight"]

    def original(self, emb: torch.Tensor, prefix: int, qtype: int) -> np.ndarray:
        """Scores at every text position from the checkpoint's Trunk and DecisionHead."""
        L = emb.shape[0]
        pad = torch.ones(1, L, dtype=torch.bool)
        with torch.no_grad():
            h = self.trunk(emb[None], pad, torch.tensor([prefix]))
            text = h[:, prefix:]
            n = text.shape[1]
            z = self.head(text, torch.ones(1, n, dtype=torch.bool), torch.arange(n)[None],
                          torch.ones(1, n, dtype=torch.bool), torch.tensor([qtype]))
        return z[0].numpy()

    def mine(self, emb: torch.Tensor, prefix: int, qtype: int, T: int, plan: str = "all") -> np.ndarray:
        v = {n: np.zeros(s, np.float32) for n, s in shapes(T, CFG["hidden_size"]).items()}
        fill(v, emb.numpy(), prefix, self.type_emb[qtype])
        x = torch.from_numpy(v["x"])
        with torch.no_grad():
            for part in parse_plan(plan, CFG["num_hidden_layers"]):
                st = Stage(self.core, T, part["layers"], part["final"], part["head"]).eval()
                x = st(*(x if n == "x" else torch.from_numpy(v[n]) for n in st.input_names()))
        return x.numpy().reshape(-1)[prefix:emb.shape[0]]

    def check(self, L: int, prefix: int, T: int, qtype: int = 0, plan: str = "all"):
        emb = torch.randn(L, CFG["hidden_size"])
        want, got = self.original(emb, prefix, qtype), self.mine(emb, prefix, qtype, T, plan)
        np.testing.assert_allclose(got, want, rtol=2e-3, atol=2e-3)

    def test_text_exact_width(self):
        self.check(L=16, prefix=0, T=16)

    def test_text_padded(self):
        self.check(L=11, prefix=0, T=32, qtype=2)

    def test_media_prefix_padded(self):
        self.check(L=20, prefix=7, T=32, qtype=1)

    def test_media_prefix_exact(self):
        self.check(L=24, prefix=9, T=24)

    def test_chunked_plans_agree(self):
        for plan in ("2", "0-1,2-3,head"):
            self.check(L=13, prefix=4, T=16, plan=plan)


class Pieces(unittest.TestCase):
    def test_fill_masks(self):
        v = {n: np.zeros(s, np.float32) for n, s in shapes(8, 4).items()}
        fill(v, np.ones((5, 4), np.float32), 2, np.full(4, 3.0, np.float32))
        self.assertEqual(v["pad"].reshape(-1).tolist(), [1, 1, 1, 1, 1, 0, 0, 0])
        self.assertEqual((v["kmask"].reshape(-1) < 0).tolist(), [False] * 5 + [True] * 3)
        self.assertEqual(v["mq"].reshape(-1).tolist(), [1, 1, 0, 0, 0, 0, 0, 0])
        self.assertEqual((v["tk"].reshape(-1) < 0).tolist(), [False, False, True, True, True, False, False, False])
        self.assertEqual(v["keep"].reshape(-1).tolist(), [1, 0, 1, 1, 1, 1, 1, 1])
        self.assertEqual((v["hmask"].reshape(-1) < 0).tolist(), [True, True, False, False, False, True, True, True])
        self.assertTrue((v["x"][0, :, 0, 5:] == 0).all() and (v["qt"] == 3).all())

    def test_fill_text_only_keeps_every_right_tap(self):
        v = {n: np.zeros(s, np.float32) for n, s in shapes(4, 2).items()}
        fill(v, np.ones((3, 2), np.float32), 0, np.zeros(2, np.float32))
        self.assertEqual(v["keep"].reshape(-1).tolist(), [1, 1, 1, 1])
        self.assertEqual(v["mq"].reshape(-1).tolist(), [0, 0, 0, 0])

    def test_parse_plan(self):
        self.assertEqual(parse_plan("all", 16), [{"layers": list(range(16)), "final": True, "head": True}])
        four = parse_plan("4", 16)
        self.assertEqual([p["layers"][0] for p in four], [0, 4, 8, 12])
        self.assertEqual([p["head"] for p in four], [False, False, False, True])
        self.assertEqual(parse_plan("0-7,8-15,head", 16)[-1], {"layers": [], "final": False, "head": True})

    @unittest.skipUnless((MODEL / "vision.py").is_file(), "checkpoint code not downloaded")
    def test_unshuffle_matches_projector(self):
        V = checkpoint_module("vision")
        proj = V.Projector(6, 8, 5)
        proj.linear_1, proj.linear_2 = torch.nn.Identity(), torch.nn.Identity()
        proj.linear_2.out_features = 24
        x = torch.randn(4, 6, 6)                                           # h=4, w=6, c=6
        with unittest.mock.patch.object(V.F, "gelu", lambda t: t):
            want = proj(x[None])[0].numpy()
        np.testing.assert_array_equal(unshuffle(x.reshape(24, 6).numpy(), 4, 6), want)


if __name__ == "__main__":
    import unittest.mock  # noqa: F401
    unittest.main()
