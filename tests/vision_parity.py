"""Vision on the ANE against the checkpoint: prefix embeddings and image decisions.

    python tests/vision_parity.py --model models/d1-omni-600M --build build/d1-omni-600M --image bench/data/cats.jpg
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from d1ane.reference import Reference  # noqa: E402
from d1ane.runtime import Decider  # noqa: E402
from d1ane.vision import VisionBackend  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", type=Path, required=True)
ap.add_argument("--build", type=Path, required=True)
ap.add_argument("--image", type=Path, required=True)
ap.add_argument("--compute", default="ane")
a = ap.parse_args()
QS = {"cats": {"type": "choice", "instructions": "How many cats are there?",
               "criteria": {"one": "One", "two": "Two", "more": "Three or more"}},
      "remote": {"type": "noul", "instructions": "Is there a remote control in the picture?"},
      "dog": {"type": "noul", "instructions": "Is there a dog in the picture?"},
      "indoors": {"type": "choice", "instructions": "Where was this taken?", "criteria": {"indoors": "Inside", "outdoors": "Outside"}}}
img = Image.open(a.image)
images = {"original": img, "small 320px": img.resize((320, 240)), "large 1600px (tiled)": img.resize((1600, 1200))}
vis = VisionBackend(a.build, a.compute)
d = Decider(a.build, a.compute)
ref = Reference(a.model)
for name, im in images.items():
    p_ref = ref.prefix(images=im)
    p, t = vis.prefix(im)
    ms = []
    for _ in range(5):
        t0 = time.perf_counter()
        vis.prefix(im)
        ms.append(1e3 * (time.perf_counter() - t0))
    pf = p.astype(np.float32)
    cos = (pf * p_ref).sum(-1) / (np.linalg.norm(pf, axis=-1) * np.linalg.norm(p_ref, axis=-1))
    print(f"{name}: {im.size} crops={t['crops']} prefix={p.shape[0]} vision {statistics.median(ms):.1f} ms "
          f"(preprocess {t['preprocess_ms']:.1f}) | min cos {cos.min():.5f} max|d| {np.abs(pf - p_ref).max():.3f} (|ref| max {np.abs(p_ref).max():.1f})")
    for qn, q in QS.items():
        want = np.array(ref.probabilities(None, q, images=[im]))
        t0 = time.perf_counter()
        got, tt = d.probabilities(None, q, prefix=p, kind="image")
        print(f"   {qn:8s} tokens={tt['tokens']:4d} width={tt['width']:4d} trunk {tt['run_ms']:6.1f} ms  ane={np.round(got, 3).tolist()} ref={np.round(want, 3).tolist()} max|dp|={np.abs(np.array(got) - want).max():.2e}")
