"""Audio on the ANE against the checkpoint: prefix embeddings and audio decisions.

    python tests/audio_parity.py --model models/d1-omni-600M --build build/d1-omni-600M --audio bench/data/speech.flac
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from d1ane.audio import AudioBackend  # noqa: E402
from d1ane.reference import Reference  # noqa: E402
from d1ane.runtime import Decider  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", type=Path, required=True)
ap.add_argument("--build", type=Path, required=True)
ap.add_argument("--audio", type=Path, required=True)
ap.add_argument("--compute", default="ane")
a = ap.parse_args()
QS = {"topic": {"type": "choice", "instructions": "What is the speaker talking about?",
                "criteria": {"food": "Food and meals", "travel": "Travel and transport", "weather": "The weather"}},
      "question": {"type": "noul", "instructions": "Is the speaker asking a question?"}}
clip, rate = sf.read(a.audio, dtype="int16")
assert rate == 16000 and clip.ndim == 1
clips = {"clip": clip, "first 2 s": clip[:32000], "x4 (cut at 30 s)": np.tile(clip, 4)}
aud = AudioBackend(a.build, a.compute)
d = Decider(a.build, a.compute)
ref = Reference(a.model)
for name, c in clips.items():
    p_ref = ref.prefix(audio=c)
    p, t = aud.prefix(c)
    ms = []
    for _ in range(5):
        t0 = time.perf_counter()
        aud.prefix(c)
        ms.append(1e3 * (time.perf_counter() - t0))
    pf = p.astype(np.float32)
    assert pf.shape == p_ref.shape, (pf.shape, p_ref.shape)
    cos = (pf * p_ref).sum(-1) / (np.linalg.norm(pf, axis=-1) * np.linalg.norm(p_ref, axis=-1))
    print(f"{name}: {len(c) / 16000:.1f} s prefix={p.shape[0]} width={t['audio_width']} total {statistics.median(ms):.1f} ms "
          f"(mel {t['mel_ms']:.1f}, encoder {t['audio_ms']:.1f}) | min cos {cos.min():.5f} max|d| {np.abs(pf - p_ref).max():.3f} (|ref| max {np.abs(p_ref).max():.1f})")
    for qn, q in QS.items():
        want = np.array(ref.probabilities("Voice note from a user.", q, audio=c))
        got, tt = d.probabilities("Voice note from a user.", q, prefix=p, kind="audio")
        print(f"   {qn:8s} tokens={tt['tokens']:4d} width={tt['width']:4d} trunk {tt['run_ms']:6.1f} ms  ane={np.round(got, 3).tolist()} ref={np.round(want, 3).tolist()} max|dp|={np.abs(np.array(got) - want).max():.2e}")
