"""The checkpoint itself (transformers, trust_remote_code): the FP32 reference and the PyTorch speed baselines."""
from __future__ import annotations

import time
from pathlib import Path


class Reference:
    def __init__(self, model: Path, device: str = "cpu", dtype: str = "float32"):
        import torch
        from transformers import AutoModel

        self.torch, self.device = torch, device
        self.model = AutoModel.from_pretrained(str(model), trust_remote_code=True, dtype=getattr(torch, dtype))
        self.model = self.model.to(device).eval()

    def probabilities(self, state, question, images=None, audio=None) -> list[float]:
        return self.model.probabilities(state, [question], images, audio)[0]

    def system_one(self, state, questions: dict, images=None, audio=None) -> dict:
        return self.model.system_one(state, questions, images, audio)

    def prefix(self, images=None, audio=None):
        """(P, D) float32 prefix embeddings of the media, as the checkpoint's encoders produce them."""
        with self.torch.no_grad():
            p = self.model.vision(images if isinstance(images, list) else [images]) if images is not None \
                else self.model.audio(audio)
        return p[0].float().cpu().numpy()

    def sync(self):
        if self.device == "mps":
            self.torch.mps.synchronize()

    def timed(self, state, question, reps: int = 10, warmup: int = 2) -> list[float]:
        out = []
        for i in range(reps + warmup):
            t0 = time.perf_counter()
            self.probabilities(state, question)          # .tolist() inside forces the result back to the host
            if i >= warmup:
                out.append(1e3 * (time.perf_counter() - t0))
        return out
