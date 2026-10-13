"""The baseline: laya-multilingual's published Core ML bucket (huggingface.co/FluidInference/laya-coreml), from Python.

    d1ane setup --laya        # puts the L128 bucket and its tokenizer in models/laya-coreml

It is the package FluidUse's Tetris demo runs, called through coremltools instead of their Swift runtime. The
sequence format and limits follow their LayaSequenceBuilder (a port of laya's build_sequence):
    [CLS] <type> question: instructions [SEP] ([MASK] option)* [SEP] state [SEP]
On the 410 rows of their published five-task set that fit 128 tokens it gives their reference's token counts on
every row and its answer on 409.
"""
import time
from pathlib import Path

import coremltools as ct
import numpy as np
from tokenizers import Tokenizer

QTYPE = {"choice": 0, "score": 1, "noul": 2}
NOUL = ["false: no, the statement does not hold", "true: yes, the statement holds"]


class Laya:
    def __init__(self, models: Path, length: int = 128, units: str = "ane", head_max: int = 256):
        unit = {"ane": ct.ComputeUnit.CPU_AND_NE, "all": ct.ComputeUnit.ALL, "gpu": ct.ComputeUnit.CPU_AND_GPU}[units]
        self.model = ct.models.CompiledMLModel(str(models / f"laya_multilingual_fp16_L{length}_options32.mlmodelc"),
                                               compute_units=unit)
        self.tok = Tokenizer.from_file(str(models / "tokenizer.json"))
        self.L, self.head_max = length, head_max
        self.cls, self.sep, self.mask, self.pad = (self.tok.token_to_id(t) for t in ("<bos>", "<eos>", "<mask>", "<pad>"))

    def enc(self, text: str) -> list[int]:
        return self.tok.encode(text.replace("<mask>", " "), add_special_tokens=False).ids

    def sequence(self, state: str, qtype: str, instructions: str, options: list[str]) -> tuple[list[int], list[int]]:
        head = self.enc(f"{qtype} question: {instructions}")
        opts = [[self.mask] + self.enc(" " + o)[:48] for o in options]
        budget = self.head_max - sum(map(len, opts))
        if budget < 16:
            per = max(4, (self.head_max - 16) // max(1, len(opts)))
            opts = [o[:per] for o in opts]
            budget = self.head_max - sum(map(len, opts))
        ids, markers = [self.cls] + head[:max(8, budget)] + [self.sep], []
        for o in opts:
            markers.append(len(ids))
            ids += o
        ids.append(self.sep)
        ids = (ids + self.enc(state)[:max(0, self.L - len(ids) - 1)] + [self.sep])[:self.L]
        if any(m >= self.L for m in markers):
            raise ValueError("prompt too long for the bucket")
        return ids, markers

    def probabilities(self, state: str, qtype: str, instructions: str, options: list[str] | None = None):
        """(probabilities over the options, model call ms, tokens). A noul's options are [false, true]."""
        ids, markers = self.sequence(state, qtype, instructions, options or NOUL)
        x = np.full((1, self.L), self.pad, np.int32)
        x[0, :len(ids)] = ids
        att = np.zeros((1, self.L), np.int32)
        att[0, :len(ids)] = 1
        mm = np.zeros((1, 32, self.L), np.float32)
        mm[0, np.arange(len(markers)), markers] = 1
        qt = np.zeros((1, 3), np.float32)
        qt[0, QTYPE[qtype]] = 1
        t0 = time.perf_counter()
        out = self.model.predict({"input_ids": x, "attention_mask": att, "marker_map": mm, "question_type": qt})
        ms = 1e3 * (time.perf_counter() - t0)
        z = np.asarray(out["logits"], np.float64).reshape(-1)[:len(markers)]
        e = np.exp(z - z.max())
        return (e / e.sum()).tolist(), ms, len(ids)
