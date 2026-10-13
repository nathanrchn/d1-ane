"""The checkpoint's own prompt code (model/prompt.py) and a tokenizer for it without transformers."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def load_prompt(model: Path):
    """model/prompt.py as a module: encode(), answer(), as_question(), QTYPES, temperature_key()."""
    import sys

    spec = importlib.util.spec_from_file_location("d1_prompt", model / "prompt.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod            # its dataclass looks the module up by name
    spec.loader.exec_module(mod)
    return mod


class Tok:
    """The three tokenizer calls prompt.encode() makes, over the `tokenizers` library."""

    def __init__(self, model: Path, bos_token_id: int = 1):
        from tokenizers import Tokenizer

        self.t = Tokenizer.from_file(str(model / "tokenizer.json"))
        self.bos_token_id = bos_token_id

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        return {"input_ids": self.t.encode(text, add_special_tokens=add_special_tokens).ids}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.t.token_to_id(token)
