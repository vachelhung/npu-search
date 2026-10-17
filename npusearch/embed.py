"""Text embedding with multilingual-e5-small on OpenVINO (NPU by default).

The NPU needs static shapes, so the model is reshaped to a fixed
[BATCH, SEQ_LEN] and every batch is padded to that size.
"""
from pathlib import Path

import numpy as np
import openvino as ov
from openvino.preprocess import PrePostProcessor
from tokenizers import Tokenizer

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "me5-small"
SEQ_LEN = 256
BATCH = 8


class Embedder:
    def __init__(self, device: str = "NPU", model_dir: Path = MODEL_DIR):
        self.device = device
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(SEQ_LEN)
        self.tokenizer.enable_padding(length=SEQ_LEN)

        core = ov.Core()
        core.set_property({"CACHE_DIR": str(model_dir / "cache")})  # speeds up NPU compile after first run
        model = core.read_model(model_dir / "openvino_model.xml")
        self.input_names = [i.get_any_name() for i in model.inputs]
        # NPU reads int64 inputs incorrectly on this driver; feed int32 instead.
        ppp = PrePostProcessor(model)
        for name in self.input_names:
            ppp.input(name).tensor().set_element_type(ov.Type.i32)
        model = ppp.build()
        model.reshape({name: [BATCH, SEQ_LEN] for name in self.input_names})
        config = {}
        if device == "NPU":
            # The compiler in NPU driver 32.0.100.4023 produces wrong results for this
            # model; the compiler bundled with OpenVINO is correct.
            config["NPU_COMPILER_TYPE"] = "PLUGIN"
        self.compiled = core.compile_model(model, device, config)

    def _run(self, texts: list[str]) -> np.ndarray:
        n = len(texts)
        texts = texts + [""] * (BATCH - n)
        enc = self.tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in enc], dtype=np.int32)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int32)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        hidden = self.compiled(feeds)[0][:n].astype(np.float32)
        m = mask[:n, :, None].astype(np.float32)
        pooled = (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
        return pooled / np.linalg.norm(pooled, axis=1, keepdims=True)

    def embed(self, texts: list[str], prefix: str) -> np.ndarray:
        """prefix is "passage: " for documents and "query: " for searches (e5 convention)."""
        texts = [prefix + t for t in texts]
        out = [self._run(texts[i:i + BATCH]) for i in range(0, len(texts), BATCH)]
        return np.vstack(out) if out else np.zeros((0, 384), dtype=np.float32)

    def embed_query(self, q: str) -> np.ndarray:
        return self.embed([q], "query: ")[0]

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self.embed(texts, "passage: ")
