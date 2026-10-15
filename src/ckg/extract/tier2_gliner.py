"""Tier 2: zero-shot NER with GLiNER, labeled with the schema's domain classes (DESIGN 5.2).

Chunk text is cut into windows that avoid masked spans and stay under the
model's input length. GLiNER returns character offsets within each window,
which are shifted back to chunk offsets and checked against the chunk text.
"""

from __future__ import annotations

import inspect
import re
from typing import Protocol

from ckg.extract.common import Span, dropped, mention, unmasked_segments

METHOD = "gliner"
# GLiNER's own word splitter; its max_len counts these tokens.
WORDS = re.compile(r"\w+(?:[-_]\w+)*|\S")


class NerModel(Protocol):
    def inference(self, texts: list[str], labels: list[str], threshold: float, batch_size: int) -> list[list[dict]]: ...


def class_labels(model_module, overrides: dict[str, str], exclude: list[str]) -> dict[str, str]:
    """Map GLiNER label text -> class name for every concrete domain class."""
    base = model_module.DomainEntity
    labels = {}
    for name, cls in sorted(inspect.getmembers(model_module, inspect.isclass)):
        if cls is base or not issubclass(cls, base) or name in exclude:
            continue
        text = overrides.get(name) or re.sub(r"(?<!^)(?=[A-Z])", " ", name).lower()
        labels[text] = name
    return labels


def windows(text: str, segments: list[tuple[int, int]], max_words: int) -> list[tuple[int, int]]:
    """Split unmasked segments into windows of at most max_words, breaking at line ends where possible."""
    out = []
    for seg_start, seg_end in segments:
        start, count = seg_start, 0
        for m in re.finditer(r"[^\n]*\n?", text[seg_start:seg_end]):
            if not m.group():
                continue
            line_start, line_end = seg_start + m.start(), seg_start + m.end()
            n = len(WORDS.findall(m.group()))
            if count and count + n > max_words:
                out.append((start, line_start))
                start, count = line_start, 0
            if n > max_words:  # one very long line: split it by words
                tokens = list(WORDS.finditer(text, line_start, line_end))
                for i in range(0, len(tokens), max_words):
                    piece = tokens[i : i + max_words]
                    end = tokens[i + max_words].start() if i + max_words < len(tokens) else line_end
                    out.append((start if i == 0 else piece[0].start(), end))
                start, count = line_end, 0
                continue
            count += n
        if start < seg_end and text[start:seg_end].strip():
            out.append((start, seg_end))
    return [(s, e) for s, e in out if text[s:e].strip()]


class GlinerExtractor:
    def __init__(self, model: NerModel, labels: dict[str, str], threshold: float, max_words: int,
                 batch_size: int, model_name: str) -> None:
        self.model = model
        self.labels = labels
        self.threshold = threshold
        self.max_words = max_words
        self.batch_size = batch_size
        self.model_name = model_name

    def extract(self, chunks: list[dict], masked: dict[str, list[Span]]) -> tuple[list[dict], list[dict]]:
        """Run on many chunks at once so windows from different chunks share batches."""
        jobs = []  # (chunk, window start, window end)
        for c in chunks:
            for s, e in windows(c["text"], unmasked_segments(c["text"], masked[c["chunk_id"]]), self.max_words):
                jobs.append((c, s, e))
        texts = [c["text"][s:e] for c, s, e in jobs]
        results = self.model.inference(texts, list(self.labels), threshold=self.threshold, batch_size=self.batch_size) if texts else []

        kept, rejected = [], []
        for (c, offset, _), ents in zip(jobs, results):
            text = c["text"]
            for ent in ents:
                start, end = offset + ent["start"], offset + ent["end"]
                record = mention(c["chunk_id"], text, start, end, METHOD, self.labels[ent["label"]],
                                 score=float(ent["score"]), model=self.model_name)
                if text[start:end] != ent["text"]:
                    rejected.append(dropped(record, "offset_mismatch"))
                elif not text[start:end].strip():
                    rejected.append(dropped(record, "empty"))
                else:
                    kept.append(record)
        key = {c["chunk_id"]: i for i, c in enumerate(chunks)}
        kept.sort(key=lambda m: (key[m["chunk_id"]], m["start"], m["end"]))
        return kept, rejected


def load_gliner(model: str, revision: str):
    """Download (once) and load a pinned GLiNER checkpoint. Returns (model, 'name@revision')."""
    import torch
    from gliner import GLiNER
    from huggingface_hub import snapshot_download

    torch.manual_seed(0)
    path = snapshot_download(model, revision=revision, allow_patterns=["gliner_config.json", "*.safetensors", "README.md"])
    m = GLiNER.from_pretrained(path)
    m.eval()
    return m, f"{model}@{revision[:12]}"
