"""Tier 3: schema-guided relationship extraction with a local LLM (DESIGN 5.2).

Follows the SPIRES pattern (Caufield et al., 2024): the prompt and the JSON
output schema are both generated from the LinkML schema, through
schema/generated/ckg.predicates.json. The model sees the chunk text and the
mentions from tiers 1 and 2, and returns assertions with evidence. Every
assertion is checked before it is kept; rejects are logged with a reason.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ckg.extract.common import Span, mention

METHOD = "llm"
PREFIX = "ckgd:"  # CURIE prefix of the domain schema's predicates
CODE_MARKER = "[code block omitted]"
# Bump when prompt construction or validation changes, so cached responses are redone.
PROMPT_VERSION = 2
CODE_INCLUDE = re.compile(r"^[ \t]*\[!(?:notebook|code)-[^\n]*$", re.M | re.I)


# --- schema -----------------------------------------------------------------------


@dataclass(frozen=True)
class Tier3Schema:
    classes: list[dict]
    predicates: dict[str, dict]
    condition_types: list[dict]

    @classmethod
    def load(cls, path: Path) -> "Tier3Schema":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data["classes"], {p["name"]: p for p in data["predicates"]}, data["condition_types"])

    @property
    def class_names(self) -> list[str]:
        return [c["name"] for c in self.classes]

    def render_system(self, template: str) -> str:
        def owner(d):
            return "any" if d in (None, "DomainEntity") else (d if isinstance(d, str) else " | ".join(d))

        classes = "\n".join(f"- {c['name']}: {_strip_cq(c['description'])}" for c in self.classes)
        preds = []
        for p in self.predicates.values():
            target = owner(p["range"]) if p["kind"] == "relation" else p["range"]
            if "values" in p:
                target = "one of " + ", ".join(p["values"])
            preds.append(f"- {p['name']} ({p['kind']}): {owner(p['domain'])} -> {target}. {_strip_cq(p['description'])}".rstrip())
        conds = "\n".join(f"- {c['name']}: {_strip_cq(c['description'])}".rstrip(": ") for c in self.condition_types)
        return template.replace("{classes}", classes).replace("{predicates}", "\n".join(preds)).replace("{condition_types}", conds)

    def output_schema(self, max_items: int | None = None) -> dict:
        cls_enum = self.class_names + ["unknown"]
        item = {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "subject_class": {"type": "string", "enum": cls_enum},
                "predicate": {"type": "string", "enum": list(self.predicates)},
                "object": {"type": "string"},
                "object_class": {"type": "string", "enum": cls_enum},
                "object_value": {"type": "string"},
                "negated": {"type": "boolean"},
                "conditions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": [c["name"] for c in self.condition_types]},
                            "value": {"type": "string"},
                        },
                        "required": ["type", "value"],
                    },
                },
                "evidence": {"type": "string"},
            },
            "required": ["subject", "subject_class", "predicate", "object", "object_class",
                         "object_value", "negated", "conditions", "evidence"],
        }
        array = {"type": "array", "items": item}
        if max_items:
            array["maxItems"] = max_items
        return {"type": "object", "properties": {"assertions": array}, "required": ["assertions"]}


def _strip_cq(text: str) -> str:
    return re.sub(r"\s*\(CQ[^)]*\)", "", text or "").strip()


# --- prompt -----------------------------------------------------------------------


def entity_lines(mentions: list[dict]) -> list[str]:
    seen, out = set(), []
    for m in mentions:
        key = (m["text"].lower(), m.get("predicted_class"))
        if key not in seen:
            seen.add(key)
            out.append(f"- {m['text']} ({m.get('predicted_class') or 'unknown'})")
    return out


def omitted_spans(text: str, code_spans: list[Span]) -> list[Span]:
    """Code blocks plus Learn code-include lines, merged and sorted."""
    spans = sorted(list(code_spans) + [Span(m.start(), m.end(), "code_include") for m in CODE_INCLUDE.finditer(text)],
                   key=lambda s: (s.start, s.end))
    merged: list[Span] = []
    for s in spans:
        if merged and s.start <= merged[-1].end:
            merged[-1] = Span(merged[-1].start, max(merged[-1].end, s.end), merged[-1].reason)
        else:
            merged.append(s)
    return merged


def user_message(chunk: dict, mentions: list[dict], code_spans: list[Span] | None) -> str:
    text = chunk["text"]
    if code_spans is not None:
        code_spans = omitted_spans(text, code_spans)
    if code_spans:
        parts, pos = [], 0
        for s in code_spans:
            parts.append(text[pos : s.start])
            parts.append(CODE_MARKER + "\n")
            pos = s.end
        parts.append(text[pos:])
        text = "".join(parts)
    heading = " > ".join(chunk["heading_path"])
    return (
        f"Section: {heading}\n\nEntities found in this section:\n" + "\n".join(entity_lines(mentions))
        + f"\n\nSection text:\n<<<\n{text.strip()}\n>>>"
    )


# --- LLM client -------------------------------------------------------------------


@dataclass(frozen=True)
class LlmResult:
    content: str
    prompt_tokens: int
    output_tokens: int
    seconds: float


class LlmClient(Protocol):
    model_id: str

    def chat(self, system: str, user: str, schema: dict) -> LlmResult: ...


class OllamaClient:
    def __init__(self, url: str, model: str, temperature: float, seed: int, num_ctx: int,
                 max_output_tokens: int, timeout: int) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.options = {"temperature": temperature, "seed": seed, "num_ctx": num_ctx, "num_predict": max_output_tokens}
        self.timeout = timeout
        tags = self._request("/api/tags", None, 30)
        digest = next((m["digest"] for m in tags.get("models", []) if m["name"] in (model, f"{model}:latest")), None)
        if digest is None:
            raise RuntimeError(f"Ollama model {model!r} is not pulled; run `ollama pull {model}`")
        self.model_id = f"{model}@{digest[:12]}"

    def _request(self, path: str, body: dict | None, timeout: int) -> dict:
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(self.url + path, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def chat(self, system: str, user: str, schema: dict) -> LlmResult:
        start = time.perf_counter()
        r = self._request("/api/chat", {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "format": schema,
            "stream": False,
            "options": self.options,
        }, self.timeout)
        return LlmResult(r["message"]["content"], r.get("prompt_eval_count", 0), r.get("eval_count", 0),
                         round(time.perf_counter() - start, 3))


# --- validation -------------------------------------------------------------------


@dataclass
class ChunkResult:
    assertions: list[dict] = field(default_factory=list)
    mentions: list[dict] = field(default_factory=list)  # new mentions created for tier 3
    rejected: list[dict] = field(default_factory=list)


def find_evidence(text: str, evidence: str, whitespace_tolerant: bool) -> tuple[int, int] | None:
    evidence = evidence.strip()
    if not evidence:
        return None
    i = text.find(evidence)
    if i >= 0:
        return i, i + len(evidence)
    if whitespace_tolerant:
        pattern = r"\s+".join(re.escape(w) for w in evidence.split())
        m = re.search(pattern, text)
        if m:
            return m.start(), m.end()
    return None


class Validator:
    def __init__(self, schema: Tier3Schema, model_id: str, whitespace_tolerant: bool) -> None:
        self.schema = schema
        self.model_id = model_id
        self.whitespace_tolerant = whitespace_tolerant

    def validate(self, chunk: dict, doc_path: str, mentions: list[dict], raw: str) -> ChunkResult:
        out = ChunkResult()
        text, cid = chunk["text"], chunk["chunk_id"]
        items, truncated = parse_items(raw)
        if items is None:
            out.rejected.append({"chunk_id": cid, "reason": "invalid_json", "raw": raw[:2000]})
            return out
        if truncated:  # complete items before the cut are still checked and kept
            out.rejected.append({"chunk_id": cid, "reason": "truncated_output", "raw": raw[-500:]})
        pool = list(mentions)
        seen = set()
        for item in items:
            reason, record = self._check(item, chunk, doc_path, pool, out)
            if reason:
                out.rejected.append({"chunk_id": cid, "reason": reason, "item": item})
                continue
            key = (record["subject"], record["predicate"], record.get("object"), record.get("object_value"),
                   record["negated"], json.dumps(record["conditions"], sort_keys=True))
            if key in seen:
                out.rejected.append({"chunk_id": cid, "reason": "duplicate", "item": item})
                continue
            seen.add(key)
            out.assertions.append(record)
        return out

    def _check(self, item: dict, chunk: dict, doc_path: str, pool: list[dict], out: ChunkResult):
        text, cid = chunk["text"], chunk["chunk_id"]
        pred = self.schema.predicates.get(item.get("predicate", ""))
        if pred is None:
            return "unknown_predicate", None
        span = find_evidence(text, item.get("evidence", ""), self.whitespace_tolerant)
        if span is None:
            return "evidence_not_in_chunk", None
        subject = self._mention_for(item.get("subject", ""), item.get("subject_class"), chunk, span, pool, out)
        if subject is None:
            return "subject_not_in_chunk", None
        obj, value = None, None
        if pred["kind"] == "relation":
            if not item.get("object", "").strip():
                return "relation_without_object", None
            obj = self._mention_for(item["object"], item.get("object_class"), chunk, span, pool, out)
            if obj is None:
                return "object_not_in_chunk", None
            if obj["id"] == subject["id"]:
                return "self_reference", None
        else:
            value = (item.get("object_value") or item.get("object") or "").strip()
            if not value:
                return "attribute_without_value", None
            if "values" in pred:
                match = next((v for v in pred["values"] if v.lower() == value.lower().replace(" ", "_")), None)
                if match is None:
                    return "invalid_value", None
                value = match
        conditions = [{"type": c["type"], "value": c["value"].strip()} for c in item.get("conditions", []) if c.get("value", "").strip()]
        negated = bool(item.get("negated"))
        provenance = {
            "source_doc": chunk["doc_id"],
            "chunk_id": cid,
            "start": span[0],
            "end": span[1],
            "evidence": text[span[0] : span[1]],
            "method": METHOD,
            "model": self.model_id,
        }
        include = _source_at(chunk, span[0])
        if include and include != doc_path:
            provenance["include_file"] = include
        target = obj["id"] if obj else value
        digest = hashlib.sha256(json.dumps([cid, subject["id"], pred["name"], target, negated, conditions], sort_keys=True).encode()).hexdigest()[:16]
        record = {
            "id": f"ckg:assertion-{digest}",
            "subject": subject["id"],
            "predicate": PREFIX + pred["name"],
            "negated": negated,
            "conditions": conditions,
            "provenance": [provenance],
        }
        if obj:
            record["object"] = obj["id"]
        else:
            record["object_value"] = value
        return None, record

    def _mention_for(self, name: str, cls: str | None, chunk: dict, span: tuple[int, int],
                     pool: list[dict], out: ChunkResult) -> dict | None:
        """Find the mention the model meant; create a tier 3 mention if the text exists but wasn't found before."""
        name = name.strip()
        # The entity list is written "name (Class)"; models sometimes copy the suffix.
        suffix = re.fullmatch(r"(.+?)\s*\((\w+)\)", name)
        if suffix and (suffix.group(2) in self.schema.class_names or suffix.group(2) == "unknown"):
            name = suffix.group(1)
        if not name:
            return None
        lname = name.lower()
        candidates = [m for m in pool if m["text"].lower() == lname]
        if candidates:  # prefer a mention inside the evidence, then the nearest one
            return min(candidates, key=lambda m: (not (span[0] <= m["start"] and m["end"] <= span[1]), abs(m["start"] - span[0])))
        text = chunk["text"]
        hits = [m.start() for m in re.finditer(re.escape(name), text, re.I)]
        if not hits:
            return None
        start = min(hits, key=lambda h: (not (span[0] <= h < span[1]), abs(h - span[0])))
        new = mention(chunk["chunk_id"], text, start, start + len(name), METHOD,
                      cls if cls and cls != "unknown" else None, model=self.model_id)
        if new["predicted_class"] is None:
            del new["predicted_class"]
        pool.append(new)
        out.mentions.append(new)
        return new


def parse_items(raw: str) -> tuple[list[dict] | None, bool]:
    """Parse {"assertions": [...]}. If the output was cut off, keep the complete items. Returns (items, truncated)."""
    try:
        items = json.loads(raw)["assertions"]
        return (items, False) if isinstance(items, list) else (None, False)
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    start = raw.find("[")
    if start < 0:
        return None, False
    decoder, pos, items = json.JSONDecoder(), start + 1, []
    while True:
        while pos < len(raw) and raw[pos] in " \t\r\n,":
            pos += 1
        try:
            obj, pos = decoder.raw_decode(raw, pos)
        except json.JSONDecodeError:
            break
        if isinstance(obj, dict):
            items.append(obj)
    return items, True


def _source_at(chunk: dict, offset: int) -> str | None:
    for p in chunk["pieces"]:
        if p["start"] <= offset < p["end"]:
            return p.get("source")
    return None
