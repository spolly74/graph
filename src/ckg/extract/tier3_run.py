"""Run tier 3 over the gated chunks, with a resumable response cache and timing.

Outputs, in paths.out_dir:
    assertions.jsonl            assertions that passed every check
    assertions.rejected.jsonl   rejected assertions (or responses), with the reason
    mentions.tier3.jsonl        mentions the model introduced that tiers 1 and 2 missed
    tier3.skipped.jsonl         chunks gated out of tier 3, with the reason
    tier3.timing.json           CPU time per chunk over every cached response

Each LLM response is stored in the state database as soon as it arrives.
A rerun reuses stored responses for chunks whose hash and settings are
unchanged, so an interrupted run picks up where it stopped.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ckg.config import Config
from ckg.extract.common import Span
from ckg.extract.tier3_llm import PROMPT_VERSION, LlmClient, OllamaClient, Tier3Schema, Validator, user_message
from ckg.state import StateDB

ASSERTIONS_FILE = "assertions.jsonl"
REJECTED_FILE = "assertions.rejected.jsonl"
T3_MENTIONS_FILE = "mentions.tier3.jsonl"
SKIPPED_FILE = "tier3.skipped.jsonl"
TIMING_FILE = "tier3.timing.json"


@dataclass
class Tier3Report:
    gated: int = 0
    skipped: int = 0
    selected: int = 0
    called: int = 0
    missing: int = 0
    assertions: int = 0
    new_mentions: int = 0
    rejected: Counter = field(default_factory=Counter)
    timing: dict = field(default_factory=dict)


def distinct_entities(mentions: list[dict]) -> int:
    return len({m.get("vocab_id") or (m.get("predicted_class"), m["text"].lower()) for m in mentions})


def spread(items: list, n: int) -> list:
    """n items spaced evenly through the list, in order."""
    if n >= len(items):
        return list(items)
    return [items[(i * len(items)) // n] for i in range(n)]


def run_tier3(cfg: Config, chunks: list[dict], base_mentions: dict[str, list[dict]], masked: dict[str, list[Span]],
              state: StateDB, limit: int | None = None, client_factory: Callable[[], LlmClient] | None = None,
              progress: Callable[[str], None] | None = None) -> Tier3Report:
    t3 = cfg.extract.tier3
    out_dir = cfg.resolve(cfg.paths.out_dir)
    schema = Tier3Schema.load(cfg.resolve(t3.predicates_file))
    system = schema.render_system(cfg.resolve(t3.prompt_file).read_text(encoding="utf-8"))
    out_schema = schema.output_schema(t3.max_assertions_per_chunk)
    docs = {d["doc_id"]: d["path"] for d in _read_jsonl(out_dir / "documents.jsonl")}
    rep = Tier3Report()

    gated, skipped = [], []
    for c in chunks:
        n = distinct_entities(base_mentions.get(c["chunk_id"], []))
        if n >= t3.min_distinct_mentions:
            gated.append(c)
        else:
            skipped.append({"chunk_id": c["chunk_id"], "reason": f"fewer_than_{t3.min_distinct_mentions}_distinct_entities", "distinct": n})
    selected = spread(gated, limit) if limit else gated
    rep.gated, rep.skipped, rep.selected = len(gated), len(skipped), len(selected)

    client = (client_factory or (lambda: OllamaClient(t3.ollama_url, t3.model, t3.temperature, t3.seed, t3.num_ctx,
                                                       t3.max_output_tokens, t3.timeout_seconds)))()
    settings = hashlib.sha256(json.dumps(
        [PROMPT_VERSION, client.model_id, system, out_schema,
         t3.model_dump(mode="json", exclude={"ollama_url", "timeout_seconds"})],
        sort_keys=True).encode("utf-8")).hexdigest()

    todo = [c for c in selected if state.llm_get(c["chunk_id"], c["sha256"], settings) is None]
    for i, c in enumerate(todo, 1):
        code = [s for s in masked[c["chunk_id"]] if s.reason == "code_block"] if t3.omit_code_blocks else None
        result = client.chat(system, user_message(c, base_mentions.get(c["chunk_id"], []), code), out_schema)
        state.llm_put(c["chunk_id"], c["sha256"], settings, result.content, result.prompt_tokens,
                      result.output_tokens, result.seconds, client.model_id)
        rep.called += 1
        if progress:
            progress(f"[{i}/{len(todo)}] {c['chunk_id']}: {result.seconds:.1f}s, "
                     f"{result.prompt_tokens} prompt + {result.output_tokens} output tokens")

    assertions, rejected, new_mentions, timings = [], [], [], []
    for c in gated:
        hit = state.llm_get(c["chunk_id"], c["sha256"], settings)
        if hit is None:
            rep.missing += 1
            continue
        timings.append(hit)
        res = Validator(schema, hit["model_id"], t3.evidence_whitespace_tolerant).validate(
            c, docs.get(c["doc_id"], ""), base_mentions.get(c["chunk_id"], []), hit["response"])
        assertions += res.assertions
        rejected += res.rejected
        new_mentions += res.mentions

    _write_jsonl(out_dir / ASSERTIONS_FILE, assertions)
    _write_jsonl(out_dir / REJECTED_FILE, rejected)
    _write_jsonl(out_dir / T3_MENTIONS_FILE, new_mentions)
    _write_jsonl(out_dir / SKIPPED_FILE, skipped)
    rep.timing = timing_summary(timings, client.model_id)
    (out_dir / TIMING_FILE).write_text(json.dumps(rep.timing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rep.assertions, rep.new_mentions = len(assertions), len(new_mentions)
    rep.rejected = Counter(r["reason"] for r in rejected)
    return rep


def timing_summary(rows: list[dict], model_id: str) -> dict:
    if not rows:
        return {"chunks": 0, "model": model_id}
    secs = sorted(r["seconds"] for r in rows)
    out_tokens = sum(r["output_tokens"] or 0 for r in rows)
    return {
        "model": model_id,
        "chunks": len(rows),
        "total_hours": round(sum(secs) / 3600, 2),
        "mean_seconds": round(statistics.mean(secs), 1),
        "median_seconds": round(statistics.median(secs), 1),
        "p90_seconds": round(secs[int(0.9 * (len(secs) - 1))], 1),
        "max_seconds": round(secs[-1], 1),
        "mean_prompt_tokens": round(statistics.mean(r["prompt_tokens"] or 0 for r in rows)),
        "mean_output_tokens": round(out_tokens / len(rows)),
    }


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path
