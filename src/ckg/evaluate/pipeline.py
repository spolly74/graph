"""Evaluate stage (DESIGN 5.6): competency questions, spot check, unmapped
mentions, pipeline metrics, and the optional gold-set comparison.

Writes:
    review/evaluation_report.md   for a person to read
    data/out/evaluation.json      every number in the report, plus all question results
    review/spot_check.csv         sample of assertions to mark correct or not
    review/unmapped_mentions.csv  entities with no vocabulary concept, by mention count
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from pyoxigraph import Store

from ckg.config import Config
from ckg.evaluate.checks import compare_gold, load_questions, run_questions, spot_check, unmapped_mentions
from ckg.load.pipeline import open_store


@dataclass
class EvaluateReport:
    questions: int = 0
    answered: int = 0
    unanswered: list[int] = field(default_factory=list)
    spot: dict = field(default_factory=dict)
    unmapped: dict = field(default_factory=dict)
    gold: dict | None = None
    outputs: list[Path] = field(default_factory=list)


def run_evaluate(cfg: Config, store: Store | None = None) -> EvaluateReport:
    ec = cfg.evaluation
    out = cfg.resolve(cfg.paths.out_dir)
    entities = {e["id"]: e for e in _read_jsonl(out / "entities.jsonl")}
    resolved = _read_jsonl(out / "assertions.resolved.jsonl")
    if not entities:
        raise FileNotFoundError(f"{out / 'entities.jsonl'} not found or empty; run `ckg resolve` and `ckg load` first")

    own = store is None
    store = open_store(cfg) if store is None else store
    try:
        results = run_questions(store, cfg.resolve(ec.queries_dir), load_questions(cfg.resolve(ec.questions_file)))
    finally:
        if own:
            del store

    rep = EvaluateReport(questions=len(results))
    rep.answered = sum(r["count"] > 0 for r in results)
    rep.unanswered = [r["n"] for r in results if r["count"] == 0]
    rep.spot = spot_check(cfg.resolve(ec.spot_check_file), resolved, entities, ec.spot_check_size, ec.seed)
    resolved_mentions = _read_jsonl(out / "mentions.resolved.jsonl")
    rep.unmapped = unmapped_mentions(cfg.resolve(ec.unmapped_file), resolved_mentions, entities)
    rep.gold = compare_gold(cfg.resolve(ec.gold_dir), _read_jsonl(out / "mentions.jsonl"),
                            _read_jsonl(out / "assertions.jsonl"), entities)

    metrics = pipeline_metrics(cfg, out)
    data = {
        "pipeline": metrics,
        "competency_questions": {"total": rep.questions, "answered": rep.answered, "unanswered": rep.unanswered,
                                 "results": results},
        "spot_check": rep.spot,
        "unmapped_mentions": rep.unmapped,
        "gold": rep.gold,
    }
    metrics_path = cfg.resolve(ec.metrics_file)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_path = cfg.resolve(ec.report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(data, ec.rows_in_report), encoding="utf-8", newline="\n")
    rep.outputs = [report_path, metrics_path, cfg.resolve(ec.spot_check_file), cfg.resolve(ec.unmapped_file)]
    return rep


def pipeline_metrics(cfg: Config, out: Path) -> dict:
    """Counts at each stage, read from the stage outputs."""
    count = lambda name: len(_read_jsonl(out / name))
    reasons = lambda name: dict(Counter(r["reason"] for r in _read_jsonl(out / name)).most_common())
    mentions = _read_jsonl(out / "mentions.jsonl")
    entities = _read_jsonl(out / "entities.jsonl")
    queue_path = cfg.resolve(cfg.resolution.queue_file)
    queue = _read_csv(queue_path)
    includes = _read_csv(cfg.resolve(cfg.paths.review_dir) / "unresolved_includes.csv")
    timing_path = out / "tier3.timing.json"
    validation_path = cfg.resolve(cfg.validation.summary_file)
    return {
        "ingest": {"documents": count("documents.jsonl"), "chunks": count("chunks.jsonl"),
                   "unresolved_includes": len(includes)},
        "mentions": {"merged": len(mentions), "by_method": dict(Counter(m["method"] for m in mentions).most_common()),
                     "dropped": reasons("mentions.dropped.jsonl")},
        "tier3": {"gated_out": count("tier3.skipped.jsonl"), "assertions": count("assertions.jsonl"),
                  "rejected": reasons("assertions.rejected.jsonl"),
                  "timing": json.loads(timing_path.read_text(encoding="utf-8")) if timing_path.is_file() else None},
        "resolution": {"entities": len(entities), "vocabulary_entities": sum("vocab_id" in e for e in entities),
                       "queue_open": sum(1 for r in queue if not r.get("decision")),
                       "queue_decided": sum(1 for r in queue if r.get("decision")),
                       "assertions": count("assertions.resolved.jsonl"),
                       "assertions_dropped": reasons("assertions.unresolved.jsonl")},
        "validation": json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.is_file() else None,
    }


def render_report(data: dict, rows_shown: int) -> str:
    p, cq, spot, um = data["pipeline"], data["competency_questions"], data["spot_check"], data["unmapped_mentions"]
    v = p["validation"] or {}
    t = (p["tier3"]["timing"] or {})
    lines = [
        "# Evaluation report", "",
        "Generated by `ckg evaluate`. Numbers come from the stage outputs in `data/out`; "
        "every value is also in `data/out/evaluation.json`.", "",
        "## Summary", "",
        "| Measure | Value |", "|---|---|",
        f"| Documents / chunks | {p['ingest']['documents']} / {p['ingest']['chunks']} |",
        f"| Mentions (merged) | {p['mentions']['merged']} ({_inline(p['mentions']['by_method'])}) |",
        f"| Tier 3 assertions kept / rejected | {p['tier3']['assertions']} / {sum(p['tier3']['rejected'].values())} |",
        f"| Tier 3 CPU time | {t.get('total_hours', '–')} h, median {t.get('median_seconds', '–')} s per chunk |",
        f"| Entities (from vocabulary) | {p['resolution']['entities']} ({p['resolution']['vocabulary_entities']}) |",
        f"| Resolved assertions | {p['resolution']['assertions']} |",
        f"| Review queue open / decided | {p['resolution']['queue_open']} / {p['resolution']['queue_decided']} |",
        f"| Validation violations | {v.get('violations', '–')} ({_inline(v.get('by_kind', {}))}) |",
        f"| Competency questions answered | {cq['answered']} of {cq['total']} |",
        f"| Spot check | {_spot_line(spot)} |",
        f"| Unmapped entities | {um['entities']} ({um['mentions']} mentions, {um['share_of_mentions']:.0%} of all; {um['singletons']} seen once) |",
        "",
        "## Competency questions", "",
        "A question counts as answered when its query returns at least one row. That is a low bar: the rows come "
        "from tier 3 assertions made by a 7B model on CPU, and many are off-topic or use the wrong predicate. "
        "Judge the rows themselves, and use the spot check for a precision estimate.", "",
    ]
    for r in cq["results"]:
        lines += [f"### CQ {r['n']}: {r['question']}", "", f"`{r['query']}`: **{r['count']} rows**", ""]
        if r["rows"]:
            lines += _table(r["rows"][:rows_shown])
            if r["count"] > rows_shown:
                lines.append(f"\n…and {r['count'] - rows_shown} more in `data/out/evaluation.json`.")
            lines.append("")
    lines += ["## Unmapped mentions", "",
              "Entities with no vocabulary concept, most mentioned first (full list in `review/unmapped_mentions.csv`). "
              "Frequent ones are candidates for the vocabulary; frequent generic words point at tier 2 noise.", "",
              "| Entity | Class | Mentions |", "|---|---|---|"]
    lines += [f"| {_cell(x['label'])} | {x['class']} | {x['mentions']} |" for x in um["top"]]
    lines += ["", "## Rejections and drops", "",
              f"- Mentions dropped: {_inline(p['mentions']['dropped'])}",
              f"- Tier 3 rejected: {_inline(p['tier3']['rejected'])}",
              f"- Chunks gated out of tier 3: {p['tier3']['gated_out']}",
              f"- Assertions dropped by resolution: {_inline(p['resolution']['assertions_dropped'])}",
              f"- Unresolved includes: {p['ingest']['unresolved_includes']}",
              f"- Validation by constraint: {_inline(v.get('by_constraint', {}))}", ""]
    gold = data["gold"]
    lines += ["## Gold set", ""]
    if gold is None:
        lines.append("No gold set in `eval/gold/`, so precision and recall were not computed (DESIGN 5.6). "
                     "The spot check above is the precision estimate.")
    else:
        for part, res in gold.items():
            m = res["micro"]
            lines.append(f"- {part}: precision {m['precision']}, recall {m['recall']}, F1 {m['f1']} "
                         f"({m['true_positive']} of {m['gold']} gold, {m['predicted']} predicted)")
    return "\n".join(lines).rstrip() + "\n"


def _spot_line(spot: dict) -> str:
    if spot.get("marked"):
        return f"{spot['marked']} of {spot['sample']} marked, precision {spot['precision']:.0%}"
    return f"{spot.get('sample', 0)} assertions in `review/spot_check.csv`, none marked yet"


def _table(rows: list[dict]) -> list[str]:
    cols = [c for c in rows[0] if any(r[c] for r in rows)]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(_cell(r[c]) for c in cols) + " |" for r in rows]
    return out


def _cell(value: str) -> str:
    value = str(value).replace("|", "\\|").replace("\n", " ")
    return value if len(value) <= 80 else value[:77] + "…"


def _inline(counts: dict) -> str:
    return ", ".join(f"{k} {v}" for k, v in counts.items()) or "none"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))
