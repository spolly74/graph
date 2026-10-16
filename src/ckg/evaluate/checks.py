"""Evaluation checks that need no gold set (DESIGN 5.6): competency questions,
the spot-check sample, and the unmapped-mention report. Plus the optional
gold-set comparison.
"""

from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from pyoxigraph import Store

SPOT_COLUMNS = ["assertion", "subject", "predicate", "object", "negated", "conditions", "evidence", "chunk", "correct", "notes"]
UNMAPPED_COLUMNS = ["entity", "label", "class", "mentions", "surface_forms", "predicted_classes", "methods", "example_chunk"]


# --- competency questions ----------------------------------------------------------


def load_questions(path: Path) -> dict[int, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"(\d+)\. \*\*(.+?)\*\*", line)
        if m:
            out[int(m.group(1))] = m.group(2)
    return out


def run_questions(store: Store, queries_dir: Path, questions: dict[int, str]) -> list[dict]:
    results = []
    for path in sorted(queries_dir.glob("cq*.rq")):
        n = int(re.sub(r"\D", "", path.stem))
        solutions = store.query(path.read_text(encoding="utf-8"), use_default_graph_as_union=True)
        names = [v.value for v in solutions.variables]
        rows = [{k: (s[k].value if s[k] is not None else "") for k in names} for s in solutions]
        rows.sort(key=lambda r: tuple(r.values()))
        results.append({"n": n, "question": questions.get(n, ""), "query": path.name, "columns": names,
                        "count": len(rows), "rows": rows})
    return results


# --- spot check ---------------------------------------------------------------------


def spot_check(path: Path, assertions: list[dict], entities: dict[str, dict], size: int, seed: int) -> dict:
    """Write a stable random sample of assertions; keep any marks a person already made."""
    marks = {}
    if path.is_file():
        with path.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r.get("correct", "").strip() or r.get("notes", "").strip():
                    marks[r["assertion"]] = (r.get("correct", "").strip().lower(), r.get("notes", ""))
    by_id = {a["id"]: a for a in assertions}
    keep = [i for i in sorted(marks) if i in by_id]
    pool = sorted(i for i in by_id if i not in marks)
    sample = keep + random.Random(seed).sample(pool, max(0, min(size - len(keep), len(pool))))
    rows = []
    for aid in sample:
        a = by_id[aid]
        label = lambda eid: entities.get(eid, {}).get("label", eid)
        rows.append({
            "assertion": aid,
            "subject": label(a["subject"]),
            "predicate": a["predicate"].split(":", 1)[-1],
            "object": label(a["object"]) if "object" in a else a.get("object_value", ""),
            "negated": "yes" if a["negated"] else "",
            "conditions": "; ".join(f"{c['type']}={c['value']}" for c in a["conditions"]),
            "evidence": a["provenance"][0]["evidence"],
            "chunk": a["provenance"][0]["chunk_id"],
            "correct": marks.get(aid, ("", ""))[0],
            "notes": marks.get(aid, ("", ""))[1],
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, SPOT_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    yes = sum(r["correct"] in ("yes", "y", "1", "true") for r in rows)
    no = sum(r["correct"] in ("no", "n", "0", "false") for r in rows)
    return {"sample": len(rows), "marked": yes + no, "correct": yes, "incorrect": no,
            "precision": round(yes / (yes + no), 3) if yes + no else None}


# --- unmapped mentions ------------------------------------------------------------------


def unmapped_mentions(path: Path, mentions: list[dict], entities: dict[str, dict]) -> dict:
    """Entities with no vocabulary concept, by mention count: candidates for the vocabulary or schema."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for m in mentions:
        if "vocab_id" not in entities.get(m["entity"], {}):
            groups[m["entity"]].append(m)
    rows = []
    for eid, ms in groups.items():
        e = entities.get(eid, {})
        rows.append({
            "entity": eid, "label": e.get("label", ""), "class": e.get("entity_type", ""), "mentions": len(ms),
            "surface_forms": "; ".join(t for t, _ in Counter(m["text"] for m in ms).most_common(5)),
            "predicted_classes": "; ".join(f"{c} {n}" for c, n in Counter(m.get("predicted_class", "") for m in ms).most_common()),
            "methods": "; ".join(f"{c} {n}" for c, n in Counter(m["method"] for m in ms).most_common()),
            "example_chunk": ms[0]["chunk_id"],
        })
    rows.sort(key=lambda r: (-r["mentions"], r["label"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, UNMAPPED_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    unmapped = sum(r["mentions"] for r in rows)
    return {
        "entities": len(rows), "mentions": unmapped, "share_of_mentions": round(unmapped / len(mentions), 3) if mentions else 0,
        "singletons": sum(r["mentions"] == 1 for r in rows),
        "by_class": dict(Counter(r["class"] for r in rows).most_common()),
        "top": [{"label": r["label"], "class": r["class"], "mentions": r["mentions"]} for r in rows[:25]],
    }


# --- optional gold set ----------------------------------------------------------------------


def compare_gold(gold_dir: Path, mentions: list[dict], assertions: list[dict], entities: dict[str, dict]) -> dict | None:
    """Precision, recall, and F1 against eval/gold, if it exists (DESIGN 5.6, optional).

    eval/gold/mentions.jsonl:   {"chunk_id", "start", "end", "class"}
    eval/gold/assertions.jsonl: {"chunk_id", "subject", "predicate", "object" or "object_value", "negated"}
    Only chunks that appear in the gold files are scored. Mentions match on
    exact span and class; assertions on chunk, predicate, negation, and the
    case-folded subject and object text.
    """
    gm_path, ga_path = gold_dir / "mentions.jsonl", gold_dir / "assertions.jsonl"
    if not gm_path.is_file() and not ga_path.is_file():
        return None
    out = {}
    text_of = {m["id"]: m["text"] for m in mentions}
    if gm_path.is_file():
        gold = [json.loads(l) for l in gm_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        chunks = {g["chunk_id"] for g in gold}
        gold_set = {(g["chunk_id"], g["start"], g["end"], g["class"]) for g in gold}
        pred_set = {(m["chunk_id"], m["start"], m["end"], entities.get(m.get("entity", ""), {}).get("entity_type", m.get("predicted_class")))
                    for m in mentions if m["chunk_id"] in chunks}
        out["mentions"] = _prf(gold_set, pred_set, key=lambda t: t[3])
    if ga_path.is_file():
        gold = [json.loads(l) for l in ga_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        chunks = {g["chunk_id"] for g in gold}

        def norm(chunk, subj, pred, obj, negated):
            return (chunk, subj.strip().lower(), pred.split(":")[-1], obj.strip().lower(), bool(negated))

        gold_set = {norm(g["chunk_id"], g["subject"], g["predicate"], g.get("object") or g.get("object_value", ""), g.get("negated"))
                    for g in gold}
        pred_set = set()
        for a in assertions:
            for p in a["provenance"]:
                if p["chunk_id"] in chunks:
                    obj = text_of.get(a.get("object", ""), "") if "object" in a else a.get("object_value", "")
                    pred_set.add(norm(p["chunk_id"], text_of.get(a["subject"], ""), a["predicate"], obj, a["negated"]))
        out["assertions"] = _prf(gold_set, pred_set, key=lambda t: t[2])
    return out


def _prf(gold: set, pred: set, key) -> dict:
    def score(g, p):
        tp = len(g & p)
        precision = tp / len(p) if p else 0.0
        recall = tp / len(g) if g else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"gold": len(g), "predicted": len(p), "true_positive": tp,
                "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}

    groups = sorted({key(t) for t in gold | pred}, key=str)
    return {"micro": score(gold, pred),
            "by_group": {str(k): score({t for t in gold if key(t) == k}, {t for t in pred if key(t) == k}) for k in groups}}
