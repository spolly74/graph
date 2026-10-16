"""Resolve stage: mentions -> canonical entities (DESIGN 5.3).

1. Normalize each mention's text.
2. A mention with a vocabulary ID (tier 1), or whose normalized text equals a
   vocabulary label, gets that concept's ID.
3. Other mentions with the same normalized text form one candidate entity;
   its class is the most common predicted class among them.
4. Within each class, candidates (and vocabulary concepts) are compared by
   rapidfuzz string similarity and embedding similarity. Pairs at or above
   merge_threshold merge; pairs between review_threshold and merge_threshold
   go to the review queue. Two vocabulary concepts never merge.
5. Decisions a person writes in the queue's `decision` column (merge or
   separate) are applied on the next run and kept in the file.

Outputs, in paths.out_dir:
    entities.jsonl              one record per entity, every surface form kept as an alias
    mentions.resolved.jsonl     mentions with `entity` set
    mentions.unresolved.jsonl   mentions not resolved, with the reason
    assertions.resolved.jsonl   assertions on entity IDs, duplicates merged
    assertions.unresolved.jsonl assertions dropped by resolution, with the reason
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.process import cdist

from ckg.config import Config
from ckg.extract.vocab import load_vocabulary
from ckg.resolve.normalize import normalize, slug

QUEUE_COLUMNS = ["class", "label_a", "label_b", "score", "string_score", "embedding_score", "mentions_a", "mentions_b",
                 "example_a", "example_b", "note", "decision", "name_a", "name_b"]
DECISIONS = {"merge", "separate"}
VARIANT_MARKS = re.compile(r"\d+|\+")


def variant_marks(name: str) -> list[str]:
    """Numbers and plus signs: Gen1 vs Gen2, 8B vs 405B, R vs R+ name different things."""
    return sorted(VARIANT_MARKS.findall(name))


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...  # rows L2-normalized


def load_embedder(model: str, revision: str) -> Embedder:
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(model, revision=revision, device="cpu")

    class _Embedder:
        def encode(self, texts: list[str]) -> np.ndarray:
            return st.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)

    return _Embedder()


@dataclass
class Node:
    key: str                      # "vocab:<iri>" or "name:<normalized name>"
    cls: str
    names: list[str]              # normalized strings compared for this node
    vocab_id: str | None = None
    vocab_label: str | None = None
    surface: Counter = field(default_factory=Counter)
    mention_ids: list[str] = field(default_factory=list)
    example: str = ""


@dataclass
class ResolveReport:
    mentions: int = 0
    resolved: int = 0
    unresolved: Counter = field(default_factory=Counter)
    entities: int = 0
    vocabulary_entities: int = 0
    auto_merges: int = 0
    human_merges: int = 0
    queued: int = 0
    decided: int = 0
    assertions_in: int = 0
    assertions_out: int = 0
    assertions_dropped: Counter = field(default_factory=Counter)
    outputs: list[Path] = field(default_factory=list)


def run_resolve(cfg: Config, embedder_factory: Callable[[], Embedder] | None = None) -> ResolveReport:
    rc = cfg.resolution
    out = cfg.resolve(cfg.paths.out_dir)
    mentions = _read_jsonl(out / "mentions.jsonl")
    assertions = _read_jsonl(out / "assertions.jsonl")
    if not mentions:
        raise FileNotFoundError(f"{out / 'mentions.jsonl'} not found or empty; run `ckg extract` first")
    concepts = load_vocabulary(cfg.resolve(cfg.extract.vocabulary))
    rep = ResolveReport(mentions=len(mentions), assertions_in=len(assertions))

    # Steps 1-3: vocabulary matches and exact-name groups.
    label_index: dict[str, object] = {}
    for c in concepts:
        for label in c.labels:
            label_index.setdefault(normalize(label, rc.singularize), c)
    by_iri = {c.iri: c for c in concepts}
    nodes: dict[str, Node] = {}
    class_votes: dict[str, Counter] = defaultdict(Counter)
    mention_node: dict[str, str] = {}
    unresolved = []
    for m in mentions:
        name = normalize(m["text"], rc.singularize)
        concept = by_iri.get(m["vocab_id"]) if m.get("vocab_id") else label_index.get(name)
        if concept is not None:
            key = f"vocab:{concept.iri}"
            if key not in nodes:
                nodes[key] = Node(key, concept.domain_class, sorted({normalize(l, rc.singularize) for l in concept.labels}),
                                  concept.iri, concept.labels[0])
        elif len(name) < rc.min_name_length:
            unresolved.append(m | {"reason": "name_too_short"})
            continue
        else:
            key = f"name:{name}"
            if key not in nodes:
                nodes[key] = Node(key, "", [name])
            class_votes[key][m.get("predicted_class") or "unknown"] += 1
        node = nodes[key]
        node.surface[m["text"]] += 1
        node.mention_ids.append(m["id"])
        node.example = node.example or m["chunk_id"]
        mention_node[m["id"]] = key
    for key, votes in class_votes.items():
        nodes[key].cls = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    for c in concepts:  # concepts never mentioned still take part, so near-miss names can merge into them
        key = f"vocab:{c.iri}"
        nodes.setdefault(key, Node(key, c.domain_class, sorted({normalize(l, rc.singularize) for l in c.labels}), c.iri, c.labels[0]))

    # Step 4: pair scores within each class.
    queue_path = cfg.resolve(rc.queue_file)
    decisions = _read_decisions(queue_path)
    embedder = (embedder_factory or (lambda: load_embedder(rc.embedding_model, rc.embedding_revision)))()
    edges, queue = _score_pairs(nodes, embedder, rc.string_weight, rc.merge_threshold, rc.review_threshold,
                                set(rc.exact_match_classes))

    # Step 5: human decisions override scores.
    for pair, decision in decisions.items():
        if pair[0] in nodes and pair[1] in nodes:
            edges.pop(pair, None)
            if decision == "merge":
                edges[pair] = (2.0, "human")
    uf = _UnionFind(nodes)
    for (a, b), (score, source) in sorted(edges.items(), key=lambda kv: (-kv[1][0], kv[0])):
        if uf.union(a, b):
            if source == "human":
                rep.human_merges += 1
            else:
                rep.auto_merges += 1
        elif uf.find(a) != uf.find(b):
            queue[(a, b)] = queue.get((a, b)) or _pair_row(nodes[a], nodes[b], score, None, None)
            queue[(a, b)]["note"] = "would join two vocabulary concepts"

    # Pairs that ended up in one entity anyway need no review, unless a person decided them.
    queue = {p: r for p, r in queue.items() if uf.find(p[0]) != uf.find(p[1]) or p in decisions}

    # Entities.
    clusters: dict[str, list[str]] = defaultdict(list)
    for key in sorted(nodes):
        if nodes[key].mention_ids or nodes[key].vocab_id:
            clusters[uf.find(key)].append(key)
    entities, node_entity, used_ids = [], {}, set()
    for members in sorted(clusters.values()):
        members = [nodes[k] for k in members]
        if not any(n.mention_ids for n in members):
            continue  # vocabulary concepts nobody mentioned stay out of the graph
        vocab = next((n for n in members if n.vocab_id), None)
        surface = sum((n.surface for n in members), Counter())
        label = vocab.vocab_label if vocab else sorted(surface.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        cls = vocab.cls if vocab else Counter({n.cls: len(n.mention_ids) for n in members}).most_common(1)[0][0]
        eid = vocab.vocab_id if vocab else f"{rc.entity_base}{slug(label)}"
        if eid in used_ids:
            eid = f"{eid}--{slug(cls)}"
        used_ids.add(eid)
        record = {"id": eid, "entity_type": cls, "label": label,
                  "aliases": sorted(s for s in surface if s != label)}
        if vocab:
            record["vocab_id"] = vocab.vocab_id
        entities.append(record)
        for n in members:
            node_entity[n.key] = eid
    rep.entities = len(entities)
    rep.vocabulary_entities = sum("vocab_id" in e for e in entities)

    resolved_mentions = []
    mention_entity = {}
    for m in mentions:
        key = mention_node.get(m["id"])
        if key is None:
            continue
        mention_entity[m["id"]] = node_entity[key]
        resolved_mentions.append(m | {"entity": node_entity[key]})
    rep.resolved = len(resolved_mentions)
    rep.unresolved = Counter(u["reason"] for u in unresolved)

    resolved_assertions, dropped = _resolve_assertions(assertions, mention_entity)
    rep.assertions_out = len(resolved_assertions)
    rep.assertions_dropped = Counter(d["reason"] for d in dropped)

    rows = _write_queue(queue_path, queue, decisions, nodes)
    rep.queued = sum(1 for r in rows if not r["decision"])
    rep.decided = sum(1 for r in rows if r["decision"])
    rep.outputs = [
        _write_jsonl(out / "entities.jsonl", sorted(entities, key=lambda e: e["id"])),
        _write_jsonl(out / "mentions.resolved.jsonl", resolved_mentions),
        _write_jsonl(out / "mentions.unresolved.jsonl", unresolved),
        _write_jsonl(out / "assertions.resolved.jsonl", resolved_assertions),
        _write_jsonl(out / "assertions.unresolved.jsonl", dropped),
        queue_path,
    ]
    return rep


def _score_pairs(nodes: dict[str, Node], embedder: Embedder, w: float, merge_at: float, review_at: float,
                 exact_only: set[str]):
    """Merge edges and queue rows for node pairs in the same class."""
    by_class: dict[str, list[Node]] = defaultdict(list)
    for key in sorted(nodes):
        by_class[nodes[key].cls].append(nodes[key])
    edges: dict[tuple[str, str], tuple[float, str]] = {}
    queue: dict[tuple[str, str], dict] = {}
    for cls, members in sorted(by_class.items()):
        if len(members) < 2 or cls in exact_only:
            continue
        rows, owner = [], []
        for i, n in enumerate(members):
            for name in n.names:
                rows.append(name)
                owner.append(i)
        string = cdist(rows, rows, scorer=fuzz.token_sort_ratio, workers=1) / 100.0
        vectors = embedder.encode(rows)
        semantic = np.clip(vectors @ vectors.T, 0, 1)
        combined = w * string + (1 - w) * semantic
        owner_arr = np.array(owner)
        marks = [variant_marks(r) for r in rows]
        best: dict[tuple[int, int], tuple[float, float, float, bool]] = {}
        ii, jj = np.where(np.triu(combined >= review_at, k=1))
        for i, j in zip(ii.tolist(), jj.tolist()):
            a, b = owner_arr[i], owner_arr[j]
            if a == b:
                continue
            a, b = min(a, b), max(a, b)
            score = round(float(combined[i, j]), 4)
            if score > best.get((a, b), (-1,))[0]:
                best[(a, b)] = (score, round(float(string[i, j]), 4), round(float(semantic[i, j]), 4), marks[i] != marks[j])
        for (a, b), (score, s, e, differ) in best.items():
            na, nb = members[a], members[b]
            if na.vocab_id and nb.vocab_id:
                continue  # distinct vocabulary concepts are distinct by definition
            if not na.mention_ids and not nb.mention_ids:
                continue
            pair = tuple(sorted((na.key, nb.key)))
            if score >= merge_at and not differ:
                edges[pair] = (score, "auto")
            else:
                queue[pair] = _pair_row(nodes[pair[0]], nodes[pair[1]], score, s, e)
                if differ:
                    queue[pair]["note"] = "numbers differ"
    return edges, queue


def _pair_row(a: Node, b: Node, score: float, s: float | None, e: float | None) -> dict:
    return {
        "class": a.cls, "label_a": _label(a), "label_b": _label(b), "name_a": a.key, "name_b": b.key, "score": score,
        "string_score": "" if s is None else s, "embedding_score": "" if e is None else e,
        "mentions_a": len(a.mention_ids), "mentions_b": len(b.mention_ids),
        "example_a": a.example, "example_b": b.example, "note": "", "decision": "",
    }


def _label(n: Node) -> str:
    if n.vocab_label:
        return f"{n.vocab_label} [vocabulary]"
    return n.surface.most_common(1)[0][0] if n.surface else n.names[0]


def _read_decisions(path: Path) -> dict[tuple[str, str], str]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as f:
        return {tuple(sorted((r["name_a"], r["name_b"]))): r["decision"].strip().lower()
                for r in csv.DictReader(f) if r.get("decision", "").strip().lower() in DECISIONS}


def _write_queue(path: Path, queue: dict, decisions: dict, nodes: dict[str, Node]) -> list[dict]:
    rows = {pair: row for pair, row in queue.items()}
    for pair, decision in decisions.items():  # keep decided rows visible even when no longer queued
        if pair not in rows and pair[0] in nodes and pair[1] in nodes:
            rows[pair] = _pair_row(nodes[pair[0]], nodes[pair[1]], "", None, None)
        if pair in rows:
            rows[pair]["decision"] = decision
    ordered = sorted(rows.values(), key=lambda r: (r["class"], -(r["score"] if r["score"] != "" else 2), r["name_a"], r["name_b"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, QUEUE_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(ordered)
    return ordered


def _resolve_assertions(assertions: list[dict], mention_entity: dict[str, str]) -> tuple[list[dict], list[dict]]:
    merged: dict[str, dict] = {}
    dropped = []
    for a in assertions:
        subject = mention_entity.get(a["subject"])
        obj = mention_entity.get(a["object"]) if "object" in a else None
        if subject is None:
            dropped.append(a | {"reason": "unresolved_subject"})
            continue
        if "object" in a and obj is None:
            dropped.append(a | {"reason": "unresolved_object"})
            continue
        if obj is not None and obj == subject:
            dropped.append(a | {"reason": "self_reference_after_resolution"})
            continue
        target = obj if obj is not None else a["object_value"]
        key = json.dumps([subject, a["predicate"], target, "object" in a, a["negated"], a["conditions"]], sort_keys=True)
        if key in merged:
            merged[key]["provenance"] += a["provenance"]
            continue
        record = {k: v for k, v in a.items() if k not in ("id", "subject", "object", "provenance")}
        record |= {"id": "ckg:assertion-" + hashlib.sha256(key.encode()).hexdigest()[:16], "subject": subject,
                   "provenance": list(a["provenance"])}
        if obj is not None:
            record["object"] = obj
        merged[key] = record
    out = []
    for r in merged.values():
        r["provenance"] = sorted(r["provenance"], key=lambda p: (p["chunk_id"], p["start"], p["end"]))
        out.append(r)
    return sorted(out, key=lambda r: r["id"]), dropped


class _UnionFind:
    def __init__(self, nodes: dict[str, Node]) -> None:
        self.parent = {k: k for k in nodes}
        self.vocab = {k: ({n.vocab_id} if n.vocab_id else set()) for k, n in nodes.items()}

    def find(self, k: str) -> str:
        while self.parent[k] != k:
            self.parent[k] = self.parent[self.parent[k]]
            k = self.parent[k]
        return k

    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.vocab[ra] and self.vocab[rb]:
            return False  # never put two vocabulary concepts in one entity
        ra, rb = sorted((ra, rb))
        self.parent[rb] = ra
        self.vocab[ra] |= self.vocab[rb]
        return True


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path
