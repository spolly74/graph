"""Load stage: resolved records -> embedded Oxigraph store (DESIGN 5.4).

Only chunks whose quads changed since the last load are touched: the
chunk's named graph is dropped and its new quads inserted in one
transaction. Chunks that no longer exist have their graphs dropped.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from pyoxigraph import NamedNode, RdfFormat, Store

from ckg.config import Config
from ckg.load.rdf import RdfMap, build_quads, chunk_graph, quads_hash
from ckg.state import StateDB

STAGE = "load"


@dataclass
class LoadReport:
    chunks: int = 0
    loaded: int = 0
    unchanged: int = 0
    removed: int = 0
    quads: int = 0
    vocabulary_loaded: bool = False
    store: Path | None = None


def open_store(cfg: Config) -> Store:
    path = cfg.resolve(cfg.store.path)
    path.mkdir(parents=True, exist_ok=True)
    return Store(str(path))


def run_load(cfg: Config, store: Store | None = None) -> LoadReport:
    out = cfg.resolve(cfg.paths.out_dir)
    entities = {e["id"]: e for e in _read_jsonl(out / "entities.jsonl")}
    mentions = _read_jsonl(out / "mentions.resolved.jsonl")
    assertions = _read_jsonl(out / "assertions.resolved.jsonl")
    if not entities:
        raise FileNotFoundError(f"{out / 'entities.jsonl'} not found or empty; run `ckg resolve` first")
    rm = RdfMap.load(cfg.resolve(cfg.store.rdf_map))
    per_chunk = build_quads(rm, cfg.store.graph_base, entities, mentions, assertions, cfg.store.load_mentions)

    rep = LoadReport(chunks=len(per_chunk), store=cfg.resolve(cfg.store.path))
    own_store = store is None
    store = open_store(cfg) if store is None else store
    state = StateDB(cfg.resolve(cfg.paths.state_db))
    try:
        settings = hashlib.sha256(cfg.resolve(cfg.store.rdf_map).read_bytes()
                                  + cfg.store.graph_base.encode() + bytes([cfg.store.load_mentions])).hexdigest()
        previous = state.processed(STAGE)
        rows = []
        for cid, quads in per_chunk.items():
            digest = quads_hash(quads)
            rows.append((cid, digest, settings))
            if previous.get(cid) == (digest, settings) and store.contains_named_graph(chunk_graph(cfg.store.graph_base, cid)):
                rep.unchanged += 1
                continue
            graph = chunk_graph(cfg.store.graph_base, cid)
            if store.contains_named_graph(graph):
                store.remove_graph(graph)
            store.extend(quads)
            rep.loaded += 1
        for cid in set(previous) - set(per_chunk):
            graph = chunk_graph(cfg.store.graph_base, cid)
            if store.contains_named_graph(graph):
                store.remove_graph(graph)
            rep.removed += 1
        state.replace(STAGE, rows)

        if cfg.store.load_vocabulary:
            rep.vocabulary_loaded = _load_vocabulary(cfg, store, state)
        elif store.contains_named_graph(NamedNode(cfg.store.graph_base + "vocabulary")):
            store.remove_graph(NamedNode(cfg.store.graph_base + "vocabulary"))  # setting turned off: drop it
            state.replace("load.vocabulary", [])
        store.flush()
        rep.quads = len(store)
    finally:
        state.close()
        if own_store:
            del store
    return rep


def _load_vocabulary(cfg: Config, store: Store, state: StateDB) -> bool:
    path = cfg.resolve(cfg.extract.vocabulary)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    graph = NamedNode(cfg.store.graph_base + "vocabulary")
    if state.processed("load.vocabulary").get("vocabulary") == (digest, "") and store.contains_named_graph(graph):
        return False
    if store.contains_named_graph(graph):
        store.remove_graph(graph)
    store.load(path=str(path), format=RdfFormat.TURTLE, to_graph=graph)
    state.replace("load.vocabulary", [("vocabulary", digest, "")])
    return True


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
