"""Phase 1: ingest and chunking (DESIGN 5.1, phase 1 done-when)."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest
import yaml

from ckg.config import load_config
from ckg.ingest.loader import normalize
from ckg.ingest.pipeline import CHUNKS_FILE, DOCUMENTS_FILE, UNRESOLVED_FILE, run_ingest

from conftest import SAMPLE_BASE, SAMPLE_REPO, sample_config_dict

BASE = SAMPLE_BASE.as_posix()


# --- helpers ---------------------------------------------------------------


def make_config(tmp: Path, repo: Path, **ingest) -> Path:
    data = sample_config_dict()
    data["corpus"]["repo_root"] = str(repo)
    data["ingest"].update(ingest)
    path = tmp / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def ingest(config: Path) -> tuple[list[dict], list[dict], list[dict]]:
    cfg = load_config(config)
    run_ingest(cfg)
    out, review = cfg.resolve(cfg.paths.out_dir), cfg.resolve(cfg.paths.review_dir)
    chunks = [json.loads(l) for l in (out / CHUNKS_FILE).read_text(encoding="utf-8").splitlines()]
    docs = [json.loads(l) for l in (out / DOCUMENTS_FILE).read_text(encoding="utf-8").splitlines()]
    with (review / UNRESOLVED_FILE).open(encoding="utf-8") as f:
        unresolved = list(csv.DictReader(f))
    return chunks, docs, unresolved


def by_doc(chunks: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for c in chunks:
        out.setdefault(c["doc_id"], []).append(c)
    return out


def write(repo: Path, rel: str, text: str) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="")


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    shutil.copytree(SAMPLE_REPO, repo)
    return repo


@pytest.fixture
def sample_run(tmp_path: Path, sample_repo: Path):
    config = make_config(tmp_path, sample_repo)
    return config, sample_repo, *ingest(config)


def synthetic(tmp_path: Path, files: dict[str, str], **ingest_settings):
    repo = tmp_path / "repo"
    for rel, text in files.items():
        write(repo, rel, text)
    return ingest(make_config(tmp_path, repo, **ingest_settings))


# --- reproducibility and change isolation ------------------------------------


def test_rerun_produces_identical_output(sample_run):
    config, *_ = sample_run
    cfg = load_config(config)
    out = cfg.resolve(cfg.paths.out_dir)
    first = {name: (out / name).read_bytes() for name in (CHUNKS_FILE, DOCUMENTS_FILE)}
    run_ingest(cfg)
    assert {name: (out / name).read_bytes() for name in first} == first


def test_editing_one_article_changes_only_its_chunks(sample_run):
    config, repo, chunks, _, _ = sample_run
    target = repo / BASE / "concept-endpoints.md"
    target.write_text(target.read_text(encoding="utf-8") + "\n## Added section\n\nNew text.\n", encoding="utf-8")
    after = by_doc(ingest(config)[0])
    before = by_doc(chunks)
    changed = {d for d in before if before[d] != after[d]}
    assert changed == {"concept-endpoints.md"}


def test_editing_an_include_changes_every_article_that_uses_it(sample_run):
    config, repo, chunks, docs, _ = sample_run
    users: dict[str, set[str]] = {}
    for d in docs:
        for inc in d["includes"]:
            users.setdefault(inc, set()).add(d["doc_id"])
    include, using = min(users.items(), key=lambda kv: (len(kv[1]) == len(docs), -len(kv[1]), kv[0]))
    assert 0 < len(using) < len(docs)
    path = repo / include
    path.write_text(path.read_text(encoding="utf-8").rstrip("\n") + " Edited.\n", encoding="utf-8")
    after = by_doc(ingest(config)[0])
    before = by_doc(chunks)
    assert {d for d in before if before[d] != after[d]} == using


def test_crlf_input_gives_identical_chunks(sample_run):
    config, repo, chunks, _, _ = sample_run
    for path in (repo / BASE).rglob("*.md"):
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert ingest(config)[0] == chunks


# --- provenance and IDs ------------------------------------------------------


def test_pieces_map_back_to_source_text(sample_run):
    _, repo, chunks, _, _ = sample_run
    cache: dict[str, str] = {}
    for c in chunks:
        covered = 0
        for p in c["pieces"]:
            assert p["start"] == covered, "pieces must tile the chunk text"
            covered = p["end"]
            if "source" in p:
                src = cache.setdefault(p["source"], normalize((repo / p["source"]).read_bytes()))
                assert src[p["source_start"] : p["source_start"] + p["end"] - p["start"]] == c["text"][p["start"] : p["end"]]
        assert covered == len(c["text"])


def test_chunk_ids_are_unique_and_well_formed(sample_run):
    _, _, chunks, docs, _ = sample_run
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
    for c in chunks:
        assert c["chunk_id"].startswith(c["doc_id"] + "#")
    assert sum(len(d["chunk_ids"]) for d in docs) == len(chunks)


def test_chunks_respect_max_size(sample_run):
    config, _, chunks, _, _ = sample_run
    cfg = load_config(config)
    assert all(c["token_estimate"] <= cfg.ingest.chunk_max_tokens for c in chunks if not c["oversize"])


def test_front_matter_metadata(sample_run):
    _, _, _, docs, _ = sample_run
    doc = next(d for d in docs if d["doc_id"] == "how-to-assign-roles.md")
    assert doc["path"] == f"{BASE}/how-to-assign-roles.md"
    assert doc["metadata"]["ms.topic"] == doc["metadata"]["ms.topic"].strip()
    assert doc["title"]


# --- includes ----------------------------------------------------------------


def test_external_include_becomes_logged_placeholder(sample_run):
    _, _, chunks, _, unresolved = sample_run
    rows = [u for u in unresolved if u["article"] == f"{BASE}/how-to-manage-quotas.md"]
    assert rows and all(u["reason"] == "external" for u in rows)
    target = rows[0]["target"]
    holders = [p for c in chunks for p in c["pieces"] if p.get("placeholder") == target]
    assert holders
    chunk = next(c for c in chunks if any(p.get("placeholder") == target for p in c["pieces"]))
    assert f"[[UNRESOLVED_INCLUDE: {target}]]" in chunk["text"]


def test_inline_and_nested_includes_resolve(tmp_path):
    chunks, docs, unresolved = synthetic(tmp_path, {
        f"{BASE}/a.md": "---\ntitle: A\n---\n# A\n\nBefore [!INCLUDE [x](includes/x.md)] after.\n",
        f"{BASE}/includes/x.md": "---\nms.topic: include\n---\nX says [!INCLUDE [y](y.md)]\n\n",
        f"{BASE}/includes/y.md": "hello\n",
    })
    assert not unresolved
    assert chunks[0]["text"] == "# A\n\nBefore X says hello after.\n"
    assert docs[0]["includes"] == [f"{BASE}/includes/x.md", f"{BASE}/includes/y.md"]


def test_include_cycle_and_missing_are_logged(tmp_path):
    chunks, _, unresolved = synthetic(tmp_path, {
        f"{BASE}/a.md": "# A\n\n[!INCLUDE [b](includes/b.md)]\n\n[!INCLUDE [m](includes/missing.md)]\n",
        f"{BASE}/includes/b.md": "B then [!INCLUDE [a](../a.md)]\n",
    })
    assert sorted(u["reason"] for u in unresolved) == ["cycle", "not_found"]
    assert "[[UNRESOLVED_INCLUDE: ../a.md]]" in chunks[0]["text"]


# --- docs-specific Markdown -----------------------------------------------------


def test_tabs_and_code_comments_are_not_headings(tmp_path):
    doc = (
        "# Title\n\nIntro.\n\n## Create\n\nPick a method.\n\n"
        "# [Azure CLI](#tab/cli)\n\n```bash\n# not a heading\naz ml workspace create\n```\n\n"
        "# [Python SDK](#tab/python)\n\n```python\n# also not a heading\nMLClient()\n```\n\n---\n\n"
        "After the tabs.\n"
    )
    chunks, _, _ = synthetic(tmp_path, {f"{BASE}/t.md": doc}, chunk_target_tokens=10, chunk_max_tokens=200)
    ids = [c["chunk_id"].split("#", 1)[1] for c in chunks]
    assert "create--cli" in ids and "create--python" in ids
    assert not any("not-a-heading" in i for i in ids)
    cli = next(c for c in chunks if c["chunk_id"].endswith("#create--cli"))
    assert cli["hints"]["interface"] == "Azure CLI"
    assert cli["heading_path"] == ["Title", "Create", "Azure CLI"]
    after = next(c for c in chunks if "After the tabs." in c["text"])
    assert after["hints"]["interface"] is None


def test_large_table_is_split_by_rows_with_header(tmp_path):
    header = "| Name | Value |\n| --- | --- |\n"
    rows = [f"| row {i:03d} | {'x' * 60} |\n" for i in range(60)]
    chunks, _, _ = synthetic(tmp_path, {f"{BASE}/t.md": "# T\n\n" + header + "".join(rows)},
                             chunk_target_tokens=200, chunk_max_tokens=300)
    assert len(chunks) > 1
    seen = []
    for c in chunks:
        assert c["token_estimate"] <= 300
        assert header in c["text"]
        seen += [l + "\n" for l in c["text"].splitlines() if l.startswith("| row")]
    assert seen == rows, "every row appears exactly once, whole and in order"


def test_heading_without_text_leads_into_next_chunk(tmp_path):
    big = "\n\n".join("Sentence number %d is here." % i for i in range(80))
    chunks, _, _ = synthetic(tmp_path, {f"{BASE}/h.md": f"# T\n\n## Parent\n\n### Child\n\n{big}\n"},
                             chunk_target_tokens=150, chunk_max_tokens=300)
    for c in chunks:
        assert any(line.strip() and not line.startswith("#") for line in c["text"].splitlines())


def test_preview_include_before_first_h2_marks_whole_article(tmp_path):
    chunks, docs, _ = synthetic(tmp_path, {
        f"{BASE}/p.md": "# P\n\n[!INCLUDE [preview](includes/machine-learning-preview-generic.md)]\n\n## One\n\nText.\n\n## Two\n\nMore.\n",
        f"{BASE}/includes/machine-learning-preview-generic.md": "This feature is in preview.\n",
    }, chunk_target_tokens=5, chunk_max_tokens=100)
    assert docs[0]["release_status_hint"] == "preview"
    assert len(chunks) == 3 and all(c["hints"]["release_status"] == "preview" for c in chunks)


def test_callouts_are_recorded(tmp_path):
    chunks, _, _ = synthetic(tmp_path, {f"{BASE}/c.md": "# C\n\n> [!IMPORTANT]\n> Read this.\n"})
    assert chunks[0]["hints"]["callouts"] == ["IMPORTANT"]
