# ckg: content knowledge graph

Builds a knowledge graph of the Azure Machine Learning documentation in [spolly74/azure-ai-docs](https://github.com/spolly74/azure-ai-docs). See [DESIGN.md](DESIGN.md) for the design and build phases.

Status: Phase 0 (scaffold). The CLI commands exist, but the stages are not built yet.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Get the corpus: clone it into a folder next to this repo (`../azure-ai-docs`, as set by `corpus.repo_root` in `config.yaml`). On Windows, `core.longpaths` is needed for some file names.

```bash
git clone -c core.longpaths=true https://github.com/spolly74/azure-ai-docs.git ../azure-ai-docs
```

## Use

```bash
uv run ckg --help
```

Every setting comes from `config.yaml` (or the file named by `--config` or `$CKG_CONFIG`). The `corpus.include` list selects the first-run slice; remove it to process the full scope.

## Tests

```bash
uv run pytest
```

`data/sample/` holds 10 articles from the slice plus the include files they use, in the repo's folder layout. Tests use it so they run without the full checkout. To rebuild it, run `tools/make_sample.py`.

## Layout

| Path | Contents |
|---|---|
| `schema/` | LinkML schemas: `core.yaml` (Phase 2), `domain.yaml` (draft) |
| `vocab/vocabulary.ttl` | SKOS vocabulary for tier 1 matching |
| `eval/competency_questions.md` | Questions the graph must answer |
| `review/` | Queues and reports for a person to check |
| `tools/` | One-time helpers: vocabulary drafting, sample building |
