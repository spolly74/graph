# ckg: content knowledge graph

Builds a knowledge graph of the Azure Machine Learning documentation in [spolly74/azure-ai-docs](https://github.com/spolly74/azure-ai-docs). See [DESIGN.md](DESIGN.md) for the design and build phases, and [review/evaluation_report.md](review/evaluation_report.md) for the latest results.

Status: all seven phases are built and have run on the first slice (77 articles).

## Setup

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), and [Ollama](https://ollama.com) for tier 3.

```bash
uv sync
```

```bash
ollama pull qwen2.5:7b-instruct
```

Get the corpus: clone it into a folder next to this repo (`../azure-ai-docs`, as set by `corpus.repo_root` in `config.yaml`). On Windows, `core.longpaths` is needed for some file names.

```bash
git clone -c core.longpaths=true https://github.com/spolly74/azure-ai-docs.git ../azure-ai-docs
```

The GLiNER checkpoint (about 780 MB) and the embedding model download on first use.

## Use

Run everything, reprocessing only what changed:

```bash
uv run ckg run
```

Or one stage at a time:

| Command | Stage | Main outputs |
|---|---|---|
| `ckg ingest` | Load articles, resolve includes, chunk | `data/out/documents.jsonl`, `chunks.jsonl` |
| `ckg extract [--tier 1\|2\|3] [--changed-only] [--limit N]` | Vocabulary matching, GLiNER, local LLM | `mentions.jsonl`, `assertions.jsonl` |
| `ckg resolve` | Merge mentions into entities | `entities.jsonl`, `assertions.resolved.jsonl`, `review/resolution_queue.csv` |
| `ckg load` | Load into the embedded Oxigraph store | `data/out/oxigraph/` |
| `ckg validate` | SHACL, domain, and datatype checks | `review/validation_report.csv` |
| `ckg evaluate` | Competency questions, spot check, reports | `review/evaluation_report.md`, `review/spot_check.csv` |

Every setting comes from `config.yaml` (or the file named by `--config` or `$CKG_CONFIG`). The `corpus.include` list selects the first-run slice; remove it to process the full scope. Tier 3 takes about a minute per chunk on CPU; its responses are cached, so interrupted runs resume.

## Things for a person to review

| File | What to do |
|---|---|
| `review/resolution_queue.csv` | Write `merge` or `separate` in `decision` for borderline entity pairs; the next `ckg resolve` applies them |
| `review/spot_check.csv` | Mark `correct` yes or no for sampled assertions; the report turns the marks into a precision estimate |
| `review/unmapped_mentions.csv` | Frequent entities with no vocabulary concept: candidates for `vocab/vocabulary.ttl` |
| `review/validation_report.csv` | Every SHACL, domain, and datatype violation, linked to its assertion and chunk |

## Querying

The store is embedded Oxigraph. Example queries are in `queries/` and `eval/competency_questions/`. Pyoxigraph 0.5 evaluates triple patterns mostly in the order written, so put the most selective pattern first and avoid `ORDER BY` (see DESIGN 5.4).

```python
from pyoxigraph import Store
store = Store("data/out/oxigraph")
rows = store.query(open("queries/assertions_with_conditions.rq").read(), use_default_graph_as_union=True)
```

## Tests

```bash
uv run pytest
```

`data/sample/` holds 10 articles from the slice plus the include files they use, in the repo's folder layout. Tests use it with fake GLiNER, LLM, and embedding models, so they run without the corpus, Ollama, or model downloads. Rebuild it with `tools/make_sample.py`.

## Layout

| Path | Contents |
|---|---|
| `schema/` | LinkML schemas (`core.yaml`, `domain.yaml`) and generated artifacts (`generated/`; rebuild with `tools/gen_schema.py`) |
| `vocab/vocabulary.ttl` | SKOS vocabulary for tier 1 matching |
| `prompts/` | Tier 3 instructions |
| `eval/` | Competency questions and their SPARQL queries |
| `queries/` | Reusable SPARQL queries |
| `review/` | Queues and reports for a person to check |
| `tools/` | Schema generation, vocabulary drafting, sample building |
