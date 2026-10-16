"""Command-line interface (DESIGN section 9).

Every command loads and validates the config, then runs its stage.
`ckg run` runs every stage in order, reprocessing changed content only.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import typer

from ckg import __version__
from ckg.config import Config, ConfigError, load_config
from ckg.ingest.pipeline import run_ingest

app = typer.Typer(
    help="Build a knowledge graph from a content corpus.",
    no_args_is_help=True,
    add_completion=False,
)


class Tier(str, Enum):
    vocab = "1"
    gliner = "2"
    llm = "3"


def _version(value: bool) -> None:
    if value:
        typer.echo(f"ckg {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    config: Annotated[
        Optional[Path],
        typer.Option("--config", "-c", help="Path to config.yaml. Defaults to $CKG_CONFIG, then ./config.yaml."),
    ] = None,
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show the version and exit.")
    ] = False,
) -> None:
    ctx.obj = config


def _load(ctx: typer.Context) -> Config:
    try:
        return load_config(ctx.obj)
    except ConfigError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=2) from e


@app.command()
def ingest(
    ctx: typer.Context,
    input: Annotated[
        Optional[Path], typer.Option("--input", help="Corpus directory. Defaults to corpus.repo_root/base_dir from the config.")
    ] = None,
) -> None:
    """Load content files and split them into chunks."""
    cfg = _load(ctx)
    if input is not None and not input.resolve().is_relative_to(cfg.repo_root):
        typer.echo(f"error: --input must be inside corpus.repo_root ({cfg.repo_root})", err=True)
        raise typer.Exit(code=2)
    try:
        report = run_ingest(cfg, input)
    except ConfigError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=2) from e
    typer.echo(f"{report.documents} documents -> {report.chunks} chunks (largest about {report.max_tokens} tokens)")
    typer.echo(f"{report.unresolved} unresolved includes, {len(report.oversize)} oversize chunks")
    for path in report.outputs:
        typer.echo(f"  wrote {path}")


@app.command()
def extract(
    ctx: typer.Context,
    tier: Annotated[Optional[Tier], typer.Option("--tier", help="Run only this tier.")] = None,
    changed_only: Annotated[bool, typer.Option("--changed-only", help="Skip chunks whose hash has not changed.")] = False,
    limit: Annotated[Optional[int], typer.Option("--limit", min=1, help="Tier 3: run at most N gated chunks, spread evenly (for benchmarks).")] = None,
) -> None:
    """Extract mentions (tiers 1 and 2) and assertions (tier 3)."""
    cfg = _load(ctx)
    from ckg.extract.pipeline import run_extract  # deferred: tier 2 imports torch

    try:
        report = run_extract(cfg, [int(tier.value)] if tier else [1, 2, 3], changed_only, limit=limit,
                             progress=lambda line: typer.echo(line, err=True))
    except (FileNotFoundError, RuntimeError, OSError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=2) from e
    for t in report.tiers:
        typer.echo(f"tier {t.tier}: {t.mentions} mentions from {t.processed} chunks processed, {t.reused} reused")
    if report.tier3:
        t3 = report.tier3
        typer.echo(f"tier 3: {t3.gated} chunks gated in, {t3.skipped} gated out; {t3.called} LLM calls this run, "
                   f"{t3.missing} gated chunks not run yet")
        typer.echo(f"tier 3: {t3.assertions} assertions kept, {sum(t3.rejected.values())} rejected, {t3.new_mentions} new mentions")
        for reason, n in sorted(t3.rejected.items()):
            typer.echo(f"  rejected {n}: {reason}")
        if t3.timing.get("chunks"):
            tm = t3.timing
            typer.echo(f"tier 3 timing over {tm['chunks']} chunks: median {tm['median_seconds']}s, "
                       f"mean {tm['mean_seconds']}s, p90 {tm['p90_seconds']}s per chunk ({tm['model']})")
    typer.echo(f"merged: {report.merged} mentions")
    for reason, n in sorted(report.dropped.items()):
        typer.echo(f"  dropped {n}: {reason}")
    for path in report.outputs:
        typer.echo(f"  wrote {path}")


@app.command()
def resolve(ctx: typer.Context) -> None:
    """Resolve mentions to canonical entities."""
    cfg = _load(ctx)
    from ckg.resolve.pipeline import run_resolve  # deferred: loads the embedding model

    try:
        r = run_resolve(cfg)
    except FileNotFoundError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=2) from e
    typer.echo(f"{r.mentions} mentions -> {r.entities} entities ({r.vocabulary_entities} from the vocabulary)")
    typer.echo(f"merges: {r.auto_merges} automatic, {r.human_merges} from review decisions")
    for reason, n in sorted(r.unresolved.items()):
        typer.echo(f"  unresolved {n} mentions: {reason}")
    typer.echo(f"review queue: {r.queued} pairs to decide, {r.decided} decided")
    typer.echo(f"{r.assertions_in} assertions -> {r.assertions_out} resolved")
    for reason, n in sorted(r.assertions_dropped.items()):
        typer.echo(f"  dropped {n}: {reason}")
    for path in r.outputs:
        typer.echo(f"  wrote {path}")


@app.command()
def load(ctx: typer.Context) -> None:
    """Load entities and assertions into the Oxigraph store."""
    cfg = _load(ctx)
    from ckg.load.pipeline import run_load

    try:
        r = run_load(cfg)
    except FileNotFoundError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=2) from e
    typer.echo(f"{r.chunks} graphs: {r.loaded} loaded, {r.unchanged} unchanged, {r.removed} removed")
    typer.echo(f"store: {r.quads} quads in {r.store}")


@app.command()
def validate(ctx: typer.Context) -> None:
    """Validate the graph against SHACL shapes generated from the schema."""
    cfg = _load(ctx)
    import logging

    logging.getLogger("rdflib").setLevel(logging.ERROR)  # ill-typed literals are reported below instead
    from ckg.validate.pipeline import run_validate

    r = run_validate(cfg)
    typer.echo(f"{r.triples} triples checked: {'conforms' if r.conforms else f'{r.violations} violations'}")
    for name, n in r.by_constraint.most_common():
        typer.echo(f"  {n:5} {name}")
    typer.echo("most affected predicates: " + ", ".join(f"{p} ({n})" for p, n in r.by_path.most_common(8)))
    for path in r.outputs:
        typer.echo(f"  wrote {path}")


@app.command()
def evaluate(ctx: typer.Context) -> None:
    """Run competency questions and write review reports."""
    cfg = _load(ctx)
    from ckg.evaluate.pipeline import run_evaluate

    try:
        r = run_evaluate(cfg)
    except FileNotFoundError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=2) from e
    typer.echo(f"competency questions: {r.answered} of {r.questions} return results"
               + (f" (no results: {', '.join(f'CQ {n}' for n in r.unanswered)})" if r.unanswered else ""))
    spot = r.spot
    typer.echo(f"spot check: {spot['sample']} assertions, {spot['marked']} marked"
               + (f", precision {spot['precision']:.0%}" if spot["precision"] is not None else ""))
    typer.echo(f"unmapped: {r.unmapped['entities']} entities without a vocabulary concept "
               f"({r.unmapped['share_of_mentions']:.0%} of mentions)")
    typer.echo("gold set: " + ("not present, skipped" if r.gold is None else
                               ", ".join(f"{k} F1 {v['micro']['f1']}" for k, v in r.gold.items())))
    for path in r.outputs:
        typer.echo(f"  wrote {path}")


@app.command()
def run(ctx: typer.Context) -> None:
    """Run all stages on changed content only."""
    for stage in (ingest, lambda c: extract(c, None, True, None), resolve, load, validate, evaluate):
        stage(ctx)


if __name__ == "__main__":
    app()
