"""Command-line interface (DESIGN section 9).

Every command loads and validates the config. Stages that are not built
yet report the phase that builds them and exit with NOT_IMPLEMENTED.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import typer

from ckg import __version__
from ckg.config import Config, ConfigError, load_config
from ckg.ingest.pipeline import run_ingest

NOT_IMPLEMENTED = 3

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


def _not_built(stage: str, phase: int) -> None:
    typer.echo(f"{stage}: not implemented yet (Phase {phase}).", err=True)
    raise typer.Exit(code=NOT_IMPLEMENTED)


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
) -> None:
    """Extract mentions (tiers 1 and 2) and assertions (tier 3)."""
    cfg = _load(ctx)
    if tier is Tier.llm:
        _not_built("extract --tier 3", 4)
    from ckg.extract.pipeline import run_extract  # deferred: tier 2 imports torch

    try:
        report = run_extract(cfg, [int(tier.value)] if tier else [1, 2], changed_only)
    except FileNotFoundError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=2) from e
    for t in report.tiers:
        typer.echo(f"tier {t.tier}: {t.mentions} mentions from {t.processed} chunks processed, {t.reused} reused")
    typer.echo(f"merged: {report.merged} mentions")
    for reason, n in sorted(report.dropped.items()):
        typer.echo(f"  dropped {n}: {reason}")
    for path in report.outputs:
        typer.echo(f"  wrote {path}")
    if tier is None:
        typer.echo("tier 3: not implemented yet (Phase 4).")


@app.command()
def resolve(ctx: typer.Context) -> None:
    """Resolve mentions to canonical entities."""
    _load(ctx)
    _not_built("resolve", 5)


@app.command()
def load(ctx: typer.Context) -> None:
    """Load entities and assertions into the Oxigraph store."""
    _load(ctx)
    _not_built("load", 6)


@app.command()
def validate(ctx: typer.Context) -> None:
    """Validate the graph against SHACL shapes generated from the schema."""
    _load(ctx)
    _not_built("validate", 6)


@app.command()
def evaluate(ctx: typer.Context) -> None:
    """Run competency questions and write review reports."""
    _load(ctx)
    _not_built("evaluate", 7)


@app.command()
def run(ctx: typer.Context) -> None:
    """Run all stages on changed content only."""
    _load(ctx)
    _not_built("run", 7)


if __name__ == "__main__":
    app()
