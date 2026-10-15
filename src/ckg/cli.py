"""Command-line interface (DESIGN section 9).

Phase 0 skeleton: every command loads and validates the config, then
reports that its stage is not built yet and exits with NOT_IMPLEMENTED.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import typer

from ckg import __version__
from ckg.config import Config, ConfigError, load_config

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
    try:
        files = cfg.corpus_files(input)
    except ConfigError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=2) from e
    typer.echo(f"{len(files)} files in scope under {input or cfg.corpus_dir}")
    _not_built("ingest", 1)


@app.command()
def extract(
    ctx: typer.Context,
    tier: Annotated[Optional[Tier], typer.Option("--tier", help="Run only this tier.")] = None,
    changed_only: Annotated[bool, typer.Option("--changed-only", help="Skip chunks whose hash has not changed.")] = False,
) -> None:
    """Extract mentions (tiers 1 and 2) and assertions (tier 3)."""
    _load(ctx)
    _not_built("extract", 4 if tier is Tier.llm else 3)


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
