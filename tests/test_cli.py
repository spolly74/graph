from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from ckg import __version__
from ckg.cli import NOT_IMPLEMENTED, app

runner = CliRunner()
COMMANDS = ["ingest", "extract", "resolve", "load", "validate", "evaluate", "run"]


def test_help_lists_every_command():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in COMMANDS:
        assert cmd in result.output


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


@pytest.mark.parametrize("cmd", COMMANDS)
def test_stages_report_not_implemented(cmd: str, sample_config: Path):
    result = runner.invoke(app, ["--config", str(sample_config), cmd])
    assert result.exit_code == NOT_IMPLEMENTED
    assert "not implemented yet" in result.output


def test_extract_accepts_tier_and_changed_only(sample_config: Path):
    result = runner.invoke(app, ["--config", str(sample_config), "extract", "--tier", "3", "--changed-only"])
    assert result.exit_code == NOT_IMPLEMENTED
    assert "Phase 4" in result.output


def test_ingest_counts_files_in_scope(sample_config: Path):
    result = runner.invoke(app, ["--config", str(sample_config), "ingest"])
    assert "10 files in scope" in result.output


def test_bad_config_exits_with_error(tmp_path: Path):
    result = runner.invoke(app, ["--config", str(tmp_path / "missing.yaml"), "resolve"])
    assert result.exit_code == 2
    assert "config file not found" in result.output
