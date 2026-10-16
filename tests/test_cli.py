from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from ckg import __version__
from ckg.cli import NOT_IMPLEMENTED, app

runner = CliRunner()
COMMANDS = ["ingest", "extract", "resolve", "load", "validate", "evaluate", "run"]
STUBS = ["evaluate", "run"]


def test_help_lists_every_command():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in COMMANDS:
        assert cmd in result.output


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


@pytest.mark.parametrize("cmd", STUBS)
def test_stages_report_not_implemented(cmd: str, sample_config: Path):
    result = runner.invoke(app, ["--config", str(sample_config), cmd])
    assert result.exit_code == NOT_IMPLEMENTED
    assert "not implemented yet" in result.output




def test_ingest_writes_outputs(sample_config: Path):
    result = runner.invoke(app, ["--config", str(sample_config), "ingest"])
    assert result.exit_code == 0
    assert "10 documents ->" in result.output
    assert (sample_config.parent / "out" / "chunks.jsonl").is_file()


def test_ingest_input_outside_repo_is_an_error(sample_config: Path, tmp_path: Path):
    result = runner.invoke(app, ["--config", str(sample_config), "ingest", "--input", str(tmp_path)])
    assert result.exit_code == 2


def test_bad_config_exits_with_error(tmp_path: Path):
    result = runner.invoke(app, ["--config", str(tmp_path / "missing.yaml"), "evaluate"])
    assert result.exit_code == 2
    assert "config file not found" in result.output
