from __future__ import annotations

from pathlib import Path

import pytest

from ckg.config import ConfigError, load_config

from conftest import PROJECT, sample_config_dict


def test_project_config_loads():
    cfg = load_config(PROJECT / "config.yaml")
    assert cfg.corpus.include is not None
    assert len(cfg.corpus.include) == len(set(cfg.corpus.include)), "duplicate entries in the slice"


@pytest.mark.skipif(not load_config(PROJECT / "config.yaml").corpus_dir.is_dir(), reason="corpus checkout not present")
def test_project_slice_files_exist():
    cfg = load_config(PROJECT / "config.yaml")
    files = cfg.corpus_files()
    assert len(files) == len(cfg.corpus.include)


def test_relative_paths_resolve_against_config_dir(sample_config: Path):
    cfg = load_config(sample_config)
    assert cfg.resolve(cfg.paths.out_dir) == (sample_config.parent / "out").resolve()


def test_full_scope_applies_excludes(sample_config: Path):
    files = [p.as_posix() for p in load_config(sample_config).corpus_files()]
    assert len(files) == 10
    assert not any(f.startswith("includes/") for f in files)
    assert "known-issues/azure-machine-learning-known-issues.md" in files
    assert files == sorted(files)


def test_include_list_selects_files(write_config):
    data = sample_config_dict()
    data["corpus"]["include"] = ["concept-endpoints.md", "how-to-assign-roles.md"]
    files = load_config(write_config(data)).corpus_files()
    assert [p.as_posix() for p in files] == ["concept-endpoints.md", "how-to-assign-roles.md"]


def test_include_list_missing_file_is_an_error(write_config):
    data = sample_config_dict()
    data["corpus"]["include"] = ["does-not-exist.md"]
    with pytest.raises(ConfigError, match="not found"):
        load_config(write_config(data)).corpus_files()


def test_missing_setting_is_an_error(write_config):
    data = sample_config_dict()
    del data["paths"]["out_dir"]
    with pytest.raises(ConfigError, match="out_dir"):
        load_config(write_config(data))


def test_unknown_setting_is_an_error(write_config):
    data = sample_config_dict()
    data["corpus"]["typo_key"] = 1
    with pytest.raises(ConfigError, match="typo_key"):
        load_config(write_config(data))


def test_missing_config_file_is_an_error(tmp_path: Path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_config_from_environment(sample_config: Path, monkeypatch):
    monkeypatch.setenv("CKG_CONFIG", str(sample_config))
    assert load_config().config_dir == sample_config.parent.resolve()
