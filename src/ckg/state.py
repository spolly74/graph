"""Pipeline state in SQLite: which chunk hash and settings each stage last processed."""

from __future__ import annotations

import sqlite3
from pathlib import Path


class StateDB:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS chunk_state ("
            " stage TEXT NOT NULL, chunk_id TEXT NOT NULL, chunk_sha TEXT NOT NULL, settings TEXT NOT NULL,"
            " PRIMARY KEY (stage, chunk_id))"
        )

        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS llm_cache ("
            " chunk_id TEXT PRIMARY KEY, chunk_sha TEXT NOT NULL, settings TEXT NOT NULL, response TEXT NOT NULL,"
            " prompt_tokens INTEGER, output_tokens INTEGER, seconds REAL, model_id TEXT)"
        )

    def llm_get(self, chunk_id: str, chunk_sha: str, settings: str) -> dict | None:
        row = self.conn.execute(
            "SELECT response, prompt_tokens, output_tokens, seconds, model_id FROM llm_cache"
            " WHERE chunk_id = ? AND chunk_sha = ? AND settings = ?", (chunk_id, chunk_sha, settings)).fetchone()
        if row is None:
            return None
        return dict(zip(("response", "prompt_tokens", "output_tokens", "seconds", "model_id"), row))

    def llm_put(self, chunk_id: str, chunk_sha: str, settings: str, response: str,
                prompt_tokens: int, output_tokens: int, seconds: float, model_id: str) -> None:
        """Store one LLM response and commit at once, so an interrupted run resumes here."""
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO llm_cache VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (chunk_id, chunk_sha, settings, response, prompt_tokens, output_tokens, seconds, model_id))

    def processed(self, stage: str) -> dict[str, tuple[str, str]]:
        rows = self.conn.execute("SELECT chunk_id, chunk_sha, settings FROM chunk_state WHERE stage = ?", (stage,))
        return {cid: (sha, settings) for cid, sha, settings in rows}

    def replace(self, stage: str, rows: list[tuple[str, str, str]]) -> None:
        """Record the current set of processed chunks for a stage, dropping chunks that no longer exist."""
        with self.conn:
            self.conn.execute("DELETE FROM chunk_state WHERE stage = ?", (stage,))
            self.conn.executemany(
                "INSERT INTO chunk_state (stage, chunk_id, chunk_sha, settings) VALUES (?, ?, ?, ?)",
                [(stage, *r) for r in rows],
            )

    def close(self) -> None:
        self.conn.close()
