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
