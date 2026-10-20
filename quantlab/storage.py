"""SQLite persistence for resumable quantization runs."""

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def validate_run_id(run_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", run_id):
        raise ValueError("Run ID must be 1-64 letters, numbers, dots, underscores, or hyphens and start with a letter or number.")
    return run_id


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    device TEXT NOT NULL,
    config_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS results (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    method TEXT NOT NULL,
    variant TEXT NOT NULL,
    status TEXT NOT NULL,
    artifact_path TEXT,
    size_gib REAL,
    perplexity REAL,
    custom_score REAL,
    task_scores_json TEXT,
    kv_cache_dtype TEXT,
    batch_size INTEGER,
    tok_s REAL,
    tok_s_std REAL,
    prefill_tok_s REAL,
    prefill_ms_estimate REAL,
    ttft_ms REAL,
    ttft_ms_std REAL,
    peak_mem_gib REAL,
    error TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (run_id, method, variant)
);
"""


class RunStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(results)")}
            if "custom_score" not in columns:
                connection.execute("ALTER TABLE results ADD COLUMN custom_score REAL")
            if "task_scores_json" not in columns:
                connection.execute("ALTER TABLE results ADD COLUMN task_scores_json TEXT")
            if "kv_cache_dtype" not in columns:
                connection.execute("ALTER TABLE results ADD COLUMN kv_cache_dtype TEXT")
            if "batch_size" not in columns:
                connection.execute("ALTER TABLE results ADD COLUMN batch_size INTEGER")
            for column in ("tok_s_std", "prefill_tok_s", "prefill_ms_estimate", "ttft_ms_std"):
                if column not in columns:
                    connection.execute(f"ALTER TABLE results ADD COLUMN {column} REAL")

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def create_run(self, run_id: str, model: str, device: str, config: dict[str, Any]) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO runs VALUES (?, ?, ?, ?, ?)",
                (run_id, model, device, json.dumps(config, sort_keys=True), _now()),
            )

    def get_result(self, run_id: str, method: str, variant: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM results WHERE run_id = ? AND method = ? AND variant = ?",
                (run_id, method, variant),
            ).fetchone()
        return dict(row) if row else None

    def save_result(self, result: dict[str, Any]) -> None:
        fields = (
            "run_id", "method", "variant", "status", "artifact_path", "size_gib",
            "perplexity", "custom_score", "task_scores_json", "kv_cache_dtype", "batch_size",
            "tok_s", "tok_s_std", "prefill_tok_s", "prefill_ms_estimate", "ttft_ms", "ttft_ms_std",
            "peak_mem_gib", "error",
        )
        values = [result.get(field) for field in fields]
        values.append(_now())
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO results (" + ", ".join(fields) + ", updated_at) "
                "VALUES (" + ", ".join("?" for _ in values) + ") "
                "ON CONFLICT(run_id, method, variant) DO UPDATE SET "
                + ", ".join(f"{field}=excluded.{field}" for field in fields[3:])
                + ", updated_at=excluded.updated_at",
                values,
            )

    def results(self, run_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM results WHERE run_id = ? ORDER BY method, variant", (run_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def run_exists(self, run_id: str) -> bool:
        with self.connect() as connection:
            return connection.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone() is not None

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        run = dict(row)
        run["config"] = json.loads(run.pop("config_json"))
        return run


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
