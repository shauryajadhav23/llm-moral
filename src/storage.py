"""
Storage layer: DuckDB tables for scenarios, runs, and turns.

All interactions go through the `DB` class. The database file lives at the
path given on construction (default: outputs/runs.db).
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import duckdb

from .schema import ManifestRow, RunStatus, Scenario

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_CREATE_SCENARIOS = """
CREATE TABLE IF NOT EXISTS scenarios (
    scenario_id         TEXT PRIMARY KEY,
    type                TEXT        NOT NULL,
    independence_class  TEXT        NOT NULL,
    n_components        INTEGER     NOT NULL,
    setup_text          TEXT        NOT NULL,
    raw_yaml            TEXT
);
"""

_CREATE_RUNS = """
CREATE TABLE IF NOT EXISTS runs (
    run_id              TEXT PRIMARY KEY,
    scenario_id         TEXT        NOT NULL,
    path_type           TEXT        NOT NULL,
    path_signature      TEXT        NOT NULL,
    perm_id             TEXT        NOT NULL,
    terminal_wording_id TEXT        NOT NULL,
    model               TEXT        NOT NULL,
    model_version       TEXT,
    provider_requested  TEXT        NOT NULL,
    provider_used       TEXT,
    temperature         REAL        NOT NULL,
    seed                INTEGER     NOT NULL,
    replicate_idx       INTEGER     NOT NULL,
    verdict             INTEGER,
    status              TEXT        NOT NULL DEFAULT 'pending',
    n_turns             INTEGER,
    total_tokens        INTEGER,
    latency_ms          INTEGER,
    timestamp           TIMESTAMP,
    FOREIGN KEY (scenario_id) REFERENCES scenarios(scenario_id)
);
"""

_CREATE_TURNS = """
CREATE TABLE IF NOT EXISTS turns (
    turn_id         TEXT PRIMARY KEY,
    run_id          TEXT        NOT NULL,
    turn_index      INTEGER     NOT NULL,
    role            TEXT        NOT NULL,
    content         TEXT        NOT NULL,
    is_terminal     INTEGER     NOT NULL DEFAULT 0,
    raw_response    TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
"""


# ---------------------------------------------------------------------------
# Database class
# ---------------------------------------------------------------------------

class DB:
    """Thread-safe wrapper around a DuckDB connection."""

    def __init__(self, db_path: str | Path = "outputs/runs.db") -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = duckdb.connect(str(self._path))
        self._init_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(_CREATE_SCENARIOS)
            self._conn.execute(_CREATE_RUNS)
            self._conn.execute(_CREATE_TURNS)

    # ------------------------------------------------------------------
    # Scenarios
    # ------------------------------------------------------------------

    def upsert_scenario(self, scenario: Scenario, raw_yaml: str = "") -> None:
        sql = """
        INSERT INTO scenarios
            (scenario_id, type, independence_class, n_components, setup_text, raw_yaml)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (scenario_id) DO UPDATE SET
            type               = excluded.type,
            independence_class = excluded.independence_class,
            n_components       = excluded.n_components,
            setup_text         = excluded.setup_text,
            raw_yaml           = excluded.raw_yaml
        """
        with self._lock:
            self._conn.execute(sql, [
                scenario.id,
                scenario.type,
                scenario.independence_class.value,
                len(scenario.considerations),
                scenario.setup.strip(),
                raw_yaml,
            ])

    def get_scenario_ids(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute("SELECT scenario_id FROM scenarios").fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # Manifest / Runs
    # ------------------------------------------------------------------

    def insert_manifest(self, rows: list[ManifestRow]) -> None:
        """Bulk-insert manifest rows (pending status). Ignores duplicates."""
        sql = """
        INSERT OR IGNORE INTO runs
            (run_id, scenario_id, path_type, path_signature, perm_id,
             terminal_wording_id, model, provider_requested, temperature,
             seed, replicate_idx, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        data = [
            (
                r.run_id,
                r.scenario_id,
                r.path_type,
                r.path_signature,
                r.perm_id,
                r.terminal_wording_id,
                r.model,
                r.provider,
                r.temperature,
                r.seed,
                r.replicate_idx,
                r.status.value,
            )
            for r in rows
        ]
        with self._lock:
            self._conn.executemany(sql, data)

    def get_pending_runs(self) -> list[dict[str, Any]]:
        """Return all runs with status = pending."""
        sql = "SELECT * FROM runs WHERE status = 'pending'"
        with self._lock:
            result = self._conn.execute(sql)
            cols = [d[0] for d in result.description]
            rows = result.fetchall()
        return [dict(zip(cols, row)) for row in rows]

    def get_all_runs(self) -> list[dict[str, Any]]:
        sql = "SELECT * FROM runs ORDER BY run_id"
        with self._lock:
            result = self._conn.execute(sql)
            cols = [d[0] for d in result.description]
            rows = result.fetchall()
        return [dict(zip(cols, row)) for row in rows]

    def update_run(
        self,
        run_id: str,
        *,
        status: RunStatus,
        verdict: Optional[int] = None,
        model_version: Optional[str] = None,
        provider_used: Optional[str] = None,
        n_turns: Optional[int] = None,
        total_tokens: Optional[int] = None,
        latency_ms: Optional[int] = None,
    ) -> None:
        """Atomically update a run row after execution."""
        ts = datetime.now(timezone.utc).isoformat()
        sql = """
        UPDATE runs SET
            status        = ?,
            verdict       = ?,
            model_version = ?,
            provider_used = ?,
            n_turns       = ?,
            total_tokens  = ?,
            latency_ms    = ?,
            timestamp     = ?
        WHERE run_id = ?
        """
        with self._lock:
            self._conn.execute(sql, [
                status.value,
                verdict,
                model_version,
                provider_used,
                n_turns,
                total_tokens,
                latency_ms,
                ts,
                run_id,
            ])

    def count_by_status(self) -> dict[str, int]:
        sql = "SELECT status, COUNT(*) FROM runs GROUP BY status"
        with self._lock:
            rows = self._conn.execute(sql).fetchall()
        return {r[0]: r[1] for r in rows}

    # ------------------------------------------------------------------
    # Turns
    # ------------------------------------------------------------------

    def insert_turns(self, turns: list[dict[str, Any]]) -> None:
        """Insert a batch of turn records for a single run."""
        sql = """
        INSERT OR IGNORE INTO turns
            (turn_id, run_id, turn_index, role, content, is_terminal, raw_response)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        data = [
            (
                t["turn_id"],
                t["run_id"],
                t["turn_index"],
                t["role"],
                t["content"],
                int(t.get("is_terminal", False)),
                t.get("raw_response"),
            )
            for t in turns
        ]
        with self._lock:
            self._conn.executemany(sql, data)

    def get_turns_for_run(self, run_id: str) -> list[dict[str, Any]]:
        sql = "SELECT * FROM turns WHERE run_id = ? ORDER BY turn_index"
        with self._lock:
            result = self._conn.execute(sql, [run_id])
            cols = [d[0] for d in result.description]
            rows = result.fetchall()
        return [dict(zip(cols, row)) for row in rows]

    # ------------------------------------------------------------------
    # Analytics helpers (used by analysis scripts)
    # ------------------------------------------------------------------

    def query(self, sql: str, params: list | None = None) -> list[dict[str, Any]]:
        """Execute an arbitrary SELECT and return dicts."""
        with self._lock:
            result = self._conn.execute(sql, params or [])
            cols = [d[0] for d in result.description]
            rows = result.fetchall()
        return [dict(zip(cols, row)) for row in rows]

    def to_dataframe(self, sql: str, params: list | None = None):
        """Return query results as a pandas DataFrame."""
        import pandas as pd  # lazy import — not required at module level
        with self._lock:
            result = self._conn.execute(sql, params or [])
            cols = [d[0] for d in result.description]
            rows = result.fetchall()
        return pd.DataFrame(rows, columns=cols)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "DB":
        return self

    def __exit__(self, *_) -> None:
        self.close()
