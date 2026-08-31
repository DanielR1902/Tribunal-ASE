"""SQLite persistence layer for the Westeros Tribunal simulation.

Both architectures write exactly one row per run into the ``trial_runs``
table of ``court_runs.db``, tagged with ``architecture_mode`` so that
single-agent and multi-agent runs can be compared later (e.g. via a simple
``SELECT`` grouping by architecture_mode).
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

# The database lives at the repository root regardless of which
# architecture's main.py is invoking it, and regardless of the caller's
# current working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = str(_REPO_ROOT / "court_runs.db")


def get_db_path() -> str:
    """Resolve the database path, honoring an optional override via env var."""
    return os.getenv("COURT_DB_PATH", DEFAULT_DB_PATH)


SCHEMA = """
CREATE TABLE IF NOT EXISTS trial_runs (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp               TEXT    NOT NULL,
    architecture_mode       TEXT    NOT NULL,
    prosecution_summary     TEXT    NOT NULL,
    defense_summary         TEXT    NOT NULL,
    judge_barak_reasoning   TEXT    NOT NULL,
    judge_barak_verdict     TEXT    NOT NULL,
    judge_elon_reasoning    TEXT    NOT NULL,
    judge_elon_verdict      TEXT    NOT NULL,
    judge_shamgar_reasoning TEXT    NOT NULL,
    judge_shamgar_verdict   TEXT    NOT NULL,
    prompt_tokens           INTEGER NOT NULL,
    completion_tokens       INTEGER NOT NULL,
    total_tokens            INTEGER NOT NULL,
    cost_usd                REAL    NOT NULL,
    cost_ils                REAL    NOT NULL,
    execution_time_sec      REAL    NOT NULL
);
"""


@contextmanager
def _connect(db_path: str | None = None) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path or get_db_path())
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str | None = None) -> str:
    """Create the trial_runs table if it does not already exist.

    Returns the resolved database path so callers can log/print it.
    """
    resolved_path = db_path or get_db_path()
    with _connect(resolved_path) as conn:
        conn.execute(SCHEMA)
    return resolved_path


@dataclass
class TrialRunRecord:
    """All fields required to log one completed tribunal run."""

    architecture_mode: str  # "single_agent" or "multi_agent"
    prosecution_summary: str
    defense_summary: str
    judge_barak_reasoning: str
    judge_barak_verdict: str
    judge_elon_reasoning: str
    judge_elon_verdict: str
    judge_shamgar_reasoning: str
    judge_shamgar_verdict: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    cost_ils: float
    execution_time_sec: float


def log_trial_run(record: TrialRunRecord, db_path: str | None = None) -> int:
    """Insert one trial run row and return the new row's primary key."""
    resolved_path = db_path or get_db_path()
    init_db(resolved_path)  # idempotent safety net if init_db wasn't called yet
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with _connect(resolved_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO trial_runs (
                timestamp, architecture_mode,
                prosecution_summary, defense_summary,
                judge_barak_reasoning, judge_barak_verdict,
                judge_elon_reasoning, judge_elon_verdict,
                judge_shamgar_reasoning, judge_shamgar_verdict,
                prompt_tokens, completion_tokens, total_tokens,
                cost_usd, cost_ils, execution_time_sec
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                record.architecture_mode,
                record.prosecution_summary,
                record.defense_summary,
                record.judge_barak_reasoning,
                record.judge_barak_verdict,
                record.judge_elon_reasoning,
                record.judge_elon_verdict,
                record.judge_shamgar_reasoning,
                record.judge_shamgar_verdict,
                record.prompt_tokens,
                record.completion_tokens,
                record.total_tokens,
                record.cost_usd,
                record.cost_ils,
                record.execution_time_sec,
            ),
        )
        run_id = cursor.lastrowid

    return int(run_id)


def fetch_run(run_id: int, db_path: str | None = None) -> sqlite3.Row | None:
    """Convenience lookup, mainly useful for tests / manual inspection."""
    resolved_path = db_path or get_db_path()
    with _connect(resolved_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM trial_runs WHERE id = ?", (run_id,)
        ).fetchone()
    return row
