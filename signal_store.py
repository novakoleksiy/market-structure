"""SQLite-backed persistence for scheduled signal runs."""

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class StoredSignal:
    source: str
    symbol: str
    cluster: str
    direction: str
    timestamp: datetime
    price: float


class SignalStore:
    """Persist emitted signals and per-job progress."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS signals (
                    source TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    cluster TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    signal_ts TEXT NOT NULL,
                    price REAL NOT NULL,
                    emitted_at TEXT NOT NULL,
                    UNIQUE (source, symbol, cluster, direction, signal_ts)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS job_state (
                    source TEXT NOT NULL,
                    cluster TEXT NOT NULL,
                    last_bar_ts TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (source, cluster)
                )
                """
            )

    def get_last_bar_ts(self, source: str, cluster: str) -> datetime | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT last_bar_ts FROM job_state WHERE source = ? AND cluster = ?",
                (source, cluster),
            ).fetchone()
        if row is None:
            return None
        return datetime.fromisoformat(row["last_bar_ts"])

    def set_last_bar_ts(self, source: str, cluster: str, timestamp: datetime) -> None:
        ts = timestamp.astimezone(UTC).isoformat()
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO job_state (source, cluster, last_bar_ts, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(source, cluster)
                DO UPDATE SET last_bar_ts = excluded.last_bar_ts,
                              updated_at = excluded.updated_at
                """,
                (source, cluster, ts, now),
            )

    def insert_signal(self, signal: StoredSignal) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO signals (
                    source, symbol, cluster, direction, signal_ts, price, emitted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    signal.source,
                    signal.symbol,
                    signal.cluster,
                    signal.direction,
                    signal.timestamp.astimezone(UTC).isoformat(),
                    signal.price,
                    datetime.now(UTC).isoformat(),
                ),
            )
        return cur.rowcount > 0
