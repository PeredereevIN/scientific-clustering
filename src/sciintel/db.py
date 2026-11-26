from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .config import settings

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    sha256 TEXT NOT NULL UNIQUE,
    path TEXT NOT NULL,
    domain TEXT NOT NULL DEFAULT 'unlabelled',
    title TEXT NOT NULL DEFAULT '',
    abstract TEXT NOT NULL DEFAULT '',
    text_path TEXT NOT NULL DEFAULT '',
    doi TEXT,
    year INTEGER,
    char_count INTEGER NOT NULL DEFAULT 0,
    page_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'ingested',
    error TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_documents_domain ON documents(domain);
CREATE INDEX IF NOT EXISTS idx_documents_doi ON documents(doi);
CREATE TABLE IF NOT EXISTS keywords (
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    method TEXT NOT NULL,
    phrase TEXT NOT NULL,
    score REAL NOT NULL,
    rank INTEGER NOT NULL,
    PRIMARY KEY(document_id, method, phrase)
);
CREATE INDEX IF NOT EXISTS idx_keywords_phrase ON keywords(phrase);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    device TEXT NOT NULL,
    batch_size INTEGER NOT NULL,
    eps REAL NOT NULL,
    min_samples INTEGER NOT NULL,
    metrics_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS cluster_labels (
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    cluster_id INTEGER NOT NULL,
    x REAL,
    y REAL,
    PRIMARY KEY(run_id, document_id)
);
"""


@contextmanager
def connect(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    path = db_path or settings.db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=60000")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    settings.ensure_dirs()
    with connect() as conn:
        conn.executescript(SCHEMA)


def rows(sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    init_db()
    with connect() as conn:
        return [dict(row) for row in conn.execute(sql, params).fetchall()]


def document_count() -> int:
    return int(rows("SELECT COUNT(*) AS n FROM documents")[0]["n"])


def latest_run() -> dict[str, Any] | None:
    found = rows("SELECT * FROM runs ORDER BY created_at DESC, rowid DESC LIMIT 1")
    if not found:
        return None
    item = found[0]
    item["metrics"] = json.loads(item.pop("metrics_json"))
    return item
