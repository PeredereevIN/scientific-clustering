from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
except ImportError:
    pass


def _path(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default))
    return value if value.is_absolute() else ROOT / value


@dataclass(frozen=True, slots=True)
class Settings:
    root: Path = ROOT
    inbox: Path = _path("SIL_INBOX", "data/inbox")
    processed: Path = _path("SIL_PROCESSED", "data/processed")
    db_path: Path = _path("SIL_DB", "data/silab.sqlite3")
    artifacts: Path = _path("SIL_ARTIFACTS", "artifacts")
    reports: Path = _path("SIL_REPORTS", "reports")
    model_name: str = os.getenv("SIL_MODEL", "sentence-transformers/allenai-specter")
    max_pages: int = int(os.getenv("SIL_MAX_PAGES", "80"))
    max_keyword_chars: int = int(os.getenv("SIL_MAX_KEYWORD_CHARS", "60000"))
    max_embedding_chars: int = int(os.getenv("SIL_MAX_EMBEDDING_CHARS", "12000"))
    keyword_top_n: int = int(os.getenv("SIL_KEYWORD_TOP_N", "25"))
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")

    def ensure_dirs(self) -> None:
        for path in (self.inbox, self.processed, self.db_path.parent,
                     self.artifacts, self.reports):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
