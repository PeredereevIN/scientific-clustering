from __future__ import annotations

import logging
import json
from pathlib import Path

from tqdm import tqdm

from . import db
from .config import settings
from .utils import (SUPPORTED, clean_text, extract_abstract, extract_doi,
                    extract_title, extract_year, infer_domain, sha256_file)

log = logging.getLogger("sciintel")


def read_document(path: Path) -> tuple[str, str, int]:
    if path.suffix.lower() in {".md", ".markdown", ".txt"}:
        raw = path.read_text(encoding="utf-8", errors="replace")
        return clean_text(raw), "", 0
    import pymupdf
    pages_text: list[str] = []
    metadata_title = ""
    page_count = 0
    with pymupdf.open(path) as pdf:
        page_count = len(pdf)
        metadata_title = (pdf.metadata or {}).get("title") or ""
        for page_index in range(min(page_count, settings.max_pages)):
            page = pdf.load_page(page_index)
            pages_text.append(page.get_text("text", sort=True))
    return clean_text("\n\n".join(pages_text)), metadata_title, page_count


def ingest(corpus_dir: Path | None = None) -> dict[str, int]:
    db.init_db()
    source_dir = corpus_dir or settings.inbox
    files = sorted(p for p in source_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)
    stats = {"found": len(files), "new": 0, "existing": 0, "failed": 0}
    if not files:
        log.warning("No PDF/MD/TXT documents found in %s", source_dir)
        return stats
    for path in tqdm(files, desc="Ingest", unit="doc"):
        try:
            digest = sha256_file(path)
            document_id = digest[:24]
            existing = db.rows("SELECT id FROM documents WHERE sha256=?", (digest,))
            if existing:
                stats["existing"] += 1
                continue
            text, metadata_title, page_count = read_document(path)
            if len(text) < 100:
                raise ValueError("Extracted text is too short (<100 characters); likely scanned or empty PDF")
            # arXiv PDFs are accompanied by a JSON sidecar from the sampler.
            # Prefer its canonical title/abstract/DOI/year while retaining the
            # extracted PDF body as the source text for audit and keyword work.
            sidecar = path.with_suffix(".json")
            source_meta = {}
            if sidecar.is_file():
                try:
                    candidate = json.loads(sidecar.read_text(encoding="utf-8"))
                    if isinstance(candidate, dict) and candidate.get("arxiv_id"):
                        source_meta = candidate
                except (OSError, json.JSONDecodeError):
                    log.warning("Could not read metadata sidecar %s", sidecar)
            title = str(source_meta.get("title") or extract_title(text, metadata_title))[:500]
            abstract = str(source_meta.get("abstract") or extract_abstract(text))[:5000]
            doi = source_meta.get("doi") or extract_doi(text)
            published = str(source_meta.get("published") or "")
            try:
                year = int(published[:4]) if published[:4].isdigit() else extract_year(text)
            except (TypeError, ValueError):
                year = extract_year(text)
            out = settings.processed / document_id
            out.mkdir(parents=True, exist_ok=True)
            text_path = out / "document.txt"
            text_path.write_text(text, encoding="utf-8")
            with db.connect() as conn:
                conn.execute("""
                    INSERT OR IGNORE INTO documents
                    (id, sha256, path, domain, title, abstract, text_path, doi, year,
                     char_count, page_count, status, error, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ready', NULL, CURRENT_TIMESTAMP)
                """, (document_id, digest, str(path.resolve()), infer_domain(path, source_dir),
                      title, abstract, str(text_path.resolve()), doi, year,
                      len(text), page_count))
            stats["new"] += 1
        except Exception as exc:
            stats["failed"] += 1
            log.exception("Could not ingest %s: %s", path, exc)
    log.info("Ingest completed: %s", stats)
    return stats
