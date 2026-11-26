from __future__ import annotations

import logging
from pathlib import Path

from .. import db
from ..config import settings
from .analyzer import MODEL_NAMES, analyze_text, detect_language, load_nlp
from .converters import convert_to_markdown
from .indexer import init_linguistic_schema, replace_document_index

log = logging.getLogger("sciintel.linguistic")


def _resolve_document_source(row) -> Path:
    """Resolve source text robustly when DB paths point to a previous project root.

    Priority: still-existing original source, current project's processed text,
    current project's processed Markdown, then any existing DB text_path.
    Stale absolute paths from another machine/project are never trusted.
    """
    doc_id = str(row["id"])
    candidates: list[Path] = []
    raw_source = str(row["path"] or "").strip()
    raw_text = str(row["text_path"] or "").strip()

    if raw_source:
        p = Path(raw_source)
        candidates.append(p if p.is_absolute() else settings.root / p)
    # Canonical locations in THIS project's configured processed directory.
    doc_dir = settings.processed / doc_id
    candidates.extend((doc_dir / "document.txt", doc_dir / "document.md"))

    if raw_text:
        p = Path(raw_text)
        # An absolute DB path is accepted only if it really exists. Relative
        # paths are interpreted relative to the current project root.
        candidates.append(p if p.is_absolute() else settings.root / p)

    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            key = str(resolved).casefold()
            if key in seen:
                continue
            seen.add(key)
            if resolved.is_file():
                return resolved
        except OSError:
            continue

    searched = "\n  - ".join(str(p) for p in candidates) or "(нет путей в записи БД)"
    raise FileNotFoundError(
        f"Не найден исходный файл документа {doc_id}. Проверены пути:\n  - {searched}\n"
        f"Проверьте наличие {doc_dir / 'document.txt'} или настройку SIL_PROCESSED."
    )


def convert_existing_documents(force: bool = False) -> dict[str, int]:
    db.init_db()
    with db.connect() as conn:
        init_linguistic_schema(conn)
        docs = conn.execute("SELECT id,path,text_path FROM documents ORDER BY id").fetchall()
    stats = {"found": len(docs), "converted": 0, "existing": 0, "failed": 0}
    for row in docs:
        output = settings.processed / row["id"] / "document.md"
        try:
            if output.exists() and not force:
                stats["existing"] += 1
                continue
            source = _resolve_document_source(row)
            markdown, _ = convert_to_markdown(source)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(markdown, encoding="utf-8")
            stats["converted"] += 1
        except Exception as exc:
            stats["failed"] += 1
            log.exception("Markdown conversion failed for %s: %s", row["id"], exc)
    return stats


def index_documents(language: str = "auto", force: bool = False, limit: int | None = None) -> dict:
    db.init_db()
    with db.connect() as conn:
        init_linguistic_schema(conn)
        sql = """SELECT d.id,d.path,d.text_path,ld.document_id AS indexed_id FROM documents d
                 LEFT JOIN linguistic_documents ld ON ld.document_id=d.id ORDER BY d.id"""
        docs = conn.execute(sql + (" LIMIT ?" if limit else ""), ((limit,) if limit else ())).fetchall()
    stats = {"found": len(docs), "indexed": 0, "skipped": 0, "failed": 0,
             "sentences": 0, "tokens": 0, "concepts": 0}
    model_cache = {}
    for row in docs:
        if row["indexed_id"] and not force and language == "auto":
            stats["skipped"] += 1
            continue
        doc_id = row["id"]
        md_path = settings.processed / doc_id / "document.md"
        try:
            if not md_path.is_file():
                source = _resolve_document_source(row)
                markdown, _ = convert_to_markdown(source)
                md_path.parent.mkdir(parents=True, exist_ok=True)
                md_path.write_text(markdown, encoding="utf-8")
            text = md_path.read_text(encoding="utf-8")
            if not text.strip():
                raise ValueError(f"Пустой текст после конвертации: {md_path}")
            lang = detect_language(text) if language == "auto" else language
            if lang not in MODEL_NAMES:
                raise ValueError(f"Unsupported language: {lang}")
            if lang not in model_cache:
                model_cache[lang] = load_nlp(lang)
            nlp = model_cache[lang]
            sentences, tokens, concepts = analyze_text(text, lang, nlp)
            # Record the actual source resolved on this machine, not the stale DB path.
            source_path = _resolve_document_source(row)
            counts = replace_document_index(doc_id, lang, MODEL_NAMES[lang], str(source_path),
                                            str(md_path), sentences, tokens, concepts)
            stats["indexed"] += 1
            for key, value in counts.items():
                stats[key] += value
        except Exception as exc:
            stats["failed"] += 1
            log.exception("Linguistic indexing failed for %s: %s", doc_id, exc)
    return stats
