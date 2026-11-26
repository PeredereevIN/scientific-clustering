#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local-first paper abstract and text enrichment for topic interpretation.

The module aligns records by document ID, OpenAlex ID, or DOI; reads local metadata
files and linked PDFs; optionally retrieves missing abstracts from OpenAlex and
Crossref; and caches all remote responses. It never changes clustering inputs.
"""
from __future__ import annotations

import html
import hashlib
import importlib.util
import json
import difflib
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ID_ALIASES = {
    "documentid", "docid", "paperid", "openalexid", "openalex", "id",
    "doi", "doiurl", "externalid", "externalids", "pmid", "pmcid", "arxivid", "arxiv",
    "corpusid", "semanticsscholarid", "s2paperid", "magid", "scopusid", "eid",
}
DOI_ALIASES = {"doi", "doiurl", "digitalobjectidentifier"}
OPENALEX_ALIASES = {"openalexid", "openalex"}
ABSTRACT_ALIASES = {
    "abstract", "abstracttext", "cleanabstract", "paperabstract", "summary",
    "abstractcontent", "descriptionabstract", "abstractinvertedindex",
}
FULLTEXT_ALIASES = {
    "fulltext", "cleantext", "extractedtext", "articletext", "papertext",
    "text", "content", "body", "plaintext", "documenttext",
}
PDF_PATH_ALIASES = {"pdfpath", "pdffile", "filepath", "localpath", "sourcepath", "filepathpdf", "filenamepdf"}


def _norm_col(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _clean_doi(value: Any) -> str:
    text = _as_text(value).lower()
    if not text or text in {"nan", "none", "null"}:
        return ""
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi:\s*", "", text)
    match = re.search(r"10\.\d{4,9}/[^\s<>\"']+", text, flags=re.I)
    if not match:
        return ""
    return match.group(0).rstrip(".,;:)").strip()


def _clean_openalex(value: Any) -> str:
    text = _as_text(value)
    match = re.search(r"(?:openalex\.org/(?:works/)?)?(W\d+)\b", text, flags=re.I)
    return match.group(1).upper() if match else ""


def _identifier_keys(value: Any, column_name: str = "") -> list[str]:
    raw = _as_text(value)
    if not raw or raw.lower() in {"nan", "none", "null"}:
        return []
    keys: list[str] = []
    oa = _clean_openalex(raw)
    doi = _clean_doi(raw)
    if oa:
        keys.append(f"oa:{oa}")
    if doi:
        keys.append(f"doi:{doi}")
    if _norm_col(column_name) in DOI_ALIASES and doi:
        keys.append(f"doi:{doi}")
    norm = raw.casefold().rstrip("/")
    if norm:
        keys.append(f"raw:{norm}")
    return list(dict.fromkeys(keys))


def _clean_text(value: Any, limit: int = 12000) -> str:
    if value is None or not isinstance(value, (str, int, float)):
        return ""
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]{1,300}>", " ", text)
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if text.lower() in {"nan", "none", "null"} or len(text) < 40:
        return ""
    return text[:limit]


def _normalize_title(value: Any) -> str:
    text = _as_text(value).casefold()
    text = re.sub(r"\b(?:https?://doi\.org/|doi:)\s*", "", text)
    text = re.sub(r"[\W_]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _title_match_score(left: Any, right: Any) -> float:
    a = _normalize_title(left)
    b = _normalize_title(right)
    if not a or not b:
        return 0.0
    seq = difflib.SequenceMatcher(None, a, b).ratio()
    ta, tb = set(a.split()), set(b.split())
    jaccard = len(ta & tb) / max(1, len(ta | tb))
    return 0.65 * seq + 0.35 * jaccard


def _title_cache_key(title: Any) -> str:
    normalized = _normalize_title(title)
    return "title:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized else ""


def _negative_title_cache_is_fresh(item: Any, *, ttl_days: int = 90) -> bool:
    """Negative title lookups expire so newly deposited abstracts can be discovered."""
    if not isinstance(item, dict) or item.get("status") != "no_abstract_match":
        return False
    checked_at = _as_text(item.get("checked_at", ""))
    if not checked_at:
        return False
    try:
        checked = datetime.fromisoformat(checked_at.replace("Z", "+00:00"))
        if checked.tzinfo is None:
            checked = checked.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - checked.astimezone(timezone.utc)
        return age.total_seconds() <= max(1, int(ttl_days)) * 86400
    except (TypeError, ValueError, OverflowError):
        return False


def reconstruct_openalex_abstract(inverted_index: Any) -> str:
    """Reconstruct plaintext from OpenAlex's word -> positions abstract index."""
    if not isinstance(inverted_index, dict) or not inverted_index:
        return ""
    words: list[tuple[int, str]] = []
    try:
        for word, positions in inverted_index.items():
            if not isinstance(positions, list):
                continue
            for position in positions:
                words.append((int(position), str(word)))
        words.sort(key=lambda item: item[0])
        return _clean_text(" ".join(word for _, word in words), limit=12000)
    except (TypeError, ValueError):
        return ""


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t", encoding="utf-8-sig", low_memory=False)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict):
            for key in ("data", "records", "results", "papers", "documents", "works"):
                if isinstance(payload.get(key), list):
                    return pd.DataFrame(payload[key])
            try:
                return pd.DataFrame(payload)
            except Exception:
                return pd.DataFrame()
    return pd.DataFrame()


def _resolve_pdf_path(root: Path, value: Any) -> Path | None:
    raw = _as_text(value).strip('"')
    if not raw or raw.lower() in {"nan", "none", "null"}:
        return None
    path = Path(raw)
    candidates = [path] if path.is_absolute() else [root / path, root / "data" / path]
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            if resolved.is_file() and resolved.suffix.lower() == ".pdf" and (resolved == root or root in resolved.parents):
                return resolved
        except OSError:
            continue
    return None


def _pdf_extractor_available() -> bool:
    """PDF extraction is optional; avoid attempting imports for every file if absent."""
    return importlib.util.find_spec("fitz") is not None or importlib.util.find_spec("pypdf") is not None


def _extract_pdf_excerpt(path: Path) -> tuple[str, str]:
    text = ""
    try:
        import fitz  # PyMuPDF, optional
        with fitz.open(path) as doc:
            text = "\n".join(page.get_text("text") for page in doc[:3])
    except Exception:
        try:
            from pypdf import PdfReader  # optional fallback
            reader = PdfReader(str(path))
            text = "\n".join((page.extract_text() or "") for page in reader.pages[:3])
        except Exception:
            return "", ""
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 100:
        return "", ""
    match = re.search(r"\babstract\b\s*[:.\-]?\s*", text, flags=re.I)
    if match:
        start = match.end()
        tail = text[start:]
        end = re.search(r"\b(?:1\s+)?(?:introduction|keywords|index terms)\b", tail, flags=re.I)
        excerpt = tail[: end.start()] if end else tail[:3500]
        cleaned = _clean_text(excerpt, 5000)
        if len(cleaned) >= 100:
            return cleaned, "pdf_abstract"
    return _clean_text(text, 2200), "pdf_frontmatter"




def _extract_abstract_from_processed_text(value: str) -> str:
    """Extract an abstract from text already produced by the main ingest pipeline."""
    text = html.unescape(str(value or ""))
    text = re.sub(r"\s+", " ", text).strip()
    match = re.search(r"\babstract\b\s*[:.\-]?\s*", text, flags=re.I)
    if not match:
        return ""
    tail = text[match.end():]
    end = re.search(
        r"\b(?:1\s+)?(?:introduction|keywords|index terms|1\.)\b",
        tail,
        flags=re.I,
    )
    candidate = tail[:end.start()] if end else tail[:5000]
    candidate = _clean_text(candidate, 5000)
    return candidate if len(candidate) >= 100 else ""


def _read_text_prefix(path: Path, limit: int = 24000) -> str:
    """Read a bounded prefix from a processed UTF-8 text file."""
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
            return _clean_text(stream.read(limit), limit=limit)
    except (OSError, UnicodeError):
        return ""


def _processed_text_path(
    root: Path, raw_path: Any, doc_id: str, configured_processed: str | None = None
) -> Path | None:
    """Resolve an ingest text_path, with the canonical processed/<document_id> fallback."""
    candidates: list[Path] = []
    raw = _as_text(raw_path)
    if raw:
        value = Path(raw)
        candidates.append(value if value.is_absolute() else root / value)
        if not value.is_absolute():
            candidates.append(root / "data" / value)
    processed_value = configured_processed or os.environ.get("SIL_PROCESSED", "data/processed")
    processed_root = Path(processed_value)
    if not processed_root.is_absolute():
        processed_root = root / processed_root
    candidates.append(processed_root / doc_id / "document.txt")
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            if resolved.is_file() and resolved.suffix.lower() == ".txt":
                return resolved
        except OSError:
            continue
    return None


def _load_main_pipeline_evidence(
    root: Path,
    ids: list[str],
    records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Read canonical abstracts and processed text from the main pipeline SQLite DB.

    The main ingest pipeline stores documents.abstract and the path to
    data/processed/<document_id>/document.txt in data/silab.sqlite3.
    This is the primary local source; external metadata APIs are only for gaps.
    """
    env_values: dict[str, str] = {}
    try:
        from dotenv import dotenv_values
        env_values = {
            str(key): str(value)
            for key, value in dotenv_values(root / ".env").items()
            if key and value is not None
        }
    except ImportError:
        pass

    configured_db = os.environ.get("SIL_DB") or env_values.get("SIL_DB") or "data/silab.sqlite3"
    configured_processed = (
        os.environ.get("SIL_PROCESSED") or env_values.get("SIL_PROCESSED") or "data/processed"
    )
    db_path = Path(configured_db)
    if not db_path.is_absolute():
        db_path = root / db_path
    stats: dict[str, Any] = {
        "sqlite_path": str(db_path),
        "sqlite_available": db_path.is_file(),
        "sqlite_rows_matched": 0,
        "sqlite_abstract_records": 0,
        "processed_text_files_found": 0,
        "processed_text_files_read": 0,
        "processed_abstracts_extracted": 0,
        "processed_text_read_errors": 0,
        "local_corpus_priority": [
            "data/silab.sqlite3:documents.abstract",
            "documents.text_path -> data/processed/<document_id>/document.txt",
            "data/inbox JSON sidecars and other local metadata",
            "local abstract cache",
            "OpenAlex/Crossref only for records with no local text",
        ],
    }
    db_rows: dict[str, dict[str, Any]] = {}
    if db_path.is_file():
        try:
            # URI mode=ro guarantees that semantic labeling cannot create or modify
            # the main ingest database.
            connection = sqlite3.connect(
                f"{db_path.resolve().as_uri()}?mode=ro", uri=True, timeout=30
            )
            connection.row_factory = sqlite3.Row
            try:
                columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(documents)")}
                if "id" not in columns:
                    raise RuntimeError("SQLite table documents has no id column")
                select_columns = ["id"]
                for optional in ("abstract", "text_path"):
                    if optional in columns:
                        select_columns.append(optional)
                for offset in range(0, len(ids), 500):
                    batch = ids[offset:offset + 500]
                    if not batch:
                        continue
                    marks = ",".join("?" for _ in batch)
                    query = f"SELECT {', '.join(select_columns)} FROM documents WHERE id IN ({marks})"
                    for row in connection.execute(query, batch).fetchall():
                        item = dict(row)
                        db_rows[str(item.get("id", ""))] = item
            finally:
                connection.close()
        except Exception as exc:
            stats["sqlite_error"] = str(exc)
            print(f"[WARN] Основная SQLite-база не прочитана ({db_path}): {exc}", file=sys.stderr)

    stats["sqlite_rows_matched"] = len(db_rows)
    for doc_id in ids:
        db_row = db_rows.get(doc_id, {})
        db_abstract = _clean_text(db_row.get("abstract", ""), 12000)
        path = _processed_text_path(
            root, db_row.get("text_path", ""), doc_id, configured_processed=configured_processed
        )
        if path:
            stats["processed_text_files_found"] += 1
            records[doc_id]["full_text_path"] = str(path)
            records[doc_id]["has_full_text"] = True

        existing_abstract = _clean_text(records[doc_id].get("abstract", ""), 12000)
        abstract = db_abstract or existing_abstract
        text_prefix = ""
        extracted_abstract = ""
        if path and not abstract:
            text_prefix = _read_text_prefix(path)
            if text_prefix:
                stats["processed_text_files_read"] += 1
                extracted_abstract = _extract_abstract_from_processed_text(text_prefix)
                if extracted_abstract:
                    stats["processed_abstracts_extracted"] += 1

        if abstract:
            records[doc_id].update({
                "abstract": abstract,
                "text": abstract,
                "source": "sqlite_abstract" if db_abstract else records[doc_id].get("source", "local_abstract"),
                "source_detail": (
                    f"{db_path.relative_to(root) if db_path.is_relative_to(root) else db_path}"
                    + (f"; {path.relative_to(root) if path.is_relative_to(root) else path}" if path else "")
                ),
                "has_abstract": True,
                "has_full_text": bool(path),
            })
            if db_abstract:
                stats["sqlite_abstract_records"] += 1
        elif extracted_abstract:
            records[doc_id].update({
                "abstract": extracted_abstract,
                "text": extracted_abstract,
                "source": "sqlite_processed_abstract",
                "source_detail": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
                "has_abstract": True,
                "has_full_text": True,
            })
        elif path:
            if not text_prefix:
                text_prefix = _read_text_prefix(path)
                if text_prefix:
                    stats["processed_text_files_read"] += 1
            if text_prefix:
                records[doc_id].update({
                    "abstract": "",
                    "text": text_prefix,
                    "source": "sqlite_processed_text",
                    "source_detail": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
                    "has_abstract": False,
                    "has_full_text": True,
                })
            else:
                stats["processed_text_read_errors"] += 1

    return stats


def _http_json(url: str, timeout: int, user_agent: str, retries: int = 2) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept": "application/json"})
    last_error: Exception | None = None
    for attempt in range(max(1, int(retries))):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read().decode("utf-8")
            value = json.loads(payload)
            return value if isinstance(value, dict) else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            last_error = RuntimeError(f"HTTP {exc.code}: {detail}")
            if exc.code not in {429, 500, 502, 503, 504}:
                break
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Запрос метаданных не выполнен: {last_error}")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


@dataclass
class EvidenceBundle:
    texts: list[str]
    text_sources: list[str]
    by_id: dict[str, dict[str, Any]]
    used_files: list[str]
    stats: dict[str, Any]


def collect_paper_evidence(
    root: Path,
    base: pd.DataFrame,
    ids: list[str],
    *,
    source: str = "auto",
    openalex_api_key: str = "",
    mailto: str = "",
    openalex_timeout: int = 25,
    max_openalex_batches: int = 50,
    max_openalex_title_lookups: int = 100,
    max_crossref_lookups: int = 1870,
    max_crossref_title_lookups: int = 0,
    crossref_fallback: bool = True,
    cache_path: Path | None = None,
) -> EvidenceBundle:
    """Find local evidence, then enrich gaps through OpenAlex and Crossref.

    `source` can be auto, local, openalex, or off. Remote requests contain only
    persistent IDs or publication titles for strict metadata search; local abstracts
    and full text are never uploaded to metadata providers. Crossref remains
    independent from OpenAlex availability and can search by DOI or title.
    """
    source = source.lower().strip()
    if source not in {"auto", "local", "openalex", "off"}:
        raise ValueError(f"Неизвестный источник текстов: {source}")
    root = root.resolve()
    id_col = next((c for c in base.columns if _norm_col(c) == "documentid"), None)
    if id_col is None:
        raise ValueError("Базовая таблица должна содержать document_id.")
    base_doc_ids = [str(value).strip() for value in base[id_col].tolist()]
    normalized_ids = [str(value).strip() for value in ids]
    if len(base_doc_ids) != len(normalized_ids) or base_doc_ids != normalized_ids:
        raise ValueError(
            "Порядок/состав document_id в evidence-таблице не совпадает с переданным списком IDs. "
            "Отказ от потенциально неверного сопоставления аннотаций."
        )
    if any(not value or value.lower() in {"nan", "none", "null"} for value in normalized_ids):
        raise ValueError("Список document_id содержит пустые идентификаторы.")
    if len(normalized_ids) != len(set(normalized_ids)):
        raise ValueError("Список document_id содержит дубликаты.")

    records: dict[str, dict[str, Any]] = {
        doc_id: {"abstract": "", "text": "", "source": "title_only", "source_detail": "", "has_abstract": False, "has_full_text": False}
        for doc_id in ids
    }
    base_key_map: dict[str, str] = {}
    id_columns = [c for c in base.columns if _norm_col(c) in ID_ALIASES]
    for row in base.to_dict(orient="records"):
        doc_id = str(row.get(id_col, "")).strip()
        if doc_id not in records:
            continue
        for col in id_columns:
            for key in _identifier_keys(row.get(col), col):
                # Conflicting identifiers are not silently reassigned.
                previous = base_key_map.get(key)
                if previous is None or previous == doc_id:
                    base_key_map[key] = doc_id
                else:
                    base_key_map.pop(key, None)

    stats: dict[str, Any] = {
        "mode": source,
        "remote_query_policy": (
            "Local metadata first; OpenAlex by DOI/OpenAlex ID and strict title search; "
            "independent Crossref fallback by DOI and strict bibliographic-title search; no user full text sent"
        ),
        "negative_title_cache_ttl_days": 90,
        "remote_full_text_sent": False,
        "local_files_scanned": 0,
        "local_text_records": 0,
        "local_pdf_records": 0,
        "pdf_extraction_available": _pdf_extractor_available(),
        "openalex_batches": 0,
        "openalex_records": 0,
        "openalex_title_lookups": 0,
        "openalex_title_selection": "round_robin_by_domain",
        "openalex_title_matches": 0,
        "openalex_title_rejected": 0,
        "openalex_title_cache_negative_hits": 0,
        "crossref_requests": 0,
        "crossref_abstracts": 0,
        "crossref_title_lookups": 0,
        "crossref_title_matches": 0,
        "crossref_title_rejected": 0,
        "crossref_title_no_abstract": 0,
        "remote_abstracts": 0,
        "remote_errors": [],
        "cache_hits": 0,
    }
    used_files: list[str] = []

    if source in {"auto", "local"}:
        search_roots = [root / "data", root / "artifacts"]
        paths: list[Path] = []
        excluded = {"runs", "embeddings", "community_detection_stage1", "community_detection_stage2", "auto_topic_discovery", "cache"}
        for folder in search_roots:
            if folder.exists():
                for suffix in ("*.csv", "*.tsv", "*.tab", "*.parquet", "*.xlsx", "*.xls", "*.jsonl", "*.ndjson", "*.json"):
                    paths.extend(folder.rglob(suffix))
        paths = [p for p in paths if not any(part.lower() in excluded for part in p.parts)]
        priority = ("paper", "document", "metadata", "abstract", "corpus", "article", "text", "catalog")
        paths.sort(key=lambda p: (0 if any(w in p.name.lower() for w in priority) else 1, len(str(p))))
        for path in paths:
            try:
                if path.stat().st_size > 600 * 1024 * 1024:
                    continue
                if path.suffix.lower() in {".csv", ".tsv", ".tab"}:
                    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig", sep="\t" if path.suffix.lower() in {".tsv", ".tab"} else ",").columns
                    if not any(_norm_col(c) in ID_ALIASES for c in header):
                        continue
                frame = _read_table(path)
                if frame.empty:
                    continue
                key_cols = [c for c in frame.columns if _norm_col(c) in ID_ALIASES]
                if not key_cols:
                    continue
                abstract_cols = [c for c in frame.columns if _norm_col(c) in ABSTRACT_ALIASES]
                fulltext_cols = [c for c in frame.columns if _norm_col(c) in FULLTEXT_ALIASES]
                pdf_cols = [c for c in frame.columns if _norm_col(c) in PDF_PATH_ALIASES]
                if not (abstract_cols or fulltext_cols or pdf_cols):
                    continue
                stats["local_files_scanned"] += 1
                added = 0
                for row in frame.to_dict(orient="records"):
                    doc_id = None
                    for col in key_cols:
                        for key in _identifier_keys(row.get(col), col):
                            doc_id = base_key_map.get(key)
                            if doc_id:
                                break
                        if doc_id:
                            break
                    if not doc_id:
                        continue
                    if abstract_cols or fulltext_cols:
                        candidates: list[tuple[int, str, str]] = []
                        for col in abstract_cols:
                            raw_abstract = row.get(col)
                            if _norm_col(col) == "abstractinvertedindex":
                                inverted = raw_abstract
                                if isinstance(inverted, str):
                                    try:
                                        inverted = json.loads(inverted)
                                    except json.JSONDecodeError:
                                        inverted = None
                                value = reconstruct_openalex_abstract(inverted)
                            else:
                                value = _clean_text(raw_abstract, 12000)
                            if value:
                                candidates.append((4, value, "abstract"))
                        for col in fulltext_cols:
                            value = _clean_text(row.get(col), 12000)
                            if value:
                                candidates.append((2, value, "full_text"))
                        if candidates:
                            rank, value, kind = max(candidates, key=lambda x: (x[0], min(len(x[1]), 5000)))
                            current = records[doc_id]
                            current_rank = 4 if current["has_abstract"] else (2 if current["has_full_text"] else 0)
                            if rank > current_rank or (rank == current_rank and len(value) > len(current["text"])):
                                current["abstract"] = value if kind == "abstract" else ""
                                current["text"] = value
                                current["source"] = kind
                                current["source_detail"] = str(path.relative_to(root))
                                current["has_abstract"] = kind == "abstract"
                                current["has_full_text"] = kind == "full_text"
                                added += 1
                    if pdf_cols and stats["pdf_extraction_available"] and not records[doc_id]["text"]:
                        for col in pdf_cols:
                            pdf_path = _resolve_pdf_path(root, row.get(col))
                            if pdf_path:
                                body, kind = _extract_pdf_excerpt(pdf_path)
                                if body:
                                    records[doc_id].update({
                                        "abstract": body if kind == "pdf_abstract" else "",
                                        "text": body, "source": kind,
                                        "source_detail": str(pdf_path.relative_to(root)),
                                        "has_abstract": kind == "pdf_abstract",
                                        "has_full_text": False,
                                    })
                                    stats["local_pdf_records"] += 1
                                    added += 1
                                    break
                if added:
                    used_files.append(str(path.relative_to(root)))
                    stats["local_text_records"] += added
            except Exception as exc:
                print(f"[WARN] Локальный источник {path} пропущен: {exc}", file=sys.stderr)

        # Also match PDFs whose filenames are exact document IDs or DOIs.
        if source in {"auto", "local"}:
            pdf_root = root / "data"
            if pdf_root.exists() and stats["pdf_extraction_available"] and any(not rec["text"] for rec in records.values()):
                file_key_map: dict[str, str] = {}
                for key, doc_id in base_key_map.items():
                    if key.startswith("raw:"):
                        file_key_map[key[4:].replace("/", "_").replace(":", "_")] = doc_id
                    elif key.startswith("oa:"):
                        file_key_map[key[3:].lower()] = doc_id
                    elif key.startswith("doi:"):
                        file_key_map[key[4:].replace("/", "_").replace(".", "_")] = doc_id
                for pdf_path in pdf_root.rglob("*.pdf"):
                    stem = pdf_path.stem.casefold()
                    doc_id = file_key_map.get(stem)
                    if not doc_id:
                        oa = _clean_openalex(stem)
                        doi = _clean_doi(stem.replace("_", "/"))
                        doc_id = base_key_map.get(f"oa:{oa}") if oa else None
                        if not doc_id and doi:
                            doc_id = base_key_map.get(f"doi:{doi}")
                    if not doc_id or records[doc_id]["text"]:
                        continue
                    body, kind = _extract_pdf_excerpt(pdf_path)
                    if body:
                        records[doc_id].update({
                            "abstract": body if kind == "pdf_abstract" else "",
                            "text": body, "source": kind,
                            "source_detail": str(pdf_path.relative_to(root)),
                            "has_abstract": kind == "pdf_abstract",
                            "has_full_text": False,
                        })
                        stats["local_pdf_records"] += 1
                        used_files.append(str(pdf_path.relative_to(root)))

    # Read the main project's canonical SQLite/processed corpus before consulting APIs.
    # Exact document_id (the PDF SHA-256 prefix) is the join key; no row-order matching.
    if source in {"auto", "local"}:
        stats.update(_load_main_pipeline_evidence(root, ids, records))

    # Use a persistent cache to avoid repeated API requests for already seen IDs.
    cache_path = cache_path or (root / "artifacts" / "cache" / "topic_abstracts" / "paper_abstract_cache.json")
    cache: dict[str, Any] = {"version": 1, "items": {}}
    if source in {"auto", "local", "openalex"} and cache_path.is_file():
        try:
            loaded = json.loads(cache_path.read_text(encoding="utf-8-sig"))
            if isinstance(loaded, dict) and isinstance(loaded.get("items"), dict):
                cache = loaded
        except Exception as exc:
            print(f"[WARN] Кэш аннотаций повреждён и будет пересоздан: {exc}", file=sys.stderr)
    cache_items: dict[str, Any] = cache.setdefault("items", {})

    # Apply positive cache entries to records with no local content.
    for row in base.to_dict(orient="records"):
        doc_id = str(row.get(id_col, "")).strip()
        if doc_id not in records or records[doc_id]["text"]:
            continue
        for col in id_columns:
            for key in _identifier_keys(row.get(col), col):
                item = cache_items.get(key)
                if isinstance(item, dict) and item.get("abstract"):
                    abstract = _clean_text(item["abstract"], 12000)
                    records[doc_id].update({"abstract": abstract, "text": abstract, "source": item.get("source", "OpenAlex"), "source_detail": item.get("source_detail", "cached API"), "has_abstract": True, "has_full_text": False})
                    stats["cache_hits"] += 1
                    break
            if records[doc_id]["text"]:
                break
        if not records[doc_id]["text"]:
            title_key = _title_cache_key(row.get("title", ""))
            item = cache_items.get(title_key) if title_key else None
            if isinstance(item, dict) and item.get("abstract"):
                abstract = _clean_text(item["abstract"], 12000)
                records[doc_id].update({"abstract": abstract, "text": abstract, "source": item.get("source", "abstract_openalex_title"), "source_detail": item.get("source_detail", "cached title lookup"), "has_abstract": True, "has_full_text": False})
                stats["cache_hits"] += 1

    openalex_failed = False
    if source in {"auto", "openalex"}:
        oa_to_docs: dict[str, list[str]] = defaultdict(list)
        doi_to_docs: dict[str, list[str]] = defaultdict(list)
        for row in base.to_dict(orient="records"):
            doc_id = str(row.get(id_col, "")).strip()
            if doc_id not in records or records[doc_id]["text"]:
                continue
            for col in id_columns:
                for key in _identifier_keys(row.get(col), col):
                    if key.startswith("oa:"):
                        oa_to_docs[key[3:]].append(doc_id)
                    elif key.startswith("doi:"):
                        doi_to_docs[key[4:]].append(doc_id)
        user_agent = f"ScientificIntelligenceLab/1.0 (mailto:{mailto})" if mailto else "ScientificIntelligenceLab/1.0"
        def fetch_filter(kind: str, identifiers: list[str]) -> None:
            nonlocal cache_items, openalex_failed
            for offset in range(0, len(identifiers), 100):
                if openalex_failed:
                    break
                if stats["openalex_batches"] >= max_openalex_batches:
                    stats["remote_errors"].append(f"Достигнут лимит OpenAlex batches={max_openalex_batches}.")
                    break
                batch = identifiers[offset:offset + 100]
                attr = "ids.openalex" if kind == "oa" else "doi"
                encoded_values = "|".join(batch)
                params = {
                    "filter": f"{attr}:{encoded_values}",
                    "per_page": "100",
                    "select": "id,ids,doi,title,abstract_inverted_index",
                }
                if openalex_api_key:
                    params["api_key"] = openalex_api_key
                if mailto:
                    params["mailto"] = mailto
                url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params, safe=":|,/")
                try:
                    stats["openalex_batches"] += 1
                    print(f"      OpenAlex {kind}: пакет {offset // 100 + 1}, идентификаторов {len(batch)}", flush=True)
                    response = _http_json(url, openalex_timeout, user_agent)
                    results = response.get("results", [])
                    if not isinstance(results, list):
                        continue
                    stats["openalex_records"] += len(results)
                    for item in results:
                        if not isinstance(item, dict):
                            continue
                        abstract = reconstruct_openalex_abstract(item.get("abstract_inverted_index"))
                        if not abstract:
                            continue
                        id_value = str(item.get("id", ""))
                        oa_id = _clean_openalex(id_value) or _clean_openalex((item.get("ids") or {}).get("openalex", ""))
                        doi_value = _clean_doi(item.get("doi", "")) or _clean_doi((item.get("ids") or {}).get("doi", ""))
                        matched_docs: set[str] = set()
                        if oa_id:
                            matched_docs.update(oa_to_docs.get(oa_id, []))
                            cache_items[f"oa:{oa_id}"] = {"abstract": abstract, "source": "abstract_openalex", "source_detail": oa_id}
                        if doi_value:
                            matched_docs.update(doi_to_docs.get(doi_value, []))
                            cache_items[f"doi:{doi_value}"] = {"abstract": abstract, "source": "abstract_openalex", "source_detail": doi_value}
                        for doc_id in matched_docs:
                            if not records[doc_id]["text"]:
                                records[doc_id].update({"abstract": abstract, "text": abstract, "source": "abstract_openalex", "source_detail": oa_id or doi_value, "has_abstract": True, "has_full_text": False})
                                stats["remote_abstracts"] += 1
                    time.sleep(0.15)
                except Exception as exc:
                    message = f"OpenAlex {kind} batch {offset // 100 + 1}: {exc}"
                    stats["remote_errors"].append(message)
                    print(f"[WARN] {message}; дальнейшие удалённые запросы OpenAlex остановлены.", file=sys.stderr)
                    openalex_failed = True
                    break

        if oa_to_docs:
            fetch_filter("oa", sorted(oa_to_docs))
        # DOI pass fills gaps for records not covered by OpenAlex ID requests.
        missing_dois = [doi for doi, doc_ids in doi_to_docs.items() if any(not records[d]["text"] for d in doc_ids)]
        if missing_dois:
            fetch_filter("doi", sorted(missing_dois))

        # A bounded title search covers corpora that do not store DOI/OpenAlex IDs.
        if not openalex_failed and max_openalex_title_lookups > 0:
            title_candidates: list[tuple[str, str, list[str]]] = []
            title_targets: dict[str, list[str]] = defaultdict(list)
            title_values: dict[str, str] = {}
            title_domains: dict[str, str] = {}
            for row in base.to_dict(orient="records"):
                doc_id = str(row.get(id_col, "")).strip()
                title = _as_text(row.get("title", ""))
                title_key = _title_cache_key(title)
                if doc_id not in records or records[doc_id]["text"] or not title or not title_key:
                    continue
                negative_cache = cache_items.get(title_key)
                if _negative_title_cache_is_fresh(negative_cache, ttl_days=90):
                    stats["openalex_title_cache_negative_hits"] += 1
                    continue
                has_stable_identifier = any(
                    key.startswith(("oa:", "doi:"))
                    for col in id_columns for key in _identifier_keys(row.get(col), col)
                )
                if has_stable_identifier:
                    continue
                title_targets[title_key].append(doc_id)
                title_values.setdefault(title_key, title)
                title_domains.setdefault(title_key, _as_text(row.get("domain", "")) or "__unknown__")
            candidates_by_domain: dict[str, list[tuple[str, str, list[str]]]] = defaultdict(list)
            for title_key, target_ids in title_targets.items():
                domain = title_domains.get(title_key, "__unknown__")
                candidates_by_domain[domain].append((target_ids[0], title_values[title_key], target_ids))
            # Don't spend a limited title-search budget only on the first domain
            # in corpus order. Round-robin by domain gives each research area a chance.
            domain_names = sorted(candidates_by_domain, key=lambda value: (value == "__unknown__", value.casefold()))
            domain_offsets = {domain: 0 for domain in domain_names}
            while True:
                added_in_round = False
                for domain in domain_names:
                    offset = domain_offsets[domain]
                    bucket = candidates_by_domain[domain]
                    if offset < len(bucket):
                        title_candidates.append(bucket[offset])
                        domain_offsets[domain] = offset + 1
                        added_in_round = True
                if not added_in_round:
                    break
            for lookup_number, (doc_id, title, target_ids) in enumerate(title_candidates[:max_openalex_title_lookups], start=1):
                if openalex_failed:
                    break
                params = {
                    "search": title[:500], "per_page": "5",
                    "select": "id,ids,doi,title,abstract_inverted_index",
                }
                if openalex_api_key:
                    params["api_key"] = openalex_api_key
                if mailto:
                    params["mailto"] = mailto
                url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
                try:
                    stats["openalex_title_lookups"] += 1
                    response = _http_json(url, openalex_timeout, user_agent)
                    results = response.get("results", [])
                    if not isinstance(results, list):
                        continue
                    ranked = sorted(
                        (( _title_match_score(title, item.get("title", "")), item) for item in results if isinstance(item, dict)),
                        key=lambda pair: pair[0], reverse=True,
                    )
                    if not ranked:
                        cache_items[_title_cache_key(title)] = {"status": "no_abstract_match", "checked_at": datetime.now(timezone.utc).isoformat()}
                        continue
                    score, item = ranked[0]
                    # Strict title agreement avoids attaching an abstract from a merely related paper.
                    if score < 0.90:
                        stats["openalex_title_rejected"] += 1
                        cache_items[_title_cache_key(title)] = {
                            "status": "no_abstract_match", "checked_at": datetime.now(timezone.utc).isoformat(),
                            "best_title_match_score": round(float(score), 4),
                        }
                        continue
                    abstract = reconstruct_openalex_abstract(item.get("abstract_inverted_index"))
                    if not abstract:
                        cache_items[_title_cache_key(title)] = {
                            "status": "no_abstract_match", "checked_at": datetime.now(timezone.utc).isoformat(),
                            "matched_title": _as_text(item.get("title", "")),
                        }
                        continue
                    matched_title = _as_text(item.get("title", ""))
                    oa_id = _clean_openalex(item.get("id", "")) or _clean_openalex((item.get("ids") or {}).get("openalex", ""))
                    doi_value = _clean_doi(item.get("doi", "")) or _clean_doi((item.get("ids") or {}).get("doi", ""))
                    for target_id in target_ids:
                        if not records[target_id]["text"]:
                            records[target_id].update({
                                "abstract": abstract, "text": abstract, "source": "abstract_openalex_title",
                                "source_detail": f"{oa_id or doi_value}; title_match={score:.3f}",
                                "has_abstract": True, "has_full_text": False,
                            })
                            stats["remote_abstracts"] += 1
                    stats["openalex_title_matches"] += 1
                    cache_key = _title_cache_key(title)
                    if cache_key:
                        cache_items[cache_key] = {
                            "abstract": abstract, "source": "abstract_openalex_title",
                            "source_detail": f"{oa_id or doi_value}; title_match={score:.3f}",
                            "matched_title": matched_title,
                        }
                    if oa_id:
                        cache_items[f"oa:{oa_id}"] = {"abstract": abstract, "source": "abstract_openalex_title", "source_detail": oa_id}
                    if doi_value:
                        cache_items[f"doi:{doi_value}"] = {"abstract": abstract, "source": "abstract_openalex_title", "source_detail": doi_value}
                    if lookup_number % 10 == 0:
                        print(f"      OpenAlex title search: {lookup_number}/{min(len(title_candidates), max_openalex_title_lookups)}", flush=True)
                    time.sleep(0.15)
                except Exception as exc:
                    message = f"OpenAlex title lookup {lookup_number}: {exc}"
                    stats["remote_errors"].append(message)
                    print(f"[WARN] {message}; дальнейшие удалённые запросы OpenAlex остановлены.", file=sys.stderr)
                    openalex_failed = True
                    break

    # Crossref is an independent provider. An OpenAlex 429/network failure must
    # not disable it; otherwise a shared OpenAlex quota outage silently prevents
    # the configured fallback from being used.
    if source in {"auto", "openalex"} and crossref_fallback:
        remaining_dois: list[tuple[str, str]] = []
        for row in base.to_dict(orient="records"):
            doc_id = str(row.get(id_col, "")).strip()
            if doc_id not in records or records[doc_id]["text"]:
                continue
            for col in id_columns:
                doi = _clean_doi(row.get(col))
                if doi:
                    remaining_dois.append((doc_id, doi))
                    break
        seen_dois: set[str] = set()
        doi_lookup: list[tuple[str, str]] = []
        for doc_id, doi in remaining_dois:
            if doi not in seen_dois:
                doi_lookup.append((doc_id, doi))
                seen_dois.add(doi)
        user_agent = f"ScientificIntelligenceLab/1.0 (mailto:{mailto})" if mailto else "ScientificIntelligenceLab/1.0"
        crossref_failures = 0
        for lookup_number, (doc_id, doi) in enumerate(doi_lookup[:max_crossref_lookups], start=1):
            cached = cache_items.get(f"doi:{doi}")
            if isinstance(cached, dict) and cached.get("abstract"):
                continue
            url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
            if mailto:
                url += "?mailto=" + urllib.parse.quote(mailto)
            try:
                stats["crossref_requests"] += 1
                response = _http_json(url, min(openalex_timeout, 8), user_agent, retries=1)
                crossref_failures = 0
                message = response.get("message", {}) if isinstance(response, dict) else {}
                abstract = _clean_text(message.get("abstract", "") if isinstance(message, dict) else "", 12000)
                if abstract:
                    for target_id, target_doi in remaining_dois:
                        if target_doi == doi and not records[target_id]["text"]:
                            records[target_id].update({"abstract": abstract, "text": abstract, "source": "abstract_crossref", "source_detail": doi, "has_abstract": True, "has_full_text": False})
                            stats["remote_abstracts"] += 1
                            stats["crossref_abstracts"] += 1
                    cache_items[f"doi:{doi}"] = {"abstract": abstract, "source": "abstract_crossref", "source_detail": doi}
                if lookup_number % 25 == 0:
                    print(f"      Crossref DOI: обработано {lookup_number}/{min(len(doi_lookup), max_crossref_lookups)}", flush=True)
                time.sleep(0.12 if mailto else 0.22)
            except Exception as exc:
                crossref_failures += 1
                stats["remote_errors"].append(f"Crossref DOI {doi}: {exc}")
                print(f"[WARN] Crossref DOI {doi}: {exc}", file=sys.stderr)
                if crossref_failures >= 2:
                    print("      Crossref DOI: остановлено после двух ошибок подряд.", flush=True)
                    break

        # Crossref title lookup fills gaps when the corpus has no DOI/OpenAlex IDs
        # or when a DOI record does not expose an abstract. The result is attached
        # only after a strict title-similarity check; no fuzzy/related paper is used.
        if max_crossref_title_lookups > 0:
            title_targets: dict[str, list[str]] = defaultdict(list)
            title_values: dict[str, str] = {}
            title_domains: dict[str, str] = {}
            for row in base.to_dict(orient="records"):
                doc_id = str(row.get(id_col, "")).strip()
                title = _as_text(row.get("title", ""))
                title_key = _title_cache_key(title)
                if doc_id not in records or records[doc_id]["text"] or not title or not title_key:
                    continue
                title_targets[title_key].append(doc_id)
                title_values.setdefault(title_key, title)
                title_domains.setdefault(title_key, _as_text(row.get("domain", "")) or "__unknown__")

            by_domain: dict[str, list[tuple[str, str, list[str]]]] = defaultdict(list)
            for title_key, target_ids in title_targets.items():
                domain = title_domains.get(title_key, "__unknown__")
                by_domain[domain].append((title_key, title_values[title_key], target_ids))
            domain_names = sorted(by_domain, key=lambda value: (value == "__unknown__", value.casefold()))
            offsets = {domain: 0 for domain in domain_names}
            title_candidates: list[tuple[str, str, list[str]]] = []
            while True:
                added = False
                for domain in domain_names:
                    offset = offsets[domain]
                    bucket = by_domain[domain]
                    if offset < len(bucket):
                        title_candidates.append(bucket[offset])
                        offsets[domain] = offset + 1
                        added = True
                if not added:
                    break

            crossref_title_failures = 0
            attempted = 0
            for lookup_number, (title_key, title, target_ids) in enumerate(
                title_candidates[:max_crossref_title_lookups], start=1
            ):
                # Another cache entry may have filled this exact title earlier in this run.
                target_ids = [doc_id for doc_id in target_ids if not records[doc_id]["text"]]
                if not target_ids:
                    continue
                params = {
                    "query.bibliographic": title[:500],
                    "rows": "5",
                    "select": "DOI,title,abstract",
                }
                if mailto:
                    params["mailto"] = mailto
                url = "https://api.crossref.org/works?" + urllib.parse.urlencode(params)
                try:
                    stats["crossref_requests"] += 1
                    stats["crossref_title_lookups"] += 1
                    attempted += 1
                    response = _http_json(url, min(openalex_timeout, 8), user_agent, retries=1)
                    crossref_title_failures = 0
                    message = response.get("message", {}) if isinstance(response, dict) else {}
                    items = message.get("items", []) if isinstance(message, dict) else []
                    if not isinstance(items, list):
                        items = []
                    ranked: list[tuple[float, dict[str, Any], str]] = []
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        raw_titles = item.get("title", [])
                        if isinstance(raw_titles, str):
                            raw_titles = [raw_titles]
                        candidate_title = max(
                            (_as_text(value) for value in raw_titles if _as_text(value)),
                            key=lambda value: _title_match_score(title, value),
                            default="",
                        )
                        if candidate_title:
                            ranked.append((_title_match_score(title, candidate_title), item, candidate_title))
                    ranked.sort(key=lambda value: value[0], reverse=True)
                    if not ranked or ranked[0][0] < 0.92:
                        stats["crossref_title_rejected"] += 1
                    else:
                        score, item, matched_title = ranked[0]
                        abstract = _clean_text(item.get("abstract", ""), 12000)
                        doi_value = _clean_doi(item.get("DOI", ""))
                        if abstract:
                            for target_id in target_ids:
                                if not records[target_id]["text"]:
                                    records[target_id].update({
                                        "abstract": abstract,
                                        "text": abstract,
                                        "source": "abstract_crossref_title",
                                        "source_detail": f"{doi_value or 'Crossref'}; title_match={score:.3f}",
                                        "has_abstract": True,
                                        "has_full_text": False,
                                    })
                                    stats["remote_abstracts"] += 1
                                    stats["crossref_abstracts"] += 1
                            stats["crossref_title_matches"] += 1
                            cache_items[title_key] = {
                                "abstract": abstract,
                                "source": "abstract_crossref_title",
                                "source_detail": f"{doi_value or 'Crossref'}; title_match={score:.3f}",
                                "matched_title": matched_title,
                            }
                            if doi_value:
                                cache_items[f"doi:{doi_value}"] = {
                                    "abstract": abstract,
                                    "source": "abstract_crossref_title",
                                    "source_detail": doi_value,
                                }
                        else:
                            stats["crossref_title_no_abstract"] += 1
                    if attempted % 25 == 0:
                        print(
                            f"      Crossref title search: {attempted}/{min(len(title_candidates), max_crossref_title_lookups)}",
                            flush=True,
                        )
                    time.sleep(0.12 if mailto else 0.22)
                except Exception as exc:
                    crossref_title_failures += 1
                    stats["remote_errors"].append(f"Crossref title lookup {lookup_number}: {exc}")
                    print(f"[WARN] Crossref title lookup {lookup_number}: {exc}", file=sys.stderr)
                    if crossref_title_failures >= 2:
                        print("      Crossref title search: остановлено после двух ошибок подряд.", flush=True)
                        break

    if source in {"auto", "openalex"}:
        _atomic_json(cache_path, cache)

    evidence_by_id: dict[str, dict[str, Any]] = {}
    texts: list[str] = []
    sources: list[str] = []
    for row, doc_id in zip(base.to_dict(orient="records"), ids, strict=True):
        record = records[doc_id]
        title = str(row.get("title", "") or "").strip()
        body = str(record.get("text", "") or "").strip()
        if body:
            combined = (title + ". " + body).strip(" .") if title else body
        else:
            combined = title
        texts.append(combined[:14000])
        sources.append(str(record.get("source", "title_only")))
        evidence_by_id[doc_id] = {
            "abstract": str(record.get("abstract", "") or ""),
            "text_excerpt": body[:3500],
            "source": str(record.get("source", "title_only")),
            "source_detail": str(record.get("source_detail", "")),
            "has_abstract": bool(record.get("has_abstract", False)),
            "has_full_text": bool(record.get("has_full_text", False)),
            "full_text_path": str(record.get("full_text_path", "") or ""),
        }
    abstract_count = sum(1 for v in evidence_by_id.values() if v["has_abstract"])
    any_text_count = sum(1 for v in evidence_by_id.values() if v["source"] != "title_only")
    stats.update({
        "documents": len(ids),
        "documents_with_abstract": int(abstract_count),
        "documents_with_any_text": int(any_text_count),
        "documents_with_full_text": int(sum(1 for v in evidence_by_id.values() if v["has_full_text"])),
        "documents_title_only": int(len(ids) - any_text_count),
        "abstract_source_counts": dict(Counter(v["source"] for v in evidence_by_id.values())),
        "cache_file": str(cache_path.relative_to(root)) if cache_path.is_relative_to(root) else str(cache_path),
        "used_local_files": sorted(set(used_files)),
    })
    return EvidenceBundle(texts, sources, evidence_by_id, sorted(set(used_files)), stats)
