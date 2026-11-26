from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

SUPPORTED = {".pdf", ".md", ".markdown", ".txt"}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def clean_text(value: str) -> str:
    value = value.replace("\x00", " ").replace("\ufeff", " ")
    value = unicodedata.normalize("NFKC", value)
    value = re.sub(r"[ \t\f\v]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def infer_domain(path: Path, inbox: Path) -> str:
    try:
        relative = path.relative_to(inbox)
        return relative.parts[0] if len(relative.parts) > 1 else "unlabelled"
    except ValueError:
        return "unlabelled"


def extract_abstract(text: str) -> str:
    patterns = [
        r"(?is)\babstract\b\s*[:\-]?\s*(.{100,5000}?)(?=\n\s*(?:1\.?\s*)?(?:introduction|keywords|index terms|1\s+introduction)\b)",
        r"(?is)\babstract\b\s*[:\-]?\s*(.{100,3000}?)(?=\n\s*\n)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return clean_text(match.group(1))[:5000]
    return ""


def extract_title(text: str, metadata_title: str = "") -> str:
    if metadata_title and len(metadata_title.strip()) > 5:
        return clean_text(metadata_title)[:500]
    for line in text.splitlines()[:30]:
        candidate = clean_text(line).strip("#*_ ")
        if 15 <= len(candidate) <= 240 and not re.match(r"^(abstract|introduction|contents|arxiv):?\b", candidate, re.I):
            return candidate
    return "Untitled scientific document"


def extract_doi(text: str) -> str | None:
    match = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", text, flags=re.I)
    return match.group(0).rstrip(".,;)").lower() if match else None


def extract_year(text: str) -> int | None:
    match = re.search(r"\b(19[89]\d|20[0-3]\d)\b", text[:8000])
    return int(match.group(1)) if match else None


def embedding_text(title: str, abstract: str, body: str, max_chars: int = 12000) -> str:
    lead = abstract.strip() or body[:max_chars].strip()
    title = title.strip()
    # SPECTER-style input: title separated from abstract/body.
    combined = f"{title} [SEP] {lead}" if title else lead
    return clean_text(combined)[:max_chars]
