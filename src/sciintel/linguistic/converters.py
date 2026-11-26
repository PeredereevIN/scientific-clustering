from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from ..utils import clean_text

SUPPORTED_DOCUMENTS = {".pdf", ".doc", ".docx", ".rtf", ".txt", ".md", ".markdown"}


def _plain_to_markdown(text: str) -> str:
    text = clean_text(text)
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    return "\n\n".join(paragraphs) + ("\n" if paragraphs else "")


def convert_to_markdown(path: Path) -> tuple[str, dict]:
    """Convert PDF/DOC/DOCX/RTF/TXT/Markdown into UTF-8 Markdown.

    DOC is converted through LibreOffice; original source files are not changed.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_DOCUMENTS:
        raise ValueError(f"Unsupported document format: {suffix or '<no extension>'}")
    metadata = {"source_format": suffix.lstrip("."), "page_count": 0}
    if suffix in {".txt", ".md", ".markdown"}:
        return _plain_to_markdown(path.read_text(encoding="utf-8-sig", errors="replace")), metadata
    if suffix == ".pdf":
        import pymupdf
        chunks = []
        with pymupdf.open(path) as pdf:
            metadata["page_count"] = len(pdf)
            title = ((pdf.metadata or {}).get("title") or "").strip()
            if title:
                chunks.append(f"# {title}")
            for page in pdf:
                value = page.get_text("text", sort=True).strip()
                if value:
                    chunks.append(value)
        return _plain_to_markdown("\n\n".join(chunks)), metadata
    if suffix == ".docx":
        from docx import Document
        doc = Document(path)
        chunks = []
        for paragraph in doc.paragraphs:
            value = paragraph.text.strip()
            if not value:
                continue
            style = (paragraph.style.name or "").lower() if paragraph.style else ""
            if style.startswith("heading"):
                match = re.search(r"(\d+)", style)
                level = min(6, int(match.group(1))) if match else 2
                chunks.append("#" * level + " " + value)
            else:
                chunks.append(value)
        for table in doc.tables:
            rows = [[cell.text.replace("\n", " ").strip() for cell in row.cells] for row in table.rows]
            if rows:
                width = max(map(len, rows))
                rows = [row + [""] * (width - len(row)) for row in rows]
                chunks.append("| " + " | ".join(rows[0]) + " |")
                chunks.append("| " + " | ".join(["---"] * width) + " |")
                chunks.extend("| " + " | ".join(row) + " |" for row in rows[1:])
        return "\n\n".join(chunks).strip() + "\n", metadata
    if suffix == ".rtf":
        try:
            from striprtf.striprtf import rtf_to_text
        except ImportError as exc:
            raise RuntimeError("RTF conversion requires striprtf; install project requirements") from exc
        payload = path.read_bytes()
        header = payload[:256].decode("ascii", errors="ignore")
        codepage = re.search(r"\\ansicpg(\d+)", header)
        encoding = f"cp{codepage.group(1)}" if codepage else "cp1252"
        raw = payload.decode(encoding, errors="replace")
        return _plain_to_markdown(rtf_to_text(raw, encoding=encoding)), metadata
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        raise RuntimeError("DOC conversion requires LibreOffice (soffice) on PATH")
    with tempfile.TemporaryDirectory(prefix="sciintel_doc_") as tmp:
        proc = subprocess.run([soffice, "--headless", "--convert-to", "docx", "--outdir", tmp, str(path)],
                              capture_output=True, text=True, timeout=180, check=False)
        converted = Path(tmp) / (path.stem + ".docx")
        if proc.returncode or not converted.is_file():
            raise RuntimeError("LibreOffice DOC conversion failed: " + (proc.stderr or proc.stdout).strip())
        markdown, _ = convert_to_markdown(converted)
        return markdown, metadata
