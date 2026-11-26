from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from . import db
from .config import settings
from .utils import embedding_text

log = logging.getLogger("sciintel")


def resolve_device(requested: str = "auto") -> str:
    import torch
    if requested == "cpu":
        return "cpu"
    available = torch.cuda.is_available()
    if requested == "cuda" and not available:
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False. Run `sciintel doctor`.")
    return "cuda" if available and requested in {"auto", "cuda"} else "cpu"


def _load_model(model_name: str, device: str, precision: str):
    from sentence_transformers import SentenceTransformer
    kwargs = {}
    if device == "cuda" and precision in {"fp16", "bf16"}:
        import torch
        kwargs["torch_dtype"] = torch.float16 if precision == "fp16" else torch.bfloat16
    effective_precision = precision if kwargs else "fp32"
    try:
        model = SentenceTransformer(model_name, device=device, model_kwargs=kwargs)
    except Exception as exc:
        if not kwargs:
            raise
        log.warning("Reduced precision model load failed (%s); retrying FP32", exc)
        model = SentenceTransformer(model_name, device=device)
        effective_precision = "fp32"
    model.max_seq_length = min(int(getattr(model, "max_seq_length", 512)), 512)
    # Expose the effective dtype so experiment metadata does not claim FP16
    # when a model loader had to fall back to FP32.
    model._sciintel_precision = effective_precision
    return model


def _cache_dir(model_name: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", model_name)
    return settings.artifacts / "embeddings" / safe


def _input_hash(text: str, model_name: str) -> str:
    return hashlib.sha256((model_name + "\0" + text).encode("utf-8")).hexdigest()


def _replace_with_retry(source: Path, destination: Path, attempts: int = 7) -> None:
    """Replace a generated artifact, tolerating short Windows/OneDrive file locks."""
    for attempt in range(attempts):
        try:
            source.replace(destination)
            return
        except PermissionError:
            if attempt + 1 >= attempts:
                raise PermissionError(
                    f"Cannot replace {destination}. Close processes that may hold the file "
                    "(dashboard, another Python run, preview/indexer), pause OneDrive sync, "
                    "and retry."
                )
            time.sleep(0.25 * (attempt + 1))


def create_embeddings(device: str = "auto", batch_size: int = 16,
                      model_name: str | None = None, precision: str = "fp16",
                      force: bool = False) -> dict:
    db.init_db()
    model_name = model_name or settings.model_name
    device = resolve_device(device)
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    docs = db.rows("SELECT id, title, abstract, text_path FROM documents WHERE status='ready' ORDER BY id")
    if not docs:
        raise RuntimeError("No ready documents. Run `sciintel ingest` first.")
    cache = _cache_dir(model_name)
    cache.mkdir(parents=True, exist_ok=True)
    vectors_path = cache / "vectors.npy"
    manifest_path = cache / "manifest.json"
    if vectors_path.exists() and manifest_path.exists() and not force:
        # Load into ordinary RAM rather than np.memmap. On Windows, an open memory-mapped
        # vectors.npy cannot be atomically replaced by Path.replace(), even after the
        # new matrix has been computed. The intended 1.8k-5k paper corpus is small
        # enough that this copy is inexpensive and avoids a persistent file lock.
        old_vectors = np.load(vectors_path, allow_pickle=False)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        old_map = {doc_id: i for i, doc_id in enumerate(manifest.get("ids", []))}
        old_hashes = manifest.get("hashes", {})
    else:
        old_vectors = np.empty((0, 0), dtype=np.float32)
        old_map, old_hashes = {}, {}
    texts, pending_ids, pending_hashes = [], [], {}
    current_ids = []
    for doc in docs:
        body = Path(doc["text_path"]).read_text(encoding="utf-8", errors="replace")
        text = embedding_text(doc["title"], doc["abstract"], body, settings.max_embedding_chars)
        digest = _input_hash(text, model_name)
        current_ids.append(doc["id"])
        if doc["id"] in old_map and old_hashes.get(doc["id"]) == digest and not force:
            continue
        texts.append(text)
        pending_ids.append(doc["id"])
        pending_hashes[doc["id"]] = digest

    stats = {"documents": len(docs), "cached": len(docs) - len(pending_ids),
             "encoded": len(pending_ids), "device": device, "model": model_name,
             "batch_size_initial": batch_size, "batch_size_final": batch_size,
             "precision": precision if device == "cuda" else "fp32"}
    if pending_ids:
        model = _load_model(model_name, device, precision if device == "cuda" else "fp32")
        stats["precision"] = getattr(model, "_sciintel_precision", "fp32")
        current_batch = batch_size
        while True:
            try:
                new_vectors = model.encode(
                    texts, batch_size=current_batch, show_progress_bar=True,
                    convert_to_numpy=True, normalize_embeddings=True,
                    device=device, precision="float32"
                ).astype(np.float32, copy=False)
                stats["batch_size_final"] = current_batch
                break
            except (RuntimeError, MemoryError) as exc:
                is_oom = "out of memory" in str(exc).lower() or "cuda" in str(exc).lower()
                if device != "cuda" or not is_oom or current_batch <= 1:
                    raise
                import torch
                torch.cuda.empty_cache()
                current_batch = max(1, current_batch // 2)
                log.warning("GPU memory pressure; retrying with batch_size=%d", current_batch)
        del model
        if device == "cuda":
            import torch
            torch.cuda.empty_cache()
    else:
        new_vectors = np.empty((0, old_vectors.shape[1] if old_vectors.ndim == 2 and old_vectors.size else 0), dtype=np.float32)

    dim = new_vectors.shape[1] if len(new_vectors) else (old_vectors.shape[1] if old_vectors.ndim == 2 and old_vectors.size else 0)
    merged: dict[str, np.ndarray] = {}
    for doc_id in current_ids:
        if doc_id in pending_hashes:
            continue
        index = old_map.get(doc_id)
        if index is not None and old_vectors.ndim == 2:
            merged[doc_id] = np.asarray(old_vectors[index], dtype=np.float32)
    for doc_id, vector in zip(pending_ids, new_vectors, strict=True):
        merged[doc_id] = vector
    ordered_ids = [doc_id for doc_id in current_ids if doc_id in merged]
    matrix = np.stack([merged[doc_id] for doc_id in ordered_ids]).astype(np.float32)
    tmp_vectors = vectors_path.with_suffix(".tmp.npy")
    np.save(tmp_vectors, matrix, allow_pickle=False)
    _replace_with_retry(tmp_vectors, vectors_path)
    hash_map = {}
    for doc in docs:
        if doc["id"] in pending_hashes:
            hash_map[doc["id"]] = pending_hashes[doc["id"]]
        elif doc["id"] in old_hashes:
            hash_map[doc["id"]] = old_hashes[doc["id"]]
        else:
            body = Path(doc["text_path"]).read_text(encoding="utf-8", errors="replace")
            hash_map[doc["id"]] = _input_hash(embedding_text(doc["title"], doc["abstract"], body,
                                                               settings.max_embedding_chars), model_name)
    manifest = {"model": model_name, "dimension": int(dim), "ids": ordered_ids,
                "hashes": hash_map, "normalized": True, "device_last_run": device,
                "batch_size_last_run": stats["batch_size_final"],
                "precision_last_run": stats["precision"]}
    tmp_manifest = manifest_path.with_suffix(".tmp")
    tmp_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _replace_with_retry(tmp_manifest, manifest_path)
    stats["dimension"] = int(dim)
    stats["path"] = str(vectors_path)
    log.info("Embeddings ready: %s", stats)
    return stats


def load_embeddings(model_name: str | None = None) -> tuple[list[str], np.ndarray, dict]:
    model_name = model_name or settings.model_name
    cache = _cache_dir(model_name)
    vectors_path, manifest_path = cache / "vectors.npy", cache / "manifest.json"
    if not vectors_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("Embeddings not found. Run `sciintel embed --device cuda` first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    vectors = np.load(vectors_path).astype(np.float32, copy=False)
    ids = manifest["ids"]
    if len(ids) != len(vectors):
        raise ValueError("Embedding manifest/vector count mismatch")
    return ids, vectors, manifest
