from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

from . import db
from .config import settings
from .embeddings import _load_model, load_embeddings, resolve_device
from .utils import clean_text


@lru_cache(maxsize=2)
def _query_model(model_name: str, device: str):
    # Keep a single model instance alive in a long-running Streamlit process.
    return _load_model(model_name, device, "fp16" if device == "cuda" else "fp32")


def semantic_search(query: str, top_k: int = 5, device: str = "auto",
                    model_name: str | None = None) -> list[dict]:
    query = clean_text(query)
    if len(query) < 3:
        raise ValueError("Search query must contain at least 3 characters")
    model_name = model_name or settings.model_name
    ids, vectors, _ = load_embeddings(model_name)
    chosen = resolve_device(device)
    model = _query_model(model_name, chosen)
    query_vector = model.encode([query], normalize_embeddings=True, convert_to_numpy=True,
                                show_progress_bar=False, device=chosen).astype(np.float32)
    matrix = torch.as_tensor(vectors, device=chosen, dtype=torch.float32)
    q = torch.as_tensor(query_vector, device=chosen, dtype=torch.float32)
    scores = (matrix @ q.T).squeeze(1)
    k = min(max(1, top_k), len(ids))
    vals, indexes = torch.topk(scores, k=k)
    selected = [(ids[int(i)], float(v)) for i, v in zip(indexes.cpu(), vals.cpu(), strict=True)]
    del matrix, q, scores, vals, indexes
    if chosen == "cuda":
        torch.cuda.empty_cache()
    by_id = {row["id"]: row for row in db.rows("SELECT id, title, domain, path, doi, year FROM documents")}
    return [{**by_id[doc_id], "score": score} for doc_id, score in selected if doc_id in by_id]
