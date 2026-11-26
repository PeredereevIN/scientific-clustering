from __future__ import annotations

import logging
import time

from . import db
from .clustering import run_dbscan
from .embeddings import create_embeddings
from .ingest import ingest
from .keywords import run_keywords

log = logging.getLogger("sciintel")


def run_all(device: str = "cuda", batch_size: int = 16, eps: float = 0.40,
            min_samples: int = 5, force_embeddings: bool = False,
            force_keywords: bool = False) -> dict:
    db.init_db()
    total_start = time.perf_counter()
    results = {}
    results["ingest"] = ingest()
    # Start with the GPU-heavy embedding pass so the accelerator is engaged early.
    results["embeddings"] = create_embeddings(device=device, batch_size=batch_size,
                                                force=force_embeddings)
    results["keywords"] = run_keywords(force=force_keywords)
    results["clustering"] = run_dbscan(eps=eps, min_samples=min_samples, device=device)
    results["total_seconds"] = round(time.perf_counter() - total_start, 2)
    log.info("Pipeline complete in %.2fs", results["total_seconds"])
    return results
