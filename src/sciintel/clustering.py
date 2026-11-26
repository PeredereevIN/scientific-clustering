from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.decomposition import PCA

from . import db
from .config import settings
from .embeddings import load_embeddings, resolve_device
from .keywords import keyword_profile

log = logging.getLogger("sciintel")


def gpu_cosine_distances(vectors: np.ndarray, device: str = "auto", chunk_size: int = 2048) -> tuple[np.ndarray, str]:
    """Compute the dense cosine-distance matrix with PyTorch on CUDA when available."""
    import torch
    chosen = resolve_device(device)
    n = vectors.shape[0]
    if n > 10000:
        raise MemoryError(
            f"Dense precomputed DBSCAN is O(N^2); got {n} papers. "
            "For this corpus size, use a sparse-neighborhood/ANN clustering backend or split the experiment."
        )
    x = torch.as_tensor(vectors, dtype=torch.float32, device=chosen)
    x = torch.nn.functional.normalize(x, p=2, dim=1)
    # Keep only one row block on the GPU. This avoids allocating an N x N
    # CUDA tensor and leaves headroom for the SPECTER model on an 8 GB GPU.
    cpu = np.empty((n, n), dtype=np.float32)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        block = (1.0 - x[start:stop] @ x.T).clamp_(min=0.0, max=2.0)
        cpu[start:stop] = block.cpu().numpy()
        del block
    np.fill_diagonal(cpu, 0.0)
    del x
    if chosen == "cuda":
        torch.cuda.empty_cache()
    return cpu, chosen


def gpu_pca_2d(vectors: np.ndarray, device: str = "auto") -> tuple[np.ndarray, str]:
    import torch
    chosen = resolve_device(device)
    try:
        x = torch.as_tensor(vectors, dtype=torch.float32, device=chosen)
        x = x - x.mean(dim=0, keepdim=True)
        # Exact SVD projection; suitable for the expected 1k-5k document corpus.
        _, _, vh = torch.linalg.svd(x, full_matrices=False)
        coords = x @ vh[:2].T
        output = coords.detach().cpu().numpy().astype(np.float32)
        del x, vh, coords
        if chosen == "cuda":
            torch.cuda.empty_cache()
        return output, chosen
    except Exception as exc:
        log.warning("GPU PCA unavailable (%s); using sklearn PCA fallback", exc)
        coords = PCA(n_components=2, random_state=42).fit_transform(vectors)
        return coords.astype(np.float32), "cpu-fallback"


def umap_projection(
    vectors: np.ndarray,
    n_components: int = 10,
    n_neighbors: int = 30,
    min_dist: float = 0.0,
    metric: str = "cosine",
    random_state: int = 42,
) -> np.ndarray:
    """Create a reproducible UMAP representation for clustering or plotting.

    UMAP is intentionally imported lazily so projects that only use DBSCAN/PCA
    do not need to load its numba stack.
    """
    try:
        import umap
    except ImportError as exc:
        raise RuntimeError(
            "UMAP is not installed. Run .\\START.bat setup or "
            "python -m pip install 'umap-learn>=0.5.12,<0.7'."
        ) from exc

    if vectors.ndim != 2 or len(vectors) < 3:
        raise ValueError("UMAP requires a 2D array with at least 3 documents")
    if n_components < 2:
        raise ValueError("UMAP n_components must be >=2")
    if n_neighbors < 2:
        raise ValueError("UMAP n_neighbors must be >=2")
    if not 0.0 <= min_dist <= 1.0:
        raise ValueError("UMAP min_dist must be between 0 and 1")

    reducer = umap.UMAP(
        n_neighbors=min(n_neighbors, len(vectors) - 1),
        n_components=n_components,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        transform_seed=random_state,
        n_jobs=1,
    )
    return np.asarray(reducer.fit_transform(vectors), dtype=np.float32)


def _calculate_metrics(distances: np.ndarray, labels: np.ndarray, domains: list[str]) -> dict:
    core = labels != -1
    unique = sorted(set(labels.tolist()) - {-1})
    metrics = {
        "documents": int(len(labels)),
        "clusters": int(len(unique)),
        "noise_count": int((~core).sum()),
        "noise_fraction": float((~core).mean()) if len(labels) else 0.0,
        "cluster_sizes": {str(int(c)): int((labels == c).sum()) for c in unique},
    }
    if len(unique) >= 2 and core.sum() > len(unique):
        idx = np.flatnonzero(core)
        if len(idx) > 1500:
            rng = np.random.default_rng(42)
            idx = np.sort(rng.choice(idx, size=1500, replace=False))
        sub = distances[np.ix_(idx, idx)]
        try:
            metrics["silhouette_non_noise"] = float(silhouette_score(sub, labels[idx], metric="precomputed"))
        except Exception:
            metrics["silhouette_non_noise"] = None
    else:
        metrics["silhouette_non_noise"] = None
    if domains and len(set(domains)) > 1:
        metrics["ari_domains"] = float(adjusted_rand_score(domains, labels))
        metrics["nmi_domains"] = float(normalized_mutual_info_score(domains, labels))
    else:
        metrics["ari_domains"] = None
        metrics["nmi_domains"] = None
    return metrics


def run_dbscan(eps: float = 0.18, min_samples: int = 5,
               device: str = "auto", model_name: str | None = None) -> dict:
    db.init_db()
    if eps <= 0 or min_samples < 2:
        raise ValueError("eps must be >0 and min_samples must be >=2")
    ids, vectors, manifest = load_embeddings(model_name)
    docs_by_id = {row["id"]: row for row in db.rows("SELECT id, title, domain FROM documents")}
    keep = [i for i, doc_id in enumerate(ids) if doc_id in docs_by_id]
    vectors = vectors[keep]
    ids = [ids[i] for i in keep]
    if len(ids) < 3:
        raise RuntimeError("At least 3 embedded documents are required for DBSCAN")
    started = time.perf_counter()
    distances, distance_device = gpu_cosine_distances(vectors, device=device)
    # DBSCAN is the vetted scikit-learn implementation. Expensive all-pairs distances
    # are computed by torch on CUDA; DBSCAN consumes that exact precomputed matrix.
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed", n_jobs=1).fit_predict(distances)
    domains = [docs_by_id[doc_id]["domain"] for doc_id in ids]
    metrics = _calculate_metrics(distances, labels, domains)
    coords, projection_device = gpu_pca_2d(vectors, device=device)
    elapsed = time.perf_counter() - started
    metrics.update({"elapsed_seconds": round(elapsed, 3), "distance_device": distance_device,
                    "projection_device": projection_device, "dbscan_backend": "scikit-learn/precomputed"})
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    model_name = model_name or manifest.get("model", settings.model_name)
    with db.connect() as conn:
        conn.execute("""
          INSERT INTO runs(run_id, model_name, device, batch_size, eps, min_samples, metrics_json)
          VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (run_id, model_name, distance_device, int(manifest.get("batch_size_last_run", 0)),
              eps, min_samples, json.dumps(metrics, ensure_ascii=False)))
        conn.executemany("""
          INSERT INTO cluster_labels(run_id, document_id, cluster_id, x, y)
          VALUES (?, ?, ?, ?, ?)
        """, [(run_id, doc_id, int(label), float(x), float(y))
              for doc_id, label, (x, y) in zip(ids, labels, coords, strict=True)])
    _write_artifacts(run_id, ids, labels, coords, docs_by_id, metrics)
    log.info("Clustering run %s: %s", run_id, metrics)
    return {"run_id": run_id, "metrics": metrics}



def run_hdbscan(min_cluster_size: int = 10, min_samples: int = 5,
                cluster_selection_method: str = "eom", allow_single_cluster: bool = False,
                device: str = "auto", model_name: str | None = None,
                reduction: str = "original", umap_components: int = 10,
                umap_neighbors: int = 30, umap_min_dist: float = 0.0,
                umap_metric: str = "cosine") -> dict:
    """Run HDBSCAN in original SPECTER space or after UMAP reduction.

    The evaluation Silhouette is always calculated using original cosine
    distances, so runs with different reduction settings remain comparable.
    """
    try:
        import hdbscan
    except ImportError as exc:
        raise RuntimeError("HDBSCAN is not installed. Install project dependencies with: pip install hdbscan") from exc

    db.init_db()
    if min_cluster_size < 2 or min_samples < 1:
        raise ValueError("min_cluster_size must be >=2 and min_samples must be >=1")
    if cluster_selection_method not in {"eom", "leaf"}:
        raise ValueError("cluster_selection_method must be 'eom' or 'leaf'")
    if reduction not in {"original", "umap"}:
        raise ValueError("reduction must be 'original' or 'umap'")
    if reduction == "umap" and umap_components < 2:
        raise ValueError("umap_components must be >=2")

    ids, vectors, manifest = load_embeddings(model_name)
    docs_by_id = {row["id"]: row for row in db.rows("SELECT id, title, domain FROM documents")}
    keep = [i for i, doc_id in enumerate(ids) if doc_id in docs_by_id]
    vectors = vectors[keep]
    ids = [ids[i] for i in keep]
    if len(ids) < min_cluster_size:
        raise RuntimeError(f"At least {min_cluster_size} embedded documents are required for HDBSCAN")

    started = time.perf_counter()
    # Keep original-space cosine distances for a consistent quality metric,
    # independent of whether clustering itself uses the UMAP representation.
    distances, distance_device = gpu_cosine_distances(vectors, device=device)
    if reduction == "umap":
        cluster_vectors = umap_projection(
            vectors, n_components=umap_components, n_neighbors=umap_neighbors,
            min_dist=umap_min_dist, metric=umap_metric,
        )
        fit_data = np.ascontiguousarray(cluster_vectors, dtype=np.float64)
        cluster_metric = "euclidean"
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=min_samples, metric=cluster_metric,
            cluster_selection_method=cluster_selection_method, allow_single_cluster=allow_single_cluster,
            approx_min_span_tree=False, max_cluster_size=350,
        )
        labels = clusterer.fit_predict(fit_data).astype(np.int64)
        # A second, independent 2D UMAP is used only for the map. It must not
        # feed back into clustering because 2D can distort density structure.
        coords = umap_projection(
            vectors, n_components=2, n_neighbors=umap_neighbors,
            min_dist=0.1, metric=umap_metric,
        )
        projection_device = "cpu-umap"
    else:
        # hdbscan-contrib's precomputed-distance MST expects C-contiguous float64.
        fit_data = np.ascontiguousarray(distances, dtype=np.float64)
        cluster_metric = "precomputed-cosine"
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=min_samples, metric="precomputed",
            cluster_selection_method=cluster_selection_method, allow_single_cluster=allow_single_cluster,
            approx_min_span_tree=False,
        )
        labels = clusterer.fit_predict(fit_data).astype(np.int64)
        coords, projection_device = gpu_pca_2d(vectors, device=device)

    domains = [docs_by_id[doc_id]["domain"] for doc_id in ids]
    metrics = _calculate_metrics(distances, labels, domains)
    if reduction == "umap":
        non_noise = labels != -1
        n_clusters = len(set(labels.tolist()) - {-1})
        if n_clusters >= 2 and int(non_noise.sum()) > n_clusters:
            try:
                metrics["silhouette_umap_non_noise"] = float(
                    silhouette_score(cluster_vectors[non_noise], labels[non_noise], metric="euclidean")
                )
            except Exception:
                metrics["silhouette_umap_non_noise"] = None
        else:
            metrics["silhouette_umap_non_noise"] = None
    elapsed = time.perf_counter() - started
    metrics.update({
        "algorithm": "HDBSCAN", "min_cluster_size": min_cluster_size,
        "min_samples": min_samples, "cluster_selection_method": cluster_selection_method,
        "allow_single_cluster": allow_single_cluster, "elapsed_seconds": round(elapsed, 3),
        "distance_device": distance_device, "clustering_device": "cpu",
        "projection_device": projection_device, "projection_method": "UMAP" if reduction == "umap" else "PCA",
        "reduction": reduction, "cluster_metric": cluster_metric,
        "hdbscan_backend": "hdbscan-contrib",
    })
    if reduction == "umap":
        metrics.update({
            "umap_cluster_components": umap_components,
            "umap_neighbors": umap_neighbors,
            "umap_min_dist": umap_min_dist,
            "umap_metric": umap_metric,
            "umap_random_state": 42,
            "silhouette_space": "original SPECTER cosine distance",
            "silhouette_umap_space": "reduced UMAP Euclidean distance",
        })
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    model_name = model_name or manifest.get("model", settings.model_name)
    # The legacy runs table requires eps; 0.0 is a neutral sentinel for non-DBSCAN runs.
    with db.connect() as conn:
        conn.execute("""
          INSERT INTO runs(run_id, model_name, device, batch_size, eps, min_samples, metrics_json)
          VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (run_id, model_name, distance_device, int(manifest.get("batch_size_last_run", 0)),
              0.0, min_samples, json.dumps(metrics, ensure_ascii=False)))
        conn.executemany("""
          INSERT INTO cluster_labels(run_id, document_id, cluster_id, x, y)
          VALUES (?, ?, ?, ?, ?)
        """, [(run_id, doc_id, int(label), float(x), float(y))
              for doc_id, label, (x, y) in zip(ids, labels, coords, strict=True)])
    _write_artifacts(run_id, ids, labels, coords, docs_by_id, metrics)
    log.info("HDBSCAN run %s: %s", run_id, metrics)
    return {"run_id": run_id, "metrics": metrics}


def sweep_dbscan(eps_values: list[float], min_samples_values: list[int],
                 device: str = "auto", model_name: str | None = None) -> dict:
    """Compare a DBSCAN parameter grid while calculating the distance matrix only once."""
    import itertools
    import time

    db.init_db()
    eps_values = sorted({float(value) for value in eps_values if float(value) > 0})
    min_samples_values = sorted({int(value) for value in min_samples_values if int(value) >= 2})
    if not eps_values or not min_samples_values:
        raise ValueError("Provide positive eps values and min_samples values >= 2")
    ids, vectors, _ = load_embeddings(model_name)
    docs_by_id = {row["id"]: row for row in db.rows("SELECT id, title, domain FROM documents")}
    keep = [i for i, doc_id in enumerate(ids) if doc_id in docs_by_id]
    vectors = vectors[keep]
    ids = [ids[i] for i in keep]
    if len(ids) < 3:
        raise RuntimeError("At least 3 embedded documents are required for DBSCAN")

    started = time.perf_counter()
    distances, distance_device = gpu_cosine_distances(vectors, device=device)
    domains = [docs_by_id[doc_id]["domain"] for doc_id in ids]
    records = []
    best_key = None
    best_config = None
    for eps, min_samples in itertools.product(eps_values, min_samples_values):
        labels = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed", n_jobs=1).fit_predict(distances)
        metrics = _calculate_metrics(distances, labels, domains)
        record = {"eps": eps, "min_samples": min_samples, **metrics}
        records.append(record)
        # A transparent ranking aid, not a claim of universal optimality. Prefer
        # source-label agreement, then silhouette, and penalize near-total noise.
        ari = metrics.get("ari_domains")
        nmi = metrics.get("nmi_domains")
        silhouette = metrics.get("silhouette_non_noise")
        agreement = nmi if nmi is not None else (ari if ari is not None else -1.0)
        sil = silhouette if silhouette is not None else -1.0
        key = (agreement, sil, -metrics["noise_fraction"])
        if best_key is None or key > best_key:
            best_key = key
            best_config = {"eps": eps, "min_samples": min_samples, **metrics}

    sweep_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    output = settings.artifacts / "sweeps" / sweep_id
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records).sort_values(
        ["nmi_domains", "silhouette_non_noise", "noise_fraction"],
        ascending=[False, False, True], na_position="last")
    frame.to_csv(output / "parameter_sweep.csv", index=False, encoding="utf-8-sig")
    # k-distance diagnostic for min_samples; the diagonal is zero and includes self.
    k = min(max(min_samples_values) - 1, len(ids) - 1)
    kth = np.partition(distances, kth=k, axis=1)[:, k]
    pd.DataFrame({"document_id": ids, "k": k, "k_distance": kth}).sort_values(
        "k_distance").to_csv(output / "k_distance.csv", index=False, encoding="utf-8-sig")
    summary = {"sweep_id": sweep_id, "distance_device": distance_device,
               "model": model_name or settings.model_name,
               "distance_matrix_computed_once": True, "tested_configurations": len(records),
               "elapsed_seconds": round(time.perf_counter() - started, 3),
               "ranking_note": "Sorted by NMI against source-folder labels, then non-noise Silhouette; inspect all metrics.",
               "top_candidate": best_config, "output": str(output)}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("DBSCAN sweep completed: %s", summary)
    return summary


def _write_artifacts(run_id: str, ids: list[str], labels: np.ndarray,
                     coords: np.ndarray, docs: dict, metrics: dict) -> None:
    out = settings.artifacts / "runs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"document_id": ids, "cluster_id": labels.astype(int),
                          "x": coords[:, 0], "y": coords[:, 1],
                          "title": [docs[x]["title"] for x in ids],
                          "domain": [docs[x]["domain"] for x in ids]})
    frame.to_csv(out / "clusters.csv", index=False, encoding="utf-8-sig")
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    profile_rows = []
    for cluster_id in sorted(set(labels.tolist()) - {-1}):
        cluster_ids = [doc_id for doc_id, label in zip(ids, labels, strict=True) if label == cluster_id]
        terms = keyword_profile(cluster_ids, limit=15)
        profile_rows.append({"cluster_id": int(cluster_id), "size": len(cluster_ids),
                             "domains": ", ".join(sorted({docs[x]["domain"] for x in cluster_ids})),
                             "top_terms": " | ".join(terms),
                             "representative_titles": " | ".join(docs[x]["title"] for x in cluster_ids[:5])})
    pd.DataFrame(profile_rows).to_csv(out / "cluster_profiles.csv", index=False, encoding="utf-8-sig")
    latest = settings.artifacts / "latest_run.txt"
    latest.write_text(run_id, encoding="utf-8")
