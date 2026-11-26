from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

from . import db
from .clustering import _calculate_metrics, _write_artifacts, gpu_pca_2d
from .config import settings
from .embeddings import load_embeddings

log = logging.getLogger("sciintel")


def build_knn_graph(vectors: np.ndarray, k: int = 15, min_similarity: float = 0.35,
                    mutual: bool = False) -> tuple[nx.Graph, pd.DataFrame]:
    """Build a weighted cosine kNN graph from normalized document embeddings."""
    if vectors.ndim != 2 or len(vectors) < 3:
        raise ValueError("At least three document vectors are required")
    if k < 1 or k >= len(vectors):
        raise ValueError("k must be between 1 and n_documents-1")
    if not -1.0 <= min_similarity <= 1.0:
        raise ValueError("min_similarity must be between -1 and 1")
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(vectors)), metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(vectors)
    distances, indices = nn.kneighbors(vectors)
    directed: dict[tuple[int, int], float] = {}
    for source, (row_dist, row_idx) in enumerate(zip(distances, indices, strict=True)):
        for distance, target in zip(row_dist, row_idx, strict=True):
            target = int(target)
            if source == target:
                continue
            similarity = float(1.0 - distance)
            if similarity < min_similarity:
                continue
            directed[(source, target)] = similarity
    graph = nx.Graph()
    graph.add_nodes_from(range(len(vectors)))
    edge_rows: list[dict] = []
    for (source, target), similarity in directed.items():
        reverse = directed.get((target, source))
        if mutual and reverse is None:
            continue
        weight = min(similarity, reverse) if mutual and reverse is not None else max(similarity, reverse or similarity)
        if graph.has_edge(source, target):
            graph[source][target]["weight"] = max(graph[source][target]["weight"], weight)
        else:
            graph.add_edge(source, target, weight=float(weight))
    for source, target, data in graph.edges(data=True):
        edge_rows.append({"source_index": source, "target_index": target,
                          "similarity": float(data["weight"])})
    edges = pd.DataFrame(edge_rows, columns=["source_index", "target_index", "similarity"])
    return graph, edges


def run_louvain(min_similarity: float = 0.35, k: int = 15, resolution: float = 1.0,
                min_community_size: int = 5, mutual: bool = False,
                device: str = "auto", model_name: str | None = None) -> dict:
    """Cluster SPECTER documents with Louvain communities on a cosine kNN graph."""
    if resolution <= 0 or min_community_size < 1:
        raise ValueError("resolution must be positive and min_community_size >= 1")
    db.init_db()
    ids, vectors, manifest = load_embeddings(model_name)
    docs_by_id = {r["id"]: r for r in db.rows("SELECT id,title,domain FROM documents")}
    keep = [i for i, doc_id in enumerate(ids) if doc_id in docs_by_id]
    ids = [ids[i] for i in keep]
    vectors = vectors[keep]
    if len(ids) < 3:
        raise RuntimeError("At least three embedded documents are required")

    started = time.perf_counter()
    graph, edges = build_knn_graph(vectors, k=k, min_similarity=min_similarity, mutual=mutual)
    if graph.number_of_edges() == 0:
        raise RuntimeError("The similarity threshold removed every edge. Lower min_similarity.")
    communities = nx.community.louvain_communities(graph, weight="weight", resolution=resolution, seed=42)
    communities = sorted(communities, key=lambda group: (-len(group), min(group)))
    labels = np.full(len(ids), -1, dtype=np.int64)
    next_label = 0
    for group in communities:
        if len(group) < min_community_size:
            continue
        for index in group:
            labels[int(index)] = next_label
        next_label += 1

    # Evaluation in the original normalized SPECTER cosine space.
    # The matrix is small for the intended 1.8k-5k corpus; do not use graph-layout coordinates.
    distances = np.clip(1.0 - (vectors @ vectors.T), 0.0, 2.0).astype(np.float32)
    domains = [docs_by_id[doc_id]["domain"] for doc_id in ids]
    metrics = _calculate_metrics(distances, labels, domains)
    non_noise = labels != -1
    if non_noise.sum() > 2 and len(set(labels[non_noise].tolist())) >= 2:
        sample_idx = np.flatnonzero(non_noise)
        if len(sample_idx) > 1500:
            rng = np.random.default_rng(42)
            sample_idx = np.sort(rng.choice(sample_idx, 1500, replace=False))
        try:
            metrics["silhouette_non_noise"] = float(
                silhouette_score(distances[np.ix_(sample_idx, sample_idx)], labels[sample_idx], metric="precomputed")
            )
        except Exception:
            metrics["silhouette_non_noise"] = None
    coords, projection_device = gpu_pca_2d(vectors, device=device)
    kept_communities = [group for group in communities if len(group) >= min_community_size]
    kept_nodes = set().union(*kept_communities) if kept_communities else set()
    final_graph = graph.subgraph(kept_nodes).copy()
    modularity = (float(nx.community.modularity(final_graph, kept_communities, weight="weight", resolution=resolution))
                  if final_graph.number_of_edges() > 0 and len(kept_communities) > 1 else None)
    raw_modularity = float(nx.community.modularity(graph, communities, weight="weight", resolution=resolution))
    degree_values = [d for _, d in graph.degree()]
    metrics.update({
        "algorithm": "Louvain kNN graph", "graph_type": "weighted cosine kNN",
        "documents": len(ids), "clusters": int(len(set(labels.tolist()) - {-1})),
        "noise_count": int((labels == -1).sum()),
        "noise_fraction": float((labels == -1).mean()),
        "cluster_sizes": {str(int(c)): int((labels == c).sum()) for c in sorted(set(labels.tolist()) - {-1})},
        "graph_nodes": graph.number_of_nodes(), "graph_edges": graph.number_of_edges(),
        "graph_components": nx.number_connected_components(graph),
        "graph_isolates": len(list(nx.isolates(graph))),
        "graph_density": float(nx.density(graph)),
        "average_degree": float(np.mean(degree_values)) if degree_values else 0.0,
        "modularity": modularity, "raw_louvain_modularity": raw_modularity, "k": k, "min_similarity": min_similarity,
        "resolution": resolution, "min_community_size": min_community_size,
        "mutual_knn": mutual, "projection_method": "PCA", "projection_device": projection_device,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "evaluation_note": "ARI/NMI use source-folder topic labels as weak external labels, not absolute ground truth.",
    })
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_graph_" + uuid.uuid4().hex[:6]
    model_name = model_name or manifest.get("model", settings.model_name)
    with db.connect() as conn:
        conn.execute("""INSERT INTO runs(run_id,model_name,device,batch_size,eps,min_samples,metrics_json)
                      VALUES(?,?,?,?,?,?,?)""",
                     (run_id, model_name, device, int(manifest.get("batch_size_last_run", 0)),
                      0.0, k, json.dumps(metrics, ensure_ascii=False)))
        conn.executemany("""INSERT INTO cluster_labels(run_id,document_id,cluster_id,x,y)
                          VALUES(?,?,?,?,?)""",
                         [(run_id, doc_id, int(label), float(x), float(y))
                          for doc_id, label, (x, y) in zip(ids, labels, coords, strict=True)])
    _write_artifacts(run_id, ids, labels, coords, docs_by_id, metrics)
    out = settings.artifacts / "runs" / run_id
    edge_rows = [{"source_id": ids[int(r.source_index)], "target_id": ids[int(r.target_index)],
                  "similarity": float(r.similarity)} for r in edges.itertuples(index=False)]
    pd.DataFrame(edge_rows, columns=["source_id", "target_id", "similarity"]).to_csv(
        out / "graph_edges.csv", index=False, encoding="utf-8-sig")
    log.info("Louvain graph run %s: %s", run_id, metrics)
    return {"run_id": run_id, "metrics": metrics}
