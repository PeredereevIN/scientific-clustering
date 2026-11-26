from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import __version__, db
from .config import settings
from .embeddings import create_embeddings
from .ingest import ingest
from .keywords import run_keywords
from .clustering import run_dbscan, run_hdbscan, sweep_dbscan
from .monitor import system_report
from .pipeline import run_all


def _json(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sciintel", description="Анализ научных публикаций и тематических сообществ")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="Создать папки проекта и инициализировать SQLite")
    doctor = sub.add_parser("doctor", help="Проверить Python, CUDA, видеокарту и оперативную память")
    doctor.add_argument("--require-cuda", action="store_true")

    bench = sub.add_parser("gpu-benchmark", help="Проверить видеокарту вычислительной нагрузкой CUDA")
    bench.add_argument("--seconds", type=int, default=20)
    bench.add_argument("--matrix-size", type=int, default=3072)

    ing = sub.add_parser("ingest", help="Обработать PDF, Markdown и TXT из папки inbox")
    ing.add_argument("--dir", type=Path, default=settings.inbox)

    kw = sub.add_parser("keywords", help="Извлечь ключевые фразы методами YAKE и RAKE")
    kw.add_argument("--force", action="store_true")
    kw.add_argument("--top-n", type=int, default=settings.keyword_top_n)

    emb = sub.add_parser("embed", help="Создать или обновить SPECTER-векторы")
    emb.add_argument("--device", choices=["cuda", "auto", "cpu"], default="cuda")
    emb.add_argument("--batch-size", type=int, default=16)
    emb.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    emb.add_argument("--force", action="store_true")

    cluster = sub.add_parser("cluster", help="Запустить DBSCAN по косинусным расстояниям")
    cluster.add_argument("--device", choices=["cuda", "auto", "cpu"], default="cuda")
    cluster.add_argument("--eps", type=float, default=0.40)
    cluster.add_argument("--min-samples", type=int, default=5)

    hdb = sub.add_parser("hdbscan", help="Запустить HDBSCAN по косинусным расстояниям")
    hdb.add_argument("--device", choices=["cuda", "auto", "cpu"], default="cuda")
    hdb.add_argument("--min-cluster-size", type=int, default=10)
    hdb.add_argument("--min-samples", type=int, default=5)
    hdb.add_argument("--selection", choices=["eom", "leaf"], default="eom")
    hdb.add_argument("--allow-single-cluster", action="store_true")
    hdb.add_argument("--reduction", choices=["original", "umap"], default="original",
                     help="Кластеризовать исходные векторы SPECTER или их проекцию UMAP")
    hdb.add_argument("--umap-components", type=int, default=10)
    hdb.add_argument("--umap-neighbors", type=int, default=30)
    hdb.add_argument("--umap-min-dist", type=float, default=0.0)
    hdb.add_argument("--umap-metric", default="cosine")

    sweep = sub.add_parser("sweep", help="Подобрать параметры DBSCAN за один проход по GPU-расстояниям")
    sweep.add_argument("--device", choices=["cuda", "auto", "cpu"], default="cuda")
    sweep.add_argument("--eps-list", default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55")
    sweep.add_argument("--min-samples-list", default="3,5,8")

    run = sub.add_parser("run", help="Выполнить полный цикл обработки корпуса")
    run.add_argument("--device", choices=["cuda", "auto", "cpu"], default="cuda")
    run.add_argument("--batch-size", type=int, default=16)
    run.add_argument("--eps", type=float, default=0.40)
    run.add_argument("--min-samples", type=int, default=5)
    run.add_argument("--force-embeddings", action="store_true")
    run.add_argument("--force-keywords", action="store_true")

    search = sub.add_parser("search", help="Найти похожие публикации по запросу")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--device", choices=["cuda", "auto", "cpu"], default="cuda")

    graph_cluster = sub.add_parser("graph-cluster", help="Построить граф косинусных соседей и выделить сообщества Louvain")
    graph_cluster.add_argument("--device", choices=["cuda", "auto", "cpu"], default="auto")
    graph_cluster.add_argument("--k", type=int, default=15)
    graph_cluster.add_argument("--min-similarity", type=float, default=0.35)
    graph_cluster.add_argument("--resolution", type=float, default=1.0)
    graph_cluster.add_argument("--min-community-size", type=int, default=5)
    graph_cluster.add_argument("--mutual", action="store_true")

    graph = sub.add_parser("export-neo4j", help="Экспортировать публикации, ключевые слова и сообщества в Neo4j")
    graph.add_argument("--run-id", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    settings.ensure_dirs()
    if args.command == "init":
        db.init_db(); _json({"status": "initialized", "root": settings.root, "db": settings.db_path})
    elif args.command == "doctor":
        report = system_report(); _json(report)
        if args.require_cuda and not report.get("cuda_available"):
            return 2
    elif args.command == "gpu-benchmark":
        from .gpu_benchmark import run_gpu_benchmark
        _json(run_gpu_benchmark(seconds=max(3, args.seconds), matrix_size=args.matrix_size))
    elif args.command == "ingest":
        _json(ingest(args.dir))
    elif args.command == "keywords":
        _json(run_keywords(force=args.force, top_n=args.top_n))
    elif args.command == "embed":
        _json(create_embeddings(device=args.device, batch_size=args.batch_size,
                                precision=args.precision, force=args.force))
    elif args.command == "cluster":
        _json(run_dbscan(eps=args.eps, min_samples=args.min_samples, device=args.device))
    elif args.command == "hdbscan":
        _json(run_hdbscan(min_cluster_size=args.min_cluster_size, min_samples=args.min_samples,
                          cluster_selection_method=args.selection, allow_single_cluster=args.allow_single_cluster,
                          device=args.device, reduction=args.reduction,
                          umap_components=args.umap_components, umap_neighbors=args.umap_neighbors,
                          umap_min_dist=args.umap_min_dist, umap_metric=args.umap_metric))
    elif args.command == "sweep":
        eps_values = [float(x.strip()) for x in args.eps_list.split(",") if x.strip()]
        min_values = [int(x.strip()) for x in args.min_samples_list.split(",") if x.strip()]
        _json(sweep_dbscan(eps_values, min_values, device=args.device))
    elif args.command == "run":
        _json(run_all(device=args.device, batch_size=args.batch_size, eps=args.eps,
                      min_samples=args.min_samples, force_embeddings=args.force_embeddings,
                      force_keywords=args.force_keywords))
    elif args.command == "search":
        from .search import semantic_search
        _json(semantic_search(args.query, top_k=args.top_k, device=args.device))
    elif args.command == "graph-cluster":
        from .graph_clustering import run_louvain
        _json(run_louvain(min_similarity=args.min_similarity, k=args.k,
                          resolution=args.resolution, min_community_size=args.min_community_size,
                          mutual=args.mutual, device=args.device))
    elif args.command == "export-neo4j":
        from .graph import export_neo4j
        _json(export_neo4j(run_id=args.run_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
