#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-label an existing Leiden result using a local Ollama model, without reclustering."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .auto_topic_discovery import (
    DEFAULT_EMBEDDING_DIR,
    OllamaTopicLabeler,
    build_profiles,
    manifest_ids,
    sha256_file,
)
from .paper_evidence import collect_paper_evidence


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise RuntimeError(f"Ожидался JSON-объект: {path}")
    return value


def resolve_source_dir(root: Path, run_id: str, source_result: str) -> Path:
    base = root / "reports" / "auto_topic_discovery" / run_id
    if source_result.lower() == "latest":
        latest_path = base / "latest.json"
        if not latest_path.is_file():
            raise FileNotFoundError(f"Не найден указатель последнего запуска: {latest_path}")
        latest = read_json(latest_path)
        folder = str(latest.get("output_dir", "")).strip()
        if not folder:
            raise RuntimeError(f"В {latest_path} отсутствует поле output_dir.")
    else:
        folder = source_result
    source = (base / folder).resolve()
    if not source.is_dir() or base.resolve() not in source.parents:
        raise FileNotFoundError(f"Каталог результата не найден или находится вне run-папки: {source}")
    for required in ("topic_profiles.csv", "document_assignments.csv", "projection_2d.csv", "summary.json"):
        if not (source / required).is_file():
            raise FileNotFoundError(f"В исходном результате отсутствует {required}: {source}")
    return source


def update_named_columns(frame: pd.DataFrame, name_by_cluster: dict[int, dict[str, str]]) -> pd.DataFrame:
    result = frame.copy()
    if "cluster_id" not in result.columns:
        return result
    ids = pd.to_numeric(result["cluster_id"], errors="raise").astype(int)
    mappings = {
        "display_name": "name_ru",
        "semantic_name": "name_en",
        "semantic_name_en": "name_en",
        "semantic_name_ru": "name_ru",
        "semantic_description_ru": "description_ru",
        "semantic_scope_note_ru": "scope_note_ru",
        "semantic_research_core_ru": "research_core_ru",
        "semantic_subtopics_ru": "subtopics_ru",
        "semantic_methods_ru": "methods_ru",
        "semantic_research_objects_ru": "research_objects_ru",
        "semantic_heterogeneity_level": "heterogeneity_level",
        "semantic_heterogeneity_note_ru": "heterogeneity_note_ru",
        "semantic_supporting_titles": "supporting_titles",
        "semantic_supporting_document_ids": "supporting_document_ids",
        "semantic_name_evidence_title_count": "name_evidence_title_count",
        "semantic_review_status": "review_status",
        "semantic_review_decision": "review_decision",
        "semantic_revision_status": "revision_status",
        "semantic_final_validation_status": "final_validation_status",
        "semantic_final_validation_decision": "final_validation_decision",
        "semantic_support_level": "support_level",
        "semantic_name_support_level": "name_support_level",
        "semantic_profile_support_level": "profile_support_level",
        "semantic_review_note_ru": "review_note_ru",
        "semantic_final_validation_note_ru": "final_validation_note_ru",
        "semantic_unsupported_name_claims": "unsupported_name_claims",
        "semantic_unsupported_profile_claims": "unsupported_profile_claims",
        "semantic_unsupported_claims": "unsupported_claims",
        "semantic_missing_directions": "missing_directions",
        "semantic_evidence_grade": "evidence_grade",
        "semantic_abstract_coverage_percent": "abstract_coverage_percent",
        "semantic_text_coverage_percent": "text_coverage_percent",
        "semantic_representative_abstract_coverage_percent": "representative_abstract_coverage_percent",
        "semantic_top_domain_share_percent": "top_domain_share_percent",
        "semantic_label_method": "label_method",
        "semantic_label_status": "label_status",
    }
    for column, key in mappings.items():
        result[column] = [name_by_cluster.get(int(cid), {}).get(key, "") for cid in ids]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Обновить семантические названия и описания готовых сообществ без повторной кластеризации."
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--run-id", default="latest", help="Идентификатор запуска; latest берётся из artifacts/latest_run.txt")
    parser.add_argument("--source-result", default="latest")
    parser.add_argument("--ollama-model", default="qwen3.5:4b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-timeout", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--abstract-source", choices=("auto", "local", "openalex", "off"), default="auto")
    parser.add_argument("--openalex-api-key-env", default="OPENALEX_API_KEY")
    parser.add_argument("--openalex-mailto", default=os.environ.get("OPENALEX_MAILTO", ""))
    parser.add_argument("--openalex-timeout", type=int, default=15)
    parser.add_argument("--max-openalex-batches", type=int, default=50)
    parser.add_argument("--max-openalex-title-lookups", type=int, default=900)
    parser.add_argument("--max-crossref-lookups", type=int, default=1870)
    parser.add_argument("--max-crossref-title-lookups", type=int, default=1870)
    parser.add_argument("--disable-crossref-fallback", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    root = Path(args.root).resolve()
    run_id = args.run_id
    if run_id == "latest":
        latest_path = root / "artifacts" / "latest_run.txt"
        if not latest_path.is_file():
            raise FileNotFoundError(f"Не найден указатель последнего запуска: {latest_path}")
        run_id = latest_path.read_text(encoding="utf-8").strip()
        if not run_id:
            raise RuntimeError(f"Файл последнего запуска пуст: {latest_path}")
    run_dir = root / "artifacts" / "runs" / run_id
    clusters_path = run_dir / "clusters.csv"
    emb_dir = root / DEFAULT_EMBEDDING_DIR
    manifest_path = emb_dir / "manifest.json"
    vectors_path = emb_dir / "vectors.npy"
    for path in (clusters_path, manifest_path, vectors_path):
        if not path.is_file():
            raise FileNotFoundError(f"Не найден обязательный входной файл: {path}")

    source_dir = resolve_source_dir(root, run_id, args.source_result)
    source_profiles = pd.read_csv(source_dir / "topic_profiles.csv", encoding="utf-8-sig")
    source_assignments = pd.read_csv(
        source_dir / "document_assignments.csv", encoding="utf-8-sig", dtype={"document_id": str}
    )
    if not {"cluster_id", "document_id"}.issubset(source_assignments.columns):
        raise RuntimeError("document_assignments.csv должен содержать cluster_id и document_id.")
    if source_assignments.document_id.isna().any() or source_assignments.document_id.duplicated().any():
        raise RuntimeError("В document_assignments.csv есть пустые или повторные document_id.")

    print(f"Исходный результат: {source_dir}")
    print("Проверяю локальную языковую модель...")
    labeler = OllamaTopicLabeler(args.ollama_url, args.ollama_model, args.ollama_timeout, args.seed)
    labeler_metadata = labeler.verify()
    labeler_metadata["output_language"] = "ru"
    print(f"Модель доступна: {args.ollama_model}")

    print("Загружаю исходный корпус и SPECTER-векторы...")
    docs = pd.read_csv(clusters_path, encoding="utf-8-sig", dtype={"document_id": str})
    docs["document_id"] = docs["document_id"].astype(str)
    if docs.document_id.isna().any() or docs.document_id.duplicated().any():
        raise RuntimeError("clusters.csv содержит пустые или повторные document_id.")
    ids = docs.document_id.tolist()
    assignment_ids = source_assignments.document_id.astype(str).tolist()
    if len(ids) != len(assignment_ids) or set(ids) != set(assignment_ids):
        raise RuntimeError(
            "Набор публикаций в исходных назначениях не совпадает с clusters.csv: "
            f"corpus={len(ids)}, assignments={len(assignment_ids)}."
        )
    labels_by_id = source_assignments.set_index("document_id")["cluster_id"]
    labels = pd.to_numeric(labels_by_id.loc[ids], errors="raise").to_numpy(dtype=np.int32)

    with manifest_path.open("r", encoding="utf-8-sig") as stream:
        manifest = json.load(stream)
    vector_ids = manifest_ids(manifest)
    if len(vector_ids) != len(set(vector_ids)) or set(vector_ids) != set(ids):
        raise RuntimeError("Manifest и clusters.csv не содержат одинаковый набор уникальных document_id.")
    matrix = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
    if matrix.ndim != 2 or matrix.shape[0] != len(vector_ids):
        raise RuntimeError(f"Неверная форма vectors.npy: {matrix.shape}")
    row_map = {doc_id: index for index, doc_id in enumerate(vector_ids)}
    X = np.asarray(matrix[[row_map[doc_id] for doc_id in ids]], dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    if not np.isfinite(X).all() or np.any(norms <= 1e-12):
        raise RuntimeError("SPECTER vectors содержат NaN/Inf или нулевые векторы.")
    U = X / norms

    docs["title"] = docs["title"].fillna("").astype(str)
    docs["domain"] = docs["domain"].fillna("").astype(str) if "domain" in docs.columns else ""
    titles = docs["title"].tolist()
    domains = docs["domain"].tolist() if "domain" in docs.columns else [""] * len(docs)
    evidence_bundle = collect_paper_evidence(
        root, docs, ids, source=args.abstract_source,
        openalex_api_key=os.environ.get(args.openalex_api_key_env, ""),
        mailto=args.openalex_mailto,
        openalex_timeout=args.openalex_timeout,
        max_openalex_batches=args.max_openalex_batches,
        max_openalex_title_lookups=args.max_openalex_title_lookups,
        max_crossref_lookups=args.max_crossref_lookups,
        max_crossref_title_lookups=args.max_crossref_title_lookups,
        crossref_fallback=not args.disable_crossref_fallback,
    )
    texts, text_sources, used_text_files = evidence_bundle.texts, evidence_bundle.text_sources, evidence_bundle.used_files
    if len(texts) != len(ids) or len(text_sources) != len(ids):
        raise RuntimeError("Нарушено выравнивание текстов и идентификаторов публикаций.")
    print(
        f"Аннотации: {evidence_bundle.stats['documents_with_abstract']}/{len(ids)}; "
        f"любой текст: {evidence_bundle.stats['documents_with_any_text']}/{len(ids)}; "
        f"полные тексты в data/processed: {evidence_bundle.stats.get('documents_with_full_text', 0)}/{len(ids)}"
    )
    print(
        f"Основной корпус: SQLite {evidence_bundle.stats.get('sqlite_rows_matched', 0)}/{len(ids)}; "
        f"аннотации из SQLite {evidence_bundle.stats.get('sqlite_abstract_records', 0)}; "
        f"processed-текстов найдено {evidence_bundle.stats.get('processed_text_files_found', 0)}; "
        f"аннотаций извлечено из processed {evidence_bundle.stats.get('processed_abstracts_extracted', 0)}"
    )
    print(
        f"Внешние API: OpenAlex batches={evidence_bundle.stats['openalex_batches']}, "
        f"title lookups={evidence_bundle.stats.get('openalex_title_lookups', 0)}; "
        f"Crossref requests={evidence_bundle.stats.get('crossref_requests', 0)}"
    )

    source_cluster_ids = set(pd.to_numeric(source_profiles["cluster_id"], errors="raise").astype(int))
    assignment_cluster_ids = set(np.unique(labels).astype(int).tolist())
    if source_cluster_ids != assignment_cluster_ids:
        raise RuntimeError(
            f"Набор сообществ в профилях и назначениях различается: "
            f"profiles={sorted(source_cluster_ids)}, assignments={sorted(assignment_cluster_ids)}"
        )

    print(f"Групп для именования: {len(source_cluster_ids)}. Перекластеризация не выполняется.")
    new_profiles, _centroids, _representatives, label_audit = build_profiles(
        labels, U, titles, domains, texts, ids, "en",
        labeler=labeler, text_sources=text_sources,
        paper_evidence_by_id=evidence_bundle.by_id,
    )
    labeler_metadata.update(labeler.runtime_stats())

    semantic_columns = [
        "cluster_id", "semantic_name", "semantic_name_en", "semantic_name_ru",
        "semantic_description_ru", "semantic_scope_note_ru", "semantic_research_core_ru",
        "semantic_subtopics_ru", "semantic_methods_ru", "semantic_research_objects_ru",
        "semantic_heterogeneity_level", "semantic_heterogeneity_note_ru", "semantic_supporting_titles",
        "semantic_supporting_document_ids",
        "semantic_review_status", "semantic_review_decision", "semantic_revision_status",
        "semantic_final_validation_status", "semantic_final_validation_decision",
        "semantic_support_level", "semantic_name_support_level", "semantic_profile_support_level",
        "semantic_review_note_ru", "semantic_final_validation_note_ru",
        "semantic_unsupported_name_claims", "semantic_unsupported_profile_claims",
        "semantic_unsupported_claims", "semantic_missing_directions", "semantic_name_evidence_title_count",
        "semantic_evidence_grade", "semantic_abstract_coverage_percent", "semantic_text_coverage_percent",
        "semantic_representative_abstract_coverage_percent", "semantic_top_domain_share_percent",
        "semantic_label_method", "semantic_label_status",
    ]
    updated_profiles = source_profiles.drop(
        columns=[c for c in semantic_columns if c != "cluster_id" and c in source_profiles.columns],
        errors="ignore",
    ).merge(new_profiles[semantic_columns], on="cluster_id", how="left", validate="one_to_one")
    if updated_profiles["semantic_name_ru"].isna().any():
        raise RuntimeError("Не удалось сформировать название для каждого сообщества.")
    # Keep the user-facing display column synchronized with the final reviewed name.
    updated_profiles["display_name"] = updated_profiles["semantic_name_ru"].astype(str)

    name_by_cluster: dict[int, dict[str, str]] = {}
    for row in new_profiles.itertuples(index=False):
        name_by_cluster[int(row.cluster_id)] = {
            "name_en": str(row.semantic_name_en),
            "name_ru": str(row.semantic_name_ru),
            "description_ru": str(row.semantic_description_ru),
            "scope_note_ru": str(row.semantic_scope_note_ru),
            "research_core_ru": str(row.semantic_research_core_ru),
            "subtopics_ru": str(row.semantic_subtopics_ru),
            "methods_ru": str(row.semantic_methods_ru),
            "research_objects_ru": str(row.semantic_research_objects_ru),
            "heterogeneity_level": str(row.semantic_heterogeneity_level),
            "heterogeneity_note_ru": str(row.semantic_heterogeneity_note_ru),
            "supporting_titles": str(row.semantic_supporting_titles),
            "supporting_document_ids": str(row.semantic_supporting_document_ids),
            "review_status": str(row.semantic_review_status),
            "review_decision": str(row.semantic_review_decision),
            "revision_status": str(row.semantic_revision_status),
            "final_validation_status": str(row.semantic_final_validation_status),
            "final_validation_decision": str(row.semantic_final_validation_decision),
            "support_level": str(row.semantic_support_level),
            "name_support_level": str(row.semantic_name_support_level),
            "profile_support_level": str(row.semantic_profile_support_level),
            "review_note_ru": str(row.semantic_review_note_ru),
            "final_validation_note_ru": str(row.semantic_final_validation_note_ru),
            "unsupported_name_claims": str(row.semantic_unsupported_name_claims),
            "unsupported_profile_claims": str(row.semantic_unsupported_profile_claims),
            "unsupported_claims": str(row.semantic_unsupported_claims),
            "name_evidence_title_count": str(row.semantic_name_evidence_title_count),
            "missing_directions": str(row.semantic_missing_directions),
            "evidence_grade": str(row.semantic_evidence_grade),
            "abstract_coverage_percent": str(row.semantic_abstract_coverage_percent),
            "text_coverage_percent": str(row.semantic_text_coverage_percent),
            "representative_abstract_coverage_percent": str(row.semantic_representative_abstract_coverage_percent),
            "top_domain_share_percent": str(row.semantic_top_domain_share_percent),
            "label_method": str(row.semantic_label_method),
            "label_status": str(row.semantic_label_status),
        }

    updated_assignments = update_named_columns(source_assignments, name_by_cluster)
    updated_projection = update_named_columns(
        pd.read_csv(source_dir / "projection_2d.csv", encoding="utf-8-sig"),
        name_by_cluster,
    )
    boundary_path = source_dir / "boundary_documents.csv"
    updated_boundary = (
        update_named_columns(pd.read_csv(boundary_path, encoding="utf-8-sig"), name_by_cluster)
        if boundary_path.is_file() else None
    )

    # Use the source folder as a lexical prefix so Streamlit's reverse name sort
    # reliably lists the relabeled snapshot before the original, regardless of timezone.
    new_name = f"{source_dir.name}_semantic_reviewed"
    output_dir = source_dir.parent / new_name
    if output_dir.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_name = f"{source_dir.name}_semantic_reviewed_{stamp}"
        output_dir = source_dir.parent / new_name
    shutil.copytree(source_dir, output_dir)
    updated_profiles.to_csv(output_dir / "topic_profiles.csv", index=False, encoding="utf-8-sig")
    updated_assignments.to_csv(output_dir / "document_assignments.csv", index=False, encoding="utf-8-sig")
    updated_projection.to_csv(output_dir / "projection_2d.csv", index=False, encoding="utf-8-sig")
    if updated_boundary is not None:
        updated_boundary.to_csv(output_dir / "boundary_documents.csv", index=False, encoding="utf-8-sig")
    paper_evidence_df = pd.DataFrame([
        {
            "document_id": doc_id,
            "abstract_available": bool(item.get("has_abstract", False)),
            "text_available": item.get("source", "title_only") != "title_only",
            "full_text_available": bool(item.get("has_full_text", False)),
            "full_text_path": item.get("full_text_path", ""),
            "text_source": item.get("source", "title_only"),
            "source_detail": item.get("source_detail", ""),
            "abstract": item.get("abstract", ""),
            "text_excerpt": item.get("text_excerpt", ""),
        }
        for doc_id, item in evidence_bundle.by_id.items()
    ])
    paper_evidence_df.to_csv(output_dir / "paper_evidence.csv", index=False, encoding="utf-8-sig")
    (output_dir / "abstract_enrichment_report.json").write_text(
        json.dumps(evidence_bundle.stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    (output_dir / "topic_label_generation.json").write_text(
        json.dumps({
            "labeler": labeler_metadata,
            "source_result": str(source_dir.relative_to(root)),
            "label_records": label_audit,
            "note": "Only semantic metadata was regenerated. Existing community assignments, resolution sweep, graph, embeddings and 2D projection coordinates were preserved.",
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = read_json(output_dir / "summary.json")
    summary["topic_labeler"] = labeler_metadata
    summary["topic_label_generation_file"] = "topic_label_generation.json"
    summary["topic_label_source_result"] = str(source_dir.relative_to(root))
    summary["topic_labeling_only"] = True
    summary["documents_with_abstract_or_text_for_labeling"] = int(evidence_bundle.stats["documents_with_any_text"])
    summary["documents_with_abstract_for_labeling"] = int(evidence_bundle.stats["documents_with_abstract"])
    summary["documents_title_only_for_labeling"] = int(evidence_bundle.stats["documents_title_only"])
    summary["labeling_text_source_files"] = used_text_files
    summary["paper_evidence"] = evidence_bundle.stats
    decision_counts = new_profiles["semantic_review_decision"].fillna("review_required").astype(str).value_counts().to_dict()
    final_decision_counts = new_profiles["semantic_final_validation_decision"].fillna("review_required").astype(str).value_counts().to_dict()
    final_status_counts = new_profiles["semantic_final_validation_status"].fillna("not_run").astype(str).value_counts().to_dict()
    evidence_grade_counts = new_profiles["semantic_evidence_grade"].fillna("unknown").astype(str).value_counts().to_dict()
    heterogeneity_counts = new_profiles["semantic_heterogeneity_level"].fillna("unknown").astype(str).value_counts().to_dict()
    total_docs_for_coverage = max(1, int(new_profiles["documents"].sum()))
    weighted_abstract_coverage = float(
        (new_profiles["semantic_abstract_coverage_percent"] * new_profiles["documents"]).sum()
        / total_docs_for_coverage
    )
    profile_records = new_profiles.to_dict(orient="records")
    summary["semantic_validation"] = {
        "reviewed_groups": int(sum(1 for r in profile_records if r.get("semantic_review_status") == "completed")),
        "groups_accepted_before_final_check": int(decision_counts.get("accepted", 0)),
        "groups_revised_before_final_check": int(decision_counts.get("revised", 0)),
        "groups_needing_review_before_final_check": int(decision_counts.get("review_required", 0)),
        "final_validation_completed_groups": int(final_status_counts.get("completed", 0)),
        "final_validation_accepted_groups": int(final_decision_counts.get("accepted", 0)),
        "final_validation_groups_needing_review": int(final_decision_counts.get("review_required", 0)),
        "groups_with_unsupported_claims": int(sum(1 for r in profile_records if str(r.get("semantic_unsupported_claims", "")).strip())),
        "groups_with_unsupported_name_claims": int(sum(1 for r in profile_records if str(r.get("semantic_unsupported_name_claims", "")).strip())),
        "groups_with_unsupported_profile_claims": int(sum(1 for r in profile_records if str(r.get("semantic_unsupported_profile_claims", "")).strip())),
        "initial_decision_counts": decision_counts,
        "final_validation_decision_counts": final_decision_counts,
        "final_validation_status_counts": final_status_counts,
        "evidence_grade_counts": evidence_grade_counts,
        "heterogeneity_counts": heterogeneity_counts,
        "weighted_abstract_coverage_percent": round(weighted_abstract_coverage, 2),
    }
    summary["labeling_source_sha256"] = {
        "topic_profiles.csv": sha256_file(source_dir / "topic_profiles.csv"),
        "document_assignments.csv": sha256_file(source_dir / "document_assignments.csv"),
        "projection_2d.csv": sha256_file(source_dir / "projection_2d.csv"),
    }
    summary["topic_labeling_runtime_seconds"] = round(time.perf_counter() - started, 3)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report_lines = [
        "# Повторное семантическое именование существующих сообществ",
        "",
        f"- Исходный результат: `{source_dir.relative_to(root)}`",
        f"- Контрольный граф: `{run_id}`",
        f"- Количество сообществ: **{len(source_cluster_ids)}**",
        f"- Публикаций: **{len(ids)}**",
        f"- Модель: **{args.ollama_model}**",
        f"- Digest: `{labeler_metadata.get('model_digest', '')}`",
        f"- Аннотации найдены для: **{summary['documents_with_abstract_for_labeling']}** публикаций",
        f"- Любой дополнительный текст найден для: **{summary['documents_with_abstract_or_text_for_labeling']}** публикаций",
        f"- Второй проход модели завершён: **{summary['semantic_validation']['reviewed_groups']}** групп",
        f"- Первичная редактура: accepted={summary['semantic_validation']['groups_accepted_before_final_check']}, revised={summary['semantic_validation']['groups_revised_before_final_check']}, review_required={summary['semantic_validation']['groups_needing_review_before_final_check']}",
        f"- Финальная независимая проверка: accepted={summary['semantic_validation']['final_validation_accepted_groups']}, review_required={summary['semantic_validation']['final_validation_groups_needing_review']}",
        f"- Взвешенное покрытие аннотациями корпуса: **{summary['semantic_validation']['weighted_abstract_coverage_percent']:.1f}%**",
        "",
        "## Семантические названия",
        "",
        "| ID | Название | Публикаций | Аннотации в группе | Аннотации в выборке | Evidence | Вердикт | Неоднородность |",
        "|---:|---|---:|---:|---:|---|---|---|",
    ]
    doc_counts = source_assignments.groupby("cluster_id").size().to_dict()
    for row in updated_profiles.sort_values("documents", ascending=False).itertuples(index=False):
        def cell(value: Any) -> str:
            return str(value if value is not None else "").replace("|", "\\|").replace("\n", " ").strip() or "—"
        report_lines.append(
            f"| {int(row.cluster_id)} | {cell(row.semantic_name_ru)} | "
            f"{int(doc_counts.get(int(row.cluster_id), 0))} | "
            f"{float(row.semantic_abstract_coverage_percent):.1f}% | "
            f"{float(row.semantic_representative_abstract_coverage_percent):.1f}% | "
            f"{cell(row.semantic_evidence_grade)} | {cell(row.semantic_review_decision)} | "
            f"{cell(row.semantic_heterogeneity_level)} |"
        )
    report_lines.extend([
        "",
        "## Что было изменено",
        "",
        "Перегенерированы только русские/английские названия, описания и примечания о неоднородности. "
        "Назначения публикаций, выбранный resolution, resolution_sweep.csv, исходный граф, "
        "SPECTER-векторы и координаты UMAP/PCA не перестраивались.",
        "",
        "Черновик, полная редакция, начальные замечания и независимая финальная проверка сохранены в `topic_label_generation.json`. "
        "Аннотации и происхождение текстов — в `paper_evidence.csv` и `abstract_enrichment_report.json`. "
        "Исходный каталог результата не изменён.",
    ])
    (output_dir / "REPORT_LLM_LABELS_RU.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    latest_path = source_dir.parent / "latest.json"
    latest_path.write_text(
        json.dumps({"output_dir": new_name, "path": str(output_dir.relative_to(root))}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nГОТОВО: {output_dir}")
    runtime = labeler.runtime_stats().get("ollama_runtime", {})
    print(
        "Ollama JSON: "
        f"requests={runtime.get('requests', 0)}, retries={runtime.get('retries', 0)}, "
        f"invalid={runtime.get('invalid_json_responses', 0)}"
    )
    print("Перекластеризация: НЕТ")
    print(f"Сообществ с новыми семантическими метками: {len(source_cluster_ids)}")
    print(f"Публикаций: {len(ids)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(2)
