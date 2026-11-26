# -*- coding: utf-8 -*-
"""Streamlit explorer for dynamically discovered scientific-paper topics."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Scientific Clustering · Тематические группы", page_icon="🧭", layout="wide")

ROOT = Path(os.environ.get("SIL_PROJECT_ROOT", Path(__file__).resolve().parents[3])).resolve()
REPORTS = ROOT / "reports" / "auto_topic_discovery"


def discover_runs() -> list[Path]:
    if not REPORTS.exists():
        return []
    paths = list(REPORTS.glob("*/*/topic_profiles.csv"))
    return sorted([p.parent for p in paths], key=lambda p: p.name, reverse=True)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def load_overrides(path: Path) -> dict[str, str]:
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return {str(k): str(v) for k, v in payload.items()}
        except Exception:
            return {}
    return {}


def save_overrides(path: Path, values: dict[str, str]) -> None:
    path.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")


def metric_value(summary: dict[str, Any], key: str, default: str = "—") -> Any:
    value = summary.get(key, default)
    if isinstance(value, float):
        return f"{value:.3f}"
    return value


st.title("🧭 Тематическая карта научных публикаций")
st.caption("Leiden автоматически определяет число сообществ. Локальная instruction-модель интерпретирует доказательную выборку публикаций, формирует названия и подтемы, а затем выполняет отдельную критическую проверку. Кластерные назначения не изменяются.")

runs = discover_runs()
if not runs:
    st.error("Не найдены результаты автоматического тематического обнаружения.")
    st.code("python -m sciintel.semantic.auto_topic_discovery --root . --run-id latest", language="powershell")
    st.stop()

run_labels = [f"{p.parent.name} · {p.name}" for p in runs]
selected_label = st.sidebar.selectbox("Запуск анализа", run_labels, index=0)
out_dir = runs[run_labels.index(selected_label)]

try:
    profiles = read_csv(out_dir / "topic_profiles.csv")
    assignments = read_csv(out_dir / "document_assignments.csv")
    projection = read_csv(out_dir / "projection_2d.csv")
    sweep = read_csv(out_dir / "resolution_sweep.csv")
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    paper_evidence_path = out_dir / "paper_evidence.csv"
    paper_evidence = read_csv(paper_evidence_path) if paper_evidence_path.is_file() else pd.DataFrame()
except Exception as exc:
    st.error(f"Не удалось прочитать результат выбранного запуска: {exc}")
    st.stop()

if profiles.empty or assignments.empty or projection.empty:
    st.error("В выбранном запуске отсутствуют данные для визуализации.")
    st.stop()

profiles["cluster_id"] = pd.to_numeric(profiles["cluster_id"], errors="coerce").astype(int)
assignments["cluster_id"] = pd.to_numeric(assignments["cluster_id"], errors="coerce").astype(int)
projection["cluster_id"] = pd.to_numeric(projection["cluster_id"], errors="coerce").astype(int)
overrides_path = out_dir / "topic_label_overrides.json"
overrides = load_overrides(overrides_path)
profiles["display_name"] = profiles.apply(
    lambda r: overrides.get(str(int(r.cluster_id)), str(r.get("semantic_name_ru") or r.get("semantic_name") or f"Тема {int(r.cluster_id)+1}")),
    axis=1,
)
name_map = dict(zip(profiles.cluster_id.astype(int), profiles.display_name.astype(str)))
projection = projection.merge(profiles[["cluster_id", "display_name"]], on="cluster_id", how="left", suffixes=("", "_profile"))
projection["display_name"] = projection["display_name"].fillna(projection["semantic_name_ru"]).fillna(projection["semantic_name"])
assignments["display_name"] = assignments["cluster_id"].map(name_map)

st.sidebar.markdown("---")
st.sidebar.caption(f"Результат: `{out_dir.relative_to(ROOT)}`")
st.sidebar.caption(f"Входной граф: `{summary.get('run_id', '—')}`")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Публикации", f"{int(summary.get('documents', len(assignments))):,}")
c2.metric("Обнаруженные темы", int(summary.get("detected_communities", len(profiles))))
c3.metric("Resolution", f"{float(summary.get('chosen_resolution', 0)):.3f}")
c4.metric("Устойчивость ARI", f"{float(summary.get('selected_candidate_metrics', {}).get('stability_ari_mean', 0)):.3f}")
c5.metric("Silhouette", f"{float(summary.get('selected_candidate_metrics', {}).get('silhouette_mean', 0)):.3f}")

with st.expander("Методика автоматического выбора числа тем и именования", expanded=False):
    st.write("Число сообществ не задаётся вручную. Для набора значений resolution выполняются повторные запуски Leiden с разными seed. Выбирается вариант по устойчивости ARI и cosine silhouette в исходном SPECTER-пространстве с проверками вырожденных разбиений.")
    labeler_info = summary.get("topic_labeler", {})
    st.write(f"Модель семантического именования: {labeler_info.get('model') or 'эвристика ключевых фраз'}")
    st.write(f"Провайдер: {labeler_info.get('provider', '—')}; digest модели: {labeler_info.get('model_digest', '—')}")
    st.caption("Qwen не формирует SPECTER-векторы, не меняет рёбра графа, resolution или назначения публикаций. Для каждой темы выполняются три этапа: генерация, полная evidence-редактура профиля и независимая финальная проверка уже исправленного результата.")
    st.code(summary.get("selection_formula", "Формула отсутствует в summary.json"))
    st.caption(summary.get("selection_note", ""))
    st.write(f"Аннотации: {summary.get('documents_with_abstract', summary.get('documents_with_abstract_for_labeling', 0))}; дополнительный текст: {summary.get('documents_with_abstract_or_text', summary.get('documents_with_abstract_or_text_for_labeling', 0))}; только заголовки: {summary.get('documents_title_only', summary.get('documents_title_only_for_labeling', 0))}.")
    evidence_stats = summary.get("paper_evidence", {})
    st.write(
        f"Источники: локальные файлы; OpenAlex ({evidence_stats.get('openalex_batches', 0)} пакетных запросов, "
        f"{evidence_stats.get('openalex_title_lookups', 0)} поисков по заголовку); "
        f"Crossref ({evidence_stats.get('crossref_requests', 0)} DOI-запросов)."
    )
    st.write(f"YAKE доступен: {summary.get('yake_available', False)}. RAKE: {summary.get('rake_available', False)}. Проекция: {summary.get('projection_method', '—')}.")

st.subheader("Контроль качества семантических меток")
qa_cols = st.columns(5)
reviewed_mask = profiles.get(
    "semantic_final_validation_status",
    profiles.get("semantic_review_status", pd.Series(["not_run"] * len(profiles))),
).astype(str) == "completed"
final_decision_series = profiles.get(
    "semantic_final_validation_decision",
    profiles.get("semantic_review_decision", pd.Series(["review_required"] * len(profiles))),
).astype(str)
initial_decision_series = profiles.get("semantic_review_decision", pd.Series(["review_required"] * len(profiles))).astype(str)
accepted_mask = final_decision_series == "accepted"
revised_mask = profiles.get("semantic_revision_status", initial_decision_series).astype(str) == "revised"
needs_review_mask = final_decision_series == "review_required"
abstract_col = pd.to_numeric(profiles.get("semantic_abstract_coverage_percent", pd.Series([0] * len(profiles))), errors="coerce").fillna(0)
document_weights = pd.to_numeric(profiles.get("documents", pd.Series([1] * len(profiles))), errors="coerce").fillna(0).clip(lower=0)
weighted_coverage = float((abstract_col * document_weights).sum() / max(1.0, float(document_weights.sum())))
qa_cols[0].metric("Финально проверено", f"{int(reviewed_mask.sum())}/{len(profiles)}")
qa_cols[1].metric("Принято финальной проверкой", int(accepted_mask.sum()))
qa_cols[2].metric("Профиль исправлялся", int(revised_mask.sum()))
qa_cols[3].metric("Нужна ручная проверка", int(needs_review_mask.sum()))
qa_cols[4].metric("Покрытие аннотациями", f"{weighted_coverage:.1f}%")

qa_table_cols = [c for c in [
    "cluster_id", "semantic_name_ru", "semantic_name_support_level", "semantic_profile_support_level",
    "semantic_review_decision", "semantic_final_validation_decision", "semantic_evidence_grade",
    "semantic_abstract_coverage_percent", "semantic_heterogeneity_level",
    "semantic_unsupported_name_claims", "semantic_unsupported_profile_claims",
    "semantic_missing_directions", "semantic_supporting_titles",
] if c in profiles.columns]
if qa_table_cols:
    qa_view = profiles[qa_table_cols].copy()
    qa_view = qa_view.rename(columns={
        "cluster_id": "ID", "semantic_name_ru": "Тема", "semantic_name_support_level": "Поддержка названия",
        "semantic_profile_support_level": "Поддержка профиля", "semantic_review_decision": "Решение редактора",
        "semantic_final_validation_decision": "Финальный вердикт", "semantic_evidence_grade": "Полнота evidence",
        "semantic_abstract_coverage_percent": "Аннотации, %", "semantic_heterogeneity_level": "Неоднородность",
        "semantic_unsupported_name_claims": "Неподтверждённое в названии",
        "semantic_unsupported_profile_claims": "Неподтверждённое в профиле",
        "semantic_missing_directions": "Пропущенные направления",
        "semantic_supporting_titles": "Публикации для проверки",
    })
    st.dataframe(qa_view, use_container_width=True, hide_index=True)
with st.expander("Как читать автоматическую проверку"):
    st.markdown(
        "- **Поддержка названия** — отдельная оценка того, насколько формулировка названия подтверждается представительными публикациями. Это экспертная оценка модели, не статистическая вероятность.\n"
        "- **Поддержка профиля** — оценка описания, исследовательского ядра, подтем, методов и объектов.\n"
        "- **Финальный вердикт** относится к профилю после внесения исправлений и третьего независимого прохода. `review_required` означает, что автоматическое принятие не подтверждено.\n"
        "- **Полнота evidence** описывает доступность аннотаций/текстовых источников, а не качество названия. Высокое покрытие само по себе не означает, что название верно.\n"
        "- **Публикации для проверки** — точные заголовки из корпуса, на которые модель опиралась при подготовке названия.\n"
        "- Внешние API получают DOI/OpenAlex ID; для записей без таких идентификаторов допускается ограниченный поиск только по заголовку. Полные тексты не отправляются."
    )

st.subheader("Интерактивная карта публикаций")
filter_col1, filter_col2, filter_col3 = st.columns([2, 1, 1])
cluster_choices = ["Все темы"] + [f"{int(r.cluster_id)} · {r.display_name} ({int(r.documents)})" for r in profiles.sort_values("documents", ascending=False).itertuples()]
with filter_col1:
    selected_topic = st.selectbox("Фокус на теме", cluster_choices)
with filter_col2:
    domain_values = sorted(x for x in assignments.domain.dropna().astype(str).unique() if x and x.lower() != "nan") if "domain" in assignments else []
    selected_domain = st.selectbox("Домен", ["Все домены"] + domain_values)
with filter_col3:
    color_options = ["Тематические группы"] + (["Домены"] if domain_values else [])
    color_mode = st.selectbox("Раскраска", color_options)

map_df = projection.copy()
# Attach domain metadata before applying filters.
if "domain" not in map_df.columns and "domain" in assignments.columns:
    map_df = map_df.merge(assignments[["document_id", "domain"]], on="document_id", how="left")
if selected_topic != "Все темы":
    selected_id = int(selected_topic.split(" · ", 1)[0])
    map_df = map_df[map_df.cluster_id == selected_id]
if selected_domain != "Все домены" and "domain" in map_df.columns:
    map_df = map_df[map_df.domain.astype(str) == selected_domain]
if color_mode == "Домены" and "domain" in map_df.columns:
    color_col = "domain"
else:
    color_col = "display_name"
if not map_df.empty:
    hover_cols = [c for c in ["document_id", "title", "domain", "cluster_id"] if c in map_df.columns]
    fig = px.scatter(
        map_df, x="x", y="y", color=color_col, hover_name="title" if "title" in map_df else None,
        hover_data=hover_cols, opacity=0.78,
        labels={"x": "Ось 1", "y": "Ось 2", "display_name": "Тематическая группа", "domain": "Домен"},
        title=f"Публикаций на карте: {len(map_df):,}",
    )
    fig.update_traces(marker={"size": 7})
    fig.update_layout(height=690, legend_title_text="Группы", margin={"l": 10, "r": 10, "t": 55, "b": 10})
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Для выбранных фильтров публикаций нет.")
st.caption("Координаты UMAP/PCA используются только для визуального расположения точек. Метрики качества рассчитаны в исходном SPECTER-пространстве и на графе, а не по координатам карты.")

st.subheader("Каталог автоматически обнаруженных тем")
view_profiles = profiles.sort_values("documents", ascending=False).copy()
cols = st.columns(3)
for pos, row in enumerate(view_profiles.itertuples(index=False)):
    with cols[pos % 3]:
        with st.container(border=True):
            st.markdown(f"**{row.display_name}**")
            st.caption(f"Сообщество {int(row.cluster_id)} · {int(row.documents):,} публикаций · {float(row.share_percent):.1f}% корпуса")
            description = getattr(row, "semantic_description_ru", "")
            if pd.notna(description) and str(description).strip():
                st.write(str(description))
            scope_note = getattr(row, "semantic_scope_note_ru", "")
            if pd.notna(scope_note) and str(scope_note).strip():
                st.warning(str(scope_note))
            research_core = getattr(row, "semantic_research_core_ru", "")
            if pd.notna(research_core) and str(research_core).strip():
                st.write("**Исследовательское ядро:** " + str(research_core))
            support = getattr(row, "semantic_name_support_level", getattr(row, "semantic_support_level", ""))
            profile_support = getattr(row, "semantic_profile_support_level", "")
            review_decision = getattr(row, "semantic_final_validation_decision", getattr(row, "semantic_review_decision", ""))
            evidence_grade = getattr(row, "semantic_evidence_grade", "")
            heterogeneity = getattr(row, "semantic_heterogeneity_level", "")
            abstract_coverage = getattr(row, "semantic_abstract_coverage_percent", 0)
            try:
                abstract_coverage = float(abstract_coverage) if pd.notna(abstract_coverage) else 0.0
            except (TypeError, ValueError):
                abstract_coverage = 0.0
            st.caption(f"Название: {support or '—'} · профиль: {profile_support or '—'} · финальный вердикт: {review_decision or '—'} · полнота evidence: {evidence_grade or '—'} · аннотации: {abstract_coverage:.1f}% · неоднородность: {heterogeneity or '—'}")
            subtopics = getattr(row, "semantic_subtopics_ru", "")
            if pd.notna(subtopics) and str(subtopics).strip():
                st.write("**Подтемы:** " + str(subtopics))
            methods = getattr(row, "semantic_methods_ru", "")
            if pd.notna(methods) and str(methods).strip():
                st.write("**Методы:** " + str(methods))
            heterogeneity_note = getattr(row, "semantic_heterogeneity_note_ru", "")
            if pd.notna(heterogeneity_note) and str(heterogeneity_note).strip():
                st.caption("Неоднородность: " + str(heterogeneity_note))
            review_note = getattr(row, "semantic_review_note_ru", "")
            if pd.notna(review_note) and str(review_note).strip():
                st.warning(str(review_note))
            unsupported_name = getattr(row, "semantic_unsupported_name_claims", "")
            unsupported_profile = getattr(row, "semantic_unsupported_profile_claims", getattr(row, "semantic_unsupported_claims", ""))
            if pd.notna(unsupported_name) and str(unsupported_name).strip():
                st.error("Неподтверждённое в названии: " + str(unsupported_name))
            if pd.notna(unsupported_profile) and str(unsupported_profile).strip():
                st.warning("Неподтверждённое в профиле: " + str(unsupported_profile))
            st.write(str(row.keywords)[:420] if pd.notna(row.keywords) else "Ключевые термины не извлечены")
            if pd.notna(row.dominant_domains) and row.dominant_domains:
                st.caption(f"Домены: {row.dominant_domains}")

st.subheader("Подробности выбранной темы")
profile_options = [int(x) for x in profiles.cluster_id.tolist()]
selected_cluster = st.selectbox(
    "Сообщество для изучения",
    profile_options,
    format_func=lambda cid: f"{cid} · {name_map.get(cid, f'Тема {cid+1}')} · {int(profiles.loc[profiles.cluster_id == cid, 'documents'].iloc[0])} публикаций",
)
selected_profile = profiles.loc[profiles.cluster_id == selected_cluster].iloc[0]
selected_docs = assignments[assignments.cluster_id == selected_cluster].copy()
left, right = st.columns([1, 2])
with left:
    st.metric("Публикаций в теме", int(selected_profile.documents))
    if "semantic_description_ru" in selected_profile and pd.notna(selected_profile.semantic_description_ru):
        st.write("**Описание**")
        st.write(str(selected_profile.semantic_description_ru))
    if "semantic_scope_note_ru" in selected_profile and pd.notna(selected_profile.semantic_scope_note_ru) and str(selected_profile.semantic_scope_note_ru).strip():
        st.warning(str(selected_profile.semantic_scope_note_ru))
    if "semantic_research_core_ru" in selected_profile and pd.notna(selected_profile.semantic_research_core_ru) and str(selected_profile.semantic_research_core_ru).strip():
        st.write("**Общее исследовательское ядро**")
        st.write(str(selected_profile.semantic_research_core_ru))
    for label, column in (("Подтемы", "semantic_subtopics_ru"), ("Методы", "semantic_methods_ru"), ("Объекты исследования", "semantic_research_objects_ru")):
        value = selected_profile.get(column, "")
        if pd.notna(value) and str(value).strip():
            st.write(f"**{label}**")
            st.write(str(value))
    selected_abstract_coverage = selected_profile.get("semantic_abstract_coverage_percent", 0)
    try:
        selected_abstract_coverage = float(selected_abstract_coverage) if pd.notna(selected_abstract_coverage) else 0.0
    except (TypeError, ValueError):
        selected_abstract_coverage = 0.0
    st.write(f"**Покрытие аннотациями:** {selected_abstract_coverage:.1f}%")
    representative_coverage = selected_profile.get("semantic_representative_abstract_coverage_percent", 0)
    try:
        representative_coverage = float(representative_coverage) if pd.notna(representative_coverage) else 0.0
    except (TypeError, ValueError):
        representative_coverage = 0.0
    st.write(f"**Покрытие аннотациями репрезентативной выборки:** {representative_coverage:.1f}%")
    st.write(
        f"**Поддержка названия:** {selected_profile.get('semantic_name_support_level', selected_profile.get('semantic_support_level', '—'))} · "
        f"**Поддержка профиля:** {selected_profile.get('semantic_profile_support_level', '—')} · "
        f"**Решение редактора:** {selected_profile.get('semantic_review_decision', '—')} · "
        f"**Финальный вердикт:** {selected_profile.get('semantic_final_validation_decision', selected_profile.get('semantic_review_decision', '—'))} · "
        f"**Полнота evidence:** {selected_profile.get('semantic_evidence_grade', '—')}"
    )
    heterogeneity_note = selected_profile.get("semantic_heterogeneity_note_ru", "")
    if pd.notna(heterogeneity_note) and str(heterogeneity_note).strip():
        st.info("**Тематическая неоднородность:** " + str(heterogeneity_note))
    if str(selected_profile.get("semantic_unsupported_name_claims", "")).strip():
        st.error("Неподтверждённые утверждения в названии: " + str(selected_profile.semantic_unsupported_name_claims))
    if str(selected_profile.get("semantic_unsupported_profile_claims", selected_profile.get("semantic_unsupported_claims", ""))).strip():
        st.warning("Неподтверждённые утверждения в профиле: " + str(selected_profile.get("semantic_unsupported_profile_claims", selected_profile.get("semantic_unsupported_claims", ""))))
    final_note = selected_profile.get("semantic_final_validation_note_ru", "")
    if pd.notna(final_note) and str(final_note).strip():
        st.caption("Финальная проверка: " + str(final_note))
    if str(selected_profile.get("semantic_missing_directions", "")).strip():
        st.warning("Возможные пропущенные направления: " + str(selected_profile.semantic_missing_directions))
    if str(selected_profile.get("semantic_supporting_titles", "")).strip():
        supporting_titles = str(selected_profile.semantic_supporting_titles).split("; ")
        supporting_ids = str(selected_profile.get("semantic_supporting_document_ids", "")).split("; ")
        with st.expander("Публикации, которыми модель обосновала название"):
            for index, supporting_title in enumerate(supporting_titles):
                document_id = supporting_ids[index] if index < len(supporting_ids) else ""
                suffix = f" · ID: `{document_id}`" if document_id else ""
                st.markdown(f"- {supporting_title}{suffix}")
    st.caption(f"Способ именования: {selected_profile.get('semantic_label_method', '—')}")
    st.write("**Ключевые фразы**")
    st.write(selected_profile.keywords or "—")
    st.write("**YAKE**")
    st.write(selected_profile.keywords_yake or "YAKE не вернул термины")
    st.write("**RAKE**")
    st.write(selected_profile.keywords_rake or "—")
    st.write("**Контрастный TF-IDF**")
    st.write(selected_profile.keywords_tfidf_contrast or "—")
    st.write("**Среднее сходство с центроидом**")
    st.write(f"{float(selected_profile.mean_cosine_to_centroid):.4f}")
with right:
    st.write("**Наиболее репрезентативные публикации**")
    evidence_by_doc = paper_evidence.set_index("document_id").to_dict(orient="index") if not paper_evidence.empty and "document_id" in paper_evidence.columns else {}
    for i, paper in enumerate(selected_docs.sort_values("cosine_to_own_centroid", ascending=False).head(12).itertuples(), start=1):
        st.markdown(f"{i}. **{paper.title}**")
        st.caption(f"{paper.document_id} · {paper.domain} · cosine={float(paper.cosine_to_own_centroid):.3f}")
        item = evidence_by_doc.get(str(paper.document_id), {})
        excerpt = str(item.get("abstract") or item.get("text_excerpt") or "").strip()
        if excerpt:
            with st.expander(f"Аннотация / текстовая выдержка · источник: {item.get('text_source', '—')}"):
                st.write(excerpt[:3500])

st.subheader("Редактирование семантического названия")
with st.form("rename_topic"):
    current_name = overrides.get(str(selected_cluster), str(selected_profile.semantic_name_ru or selected_profile.semantic_name))
    new_name = st.text_input("Название темы", value=current_name, max_chars=120)
    submitted = st.form_submit_button("Сохранить название")
    if submitted:
        overrides[str(selected_cluster)] = new_name.strip() or current_name
        save_overrides(overrides_path, overrides)
        st.success("Название сохранено для этого запуска анализа.")
        st.rerun()

with st.expander("Пограничные публикации"):
    boundary_path = out_dir / "boundary_documents.csv"
    if boundary_path.exists():
        boundary = read_csv(boundary_path)
        if "cluster_id" in boundary:
            boundary = boundary[boundary.cluster_id == selected_cluster]
        st.dataframe(boundary.head(50), use_container_width=True, hide_index=True)

with st.expander("Автоматический выбор resolution"):
    display_sweep = sweep.sort_values("selection_score", ascending=False)
    st.dataframe(display_sweep, use_container_width=True, hide_index=True)
    if {"resolution", "selection_score", "stability_ari_mean", "silhouette_mean", "modularity_mean"}.issubset(sweep.columns):
        chart_df = sweep.melt(id_vars="resolution", value_vars=["selection_score", "stability_ari_mean", "silhouette_mean"], var_name="metric", value_name="score")
        fig_sweep = px.line(chart_df, x="resolution", y="score", color="metric", markers=True, title="Поведение критериев при изменении resolution")
        fig_sweep.update_layout(height=360)
        st.plotly_chart(fig_sweep, use_container_width=True)

st.subheader("Экспорт данных")
d1, d2, d3, d4 = st.columns(4)
d1.download_button("Скачать назначения CSV", assignments.to_csv(index=False).encode("utf-8-sig"), "document_assignments.csv", "text/csv", use_container_width=True)
d2.download_button("Скачать профили тем CSV", profiles.to_csv(index=False).encode("utf-8-sig"), "topic_profiles.csv", "text/csv", use_container_width=True)
d3.download_button("Скачать summary JSON", json.dumps(summary, ensure_ascii=False, indent=2).encode("utf-8"), "summary.json", "application/json", use_container_width=True)
if not paper_evidence.empty:
    d4.download_button("Скачать аннотации CSV", paper_evidence.to_csv(index=False).encode("utf-8-sig"), "paper_evidence.csv", "text/csv", use_container_width=True)
else:
    d4.caption("Файл аннотаций отсутствует в этом запуске")

st.caption("Автоматические названия и их критическая проверка являются модельной интерпретацией, а не ground truth. При низком покрытии аннотациями проверяйте оригинальные публикации.")
