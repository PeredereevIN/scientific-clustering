from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sciintel import db  # noqa: E402
from sciintel.config import settings  # noqa: E402
from sciintel.topic_naming import translate_term  # noqa: E402

st.set_page_config(page_title="Scientific Clustering", page_icon="◈", layout="wide",
                   initial_sidebar_state="expanded")

# Calm, high-contrast design system: neutral surfaces + one cool primary + restrained warm accent.
PALETTE = ["#356B8C", "#D17A48", "#3A8D78", "#8064A2", "#C4A23A", "#5277B8",
           "#B85C72", "#6B8795", "#8A9B43", "#A66D3F", "#5C6B8A", "#2F8F9D",
           "#A45D9A", "#778899", "#C17C9C", "#557C55", "#B59B45", "#6574A6",
           "#C06C4D", "#4F9B8C", "#8B79B5", "#A0A0A0"]
NOISE = "#A8B0BA"

st.markdown("""
<style>
:root { --sil-ink:#182B3A; --sil-muted:#617384; --sil-line:#E2E8EE; --sil-bg:#F4F7F9;
        --sil-primary:#356B8C; --sil-teal:#3A8D78; --sil-warm:#D17A48; }
.stApp { background:var(--sil-bg); color:var(--sil-ink); }
.block-container { max-width: 1660px; padding-top: 1.15rem; padding-bottom: 2.4rem; }
[data-testid="stSidebar"] { background:#EDF2F5; border-right:1px solid var(--sil-line); }
[data-testid="stSidebar"] h1 { letter-spacing:-.04em; color:var(--sil-ink); }
[data-testid="stMetric"] { background:#FFFFFF; border:1px solid var(--sil-line); padding:15px 17px;
  border-radius:14px; box-shadow:0 2px 8px rgba(24,43,58,.035); }
[data-testid="stMetricLabel"] { color:var(--sil-muted); font-weight:600; }
[data-testid="stMetricValue"] { color:var(--sil-ink); font-weight:750; }
.stButton > button[kind="primary"] { background:var(--sil-primary); border-color:var(--sil-primary); }
.stButton > button { border-radius:10px; font-weight:650; }
[data-testid="stTabs"] button { font-weight:650; }
.sil-eyebrow { color:var(--sil-teal); font-size:.72rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; }
.sil-title { color:var(--sil-ink); font-size:2.05rem; font-weight:780; letter-spacing:-.045em; margin:.1rem 0 .2rem; }
.sil-subtitle { color:var(--sil-muted); font-size:.95rem; margin-bottom:1.15rem; }
.sil-panel { background:#fff; border:1px solid var(--sil-line); border-radius:15px; padding:16px 18px; }
.sil-note { color:var(--sil-muted); font-size:.83rem; }
hr { border-color:var(--sil-line); }
</style>
""", unsafe_allow_html=True)


def load_latest():
    run = db.latest_run()
    if not run:
        return None, pd.DataFrame(), pd.DataFrame(), None
    out = settings.artifacts / "runs" / run["run_id"]
    cp, pp, mp = out / "clusters.csv", out / "cluster_profiles.csv", out / "metrics.json"
    clusters = pd.read_csv(cp) if cp.exists() else pd.DataFrame()
    profiles = pd.read_csv(pp) if pp.exists() else pd.DataFrame()
    if mp.exists():
        run["metrics"] = json.loads(mp.read_text(encoding="utf-8"))
    return run, clusters, profiles, out


def gpu_snapshot():
    try:
        cmd = ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
               "--format=csv,noheader,nounits"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3, check=False)
        if result.returncode == 0 and result.stdout.strip():
            return [x.strip() for x in result.stdout.strip().splitlines()[0].split(",")]
    except Exception:
        pass
    return None


def load_docs():
    rows = db.rows("SELECT id AS document_id,title,abstract,domain,year,doi,path,text_path,char_count,page_count,status FROM documents ORDER BY title")
    return pd.DataFrame(rows)


def load_semantic_profile_data(run_dir: Path | None):
    """Load topic interpretation artifacts tied to this exact clustering run."""
    empty = (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {})
    if run_dir is None:
        return empty
    folder = run_dir / "semantic_profiles"
    paths = {
        "profiles": folder / "group_topic_profiles.csv",
        "documents": folder / "document_topic_analysis.csv",
        "subtopics": folder / "internal_subtopics.csv",
        "candidates": folder / "topic_candidates.csv",
        "metadata": folder / "topic_understanding_metadata.json",
    }
    def safe_read_csv(path: Path) -> pd.DataFrame:
        """An existing but zero-byte CSV is a valid empty artifact, not a crash."""
        if not path.exists():
            return pd.DataFrame()
        try:
            if path.stat().st_size == 0:
                return pd.DataFrame()
            return pd.read_csv(path, encoding="utf-8-sig")
        except (pd.errors.EmptyDataError, OSError, UnicodeDecodeError):
            return pd.DataFrame()

    frames = []
    for key in ("profiles", "documents", "subtopics", "candidates"):
        frames.append(safe_read_csv(paths[key]))
    metadata = {}
    if paths["metadata"].exists():
        try:
            metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}
    return (*frames, metadata)


def _first_text(row: dict, *keys: str, default: str = "") -> str:
    """Return the first non-empty scalar value, treating CSV NaN as missing."""
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        value = str(value).strip()
        if value and value.casefold() not in {"nan", "none", "null"}:
            return value
    return default


PROFILE_STATUS_RU = {
    "STRONG_PROFILE": "Устойчивый тематический профиль",
    "MODERATE_PROFILE": "Тема подтверждается умеренно",
    "REVIEW_PROFILE": "Недостаточно свидетельств — нужна проверка",
    "MIXED_TOPIC_REVIEW": "Смешанная тематика — нужна проверка",
    "INTERNAL_SUBTOPICS_DETECTED": "Внутри сообщества выявлены подтемы",
}


def make_semantic_label_map(topic_profiles: pd.DataFrame, base_profiles: pd.DataFrame) -> dict[int, str]:
    """Prefer validated semantic titles; otherwise show explicitly preliminary phrase labels."""
    result: dict[int, str] = {}
    if not topic_profiles.empty and {"group_id", "automatic_label"}.issubset(topic_profiles.columns):
        for row in topic_profiles.to_dict("records"):
            try:
                group_id = int(str(row["group_id"]))
            except (TypeError, ValueError):
                continue
            label = _first_text(row, "automatic_label_ru", "automatic_label_en", "automatic_label")
            if label and label.lower() not in {"nan", "none"}:
                result[group_id] = label
    if not base_profiles.empty and {"cluster_id", "top_terms"}.issubset(base_profiles.columns):
        for row in base_profiles.to_dict("records"):
            try:
                group_id = int(row["cluster_id"])
            except (TypeError, ValueError):
                continue
            if group_id in result:
                continue
            terms = [x.strip() for x in str(row.get("top_terms") or "").split("|") if x.strip()]
            translated = [translate_term(x) for x in terms[:2]]
            result[group_id] = "Предварительно: " + ", ".join(translated) if translated else "Тема ещё не определена"
    return result


def show_semantic_group_cards(topic_profiles: pd.DataFrame, topic_documents: pd.DataFrame,
                              subtopics: pd.DataFrame, source_docs: pd.DataFrame,
                              max_groups: int | None = None):
    """Render human-readable group cards with evidence and representative papers."""
    if topic_profiles.empty:
        st.info("Семантические названия ещё не сформированы. Запусти анализ тематических профилей выше.")
        return
    frame = topic_profiles.copy()
    if "document_count" in frame:
        frame = frame.sort_values("document_count", ascending=False)
    if max_groups is not None:
        frame = frame.head(max_groups)
    for row in frame.to_dict("records"):
        group_id = str(row.get("group_id", "?"))
        label = _first_text(row, "automatic_label_ru", "automatic_label", default="Тема не определена")
        status = str(row.get("topic_profile_status") or "REVIEW_PROFILE")
        count = int(row.get("document_count") or 0)
        with st.container(border=True):
            st.markdown(f"### {label}")
            st.caption(f"{PROFILE_STATUS_RU.get(status, status)} · {count:,} публикаций · технический ID сообщества {group_id}")
            description = _first_text(row, "automatic_description_ru", "automatic_description")
            if description:
                st.write(description)
            aspects_source = _first_text(row, "topic_aspects_ru", "topic_aspects")
            aspects = [x.strip() for x in aspects_source.split("|") if x.strip()]
            if aspects:
                st.markdown("**Подтверждённые тематические аспекты:** " + " · ".join(aspects))
            c1, c2, c3 = st.columns(3)
            c1.metric("Поддержка основной темы", f"{float(row.get('primary_topic_support') or 0):.1%}")
            c2.metric("Сходство с центроидом", f"{float(row.get('mean_title_abstract_centroid_similarity') or 0):.3f}")
            c3.metric("Индекс подтверждения", f"{float(row.get('topic_profile_score') or 0):.3f}")
            st.caption("Индекс подтверждения — эвристическая сводка свидетельств, а не вероятность правильности названия.")
            confidence = _first_text(row, "naming_confidence")
            if confidence:
                confidence_score = row.get("naming_confidence_score")
                score_text = (
                    f" (эвристическая оценка {float(confidence_score):.2f}/1)"
                    if confidence_score is not None and pd.notna(confidence_score) else ""
                )
                st.caption(
                    f"Уверенность в названии: {confidence.lower()}{score_text}. "
                    "Это оценка терминологических свидетельств, а не вероятность правильности "
                    "и не показатель однородности сообщества."
                )

            if not topic_documents.empty and "group_id" in topic_documents.columns and "topic_label" in source_docs.columns:
                member_ids = topic_documents.loc[topic_documents["group_id"].astype(str) == group_id, "document_id"].astype(str)
                source_mix = source_docs[source_docs["document_id"].astype(str).isin(set(member_ids))]["topic_label"].value_counts(normalize=True).head(3)
                if not source_mix.empty:
                    mix_text = " · ".join(f"{name}: {share:.0%}" for name, share in source_mix.items())
                    st.caption("Состав по исходным категориям arXiv (вспомогательная проверка): " + mix_text)

            if not topic_documents.empty and "group_id" in topic_documents.columns:
                reps = topic_documents[topic_documents["group_id"].astype(str) == group_id].copy()
                if not reps.empty:
                    reps = reps.sort_values("centroid_similarity", ascending=False).head(4)
                    extra_cols = [c for c in ["document_id", "domain", "year", "doi", "abstract"] if c in source_docs.columns]
                    if extra_cols and "document_id" in source_docs.columns:
                        reps = reps.merge(source_docs[extra_cols].drop_duplicates("document_id"), on="document_id", how="left")
                    st.markdown("**Репрезентативные публикации**")
                    for i, paper in enumerate(reps.to_dict("records"), 1):
                        title = str(paper.get("title") or "Без названия")
                        meta = [str(paper.get(k)) for k in ("domain", "year") if pd.notna(paper.get(k)) and str(paper.get(k)) not in {"", "nan"}]
                        sim = paper.get("centroid_similarity")
                        sim_text = f" · сходство с центроидом {float(sim):.3f}" if pd.notna(sim) else ""
                        st.markdown(f"{i}. **{title}**")
                        st.caption(" · ".join(meta) + sim_text if meta else sim_text.strip(" ·"))
                        doi = str(paper.get("doi") or "").strip()
                        if doi and doi.lower() not in {"nan", "none"}:
                            st.markdown(f"[Открыть публикацию по DOI](https://doi.org/{doi})")
                        abstract = str(paper.get("abstract") or "").strip()
                        if abstract and abstract.lower() != "nan":
                            with st.expander(f"Аннотация публикации {i}"):
                                st.write(abstract)
            review_titles = [x.strip() for x in str(row.get("review_titles") or "").split("|") if x.strip()]
            if review_titles:
                with st.expander("Пограничные публикации для ручной проверки"):
                    st.caption("Это документы с наименьшим сходством к TF-IDF-центроиду группы. Они помогают обнаружить возможные тематические примеси, но не считаются автоматически ошибочными назначениями.")
                    for title in review_titles[:5]:
                        st.write(f"• {title}")
            if not subtopics.empty and "group_id" in subtopics.columns:
                inner = subtopics[subtopics["group_id"].astype(str) == group_id]
                if not inner.empty:
                    st.markdown("**Внутренние тематические направления**")
                    for sub in inner.to_dict("records"):
                        sub_label = _first_text(sub, "subtopic_label_ru", "subtopic_label", default="Подтема")
                        st.write(f"• {sub_label}: {int(sub.get('document_count') or 0)} публикаций")


def build_semantic_profiles(run_dir: Path, docs_frame: pd.DataFrame):
    """Generate deterministic evidence-based names for the selected run."""
    from sciintel.topic_understanding import TopicConfig, run_topic_understanding

    abstract_by_id = {}
    if not docs_frame.empty and {"document_id", "abstract"}.issubset(docs_frame.columns):
        abstract_by_id = {
            str(row.document_id): (str(row.abstract) if pd.notna(row.abstract) else "")
            for row in docs_frame[["document_id", "abstract"]].itertuples(index=False)
        }
    output = run_dir / "semantic_profiles"
    return run_topic_understanding(
        run_dir / "clusters.csv", settings.processed, output, TopicConfig(),
        abstract_by_id=abstract_by_id,
    )


def load_topic_manifest():
    path = ROOT / "data" / "metadata" / "arxiv_random_sample.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame()


def cluster_color(value):
    try:
        value = int(value)
    except Exception:
        return NOISE
    return NOISE if value == -1 else PALETTE[value % len(PALETTE)]


def add_page_header(eyebrow: str, title: str, subtitle: str):
    st.markdown(f'<div class="sil-eyebrow">{eyebrow}</div><div class="sil-title">{title}</div><div class="sil-subtitle">{subtitle}</div>',
                unsafe_allow_html=True)


def run_cluster_job(kind: str, params: dict):
    with st.spinner("Выполняется анализ эмбеддингов. Это может занять несколько минут..."):
        if kind == "HDBSCAN":
            from sciintel.clustering import run_hdbscan
            result = run_hdbscan(**params)
        else:
            from sciintel.graph_clustering import run_louvain
            result = run_louvain(**params)
    st.session_state["last_job_result"] = result
    st.success(f"Эксперимент завершён: {result['run_id']}")
    st.rerun()


def ensure_graph_edges(run_id: str, clusters: pd.DataFrame, k: int, min_similarity: float, mutual: bool):
    out = settings.artifacts / "runs" / run_id
    edge_path = out / f"graph_edges_k{k}_s{min_similarity:.2f}_{'mutual' if mutual else 'union'}.csv"
    if edge_path.exists():
        return pd.read_csv(edge_path)
    from sciintel.embeddings import load_embeddings
    from sciintel.graph_clustering import build_knn_graph
    ids, vectors, _ = load_embeddings()
    allowed = set(clusters["document_id"].astype(str))
    keep = [i for i, doc_id in enumerate(ids) if doc_id in allowed]
    ids = [ids[i] for i in keep]
    vectors = vectors[keep]
    _, edges = build_knn_graph(vectors, k=min(k, len(vectors)-1), min_similarity=min_similarity, mutual=mutual)
    edges["source_id"] = edges["source_index"].map(lambda i: ids[int(i)])
    edges["target_id"] = edges["target_index"].map(lambda i: ids[int(i)])
    edges = edges[["source_id", "target_id", "similarity"]]
    edges.to_csv(edge_path, index=False, encoding="utf-8-sig")
    return edges



def draw_cluster_similarity_graph(clusters: pd.DataFrame, threshold: float = 0.45, neighbors: int = 3,
                                  label_map: dict[int, str] | None = None):
    """Aggregate document embeddings into cluster centroids and draw their similarity graph."""
    from sciintel.embeddings import load_embeddings
    ids, vectors, _ = load_embeddings()
    id_to_index = {doc_id: i for i, doc_id in enumerate(ids)}
    frame = clusters[clusters["cluster_id"] != -1].copy()
    grouped = []
    for cluster_id, part in frame.groupby("cluster_id"):
        idx = [id_to_index[x] for x in part["document_id"].astype(str) if x in id_to_index]
        if not idx:
            continue
        centroid = vectors[idx].mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm:
            centroid = centroid / norm
        cluster_int = int(cluster_id)
        grouped.append({"cluster_id": cluster_int, "size": len(idx), "centroid": centroid,
                        "label": (label_map or {}).get(cluster_int, "Тема ещё не определена"),
                        "sample_title": str(part.iloc[0].get("title", ""))})
    if len(grouped) < 2:
        st.info("Нужно не менее двух непустых кластеров.")
        return
    centroids = np.stack([x["centroid"] for x in grouped])
    similarity = centroids @ centroids.T
    graph = nx.Graph()
    for i, item in enumerate(grouped):
        graph.add_node(i, **item)
    for i in range(len(grouped)):
        order = np.argsort(similarity[i])[::-1]
        added = 0
        for j in order:
            if i == j:
                continue
            score = float(similarity[i, j])
            if score < threshold:
                continue
            graph.add_edge(i, int(j), weight=score)
            added += 1
            if added >= neighbors:
                break
    pos = nx.spring_layout(graph, seed=42, weight="weight", k=1.2 / np.sqrt(max(1, graph.number_of_nodes())), iterations=100)
    edge_x, edge_y = [], []
    for u, v, data in graph.edges(data=True):
        x0,y0 = pos[u]; x1,y1 = pos[v]
        edge_x.extend([x0,x1,None]); edge_y.extend([y0,y1,None])
    fig = go.Figure()
    if edge_x:
        fig.add_trace(go.Scatter(x=edge_x,y=edge_y,mode="lines",line=dict(color="rgba(75,103,125,.35)",width=1.2),
                                 hoverinfo="skip",name="Сходство тематических групп"))
    node_x,node_y,node_size,node_color,node_text,node_custom = [],[],[],[],[],[]
    for i,item in enumerate(grouped):
        x,y = pos[i]
        node_x.append(x); node_y.append(y); node_size.append(18 + min(30, np.sqrt(item["size"])*2))
        node_color.append(cluster_color(item["cluster_id"]))
        node_text.append(f"{item['label']} · {item['size']} публикаций · ID {item['cluster_id']}")
        node_custom.append(item["sample_title"])
    fig.add_trace(go.Scatter(x=node_x,y=node_y,mode="markers+text",text=[(x["label"][:26] + ("…" if len(x["label"]) > 26 else "")) for x in grouped],
        textposition="top center",marker=dict(size=node_size,color=node_color,line=dict(width=1.5,color="white")),
        customdata=node_custom,hovertext=node_text,hovertemplate="<b>%{hovertext}</b><br>%{customdata}<extra></extra>",
        name="Тематические группы"))
    fig.update_layout(height=580,template="plotly_white",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#FFFFFF",
        margin=dict(l=10,r=10,t=20,b=10),xaxis=dict(visible=False),yaxis=dict(visible=False,scaleanchor="x",scaleratio=1))
    st.plotly_chart(fig,use_container_width=True)
    st.caption(f"Вершины — центроиды кластеров в исходном SPECTER-пространстве; рёбра — top-{neighbors} межкластерных сходств выше {threshold:.2f}. Размер вершины отражает число публикаций.")

def draw_document_graph(nodes: pd.DataFrame, edges: pd.DataFrame, max_edges: int = 12000,
                        label_map: dict[int, str] | None = None):
    if nodes.empty:
        st.info("Нет узлов для визуализации.")
        return
    lookup = nodes.set_index("document_id")[["x", "y"]].to_dict("index")
    e = edges[edges["source_id"].isin(lookup) & edges["target_id"].isin(lookup)].copy()
    if len(e) > max_edges:
        e = e.nlargest(max_edges, "similarity")
    xs, ys = [], []
    for row in e.itertuples(index=False):
        a, b = lookup[row.source_id], lookup[row.target_id]
        xs.extend([a["x"], b["x"], None])
        ys.extend([a["y"], b["y"], None])
    fig = go.Figure()
    if xs:
        fig.add_trace(go.Scattergl(x=xs, y=ys, mode="lines", line=dict(color="rgba(86,111,130,.22)", width=.65),
                                   hoverinfo="skip", name="Сходство документов"))
    node_colors = [cluster_color(x) for x in nodes["cluster_id"]]
    domain_values = nodes["topic_label"] if "topic_label" in nodes.columns else nodes["domain"]
    semantic_names = nodes["cluster_id"].map(lambda value: (label_map or {}).get(int(value), "Тема ещё не определена"))
    fig.add_trace(go.Scattergl(
        x=nodes["x"], y=nodes["y"], mode="markers",
        marker=dict(size=8, color=node_colors, line=dict(width=.6, color="#FFFFFF"), opacity=.92),
        text=nodes["title"], customdata=np.stack([nodes["document_id"], domain_values, nodes["cluster_id"], semantic_names], axis=-1),
        hovertemplate="<b>%{text}</b><br>Документ: %{customdata[0]}<br>Исходная категория: %{customdata[1]}<br>Тематическая группа: %{customdata[3]}<br>Технический ID: %{customdata[2]}<extra></extra>",
        name="Научные публикации"))
    fig.update_layout(height=700, template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="#FFFFFF", margin=dict(l=10,r=10,t=25,b=10),
                      xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
                      legend=dict(orientation="h", y=1.02))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Узлов: {len(nodes):,} · рёбер: {len(e):,}. Рёбра — cosine kNN по SPECTER; координаты — проекция из выбранного запуска. Проекция нужна для отображения, а не для вычисления весов рёбер.")


db.init_db()
docs = load_docs()
manifest = load_topic_manifest()
topic_name_map = dict(zip(manifest["topic_slug"].astype(str), manifest["topic_label"].astype(str))) if not manifest.empty and {"topic_slug","topic_label"}.issubset(manifest.columns) else {}
# Load the latest run before decorating its cluster assignments with topic labels.
# The previous order referenced `clusters` before it had been initialized, so
# Streamlit crashed immediately after opening the dashboard.
run, clusters, profiles, run_dir = load_latest()
semantic_profiles, topic_document_analysis, internal_subtopics, topic_candidates, topic_metadata = load_semantic_profile_data(run_dir)
semantic_label_map = make_semantic_label_map(semantic_profiles, profiles)
if not clusters.empty:
    clusters = clusters.copy()
    clusters["semantic_name"] = clusters["cluster_id"].map(
        lambda value: "Не распределено" if int(value) == -1 else semantic_label_map.get(int(value), "Тема ещё не определена")
    )
if not docs.empty:
    docs["topic_label"] = docs["domain"].map(topic_name_map).fillna(docs["domain"])
if not clusters.empty and "domain" in clusters.columns:
    clusters["topic_label"] = clusters["domain"].map(topic_name_map).fillna(clusters["domain"])
keyword_count = db.rows("SELECT COUNT(*) AS n FROM keywords")[0]["n"]
gpu = gpu_snapshot()

with st.sidebar:
    st.markdown("<div class='sil-eyebrow'>Research analytics platform</div>", unsafe_allow_html=True)
    st.markdown("# ◈ SIL / LAB")
    page = st.radio("РАЗДЕЛЫ", ["Обзор", "Корпус", "Кластеризация", "Тематические профили",
                                "Граф публикаций", "Графовые сообщества", "Проверка качества",
                                "Ключевые термины", "Семантический поиск", "Лингвистический анализ"], label_visibility="collapsed")
    st.divider()
    if gpu:
        st.success(f"GPU · {gpu[0]}")
        st.caption(f"Загрузка {gpu[1]}% · VRAM {gpu[2]}/{gpu[3]} MB · {gpu[4]} °C")
    else:
        st.caption("GPU не обнаружена · доступен CPU-режим")
    st.caption(f"Корпус: {len(docs):,} документов")
    if run:
        st.caption(f"Последний запуск: {run['run_id']}")

add_page_header("SCIENTIFIC INTELLIGENCE LAB", page,
                "Аналитика научных публикаций · SPECTER embeddings · HDBSCAN · kNN graph · Louvain")

if page == "Обзор":
    embedded = 0
    try:
        from sciintel.embeddings import load_embeddings
        embedded = len(load_embeddings()[0])
    except Exception:
        pass
    topic_count = docs["domain"].nunique() if not docs.empty else 0
    cluster_count = run["metrics"].get("clusters", "—") if run else "—"
    noise = run["metrics"].get("noise_fraction") if run else None
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Публикации", f"{len(docs):,}")
    c2.metric("SPECTER-векторы", f"{embedded:,}")
    c3.metric("Исходные темы", f"{topic_count:,}")
    c4.metric("Группы", cluster_count)
    c5.metric("Шум", f"{noise*100:.1f}%" if noise is not None else "—")
    if not docs.empty:
        left,right = st.columns([1.1,1])
        with left:
            st.markdown("#### Состав корпуса")
            counts = docs.groupby("topic_label", dropna=False).size().reset_index(name="documents").sort_values("documents", ascending=True)
            fig = px.bar(counts, x="documents", y="topic_label", orientation="h", text="documents",
                         color_discrete_sequence=["#356B8C"], height=460)
            fig.update_layout(template="plotly_white", showlegend=False, margin=dict(l=5,r=10,t=10,b=5))
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.markdown("#### Последний эксперимент")
            if run:
                m = run["metrics"]
                st.metric("Алгоритм", m.get("algorithm", "DBSCAN"))
                a,b = st.columns(2)
                a.metric("Silhouette", f"{m['silhouette_non_noise']:.3f}" if m.get("silhouette_non_noise") is not None else "н/д")
                b.metric("Покрытие", f"{(1-m.get('noise_fraction',0))*100:.1f}%")
                if m.get("modularity") is not None:
                    a.metric("Modularity", f"{m['modularity']:.3f}")
                    b.metric("Рёбра графа", f"{m.get('graph_edges',0):,}")
                st.json(m, expanded=False)
            else:
                st.info("Пока нет результатов. Перейди в «Кластеризация» и запусти первый эксперимент.")
    else:
        st.info("Корпус пока пуст. Скопируй PDF/MD/TXT в data/inbox/<topic>/ и выполни ingest.")
        st.code(".\\.venv\\Scripts\\python.exe .\\-m sciintel.cli ingest", language="powershell")
    st.markdown("#### Этапы аналитического конвейера")
    stages = ["Сбор PDF и контроль источников", "Извлечение текста и метаданных", "SPECTER-векторизация",
              "Построение плотностных / графовых групп", "Тематическая интерпретация", "Проверка качества и визуализация"]
    cols = st.columns(len(stages))
    for i,(col,label) in enumerate(zip(cols,stages,strict=True),1):
        col.markdown(f"<div class='sil-panel'><div class='sil-eyebrow'>STEP {i:02d}</div><b>{label}</b></div>", unsafe_allow_html=True)

elif page == "Корпус":
    if docs.empty:
        st.info("Публикаций пока нет. Положи PDF/MD/TXT в data/inbox/<topic>/ и выполни ingest.")
        st.code(r".\.venv\Scripts\python.exe -m sciintel.cli ingest", language="powershell")
    else:
        q = st.text_input("Поиск по названию или ID", placeholder="Например, graph neural networks")
        topics = sorted(docs["topic_label"].dropna().unique().tolist())
        chosen = st.multiselect("Исходные темы", topics, default=topics)
        subset = docs[docs["topic_label"].isin(chosen)]
        if q.strip():
            mask = subset["title"].fillna("").str.contains(q, case=False, regex=False) | subset["document_id"].str.contains(q, case=False, regex=False)
            subset = subset[mask]
        a,b,c = st.columns(3)
        a.metric("В выборке", f"{len(subset):,}")
        b.metric("Всего документов", f"{len(docs):,}")
        c.metric("Тематических категорий", f"{subset['domain'].nunique():,}")
        st.dataframe(subset, use_container_width=True, hide_index=True, height=560)
        st.download_button("Экспортировать отфильтрованный корпус", subset.to_csv(index=False).encode("utf-8-sig"),
                           "scientific_corpus.csv", "text/csv")
        if not manifest.empty:
            with st.expander("План случайной выборки arXiv"):
                st.dataframe(manifest[[c for c in ["arxiv_id","topic_category","topic_label","title","published","pdf_path"] if c in manifest.columns]],
                             use_container_width=True, hide_index=True)

elif page == "Кластеризация":
    st.markdown("#### Параметры эксперимента")
    kind = st.segmented_control("Метод", ["HDBSCAN", "Louvain graph"], default="HDBSCAN")
    if kind == "HDBSCAN":
        c1,c2,c3 = st.columns(3)
        mcs = c1.number_input("min_cluster_size", 3, 200, 12, 1)
        ms = c2.number_input("min_samples", 1, 100, 5, 1)
        selection = c3.selectbox("Selection method", ["eom","leaf"])
        reduction = st.selectbox("Пространство кластеризации", ["umap","original"], index=0,
                                 format_func=lambda x: "UMAP 10D → HDBSCAN" if x == "umap" else "Исходное SPECTER 768D")
        if reduction == "umap":
            c1,c2,c3 = st.columns(3)
            components = c1.number_input("UMAP components", 2, 50, 10)
            neighbors = c2.number_input("UMAP neighbors", 5, 100, 30)
            min_dist = c3.slider("UMAP min_dist", 0.0, 0.99, 0.0, 0.01)
        else:
            components,neighbors,min_dist = 10,30,0.0
        device = st.selectbox("Устройство расчёта расстояний и PCA", ["cuda","auto","cpu"], index=0)
        if st.button("Запустить HDBSCAN", type="primary", disabled=len(docs)<3):
            run_cluster_job("HDBSCAN", {"min_cluster_size":int(mcs),"min_samples":int(ms),
                "cluster_selection_method":selection,"device":device,"reduction":reduction,
                "umap_components":int(components),"umap_neighbors":int(neighbors),"umap_min_dist":float(min_dist)})
    else:
        c1,c2,c3,c4 = st.columns(4)
        k = c1.number_input("k соседей", 2, 100, 15)
        threshold = c2.slider("Минимальное cosine similarity", 0.0, 0.99, 0.35, 0.01)
        resolution = c3.slider("Louvain resolution", 0.1, 3.0, 1.0, 0.05)
        min_size = c4.number_input("Минимальный размер сообщества", 1, 100, 5)
        mutual = st.checkbox("Mutual kNN (ребро только при взаимном соседстве)", value=False)
        device = st.selectbox("Устройство PCA", ["auto","cuda","cpu"], index=0)
        if st.button("Построить граф и найти сообщества", type="primary", disabled=len(docs)<3):
            run_cluster_job("Louvain", {"k":int(k),"min_similarity":float(threshold),"resolution":float(resolution),
                "min_community_size":int(min_size),"mutual":mutual,"device":device})
    st.divider()
    if run:
        st.markdown("#### Последний запуск")
        m = run["metrics"]
        a,b,c,d,e = st.columns(5)
        a.metric("Алгоритм", m.get("algorithm","—"))
        b.metric("Тематических групп", m.get("clusters","—"))
        c.metric("Покрытие", f"{(1-m.get('noise_fraction',0))*100:.1f}%")
        d.metric("Silhouette", f"{m['silhouette_non_noise']:.3f}" if m.get("silhouette_non_noise") is not None else "н/д")
        e.metric("Время", f"{m.get('elapsed_seconds',0):.1f} с")
        if run_dir and st.button("Сформировать семантические названия и проверить сообщества", key="label_run"):
            try:
                with st.spinner("Анализирую заголовки, аннотации и внутригрупповую структуру..."):
                    result = build_semantic_profiles(run_dir, docs)
                st.success(f"Семантические профили сохранены: {result['profiles']}")
                st.rerun()
            except Exception as exc:
                st.error(f"Не удалось сформировать семантические профили: {exc}")
    else:
        st.info("После подготовки корпуса выбери алгоритм и запусти эксперимент. Исторические назначения не подмешиваются.")

elif page == "Тематические профили":
    if clusters.empty:
        st.info("Сначала выполни HDBSCAN или Louvain.")
    else:
        st.markdown("#### Семантическая интерпретация тематических групп")
        st.caption("Русское название формируется из повторяющихся научных понятий в заголовках и аннотациях. Ранжирование учитывает распространённость, присутствие в заголовках, контрастность и специфичность терминов. Исходные английские термины и доказательства сохраняются для проверки. Это интерпретационный слой: он не меняет назначения HDBSCAN или Louvain.")
        c1, c2 = st.columns([1, 2])
        with c1:
            if st.button("Сформировать / обновить семантические названия", type="primary", key="build_topic_profiles"):
                try:
                    with st.spinner("Извлекаю устойчивые понятия, оцениваю однородность и ищу внутренние подтемы..."):
                        result = build_semantic_profiles(run_dir, docs)
                    st.success(f"Профили сформированы: {result['profiles']}")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Ошибка анализа тематик: {exc}")
        with c2:
            if not semantic_profiles.empty:
                statuses = semantic_profiles.get("topic_profile_status", pd.Series(dtype=str)).value_counts()
                st.caption(" · ".join(f"{PROFILE_STATUS_RU.get(k, k)}: {v}" for k, v in statuses.items()))
            else:
                st.info("Пока показаны только предварительные фразы из YAKE/RAKE. Нажми кнопку, чтобы сформировать проверенные профили.")

        if not semantic_profiles.empty:
            meta_cols = st.columns(4)
            strong = int(semantic_profiles["topic_profile_status"].eq("STRONG_PROFILE").sum()) if "topic_profile_status" in semantic_profiles else 0
            mixed = int(semantic_profiles["topic_profile_status"].isin(["MIXED_TOPIC_REVIEW", "INTERNAL_SUBTOPICS_DETECTED"]).sum()) if "topic_profile_status" in semantic_profiles else 0
            meta_cols[0].metric("Групп с названиями", len(semantic_profiles))
            meta_cols[1].metric("Устойчивые профили", strong)
            meta_cols[2].metric("Смешанные / с подтемами", mixed)
            meta_cols[3].metric("Публикаций проанализировано", int(topic_metadata.get("documents_analyzed", 0)))
            if topic_metadata:
                source_modes = topic_metadata.get("text_source_modes", {})
                st.caption("Источники текста для интерпретации: " + " · ".join(f"{k}: {v}" for k, v in source_modes.items()))
            st.markdown("#### Тематические группы")
            show_semantic_group_cards(semantic_profiles, topic_document_analysis, internal_subtopics, docs)
            if not topic_candidates.empty:
                with st.expander("Проверить доказательные фразы и кандидаты названий"):
                    cols = [c for c in [
                        "group_id", "canonical_phrase", "canonical_phrase_ru", "translation_source",
                        "support_docs", "literal_support_docs", "ontology_only_support_docs",
                        "prevalence", "title_support_docs", "literal_title_support_docs",
                        "ontology_title_support_docs", "contrast_score", "score", "naming_score",
                        "generic_penalty",
                    ] if c in topic_candidates.columns]
                    candidate_view = topic_candidates[cols].copy()
                    if "canonical_phrase_ru" in candidate_view.columns:
                        candidate_view = candidate_view.rename(columns={
                            "canonical_phrase_ru": "Термин по-русски",
                            "translation_source": "Источник перевода",
                        })
                    elif "canonical_phrase" in candidate_view.columns:
                        candidate_view.insert(2, "Термин по-русски", candidate_view["canonical_phrase"].map(translate_term))
                    candidate_view = candidate_view.sort_values([c for c in ["group_id", "naming_score"] if c in candidate_view.columns], ascending=[True, False] if "naming_score" in candidate_view.columns else True)
                    st.dataframe(candidate_view, use_container_width=True, hide_index=True)
            if not internal_subtopics.empty:
                with st.expander("Таблица внутренних тематических подгрупп"):
                    st.dataframe(internal_subtopics, use_container_width=True, hide_index=True)
        else:
            st.markdown("#### Предварительные тематические подсказки")
            preview = clusters[clusters["cluster_id"] != -1].groupby(["cluster_id", "semantic_name"], as_index=False).agg(
                publications=("document_id", "count"), examples=("title", lambda x: " | ".join(x.head(3).astype(str)))
            ).sort_values("publications", ascending=False)
            st.dataframe(preview.rename(columns={"semantic_name": "Предварительное название", "cluster_id": "Технический ID"}),
                         use_container_width=True, hide_index=True)

        st.markdown("#### Карта тематических групп в пространстве публикаций")
        view = clusters.copy()
        group_names = view[view.cluster_id != -1][["cluster_id", "semantic_name"]].drop_duplicates()
        duplicate_names = group_names["semantic_name"].duplicated(keep=False)
        for idx in group_names[duplicate_names].index:
            cid = int(group_names.loc[idx, "cluster_id"])
            group_names.loc[idx, "semantic_name"] = f"{group_names.loc[idx, 'semantic_name']} · направление {cid + 1}"
        display_map = dict(zip(group_names["cluster_id"].astype(int), group_names["semantic_name"].astype(str)))
        view["cluster_display"] = view["cluster_id"].map(lambda x: "Не распределено" if int(x) == -1 else display_map.get(int(x), "Тема ещё не определена"))
        options = ["Все группы"] + sorted(view[view.cluster_id != -1]["cluster_display"].unique().tolist())
        if (view.cluster_id == -1).any():
            options.append("Не распределено")
        selected = st.selectbox("Фильтр по тематической группе", options)
        if selected != "Все группы":
            view = view[view.cluster_display == selected]
        color_map = {str(row.cluster_display): cluster_color(row.cluster_id)
                     for row in view[["cluster_id", "cluster_display"]].drop_duplicates().itertuples(index=False)}
        fig = px.scatter(view, x="x", y="y", color="cluster_display", hover_name="title",
                         hover_data={"document_id": True, "topic_label": True, "cluster_id": True,
                                     "semantic_name": True, "x": False, "y": False},
                         color_discrete_map=color_map, height=640)
        fig.update_traces(marker=dict(size=8, opacity=.88, line=dict(width=.5, color="white")))
        fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=5,r=5,t=20,b=5),
                          xaxis_title="Координата 1", yaxis_title="Координата 2", legend_title="Тематическая группа")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Каждая точка — публикация. Цвет обозначает семантически названную группу; координаты PCA/UMAP служат только для визуализации.")
        st.markdown("#### Публикации выбранной группы")
        display_cols = [c for c in ["title", "semantic_name", "topic_label", "year", "doi", "document_id"] if c in view.columns]
        st.dataframe(view[display_cols], use_container_width=True, hide_index=True, height=360)
        st.download_button("Скачать назначения и семантические названия", view.to_csv(index=False).encode("utf-8-sig"),
                           "semantic_cluster_assignments.csv", "text/csv")

elif page == "Граф публикаций":
    if clusters.empty:
        st.info("Сначала выполни кластеризацию, чтобы наложить её метки на граф документов.")
    else:
        c1,c2,c3 = st.columns(3)
        k = c1.slider("Число ближайших соседей k", 3, 40, 12)
        threshold = c2.slider("Минимальное сходство рёбер", 0.10, 0.95, 0.45, 0.01)
        mutual = c3.checkbox("Только взаимные соседи", False)
        if st.button("Построить / обновить граф сходства", type="primary"):
            with st.spinner("Строю взвешенный cosine kNN-граф из SPECTER-векторов..."):
                edges = ensure_graph_edges(run["run_id"], clusters, k, threshold, mutual)
            st.success(f"Граф готов: {len(clusters):,} узлов, {len(edges):,} рёбер")
        edge_candidates = sorted(run_dir.glob("graph_edges*.csv")) if run_dir else []
        if edge_candidates:
            edge_file = st.selectbox("Набор рёбер", edge_candidates, format_func=lambda p:p.name)
            edges = pd.read_csv(edge_file)
            selectable = clusters[["cluster_id", "semantic_name"]].drop_duplicates().copy()
            selectable["selector_label"] = selectable.apply(
                lambda row: "Не распределено" if int(row.cluster_id) == -1 else str(row.semantic_name), axis=1
            )
            duplicated = selectable["selector_label"].duplicated(keep=False) & (selectable["cluster_id"] != -1)
            selectable.loc[duplicated, "selector_label"] = selectable.loc[duplicated].apply(
                lambda row: f"{row.selector_label} · направление {int(row.cluster_id) + 1}", axis=1
            )
            labels = selectable["selector_label"].tolist()
            selected_labels = st.multiselect("Показывать тематические группы", labels, default=labels)
            selected_ids = selectable.loc[selectable["selector_label"].isin(selected_labels), "cluster_id"].tolist()
            nodes = clusters[clusters["cluster_id"].isin(selected_ids)].copy()
            tab_docs, tab_clusters = st.tabs(["Граф документов", "Граф тематических групп"])
            with tab_docs:
                draw_document_graph(nodes, edges, label_map=semantic_label_map)
                st.dataframe(edges.nlargest(100,"similarity"), use_container_width=True, hide_index=True)
                st.download_button("Скачать рёбра графа", edges.to_csv(index=False).encode("utf-8-sig"), "document_knn_edges.csv", "text/csv")
            with tab_clusters:
                c1,c2 = st.columns(2)
                group_threshold = c1.slider("Минимальное сходство групп", 0.10, 0.95, 0.45, 0.01)
                group_neighbors = c2.slider("Число связей на группу", 1, 8, 3)
                draw_cluster_similarity_graph(clusters, group_threshold, group_neighbors, semantic_label_map)
        else:
            st.info("Построй граф. Каждая вершина — публикация; ребро соединяет близкие SPECTER-векторы; вес ребра — cosine similarity.")

elif page == "Графовые сообщества":
    if clusters.empty:
        st.info("Сначала выполни HDBSCAN или Louvain. Для графового разбиения открой «Кластеризация» и выбери Louvain graph.")
    else:
        if run and run["metrics"].get("algorithm", "").startswith("Louvain"):
            m = run["metrics"]
            a,b,c,d = st.columns(4)
            a.metric("Вершины", f"{m.get('graph_nodes',0):,}")
            b.metric("Рёбра", f"{m.get('graph_edges',0):,}")
            c.metric("Компоненты", m.get("graph_components","—"))
            d.metric("Modularity", f"{m.get('modularity',0):.3f}")
            st.caption("Louvain максимизирует модульность взвешенного графа; min_community_size переводит слишком малые сообщества в noise.")
        else:
            st.info("Последний запуск не является графовым разбиением. Можно запустить Louvain в разделе «Кластеризация».")
        if run_dir and (run_dir/"graph_edges.csv").exists():
            edges = pd.read_csv(run_dir/"graph_edges.csv")
            st.markdown("#### Граф публикаций и выделенные сообщества")
            draw_document_graph(clusters, edges, label_map=semantic_label_map)
        if not semantic_profiles.empty:
            st.markdown("#### Семантическое содержание сообществ")
            show_semantic_group_cards(semantic_profiles, topic_document_analysis, internal_subtopics, docs, max_groups=10)
        elif not profiles.empty:
            st.markdown("#### Предварительные фразы сообществ")
            st.dataframe(profiles.rename(columns={"cluster_id": "Технический ID", "size": "Публикаций", "top_terms": "Ключевые фразы"}),
                         use_container_width=True, hide_index=True)

elif page == "Проверка качества":
    if clusters.empty:
        st.info("После первого запуска здесь появятся внутренние и внешние показатели качества.")
    else:
        m = run["metrics"]
        st.markdown("#### Метрики текущего разбиения")
        keys = ["silhouette_non_noise","ari_domains","nmi_domains","noise_fraction","clusters","documents",
                "modularity","graph_density","average_degree"]
        metric_titles = {
            "silhouette_non_noise": "Silhouette · SPECTER",
            "ari_domains": "ARI · категории источника",
            "nmi_domains": "NMI · категории источника",
            "noise_fraction": "Доля нераспределённых",
            "clusters": "Тематических групп",
            "documents": "Публикаций",
            "modularity": "Модульность графа",
            "graph_density": "Плотность графа",
            "average_degree": "Средняя степень вершины",
        }
        cols = st.columns(3)
        for i,key in enumerate(keys):
            if key in m and m[key] is not None:
                value = m[key]
                display = f"{value:.4f}" if isinstance(value, (float, np.floating)) else f"{value:,}" if isinstance(value, (int, np.integer)) else str(value)
                cols[i%3].metric(metric_titles.get(key, key), display)
        st.warning("ARI/NMI сравнивают кластеры с категорией источника (папка arXiv). Это слабая внешняя метка, а не абсолютная истина: категории пересекаются, а реальная тема статьи может быть уже категории.")
        if not semantic_profiles.empty:
            st.markdown("#### Проверка семантической целостности")
            profile_view = semantic_profiles.copy()
            profile_view["display_name"] = profile_view["automatic_label"]
            profile_view["status_ru"] = profile_view["topic_profile_status"].map(lambda x: PROFILE_STATUS_RU.get(str(x), str(x)))
            status_counts = profile_view["status_ru"].value_counts().rename_axis("Статус").reset_index(name="Групп")
            a, b = st.columns([1, 1.3])
            with a:
                fig = px.bar(status_counts, x="Групп", y="Статус", orientation="h", color="Статус",
                             color_discrete_sequence=["#3A8D78", "#D17A48", "#8064A2", "#5277B8"], height=320)
                fig.update_layout(template="plotly_white", showlegend=False, margin=dict(l=5,r=5,t=10,b=5))
                st.plotly_chart(fig, use_container_width=True)
            with b:
                score_cols = [c for c in ["display_name", "document_count", "primary_topic_support",
                                          "mean_title_abstract_centroid_similarity", "mean_pairwise_title_abstract_similarity",
                                          "topic_profile_score", "status_ru"] if c in profile_view.columns]
                st.dataframe(profile_view[score_cols].sort_values("topic_profile_score", ascending=False),
                             use_container_width=True, hide_index=True)
            st.caption("Поддержка темы — доля публикаций группы с соответствующей фразой; сходство — TF-IDF по заголовкам и аннотациям; индекс профиля эвристический, не вероятность.")
            if not internal_subtopics.empty:
                st.markdown("#### Группы с внутренней тематической неоднородностью")
                st.dataframe(internal_subtopics, use_container_width=True, hide_index=True)
        else:
            st.info("Для содержательной проверки сформируй семантические профили на вкладке «Тематические профили». До этого доступны только метрики кластеризации и сравнение с исходными категориями.")
        quality = clusters.drop(columns=["topic_label"], errors="ignore").merge(docs[["document_id","topic_label"]].drop_duplicates(), on="document_id", how="left")
        non_noise = quality[quality.cluster_id!=-1]
        if not non_noise.empty:
            cross = pd.crosstab(non_noise["cluster_id"], non_noise["topic_label"])
            purity = cross.max(axis=1) / cross.sum(axis=1)
            summary = pd.DataFrame({"cluster_id":purity.index,"size":cross.sum(axis=1).values,
                                    "dominant_source_topic":cross.idxmax(axis=1).values,"purity":purity.values})
            summary["semantic_group"] = summary["cluster_id"].map(lambda x: semantic_label_map.get(int(x), f"Тема не названа · ID {x}"))
            st.markdown("#### Чистота кластеров относительно исходной темы")
            fig = px.bar(summary.sort_values("purity"), x="purity", y="semantic_group", orientation="h",
                         color="purity", color_continuous_scale=["#E7B36B","#77A99B","#356B8C"],
                         hover_data=["cluster_id","size","dominant_source_topic"], height=520)
            fig.update_layout(template="plotly_white", coloraxis_showscale=False, xaxis_range=[0,1])
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(summary.sort_values("purity",ascending=False).rename(columns={"semantic_group":"Тематическая группа", "cluster_id":"Технический ID", "size":"Публикаций", "dominant_source_topic":"Преобладающая категория источника", "purity":"Доля крупнейшей категории"}), use_container_width=True, hide_index=True)
            st.markdown("#### Матрица соответствия тематических групп исходным категориям")
            cross.index = [semantic_label_map.get(int(x), f"Тема не названа · ID {x}") for x in cross.index]
            fig = px.imshow(cross, aspect="auto", color_continuous_scale="Blues", labels=dict(x="Исходная тема",y="Семантическая группа",color="Документы"), height=600)
            st.plotly_chart(fig, use_container_width=True)

elif page == "Ключевые термины":
    method = st.selectbox("Метод извлечения", ["YAKE","RAKE"])
    rows = db.rows("""SELECT k.phrase,COUNT(*) AS documents,AVG(k.rank) AS avg_rank,AVG(k.score) AS avg_score
                      FROM keywords k WHERE k.method=? GROUP BY k.phrase ORDER BY documents DESC,avg_rank ASC LIMIT 200""", (method,))
    if rows:
        frame = pd.DataFrame(rows)
        fig = px.bar(frame.head(35).sort_values("documents"), x="documents", y="phrase", orientation="h",
                     color_discrete_sequence=["#3A8D78"], height=700)
        fig.update_layout(template="plotly_white", margin=dict(l=5,r=5,t=15,b=5))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(frame, use_container_width=True, hide_index=True)
    else:
        st.info("Термины ещё не извлечены. Выполни `python -m sciintel.cli keywords`.")

elif page == "Лингвистический анализ":
    st.markdown("### Морфология, лемматизация и позиционный индекс")
    st.caption("Английская и русская spaCy-модели. Пунктуация сохраняется как токены; позиции отсчитываются от начала Markdown-документа.")
    language = st.selectbox("Язык корпуса", ["auto", "en", "ru"], format_func=lambda x: {"auto":"Автоопределение","en":"Английский","ru":"Русский"}[x])
    force = st.checkbox("Переиндексировать уже обработанные документы", value=False)
    if st.button("Построить / обновить лингвистический индекс", type="primary"):
        with st.spinner("Конвертирую документы, выполняю морфологический и синтаксический анализ..."):
            from sciintel.linguistic.pipeline import index_documents
            result = index_documents(language=language, force=force)
        st.json(result)
        if result.get("failed"):
            st.warning("Часть документов не обработана. Проверьте журнал и наличие языковых моделей.")
    st.divider()
    tab_sent, tab_tokens, tab_concepts = st.tabs(["Поиск по предложениям", "Слова и леммы", "Предметные понятия"])
    with tab_sent:
        sentence_query = st.text_input("Текстовый запрос", key="ling_sentence_query", placeholder="например: neural network")
        if st.button("Найти предложения", key="ling_sentence_search", disabled=not sentence_query.strip()):
            from sciintel.linguistic.indexer import search_sentences
            try:
                found = search_sentences(sentence_query.strip(), 100)
                st.write(f"Найдено: {len(found)}")
                for item in found:
                    with st.container(border=True):
                        st.markdown(f"**{item['title'] or item['document_id']}**")
                        st.write(item["text"])
                        st.caption(f"Документ {item['document_id']} · предложение {item['sentence_index']} · символы [{item['char_start']}, {item['char_end']})")
            except Exception as exc:
                st.error(f"Ошибка поискового запроса FTS5: {exc}")
    with tab_tokens:
        token_query = st.text_input("Словоформа или лемма", key="ling_token_query", placeholder="например: networks или network")
        if st.button("Найти токены", key="ling_token_search", disabled=not token_query.strip()):
            from sciintel.linguistic.indexer import search_tokens
            try:
                found = search_tokens(token_query.strip(), 200)
                st.write(f"Найдено токенов: {len(found)}")
                if found:
                    st.dataframe(pd.DataFrame(found), use_container_width=True, hide_index=True)
                else:
                    st.info("Совпадений нет. Проверьте лемму или сначала постройте индекс.")
            except Exception as exc:
                st.error(f"Ошибка поиска токенов: {exc}")
    with tab_tokens:
        token_query = st.text_input("Словоформа или лемма", key="ling_token_query", placeholder="например: networks или network")
        if st.button("Найти токены", key="ling_token_search", disabled=not token_query.strip()):
            from sciintel.linguistic.indexer import search_tokens
            try:
                found = search_tokens(token_query.strip(), 200)
                st.write(f"Найдено токенов: {len(found)}")
                if found:
                    st.dataframe(pd.DataFrame(found), use_container_width=True, hide_index=True)
                else:
                    st.info("Совпадений нет. Проверьте лемму или сначала постройте индекс.")
            except Exception as exc:
                st.error(f"Ошибка поиска токенов: {exc}")
    with tab_concepts:
        concept_query = st.text_input("Понятие или лемма", key="ling_concept_query", placeholder="например: neural network")
        if st.button("Найти понятия и вхождения", key="ling_concept_search", disabled=not concept_query.strip()):
            from sciintel.linguistic.indexer import search_concepts
            try:
                found = search_concepts(concept_query.strip(), 100)
                if found:
                    st.dataframe(pd.DataFrame(found), use_container_width=True, hide_index=True)
                    with db.connect() as conn:
                        from sciintel.linguistic.indexer import init_linguistic_schema
                        init_linguistic_schema(conn)
                        ids = [int(x["id"]) for x in found]
                        marks = ",".join("?" for _ in ids)
                        occurrences = conn.execute(f"""SELECT c.canonical_form,o.surface_form,d.title,o.document_id,
                            s.sentence_index,s.text,o.char_start,o.char_end,o.token_start,o.token_end
                            FROM concept_occurrences o JOIN concepts c ON c.id=o.concept_id
                            JOIN documents d ON d.id=o.document_id JOIN sentences s ON s.id=o.sentence_id
                            WHERE c.id IN ({marks}) ORDER BY c.canonical_form,d.title LIMIT 300""", ids).fetchall()
                    st.markdown("#### Найденные вхождения")
                    st.dataframe(pd.DataFrame([dict(x) for x in occurrences]), use_container_width=True, hide_index=True)
                else:
                    st.info("Понятия не найдены. Сначала постройте лингвистический индекс.")
            except Exception as exc:
                st.error(f"Ошибка поиска понятий: {exc}")

elif page == "Семантический поиск":
    query = st.text_area("Опиши научную тему", placeholder="Например: graph neural networks for scientific document clustering")
    top_k = st.slider("Количество результатов", 3, 30, 8)
    device = st.selectbox("Устройство", ["cuda","auto","cpu"], index=0)
    if st.button("Найти похожие публикации", type="primary", disabled=len(query.strip())<3):
        with st.spinner("Кодирую запрос моделью SPECTER и ищу ближайшие документы..."):
            from sciintel.search import semantic_search
            found = semantic_search(query, top_k=top_k, device=device)
        for i,item in enumerate(found,1):
            st.markdown(f"**{i}. {item['title']}**")
            st.caption(f"{item['domain']} · {item.get('year') or 'год неизвестен'} · cosine similarity {item['score']:.4f}")
            if item.get("doi"):
                st.markdown(f"[DOI](https://doi.org/{item['doi']})")
            st.divider()

st.divider()
st.caption("SPECTER формирует векторное представление публикации. HDBSCAN выделяет плотностные группы; Louvain работает на взвешенном cosine kNN-графе. PCA/UMAP используются для визуального размещения точек, но не подменяют исходные векторы при оценке сходства.")
