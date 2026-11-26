"""Cluster Topic Understanding v1.2.1

Analytical interpretation layer for document clusters.
Uses titles + abstracts (not arbitrary full-paper body frequency),
cluster-level phrase statistics, text coherence, representative papers,
and optional within-cluster subtopic diagnostics. No LLMs or per-cluster rules.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from .topic_naming import build_topic_name, detect_ontology_concepts, select_naming_terms, translate_term, translation_source, score_naming_candidates

BOUNDARY = "zzboundarytokenzz"

# Corpus-independent linguistic / publication noise, not cluster-specific labels.
STOPWORDS = set(ENGLISH_STOP_WORDS) | {
    "abstract", "introduction", "conclusion", "references", "reference", "bibliography",
    "figure", "figures", "table", "tables", "section", "appendix", "supplementary",
    "copyright", "paper", "papers", "article", "articles", "study", "studies",
    "work", "works", "research", "researchers", "method", "methods", "methodology",
    "approach", "approaches", "propose", "proposed", "proposes", "present", "presents",
    "presented", "show", "shows", "shown", "demonstrate", "demonstrates", "result",
    "results", "experiment", "experiments", "experimental", "dataset", "datasets", "data",
    "task", "tasks", "problem", "problems", "using", "used", "use", "based", "provide",
    "provides", "including", "however", "therefore", "thus", "also", "well", "new", "different",
    "first", "second", "third", "one", "two", "three", "et", "al", "fig", "eq", "arxiv",
    "preprint", "university", "department", "institute", "proceedings", "conference", "workshop",
    "vol", "volume", "issue", "pages", "page", "doi", "isbn", "issn", "corresponding",
    "author", "authors", "affiliation", "affiliations", "accepted", "submitted", "published",
    "publication", "license", "version", "manuscript", "editor", "editors", "creative", "commons",
    "permission", "permissions", "cs", "ds", "cl", "cv", "lg", "cr", "ir", "ne", "ro", "eess",
    "stat", "econ", "quant", "neurips", "nips", "icml", "iclr", "acl", "emnlp", "naacl",
    "does", "given", "single", "let", "directly", "standard", "process", "processes", "support",
    "supports", "generate", "generates", "generated", "compute", "computes", "computed", "predicts",
    "predicted", "improve", "improves", "improved", "include", "includes", "included", "consider",
    "considers", "considered", "require", "requires", "required", "among", "across", "within",
    "without", "around", "often", "generally", "several", "various", "specific", "particular",
    "certain", "another", "whether", "although", "whereas", "hence", "thereby", "respectively",
    "rather", "instead", "much", "many", "less", "more", "most", "least", "highly", "following",
    "previous", "existing", "suppose", "assume", "assuming", "denote", "denoted", "define", "defined",
    "observe", "observed", "obtained", "considering", "corresponding", "set", "sets", "number", "numbers",
    "fixed", "constant", "known", "case", "cases", "size", "sizes", "mean", "means", "random", "target",
    "targets", "linear", "single", "directly", "information processing systems", "computing methodologies",
    "evaluate", "evaluates", "evaluated", "evaluating", "use", "uses", "employ", "employs", "employed",
    "learn", "learns", "learned", "optimize", "optimizes", "optimized", "optimizing",
    "combine", "combines", "combined", "combining", "transfer", "transfers", "transferred",
    "affect", "affects", "affected", "affecting", "test", "tests", "tested", "testing",
    "modeling", "modeled", "modelling", "investigate", "investigates", "investigated",
    "analyze", "analyzes", "analyzed", "analyse", "analyses", "analyzed", "measure", "measures",
    "measured", "measuring", "compare", "compares", "compared", "comparing", "showing",
}

GENERIC_TERMS = {
    "system", "systems", "model", "models", "method", "methods", "approach", "approaches",
    "data", "research", "study", "studies", "language", "languages", "text", "texts", "result",
    "results", "problem", "problems", "algorithm", "algorithms", "network", "networks",
    "feature", "features", "performance", "accuracy", "application", "applications", "agent", "agents",
    "set", "sets", "number", "numbers", "fixed", "constant", "known", "case", "cases", "size", "sizes",
    "target", "targets", "mean", "means", "random", "given", "single", "standard", "directly",
    "audio", "brain", "human", "molecular", "chemical", "retrieval", "reconstruction", "decoding",
    "tokenizer", "tokenization", "token", "tokens", "reasoning", "evaluation", "annotation", "analysis",
    "judge", "trading", "optimal", "approximation", "polynomial", "linear", "gradient", "gaussian",
    "noise", "mechanism", "mechanisms", "policy", "policies", "social", "enterprise", "evidence",
    "question", "answer", "answers", "similarity", "influence", "effects", "impact", "memory", "knowledge",
}

INVALID_START = {"using", "based", "results", "data", "models", "systems", "effects", "challenges"}
INVALID_END = {
    "large", "small", "global", "efficient", "distributed", "financial", "multimodal", "mathematical",
    "political", "visual", "retrieved", "dense", "inertial", "noisy", "cyrillic", "russian",
    "multilingual", "cross-lingual", "clinical", "medical", "neural", "social", "mechanistic",
    "spiking", "acoustic", "causal", "temporal", "synthetic", "spoken", "verifiable", "robotic",
    "retinal", "long-horizon", "memory-efficient", "hybrid-attention", "language", "information",
}

ALIAS_TO_CANONICAL = {
    "asr": "speech recognition",
    "grpo": "group relative policy optimization",
    "rlvr": "verifiable reward learning",
    "rag": "retrieval augmented generation",
    "vlm": "vision-language models",
    "vlms": "vision-language models",
    "marl": "multi agent reinforcement learning",
    "automatic speech recognition": "speech recognition",
    "vlm": "vision-language models",
    "vlms": "vision-language models",
    "vision language models": "vision-language models",
    "vision-language model": "vision-language models",
    "vision-language-action model": "vision-language-action models",
    "vision-language-action models": "vision-language-action models",
    "vla": "vision-language-action models",
    "vlas": "vision-language-action models",
    "speech recognition system": "speech recognition",
    "retrieval augmented generation": "retrieval augmented generation",
    "grpo": "group relative policy optimization",
    "rlvr": "verifiable reward learning",
    "rag": "retrieval augmented generation",
    "retrieval-augmented generation": "retrieval augmented generation",
    "marl": "multi agent reinforcement learning",
    "electrocardiogram": "ecg",
    # Carefully selected economic single-word concepts. They are admitted as
    # candidates only when the group-level support threshold is met.
    "market": "markets",
    "markets": "markets",
    "trade": "trade",
    "labor": "labor",
    "labour": "labor",
    "wage": "wages",
    "wages": "wages",
    "inequality": "inequality",
    "pricing": "pricing",
    "employment": "employment",
    "investment": "investment",
    "finance": "finance",
    "fiscal": "fiscal policy",
}

ACRONYM_DISPLAY = {
    "ecg": "ECG", "llm": "LLM", "llms": "LLMs", "asr": "ASR", "vlm": "VLM",
    "grpo": "GRPO", "rlvr": "RLVR", "rag": "RAG", "gpu": "GPU", "kv": "KV",
    "marl": "MARL", "dft": "DFT", "poi": "POI",
}

REFERENCE_RE = re.compile(r"(?im)^\s*(?:\d+(?:\.\d+)*\s+)?(?:references|bibliography|works cited)\s*$")
INTRO_RE = re.compile(r"(?im)^\s*(?:\d+(?:\.\d+)*\s+)?introduction\s*$")
KEYWORDS_RE = re.compile(r"(?im)^\s*(?:keywords?|index terms|key words)\s*[:\-]")


@dataclass
class TopicConfig:
    max_abstract_chars: int = 5000
    fallback_intro_chars: int = 2200
    max_features: int = 60000
    ngram_max: int = 4
    min_candidate_prevalence: float = 0.03
    # Retained for compatibility with older configs; current multi-term naming
    # uses the adaptive support/title/contrast thresholds in topic_naming.py.
    min_primary_prevalence: float = 0.12
    min_secondary_prevalence: float = 0.08
    max_topics_per_group: int = 3
    min_subtopic_group_size: int = 24
    max_subtopics: int = 5
    min_subtopic_silhouette: float = 0.16
    min_subtopic_size: int = 5
    random_state: int = 42


class ClusterTopicUnderstanding:
    def __init__(self, config: Optional[TopicConfig] = None):
        self.config = config or TopicConfig()
        self.stop_words = STOPWORDS

    @staticmethod
    def _is_noise(value: object) -> bool:
        return str(value).strip().lower() in {"noise", "-1", "none", "nan", "unassigned", "outlier"}

    @staticmethod
    def _clean_lines(raw: str) -> list[str]:
        text = unicodedata.normalize("NFKC", str(raw or "")).replace("\x00", " ").replace("\f", "\n")
        text = re.sub(r"-\s*\n\s*", "", text)
        text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.I)
        text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", " ", text)
        text = re.sub(r"\b10\.\d{4,9}/[^\s]+", " ", text)
        lines = []
        seen = set()
        for line in text.splitlines():
            line = re.sub(r"\s+", " ", line).strip()
            if not line:
                continue
            low = line.casefold()
            if len(line) < 180 and re.search(r"\b(university|department|faculty|institute|laboratory|college|proceedings of)\b", low):
                continue
            if re.search(r"\b(arxiv preprint|copyright|all rights reserved|creative commons|submitted to|accepted at|published as a conference paper)\b", low):
                continue
            key = re.sub(r"\W+", " ", low).strip()
            if len(key) < 160 and key in seen:
                continue
            seen.add(key)
            lines.append(line)
        return lines

    def _extract_abstract(self, raw: str, title: str) -> tuple[str, str]:
        lines = self._clean_lines(raw)
        text = "\n".join(lines)
        # Strip a possible duplicate title at the very beginning.
        if title:
            title_norm = re.sub(r"\W+", " ", title.lower()).strip()
            first = re.sub(r"\W+", " ", text[:500].lower()).strip()
            if title_norm and title_norm in first:
                text = re.sub(re.escape(title), " ", text, count=1, flags=re.I)
        # Explicit abstract heading is the preferred source.
        match = re.search(r"(?im)^\s*(?:abstract|summary)\s*[:.\-–—]?\s*$", text)
        if match:
            tail = text[match.end():]
            cuts = []
            for pattern in (INTRO_RE, KEYWORDS_RE, REFERENCE_RE):
                m = pattern.search(tail)
                if m and m.start() > 50:
                    cuts.append(m.start())
            if cuts:
                tail = tail[:min(cuts)]
            abstract = re.sub(r"\s+", " ", tail).strip()[: self.config.max_abstract_chars]
            if len(abstract) >= 60:
                return abstract, "explicit_abstract"
        # Some converters put the Abstract heading and text on the same line.
        match = re.search(r"(?i)\b(?:abstract|summary)\s*[:.\-–—]\s+", text)
        if match:
            tail = text[match.end():]
            cuts = []
            for pattern in (INTRO_RE, KEYWORDS_RE, REFERENCE_RE):
                m = pattern.search(tail)
                if m and m.start() > 50:
                    cuts.append(m.start())
            if cuts:
                tail = tail[:min(cuts)]
            abstract = re.sub(r"\s+", " ", tail).strip()[: self.config.max_abstract_chars]
            if len(abstract) >= 60:
                return abstract, "inline_abstract"
        # Conservative fallback: first-page/intro excerpt only. This mode is
        # explicitly exposed in output so it is not mistaken for an abstract.
        fallback = []
        for line in lines[:35]:
            low = line.lower()
            if re.match(r"^(?:\d+(?:\.\d+)*\s+)?(?:introduction|contents|keywords?)\b", low):
                continue
            if len(line) < 180 and re.search(r"\b(university|department|institute|laboratory|conference|workshop)\b", low):
                continue
            if title and line.strip().lower() == title.strip().lower():
                continue
            fallback.append(line)
        excerpt = re.sub(r"\s+", " ", " ".join(fallback)).strip()[: self.config.fallback_intro_chars]
        if len(excerpt) < 80:
            return "", "title_only"
        return excerpt, "intro_fallback"

    @staticmethod
    def _segments(text: str):
        text = unicodedata.normalize("NFKD", str(text or ""))
        text = text.encode("ascii", "ignore").decode("ascii").lower()
        # Punctuation and sentence boundaries remain hard boundaries. Hyphens
        # remain inside scientific compounds such as vision-language.
        for segment in re.split(rf"{re.escape(BOUNDARY)}|[\n\r.!?;:,()\[\]{{}}]+", text):
            tokens = re.findall(r"(?u)\b[a-z][a-z0-9]*(?:-[a-z0-9]+)*\b", segment)
            for n in range(1, 5):
                if len(tokens) < n:
                    continue
                for i in range(len(tokens) - n + 1):
                    chunk = tokens[i:i+n]
                    if any(t in STOPWORDS for t in chunk):
                        continue
                    yield " ".join(chunk)

    @staticmethod
    def _valid_phrase(phrase: str) -> bool:
        p = re.sub(r"\s+", " ", str(phrase).lower()).strip()
        toks = p.split()
        if not toks or len(toks) > 4:
            return False
        if len(toks) == 1:
            return p in ALIAS_TO_CANONICAL
        if any(t in STOPWORDS for t in toks):
            return False
        if toks[0] in INVALID_START or toks[-1] in INVALID_END:
            return False
        if len(toks) >= 3 and toks[0] in {"model", "models", "system", "systems", "benchmark", "benchmarks", "dataset", "datasets"}:
            return False
        if len(toks) >= 3 and toks[-1] in {"evaluation", "benchmark", "benchmarks"} and toks[0] not in {"llm", "agent", "safety", "clinical"}:
            return False
        if any(re.fullmatch(r"[a-z]{1,2}", t) for t in toks):
            return False
        if any(re.search(r"\d", t) for t in toks):
            return False
        if p in {"information processing systems", "information processing systems neurips", "general and reference"}:
            return False
        if len(set(toks)) < len(toks):
            return False
        # Avoid malformed modifier stacks and fragments.
        if toks[0] in {"large", "small", "new", "general", "efficient", "distributed", "target"} and len(toks) == 2:
            if toks[1] not in {"language", "language models", "scale", "dataset"}:
                return False
        return True

    @staticmethod
    def _canonical(phrase: str) -> str:
        p = re.sub(r"\s+", " ", str(phrase).lower()).strip()
        p = p.replace("retrieval-augmented generation", "retrieval augmented generation")
        return ALIAS_TO_CANONICAL.get(p, p)

    @staticmethod
    def _display(phrase: str) -> str:
        p = ClusterTopicUnderstanding._canonical(phrase)
        if p in {"speech recognition"}:
            return "Speech Recognition"
        if p == "vision-language models":
            return "Vision-Language Models"
        if p == "retrieval augmented generation":
            return "Retrieval-Augmented Generation"
        if p == "group relative policy optimization":
            return "Group Relative Policy Optimization (GRPO)"
        if p == "verifiable reward learning":
            return "Verifiable Reward Learning (RLVR)"
        words = []
        for i, token in enumerate(p.split()):
            if token in ACRONYM_DISPLAY:
                words.append(ACRONYM_DISPLAY[token])
            elif token in {"vision-language", "cross-lingual", "on-policy", "time-series", "next-poi"}:
                words.append(token.title())
            elif i > 0 and token in {"and", "of", "for", "with", "in", "on", "to", "via"}:
                words.append(token)
            else:
                words.append(token[:1].upper() + token[1:])
        return " ".join(words)

    @staticmethod
    def _phrase_similarity(a: str, b: str) -> float:
        aa, bb = set(a.split()), set(b.split())
        return len(aa & bb) / max(1, len(aa | bb))

    def _build_candidates(
        self,
        group_id: str,
        member_idx: np.ndarray,
        binary_matrix: sparse.csr_matrix,
        title_matrix: sparse.csr_matrix,
        feature_names: np.ndarray,
        all_group_members: list[np.ndarray],
        group_prevalence: np.ndarray,
        group_position: int,
        n_groups: int,
        literal_binary_matrix: sparse.csr_matrix,
        literal_title_matrix: sparse.csr_matrix,
    ) -> pd.DataFrame:
        n = len(member_idx)
        own = np.asarray(binary_matrix[member_idx].sum(axis=0)).ravel()
        title = np.asarray(title_matrix[member_idx].sum(axis=0)).ravel()
        literal_own = np.asarray(literal_binary_matrix[member_idx].sum(axis=0)).ravel()
        literal_title = np.asarray(literal_title_matrix[member_idx].sum(axis=0)).ravel()
        global_doc = np.asarray(binary_matrix.sum(axis=0)).ravel()
        total_docs = binary_matrix.shape[0]
        global_prev = global_doc / max(total_docs, 1)
        group_counts = np.asarray(binary_matrix[member_idx].sum(axis=0)).ravel()
        group_term_groups = (group_prevalence > 0).sum(axis=0)
        ctfidf = (group_counts / max(group_counts.sum(), 1.0)) * np.log1p(n_groups / np.maximum(group_term_groups, 1))
        min_support = max(2, int(math.ceil(self.config.min_candidate_prevalence * n)))
        rows = []
        for j, phrase in enumerate(feature_names):
            phrase = str(phrase)
            if not self._valid_phrase(phrase):
                continue
            support = int(own[j])
            if support < min_support:
                continue
            phrase_len = len(phrase.split())
            # Four-word fragments are frequently clause remnants in PDF text.
            # Keep them only when independently corroborated by at least two titles.
            if phrase_len >= 4 and int(title[j]) < 2:
                continue
            prev = support / max(n, 1)
            title_support = int(title[j])
            title_prev = title_support / max(n, 1)
            other = np.delete(group_prevalence[:, j], group_position)
            other_max = float(other.max()) if len(other) else 0.0
            lift = math.log((prev + 0.01) / (float(global_prev[j]) + 0.01))
            contrast_margin = prev - other_max
            contrast = min(1.0, max(0.0, (math.log((prev + 0.015)/(other_max + 0.015)) / math.log(8.0))))
            rows.append({
                "group_id": group_id, "phrase": phrase, "canonical_phrase": self._canonical(phrase),
                "support_docs": support,
                "literal_support_docs": int(literal_own[j]),
                "ontology_only_support_docs": max(0, support - int(literal_own[j])),
                "prevalence": prev,
                "title_support_docs": title_support,
                "literal_title_support_docs": int(literal_title[j]),
                "ontology_title_support_docs": max(0, title_support - int(literal_title[j])),
                "title_prevalence": title_prev, "global_prevalence": float(global_prev[j]),
                "max_other_group_prevalence": other_max, "contrast_margin": contrast_margin,
                "contrast_score": contrast, "lift": lift, "ctfidf": float(ctfidf[j]),
            })
        if not rows:
            return pd.DataFrame()
        out = pd.DataFrame(rows)
        # Collapse aliases and trivial singular/plural duplicates, keeping the
        # strongest real-source phrase rather than adding supports together.
        out["_canonical_len"] = out["canonical_phrase"].str.split().str.len()
        out = out.sort_values(
            ["support_docs", "title_support_docs", "contrast_score", "_canonical_len"],
            ascending=[False, False, False, False],
        ).drop_duplicates("canonical_phrase", keep="first")
        ctfidf_rank = out["ctfidf"].rank(pct=True).to_numpy()
        prev_score = np.minimum(1.0, out["prevalence"].to_numpy() / 0.40)
        title_score = np.minimum(1.0, out["title_prevalence"].to_numpy() / 0.10)
        length_score = np.where(out["canonical_phrase"].str.split().str.len().to_numpy() >= 2, 1.0, 0.8)
        out["score"] = (
            0.34 * prev_score
            + 0.18 * title_score
            + 0.18 * out["contrast_score"].to_numpy()
            + 0.18 * ctfidf_rank
            + 0.12 * length_score
        )
        out = out.sort_values(["score", "support_docs", "title_support_docs"], ascending=False)
        return out.drop(columns=["_canonical_len"]).head(180).reset_index(drop=True)

    def _select_topics(self, candidates: pd.DataFrame, group_size: int) -> list[dict]:
        """Select naming terms with a title/specificity-aware score.

        The original extractor score remains in the candidate table for audit.
        Naming uses a separate score so common but generic phrases do not
        automatically dominate the user-facing title.
        """
        return select_naming_terms(candidates, group_size, self.config.max_topics_per_group)

    @staticmethod
    def _centroid_coherence(matrix: sparse.csr_matrix) -> tuple[float, float, float, np.ndarray]:
        if matrix.shape[0] <= 1:
            return 1.0, 1.0, 1.0, np.ones(matrix.shape[0])
        centroid = normalize(np.asarray(matrix.mean(axis=0)))
        sims = np.asarray(matrix @ centroid.T).ravel()
        if matrix.shape[0] <= 450:
            pairwise = (matrix @ matrix.T).toarray()
            tri = pairwise[np.triu_indices(matrix.shape[0], k=1)]
            pair_mean = float(tri.mean()) if len(tri) else 1.0
        else:
            rng = np.random.default_rng(42)
            pairs_a = rng.integers(0, matrix.shape[0], 12000)
            pairs_b = rng.integers(0, matrix.shape[0], 12000)
            vals = np.asarray(matrix[pairs_a].multiply(matrix[pairs_b]).sum(axis=1)).ravel()
            pair_mean = float(vals.mean())
        return float(sims.mean()), float(np.median(sims)), pair_mean, sims

    def _subtopic_diagnostic(self, matrix: sparse.csr_matrix, doc_ids: np.ndarray) -> tuple[dict, np.ndarray]:
        n = matrix.shape[0]
        assignments = np.full(n, -1, dtype=int)
        empty = {"detected": False, "k": 1, "silhouette": None, "reason": "not_tested"}
        if n < self.config.min_subtopic_group_size:
            empty["reason"] = "group_too_small"
            return empty, assignments
        best = None
        upper = min(self.config.max_subtopics, max(2, n // self.config.min_subtopic_size))
        signatures = set()
        csr = matrix.tocsr()
        for i in range(csr.shape[0]):
            start, end = csr.indptr[i], csr.indptr[i + 1]
            signatures.add((csr.indices[start:end].tobytes(), np.round(csr.data[start:end], 6).tobytes()))
        upper = min(upper, len(signatures))
        for k in range(2, upper + 1):
            try:
                km = KMeans(n_clusters=k, random_state=self.config.random_state, n_init=10)
                labels = km.fit_predict(matrix)
                if len(np.unique(labels)) != k:
                    continue
                sizes = np.bincount(labels, minlength=k)
                if sizes.min() < self.config.min_subtopic_size:
                    continue
                sil = float(silhouette_score(
                    matrix, labels, metric="cosine",
                    sample_size=min(250, n), random_state=self.config.random_state,
                ))
                if best is None or sil > best[0]:
                    best = (sil, k, labels, sizes)
            except Exception:
                continue
        if best is None:
            empty["reason"] = "no_valid_partition"
            return empty, assignments
        sil, k, labels, sizes = best
        if sil < self.config.min_subtopic_silhouette:
            empty["reason"] = "no_stable_subtopics"
            empty["best_tested_silhouette"] = round(sil, 4)
            return empty, assignments
        empty = {
            "detected": True, "k": int(k), "silhouette": round(sil, 4),
            "sizes": [int(x) for x in sizes], "reason": "internal_structure_detected",
        }
        return empty, labels.astype(int)

    def fit_predict(self, assignments: pd.DataFrame, texts: pd.DataFrame) -> dict[str, pd.DataFrame | dict]:
        required = {"document_id", "group_id", "title"}
        if not required.issubset(assignments.columns):
            raise ValueError(f"Assignments must contain columns: {sorted(required)}")
        if not {"document_id", "text"}.issubset(texts.columns):
            raise ValueError("Texts must contain document_id and text columns")
        joined = assignments.merge(texts[["document_id", "text"]], on="document_id", how="left")
        joined["text"] = joined["text"].fillna("").astype(str)
        joined["title"] = joined["title"].fillna("").astype(str)
        joined["abstract"], joined["text_source_mode"] = zip(*[
            self._extract_abstract(t, title) for t, title in zip(joined["text"], joined["title"])
        ])
        joined["analysis_text"] = [
            f"{title} {BOUNDARY} {abstract}" if abstract else title
            for title, abstract in zip(joined["title"], joined["abstract"])
        ]
        eligible_mask = ~joined["group_id"].map(self._is_noise)
        work = joined.loc[eligible_mask].copy().reset_index(drop=True)
        noise = joined.loc[~eligible_mask].copy()
        if len(work) < 2:
            raise ValueError("Not enough assigned documents with non-noise cluster labels")

        # Add a small, corpus-independent ontology layer only to the phrase
        # evidence matrices. The coherence matrix below still uses the original
        # title+abstract text, so synthetic tags cannot inflate similarity.
        ontology = [detect_ontology_concepts(title, abstract)
                    for title, abstract in zip(work["title"], work["abstract"])]
        # Keep title evidence honest: concepts discovered only in an abstract
        # may enrich document-level candidates, but must not inflate title_support.
        title_ontology = [detect_ontology_concepts(title, "") for title in work["title"]]
        work["ontology_concepts"] = [" | ".join(items) for items in ontology]
        work["candidate_text"] = [
            f"{text} {BOUNDARY} {' '.join(items)}" if items else text
            for text, items in zip(work["analysis_text"], ontology)
        ]
        work["candidate_title_text"] = [
            f"{title} {BOUNDARY} {' '.join(items)}" if items else title
            for title, items in zip(work["title"], title_ontology)
        ]

        analyzer = lambda text: list(self._segments(text))
        count_vec = CountVectorizer(analyzer=analyzer, binary=True, max_features=self.config.max_features, min_df=2)
        binary = count_vec.fit_transform(work["candidate_text"]).tocsr()
        feature_names = count_vec.get_feature_names_out()
        title_binary = count_vec.transform(work["candidate_title_text"]).tocsr()
        # Parallel lexical-only matrices let the CSV distinguish literal text
        # matches from deterministic ontology tags injected for candidate recall.
        literal_binary = count_vec.transform(work["analysis_text"]).tocsr()
        literal_title_binary = count_vec.transform(work["title"]).tocsr()
        tfidf_vec = TfidfVectorizer(
            analyzer=analyzer, max_features=self.config.max_features,
            min_df=1, max_df=0.99, sublinear_tf=True, norm="l2",
        )
        doc_matrix = tfidf_vec.fit_transform(work["analysis_text"]).tocsr()
        groups = sorted(work["group_id"].astype(str).unique(), key=self._sort_group)
        group_members = [np.flatnonzero(work["group_id"].astype(str).to_numpy() == g) for g in groups]
        group_prevs = []
        for idx in group_members:
            group_prevs.append(np.asarray(binary[idx].sum(axis=0)).ravel() / max(len(idx), 1))
        group_prevalence = np.vstack(group_prevs)

        profiles = []
        candidate_frames = []
        document_rows = []
        subtopic_rows = []
        subtopic_assignment_rows = []

        for pos, (group, idx) in enumerate(zip(groups, group_members)):
            n = len(idx)
            candidates = self._build_candidates(
                group, idx, binary, title_binary, feature_names,
                group_members, group_prevalence, pos, len(groups),
                literal_binary, literal_title_binary,
            )
            if not candidates.empty:
                candidates = score_naming_candidates(candidates, n)
                candidates["canonical_phrase_ru"] = candidates["canonical_phrase"].map(translate_term)
                candidates["translation_source"] = candidates["canonical_phrase"].map(translation_source)
                candidate_frames.append(candidates)
            selected = self._select_topics(candidates, n)
            coh_mean, coh_median, pair_mean, similarities = self._centroid_coherence(doc_matrix[idx])
            subdiag, local_subtopics = self._subtopic_diagnostic(doc_matrix[idx], idx)
            group_subtopic_start = len(subtopic_rows)

            # Select representative and review candidates by similarity to the
            # title+abstract centroid, not by phrase frequency alone.
            order = np.argsort(similarities)[::-1]
            rep_titles = [str(work.iloc[idx[j]]["title"]) for j in order[:5]]
            out_titles = [str(work.iloc[idx[j]]["title"]) for j in order[-5:][::-1]]

            if selected:
                display_topics = [self._display(x["canonical_phrase"]) for x in selected]
                label = ", ".join(display_topics)
                description = (
                    f"Основные тематические направления группы: {display_topics[0]}. "
                    + ("Дополнительные подтверждённые аспекты: " + "; ".join(display_topics[1:]) + ". " if len(display_topics) > 1 else "")
                    + "Темы извлечены из заголовков и аннотаций публикаций; исходные назначения кластеризации не изменялись."
                )
                primary = selected[0]
                support = float(primary["prevalence"])
                title_support = int(primary["title_support_docs"])
                distinct = float(primary["contrast_score"])
                evidence = 0.45 * support + 0.30 * distinct + 0.25 * coh_mean
                if evidence >= 0.58 and support >= 0.20 and title_support >= 2 and coh_mean >= 0.22:
                    quality = "STRONG_PROFILE"
                elif evidence >= 0.36 and support >= 0.10:
                    quality = "MODERATE_PROFILE"
                else:
                    quality = "REVIEW_PROFILE"
            else:
                display_topics = []
                label = "Mixed research topics — review recommended"
                description = "Не найдено одно достаточно устойчивое тематическое понятие с необходимой поддержкой. Группа требует просмотра нескольких тематических аспектов и, возможно, внутреннего разделения."
                support = 0.0
                title_support = 0
                distinct = 0.0
                evidence = 0.0
                quality = "MIXED_TOPIC_REVIEW"

            if subdiag["detected"]:
                quality = "INTERNAL_SUBTOPICS_DETECTED"
                for sub_id in sorted(set(local_subtopics.tolist())):
                    sub_local = np.flatnonzero(local_subtopics == sub_id)
                    global_indices = idx[sub_local]
                    submatrix = doc_matrix[global_indices]
                    submean, submedian, subpair, subsims = self._centroid_coherence(submatrix)
                    sub_order = np.argsort(subsims)[::-1]
                    # Re-rank genuine phrases within the subtopic, rather than
                    # taking the most frequent individual tokens.
                    sub_counts = np.asarray(binary[global_indices].sum(axis=0)).ravel()
                    sub_titles = np.asarray(title_binary[global_indices].sum(axis=0)).ravel()
                    sub_support = max(2, int(math.ceil(0.12 * len(global_indices))))
                    candidates_local = []
                    for j, count in enumerate(sub_counts):
                        if count < sub_support:
                            continue
                        phrase = str(feature_names[j])
                        if not self._valid_phrase(phrase):
                            continue
                        prev_local = float(count / len(global_indices))
                        title_local = float(sub_titles[j] / len(global_indices))
                        score_local = 0.68 * min(1.0, prev_local / 0.45) + 0.32 * min(1.0, title_local / 0.25)
                        candidates_local.append((score_local, phrase, int(count)))
                    candidates_local.sort(reverse=True)
                    candidate_terms = []
                    canonical_seen = set()
                    for _, phrase, _ in candidates_local:
                        canonical = self._canonical(phrase)
                        if canonical in canonical_seen:
                            continue
                        if any(self._phrase_similarity(canonical, old) >= 0.55 for old in canonical_seen):
                            continue
                        candidate_terms.append(phrase)
                        canonical_seen.add(canonical)
                        if len(candidate_terms) == 3:
                            break
                    sub_label = ", ".join(self._display(x) for x in candidate_terms) if candidate_terms else "Unresolved subtopic"
                    sub_label_ru = " и ".join(translate_term(self._canonical(x)) for x in candidate_terms) if candidate_terms else "Подтема не определена"
                    subtopic_rows.append({
                        "group_id": group, "subtopic_id": f"{group}.T{sub_id+1}",
                        "document_count": len(global_indices), "subtopic_label": sub_label_ru,
                        "subtopic_label_en": sub_label, "subtopic_label_ru": sub_label_ru,
                        "mean_centroid_similarity": round(submean, 4),
                        "mean_pairwise_similarity": round(subpair, 4),
                        "representative_titles": " | ".join(str(work.iloc[global_indices[j]]["title"]) for j in sub_order[:4]),
                    })
                    for global_i in global_indices:
                        subtopic_assignment_rows.append({
                            "document_id": work.iloc[global_i]["document_id"],
                            "group_id": group, "subtopic_id": f"{group}.T{sub_id+1}",
                        })
                group_subtopic_labels = [
                    x.get("subtopic_label_en", x["subtopic_label"])
                    for x in subtopic_rows[group_subtopic_start:]
                    if x["group_id"] == group
                ]
                label = "Mixed research directions: " + "; ".join(group_subtopic_labels)
                description = (
                    f"Внутри группы автоматически обнаружено {len(group_subtopic_labels)} устойчивых тематических подгрупп: "
                    + "; ".join(group_subtopic_labels)
                    + ". Это диагностическое внутреннее разбиение; исходные назначения кластеризации не изменялись."
                )

            naming = build_topic_name(candidates, n, quality)
            label_ru = naming["automatic_label_ru"]
            description_ru = naming["automatic_description_ru"]
            if subdiag["detected"]:
                sub_labels_ru = [
                    x.get("subtopic_label_ru", x.get("subtopic_label", "Подтема не определена"))
                    for x in subtopic_rows[group_subtopic_start:]
                    if x["group_id"] == group
                ]
                label_ru = "Несколько тематических направлений: " + "; ".join(sub_labels_ru)
                naming["automatic_label_en"] = label
                description_ru = (
                    f"Внутри сообщества обнаружено {len(sub_labels_ru)} внутренних тематических направлений: "
                    + "; ".join(sub_labels_ru)
                    + ". Это диагностическая структура; исходные назначения графовой кластеризации не изменялись."
                )
            # A selected phrase may be too weak for a declarative label. Keep
            # the Russian abstention when the profile itself requires review.
            if quality == "MIXED_TOPIC_REVIEW" and not naming["selected_topics"]:
                label_ru = "Разнородные исследования — требуется тематическая проверка"

            profiles.append({
                "group_id": group,
                "document_count": n,
                "automatic_label": label_ru,
                "automatic_label_en": naming["automatic_label_en"],
                "automatic_label_ru": label_ru,
                "automatic_description": description_ru,
                "automatic_description_ru": description_ru,
                "topic_aspects": naming["topic_aspects_ru"],
                "topic_aspects_en": naming["topic_aspects_en"],
                "topic_aspects_ru": naming["topic_aspects_ru"],
                "naming_confidence": naming["naming_confidence"],
                "naming_confidence_score": naming["naming_confidence_score"],
                "naming_confidence_basis": naming["naming_confidence_basis"],
                "top_term_naming_score": naming["top_term_naming_score"],
                "naming_evidence": json.dumps(naming["naming_evidence"], ensure_ascii=False),
                "primary_topic_support": round(support, 4),
                "primary_topic_title_support_docs": title_support,
                "primary_topic_distinctiveness": round(distinct, 4),
                "mean_title_abstract_centroid_similarity": round(coh_mean, 4),
                "median_title_abstract_centroid_similarity": round(coh_median, 4),
                "mean_pairwise_title_abstract_similarity": round(pair_mean, 4),
                "topic_profile_score": round(evidence, 4),
                "topic_profile_status": quality,
                "abstract_source_distribution": json.dumps(work.iloc[idx]["text_source_mode"].value_counts().to_dict(), ensure_ascii=False),
                "internal_subtopic_diagnostic": json.dumps(subdiag, ensure_ascii=False),
                "representative_titles": " | ".join(rep_titles),
                "review_titles": " | ".join(out_titles),
            })

            submap = {}
            if subdiag["detected"]:
                submap = {int(local_i): int(sub_id) for local_i, sub_id in enumerate(local_subtopics)}
            for local_i, global_i in enumerate(idx):
                row = work.iloc[global_i]
                document_rows.append({
                    "document_id": row["document_id"],
                    "group_id": group,
                    "title": row["title"],
                    "text_source_mode": row["text_source_mode"],
                    "abstract_chars": len(row["abstract"]),
                    "centroid_similarity": round(float(similarities[local_i]), 4),
                    "automatic_label": label_ru,
                    "automatic_label_en": naming["automatic_label_en"],
                    "automatic_label_ru": label_ru,
                    "topic_profile_status": quality,
                    "subtopic_id": f"{group}.T{submap[local_i]+1}" if local_i in submap else "",
                    "ontology_concepts": row.get("ontology_concepts", ""),
                })

        labels_df = pd.DataFrame(profiles)
        candidates_df = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
        doc_df = pd.DataFrame(document_rows)
        subtopics_df = pd.DataFrame(subtopic_rows)
        subtopic_assign_df = pd.DataFrame(subtopic_assignment_rows)
        noise_df = noise[["document_id", "group_id", "title"]].copy()
        if not noise_df.empty:
            noise_df["text_source_mode"] = "not_analyzed_noise"
            noise_df["abstract_chars"] = 0
            noise_df["centroid_similarity"] = np.nan
            noise_df["automatic_label"] = "Не распределено"
            noise_df["automatic_label_en"] = "Unassigned noise"
            noise_df["automatic_label_ru"] = "Не распределено"
            noise_df["topic_profile_status"] = "NOT_APPLICABLE"
            noise_df["subtopic_id"] = ""
            noise_df["ontology_concepts"] = ""
            doc_df = pd.concat([doc_df, noise_df], ignore_index=True)

        metadata = {
            "engine": "Cluster Topic Understanding",
            "version": "1.2.1",
            "method": "title + canonical abstract topic profiling; title-aware and specificity-aware term ranking; lexical-versus-ontology evidence provenance; evidence-sensitive bilingual naming; separate name-confidence and profile-coherence diagnostics; TF-IDF coherence; optional within-cluster KMeans/silhouette diagnostics",
            "uses_llm": False,
            "uses_external_translation_service": False,
            "uses_group_id_specific_topic_rules": False,
            "uses_corpus_independent_concept_patterns": True,
            "synthetic_concept_tags_used_for_candidate_extraction_only": True,
            "cluster_assignments_changed": False,
            "uses_legacy_automatic_labels": False,
            "source_documents": int(len(joined)),
            "documents_analyzed": int(len(work)),
            "noise_documents": int(len(noise)),
            "groups_analyzed": int(len(groups)),
            "text_source_modes": joined["text_source_mode"].value_counts().to_dict(),
            "config": asdict(self.config),
            "important_limitations": [
                "Topic evidence is extracted primarily from title and canonical abstract; when no abstract is available, the PDF intro fallback is used and flagged.",
                "Russian naming is deterministic and dictionary-backed; unknown terms may remain in English rather than being mistranslated.",
                "Candidate CSV distinguishes literal lexical support from ontology-only support; title_support_docs includes both forms of title evidence, with literal_title_support_docs and ontology_title_support_docs reported separately.",
                "Naming confidence is a separate heuristic about term evidence; topic_profile_status describes textual profile support/coherence. Neither score is a calibrated probability.",
                "Ontology-derived synthetic tags are used only in the candidate-count matrices; the TF-IDF coherence matrix uses unmodified title+abstract text.",
                "Naming scores and topic profile scores are heuristic evidence summaries, not calibrated probabilities.",
                "Internal KMeans subtopics are diagnostic and do not change the original graph/density-clustering assignments.",
                "The 2D UMAP x/y coordinates are not treated as semantic embeddings.",
                "No SPECTER embedding matrix is loaded unless an explicit verified embedding integration is added; current coherence uses TF-IDF on titles and abstracts.",
                "No automatic cluster reassignment is performed.",
            ],
        }
        return {
            "profiles": labels_df,
            "candidates": candidates_df,
            "documents": doc_df,
            "subtopics": subtopics_df,
            "subtopic_assignments": subtopic_assign_df,
            "metadata": metadata,
        }

    @staticmethod
    def _sort_group(value: str):
        m = re.fullmatch(r"([A-Za-z]+)(\d+)", str(value))
        return (m.group(1).lower(), int(m.group(2))) if m else (str(value).lower(), 0)


def load_document_texts(
    assignments: pd.DataFrame,
    processed_dir: Path,
    abstract_by_id: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """Load extracted bodies and prepend canonical source abstracts when available.

    The arXiv metadata abstract is a cleaner semantic signal than PDF text extraction.
    It is kept separate from the extracted body and explicitly marked so the topic
    parser always analyses it first. Missing body files are still reported.
    """
    texts = []
    missing = []
    abstract_by_id = abstract_by_id or {}
    for doc_id in assignments["document_id"].fillna("").astype(str):
        path = Path(processed_dir) / doc_id / "document.txt"
        try:
            body = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
        except OSError:
            body = ""
        abstract = str(abstract_by_id.get(doc_id, "") or "").strip()
        if not body:
            missing.append(doc_id)
        # Prefer the canonical arXiv abstract as a standalone source. Appending
        # extracted PDF text could introduce a second Abstract heading, causing
        # the parser to select PDF-extracted text instead of the canonical value.
        # When no canonical abstract exists, retain the previous PDF-body route.
        if abstract:
            text = f"Abstract\n{abstract}\n\nIntroduction\n"
        else:
            text = body
        texts.append(text)
    out = assignments[["document_id"]].copy()
    out["text"] = texts
    out.attrs["missing_document_ids"] = missing
    out.attrs["canonical_abstract_count"] = sum(bool(str(abstract_by_id.get(x, "") or "").strip()) for x in assignments["document_id"].fillna("").astype(str))
    return out


def run_topic_understanding(
    assignments_path: Path,
    processed_dir: Path,
    output_dir: Path,
    config: Optional[TopicConfig] = None,
    abstract_by_id: Optional[dict[str, str]] = None,
) -> dict[str, Path]:
    assignments_path = Path(assignments_path)
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments = pd.read_csv(assignments_path, dtype={"document_id": str}).fillna("")
    if "group_id" not in assignments.columns and "cluster_id" in assignments.columns:
        assignments = assignments.rename(columns={"cluster_id": "group_id"})
    texts = load_document_texts(assignments, processed_dir, abstract_by_id=abstract_by_id)
    missing = len(texts.attrs.get("missing_document_ids", []))
    canonical_abstract_count = int(texts.attrs.get("canonical_abstract_count", 0))
    engine = ClusterTopicUnderstanding(config)
    result = engine.fit_predict(assignments, texts)
    files = {
        "profiles": output_dir / "group_topic_profiles.csv",
        "candidates": output_dir / "topic_candidates.csv",
        "documents": output_dir / "document_topic_analysis.csv",
        "subtopics": output_dir / "internal_subtopics.csv",
        "subtopic_assignments": output_dir / "internal_subtopic_assignments.csv",
        "metadata": output_dir / "topic_understanding_metadata.json",
    }
    for key in ("profiles", "candidates", "documents", "subtopics", "subtopic_assignments"):
        result[key].to_csv(files[key], index=False, encoding="utf-8-sig")
    result["metadata"]["missing_text_files"] = missing
    result["metadata"]["canonical_source_abstracts"] = canonical_abstract_count
    files["metadata"].write_text(json.dumps(result["metadata"], ensure_ascii=False, indent=2), encoding="utf-8")
    cards = []
    for _, row in result["profiles"].iterrows():
        cards.append({
            "group_id": row["group_id"], "title": row["automatic_label"],
            "description": row["automatic_description"],
            "document_count": int(row["document_count"]),
            "status": row["topic_profile_status"],
            "aspects": [x for x in str(row["topic_aspects"]).split(" | ") if x],
            "representative_titles": [x for x in str(row["representative_titles"]).split(" | ") if x],
            "review_titles": [x for x in str(row["review_titles"]).split(" | ") if x],
        })
    files["cards"] = output_dir / "topic_cards.json"
    files["cards"].write_text(json.dumps(cards, ensure_ascii=False, indent=2), encoding="utf-8")
    return files
