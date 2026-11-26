from __future__ import annotations

import math
import logging
import re
from collections import Counter

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from tqdm import tqdm

from . import db
from .config import settings
from .utils import clean_text

log = logging.getLogger("sciintel")


def extract_for_document(text: str, top_n: int = 25) -> dict[str, list[tuple[str, float]]]:
    text = clean_text(text)
    if not text:
        return {"YAKE": [], "RAKE": []}
    import yake
    from rake_nltk import Rake

    sample = text[:settings.max_keyword_chars]
    yake_extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.9, top=top_n)
    yake_items = [(phrase.strip(), float(score)) for phrase, score in yake_extractor.extract_keywords(sample)]

    rake = Rake(
        stopwords=set(ENGLISH_STOP_WORDS),
        min_length=1,
        max_length=4,
        sentence_tokenizer=lambda value: re.split(r"(?<=[.!?])\s+", value),
        word_tokenizer=lambda value: re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", value),
    )
    rake.extract_keywords_from_text(sample)
    rake_items = [(phrase.strip(), float(score))
                  for score, phrase in rake.get_ranked_phrases_with_scores()]
    # Remove empty / near-duplicate phrases while preserving method-native scores.
    def unique(items: list[tuple[str, float]]) -> list[tuple[str, float]]:
        output, seen = [], set()
        for phrase, score in items:
            key = re.sub(r"\W+", " ", phrase.casefold()).strip()
            if len(key) < 3 or key in seen:
                continue
            seen.add(key)
            output.append((phrase, score))
            if len(output) >= top_n:
                break
        return output
    return {"YAKE": unique(yake_items), "RAKE": unique(rake_items)}


def run_keywords(force: bool = False, top_n: int | None = None) -> dict[str, int]:
    db.init_db()
    top_n = top_n or settings.keyword_top_n
    docs = db.rows("SELECT id, title, abstract, text_path FROM documents WHERE status='ready' ORDER BY id")
    stats = {"documents": len(docs), "processed": 0, "cached": 0, "failed": 0, "terms": 0}
    cached_ids = {row["document_id"] for row in db.rows("SELECT DISTINCT document_id FROM keywords")}
    for doc in tqdm(docs, desc="YAKE + RAKE", unit="doc"):
        if doc["id"] in cached_ids and not force:
            stats["cached"] += 1
            continue
        try:
            body = open(doc["text_path"], "r", encoding="utf-8").read()
            # Title and canonical abstract receive priority; the PDF body adds context.
            text = "\n\n".join(x for x in [doc.get("title", ""), doc.get("abstract", ""), body] if x)
            result = extract_for_document(text, top_n=top_n)
            with db.connect() as conn:
                if force:
                    conn.execute("DELETE FROM keywords WHERE document_id=?", (doc["id"],))
                for method, items in result.items():
                    for rank, (phrase, score) in enumerate(items, start=1):
                        conn.execute("""
                            INSERT OR REPLACE INTO keywords(document_id, method, phrase, score, rank)
                            VALUES (?, ?, ?, ?, ?)
                        """, (doc["id"], method, phrase, score, rank))
                        stats["terms"] += 1
            stats["processed"] += 1
        except Exception as exc:
            stats["failed"] += 1
            log.exception("Keyword extraction failed for %s: %s", doc["id"], exc)
    log.info("Keyword extraction completed: %s", stats)
    return stats



_PROFILE_STOP_PHRASES = {
    "arxiv preprint",
    "arxiv preprint arxiv",
    "preprint arxiv",
    "international conference",
    "conference on",
    "proceedings",
    "machine learning research",
    "association for computational linguistics",
    "computational linguistics",
    "neural information processing systems",
    "advances in neural information processing systems",
    "information processing systems",
}

_PROFILE_STOP_SINGLE = {
    "model", "models",
    "data",
    "task", "tasks",
    "method", "methods",
    "approach", "approaches",
    "result", "results",
    "paper", "papers",
    "study", "studies",
    "work", "works",
    "learning",
    "using", "based",
    "new", "different", "proposed",
    "section", "table", "figure", "appendix",
    "reference", "references",
    "arxiv", "preprint", "url", "proc",
    "conference", "doi", "volume", "page", "pages",
    "author", "authors",
    "lemma", "theorem", "log", "set", "time",
}


def keyword_profile(document_ids: list[str], limit: int = 12) -> list[str]:
    """
    Build an interpretable cluster profile using cluster-level
    document frequency and inverse document frequency.
    """
    if not document_ids:
        return []

    placeholders = ",".join("?" for _ in document_ids)
    params = tuple(document_ids)

    total_rows = db.rows(
        "SELECT COUNT(*) AS n FROM documents WHERE status='ready'"
    )
    total_docs = max(
        1,
        int(total_rows[0]["n"]) if total_rows else len(document_ids)
    )

    # Number of documents containing each phrase in the whole corpus.
    global_rows = db.rows("""
        SELECT LOWER(TRIM(phrase)) AS phrase_key,
               COUNT(DISTINCT document_id) AS global_df
        FROM keywords
        GROUP BY LOWER(TRIM(phrase))
    """)

    global_df = {
        row["phrase_key"]: int(row["global_df"])
        for row in global_rows
    }

    # Number of documents containing each phrase in this cluster.
    found = db.rows(f"""
        SELECT LOWER(TRIM(phrase)) AS phrase_key,
               MIN(TRIM(phrase)) AS phrase,
               COUNT(DISTINCT document_id) AS cluster_df,
               AVG(rank) AS avg_rank
        FROM keywords
        WHERE document_id IN ({placeholders})
        GROUP BY LOWER(TRIM(phrase))
    """, params)

    candidates = []
    cluster_size = max(1, len(document_ids))

    for item in found:
        phrase = str(item["phrase"]).strip()
        key = str(item["phrase_key"]).casefold().strip()

        clean_key = re.sub(r"[^a-z0-9+#]+", " ", key).strip()
        words = clean_key.split()

        if len(clean_key) < 3:
            continue

        if key in _PROFILE_STOP_PHRASES or clean_key in _PROFILE_STOP_PHRASES:
            continue

        if len(words) == 1 and words[0] in _PROFILE_STOP_SINGLE:
            continue

        # Avoid terms appearing in only one publication.
        df = int(item["cluster_df"])
        if df < 2:
            continue

        corpus_df = max(1, global_df.get(key, df))

        # Cluster term frequency multiplied by inverse corpus frequency.
        tf = df / cluster_size
        idf = math.log((total_docs + 1) / (corpus_df + 1))
        score = tf * idf

        if score <= 0:
            continue

        avg_rank = float(item["avg_rank"] or 0)

        candidates.append(
            (score, df, -avg_rank, phrase)
        )

    candidates.sort(reverse=True)

    return [
        phrase
        for _, _, _, phrase in candidates[:limit]
    ]
