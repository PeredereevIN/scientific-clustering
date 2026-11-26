from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

MODEL_NAMES = {"en": "en_core_web_sm", "ru": "ru_core_news_sm"}


@dataclass(frozen=True)
class SentenceRecord:
    sentence_index: int
    text: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class TokenRecord:
    token_index: int
    text: str
    lemma: str
    pos: str
    tag: str
    morph: str
    dep: str
    head_index: int
    char_start: int
    char_end: int
    is_punct: bool
    is_stop: bool


@dataclass(frozen=True)
class ConceptRecord:
    canonical_form: str
    surface_form: str
    token_start: int
    token_end: int
    char_start: int
    char_end: int
    extraction_method: str


@lru_cache(maxsize=4)
def load_nlp(language: str):
    try:
        import spacy
    except ImportError as exc:
        raise RuntimeError("spaCy is required; install project requirements") from exc
    model = MODEL_NAMES.get(language)
    if not model:
        raise ValueError(f"Unsupported language: {language}")
    try:
        return spacy.load(model)
    except OSError as exc:
        raise RuntimeError(f"Language model {model} is missing. Run: python -m spacy download {model}") from exc


def detect_language(text: str) -> str:
    cyrillic = len(re.findall(r"[А-Яа-яЁё]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    return "ru" if cyrillic > latin else "en"


def analyze_text(text: str, language: str = "auto", nlp=None):
    """Return sentence/token/concept records with half-open offsets into input text."""
    if not text:
        return [], [], []
    lang = detect_language(text) if language == "auto" else language
    pipeline = nlp or load_nlp(lang)
    doc = pipeline(text)
    spans = list(doc.sents)
    if not spans and len(doc):
        spans = [doc[:]]
    sentences, tokens, concepts = [], [], []
    for sentence_no, sent in enumerate(spans):
        sentences.append(SentenceRecord(sentence_no, sent.text, sent.start_char, sent.end_char))
        for token in sent:
            tokens.append(TokenRecord(
                token_index=token.i, text=token.text,
                lemma=(token.lemma_ or token.text).casefold(), pos=token.pos_ or "X",
                tag=token.tag_ or "", morph=str(token.morph), dep=token.dep_ or "",
                head_index=token.head.i if token.head is not None else token.i,
                char_start=token.idx, char_end=token.idx + len(token),
                is_punct=bool(token.is_punct or token.pos_ == "PUNCT"),
                is_stop=bool(token.is_stop),
            ))
        candidates = []
        try:
            candidates.extend((chunk.start, chunk.end, "noun_chunk") for chunk in sent.noun_chunks)
        except (ValueError, NotImplementedError):
            pass
        if not candidates:
            start = None
            for token in sent:
                if token.pos_ in {"NOUN", "PROPN", "ADJ"}:
                    if start is None:
                        start = token.i
                elif start is not None:
                    if token.i > start:
                        candidates.append((start, token.i, "pos_sequence"))
                    start = None
            if start is not None and sent.end > start:
                candidates.append((start, sent.end, "pos_sequence"))
        seen = set()
        for start, end, method in candidates:
            if (start, end) in seen:
                continue
            seen.add((start, end))
            span = doc[start:end]
            content = [t for t in span if t.pos_ in {"NOUN", "PROPN", "ADJ"} and not t.is_stop]
            if not content:
                continue
            canonical = " ".join((t.lemma_ or t.text).casefold() for t in content).strip()
            if canonical:
                concepts.append(ConceptRecord(canonical, span.text, start, end,
                                              span.start_char, span.end_char, method))
    return sentences, tokens, concepts
