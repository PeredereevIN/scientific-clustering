from __future__ import annotations

import sqlite3

from .. import db

LINGUISTIC_SCHEMA = """
CREATE TABLE IF NOT EXISTS linguistic_documents (
 document_id TEXT PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
 language TEXT NOT NULL, model_name TEXT NOT NULL, source_path TEXT NOT NULL,
 markdown_path TEXT NOT NULL, indexed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS sentences (
 id INTEGER PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
 sentence_index INTEGER NOT NULL, text TEXT NOT NULL, char_start INTEGER NOT NULL, char_end INTEGER NOT NULL,
 UNIQUE(document_id,sentence_index));
CREATE INDEX IF NOT EXISTS idx_sentences_document ON sentences(document_id,sentence_index);
CREATE TABLE IF NOT EXISTS tokens (
 id INTEGER PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
 sentence_id INTEGER NOT NULL REFERENCES sentences(id) ON DELETE CASCADE, token_index INTEGER NOT NULL,
 text TEXT NOT NULL, lemma TEXT NOT NULL, pos TEXT NOT NULL, tag TEXT NOT NULL, morph TEXT NOT NULL DEFAULT '', dep TEXT NOT NULL,
 head_token_index INTEGER NOT NULL, char_start INTEGER NOT NULL, char_end INTEGER NOT NULL,
 is_punct INTEGER NOT NULL DEFAULT 0, is_stop INTEGER NOT NULL DEFAULT 0,
 UNIQUE(document_id,token_index));
CREATE INDEX IF NOT EXISTS idx_tokens_lemma ON tokens(lemma);
CREATE INDEX IF NOT EXISTS idx_tokens_pos ON tokens(pos);
CREATE INDEX IF NOT EXISTS idx_tokens_sentence ON tokens(sentence_id,token_index);
CREATE TABLE IF NOT EXISTS concepts (
 id INTEGER PRIMARY KEY, canonical_form TEXT NOT NULL UNIQUE, display_form TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_concepts_canonical ON concepts(canonical_form);
CREATE TABLE IF NOT EXISTS concept_occurrences (
 id INTEGER PRIMARY KEY, concept_id INTEGER NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
 document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
 sentence_id INTEGER NOT NULL REFERENCES sentences(id) ON DELETE CASCADE,
 token_start INTEGER NOT NULL, token_end INTEGER NOT NULL, char_start INTEGER NOT NULL, char_end INTEGER NOT NULL,
 surface_form TEXT NOT NULL, extraction_method TEXT NOT NULL,
 UNIQUE(document_id,token_start,token_end,concept_id));
CREATE INDEX IF NOT EXISTS idx_occurrences_concept ON concept_occurrences(concept_id);
CREATE INDEX IF NOT EXISTS idx_occurrences_document ON concept_occurrences(document_id,sentence_id);
CREATE VIRTUAL TABLE IF NOT EXISTS sentence_fts USING fts5(document_id UNINDEXED,sentence_id UNINDEXED,text,tokenize='unicode61');
CREATE VIRTUAL TABLE IF NOT EXISTS token_fts USING fts5(document_id UNINDEXED,sentence_id UNINDEXED,token_id UNINDEXED,text,lemma,tokenize='unicode61');
CREATE VIRTUAL TABLE IF NOT EXISTS concept_fts USING fts5(canonical_form,display_form,tokenize='unicode61');
"""


def init_linguistic_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(LINGUISTIC_SCHEMA)
    # Safe migration for an index created by an earlier patch revision.
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(tokens)")}
    if "morph" not in columns:
        conn.execute("ALTER TABLE tokens ADD COLUMN morph TEXT NOT NULL DEFAULT ''")


def replace_document_index(document_id, language, model_name, source_path, markdown_path,
                           sentences, tokens, concepts) -> dict[str, int]:
    db.init_db()
    with db.connect() as conn:
        init_linguistic_schema(conn)
        if not conn.execute("SELECT 1 FROM documents WHERE id=?", (document_id,)).fetchone():
            raise ValueError(f"Unknown document_id: {document_id}")
        conn.execute("DELETE FROM sentence_fts WHERE document_id=?", (document_id,))
        conn.execute("DELETE FROM token_fts WHERE document_id=?", (document_id,))
        conn.execute("DELETE FROM concept_occurrences WHERE document_id=?", (document_id,))
        conn.execute("DELETE FROM tokens WHERE document_id=?", (document_id,))
        conn.execute("DELETE FROM sentences WHERE document_id=?", (document_id,))
        conn.execute("DELETE FROM linguistic_documents WHERE document_id=?", (document_id,))
        sentence_ids = {}
        for item in sentences:
            cur = conn.execute("INSERT INTO sentences(document_id,sentence_index,text,char_start,char_end) VALUES(?,?,?,?,?)",
                               (document_id,item.sentence_index,item.text,item.char_start,item.char_end))
            sentence_ids[item.sentence_index] = cur.lastrowid
            conn.execute("INSERT INTO sentence_fts(document_id,sentence_id,text) VALUES(?,?,?)",
                         (document_id,str(cur.lastrowid),item.text))
        for item in tokens:
            sent_no = next((s.sentence_index for s in sentences
                            if s.char_start <= item.char_start < s.char_end), None)
            if sent_no is None:
                continue
            cur = conn.execute("""INSERT INTO tokens(document_id,sentence_id,token_index,text,lemma,pos,tag,morph,dep,
                head_token_index,char_start,char_end,is_punct,is_stop) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (document_id,sentence_ids[sent_no],item.token_index,item.text,item.lemma,item.pos,item.tag,
                 item.morph,item.dep,item.head_index,item.char_start,item.char_end,int(item.is_punct),int(item.is_stop)))
            conn.execute("INSERT INTO token_fts(document_id,sentence_id,token_id,text,lemma) VALUES(?,?,?,?,?)",
                         (document_id,str(sentence_ids[sent_no]),str(cur.lastrowid),item.text,item.lemma))
        for item in concepts:
            conn.execute("""INSERT INTO concepts(canonical_form,display_form) VALUES(?,?)
                ON CONFLICT(canonical_form) DO UPDATE SET display_form=CASE
                WHEN length(excluded.display_form)>length(concepts.display_form)
                THEN excluded.display_form ELSE concepts.display_form END""",
                (item.canonical_form,item.surface_form))
            concept_id = conn.execute("SELECT id FROM concepts WHERE canonical_form=?",
                                      (item.canonical_form,)).fetchone()["id"]
            sentence = conn.execute("""SELECT id FROM sentences WHERE document_id=? AND char_start<=?
                AND char_end>=? ORDER BY sentence_index LIMIT 1""",
                (document_id,item.char_start,item.char_end)).fetchone()
            if sentence:
                conn.execute("""INSERT OR IGNORE INTO concept_occurrences
                    (concept_id,document_id,sentence_id,token_start,token_end,char_start,char_end,
                     surface_form,extraction_method) VALUES(?,?,?,?,?,?,?,?,?)""",
                    (concept_id,document_id,sentence["id"],item.token_start,item.token_end,item.char_start,
                     item.char_end,item.surface_form,item.extraction_method))
        conn.execute("DELETE FROM concepts WHERE NOT EXISTS (SELECT 1 FROM concept_occurrences o WHERE o.concept_id=concepts.id)")
        conn.execute("DELETE FROM concept_fts")
        conn.execute("INSERT INTO concept_fts(canonical_form,display_form) SELECT canonical_form,display_form FROM concepts")
        conn.execute("""INSERT INTO linguistic_documents(document_id,language,model_name,source_path,markdown_path)
            VALUES(?,?,?,?,?)""",(document_id,language,model_name,source_path,markdown_path))
    return {"sentences":len(sentences),"tokens":len(tokens),"concepts":len(concepts)}


def search_sentences(query: str, limit: int = 50) -> list[dict]:
    db.init_db()
    with db.connect() as conn:
        init_linguistic_schema(conn)
        rows = conn.execute("""SELECT s.document_id,d.title,s.sentence_index,s.text,s.char_start,s.char_end
            FROM sentence_fts f JOIN sentences s ON s.id=CAST(f.sentence_id AS INTEGER)
            JOIN documents d ON d.id=s.document_id WHERE sentence_fts MATCH ? ORDER BY rank LIMIT ?""",
            (query,limit)).fetchall()
        return [dict(row) for row in rows]


def search_concepts(query: str, limit: int = 50) -> list[dict]:
    # Normalize inflected query forms to the same lemmas used during indexing.
    from .analyzer import detect_language, load_nlp
    try:
        nlp = load_nlp(detect_language(query))
        parsed = nlp(query)
        terms = [(token.lemma_ or token.text).casefold() for token in parsed
                 if not token.is_punct and not token.is_space and not token.is_stop]
        normalized_query = '"' + " ".join(terms).replace('"', '""') + '"' if terms else query
    except (RuntimeError, OSError):
        # Keep search usable for an already-built index if a model is temporarily unavailable.
        normalized_query = query
    db.init_db()
    with db.connect() as conn:
        init_linguistic_schema(conn)
        rows = conn.execute("""SELECT c.id,c.canonical_form,c.display_form,COUNT(o.id) AS occurrences,
            COUNT(DISTINCT o.document_id) AS documents FROM concept_fts f
            JOIN concepts c ON c.canonical_form=f.canonical_form
            LEFT JOIN concept_occurrences o ON o.concept_id=c.id WHERE concept_fts MATCH ?
            GROUP BY c.id ORDER BY occurrences DESC LIMIT ?""",(normalized_query,limit)).fetchall()
        return [dict(row) for row in rows]


def search_tokens(query: str, limit: int = 100) -> list[dict]:
    """Search surface forms and lemmas, returning exact document offsets."""
    db.init_db()
    with db.connect() as conn:
        init_linguistic_schema(conn)
        rows = conn.execute("""SELECT t.document_id,d.title,s.sentence_index,s.text AS sentence_text,
            t.token_index,t.text,t.lemma,t.pos,t.tag,t.morph,t.dep,t.char_start,t.char_end,t.is_punct
            FROM token_fts f JOIN tokens t ON t.id=CAST(f.token_id AS INTEGER)
            JOIN sentences s ON s.id=t.sentence_id JOIN documents d ON d.id=t.document_id
            WHERE token_fts MATCH ? ORDER BY rank LIMIT ?""", (query,limit)).fetchall()
        return [dict(row) for row in rows]
