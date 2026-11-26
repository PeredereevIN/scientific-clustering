import pytest

spacy = pytest.importorskip("spacy")

from sciintel.linguistic.analyzer import analyze_text, detect_language
from sciintel.linguistic.converters import convert_to_markdown


def test_language_detection_for_english_and_russian():
    assert detect_language("Neural networks learn representations.") == "en"
    assert detect_language("Нейронные сети изучают представления.") == "ru"


def test_analyzer_keeps_punctuation_and_exact_offsets():
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    text = "Graph models work. They learn!"
    sentences, tokens, concepts = analyze_text(text, "en", nlp=nlp)
    assert [s.text for s in sentences] == ["Graph models work.", "They learn!"]
    punctuation = [t for t in tokens if t.is_punct]
    assert [t.text for t in punctuation] == [".", "!"]
    for token in tokens:
        assert text[token.char_start:token.char_end] == token.text
    assert all(token.char_start < token.char_end for token in tokens)


def test_text_converter_creates_markdown(tmp_path):
    source = tmp_path / "paper.txt"
    source.write_text("Title\n\nFirst paragraph.\n\nSecond paragraph.", encoding="utf-8")
    markdown, metadata = convert_to_markdown(source)
    assert "First paragraph." in markdown
    assert "\n\n" in markdown
    assert metadata["source_format"] == "txt"


def test_docx_converter_preserves_heading_and_paragraph(tmp_path):
    from docx import Document
    source = tmp_path / "paper.docx"
    doc = Document()
    doc.add_heading("Methods", level=1)
    doc.add_paragraph("We describe the method.")
    doc.save(source)
    markdown, _ = convert_to_markdown(source)
    assert "# Methods" in markdown
    assert "We describe the method." in markdown


def test_sqlite_index_stores_sentence_token_and_concept_offsets(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from sciintel import db
    from sciintel.linguistic.analyzer import SentenceRecord, TokenRecord, ConceptRecord
    from sciintel.linguistic.indexer import replace_document_index, search_sentences, search_concepts, search_tokens

    class TestSettings:
        db_path = tmp_path / "test.sqlite3"
        def ensure_dirs(self):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(db, "settings", TestSettings())
    db.init_db()
    with db.connect() as conn:
        conn.execute("""INSERT INTO documents(id,sha256,path,title,text_path,status)
            VALUES('doc1','hash1','source.txt','Test paper','body.txt','ready')""")
    sentences = [SentenceRecord(0, "Neural networks work.", 0, 21)]
    tokens = [
        TokenRecord(0, "Neural", "neural", "ADJ", "JJ", "Degree=Pos", "amod", 1, 0, 6, False, False),
        TokenRecord(1, "networks", "network", "NOUN", "NNS", "Number=Plur", "nsubj", 2, 7, 15, False, False),
        TokenRecord(2, "work", "work", "VERB", "VBP", "Tense=Pres|VerbForm=Fin", "ROOT", 2, 16, 20, False, False),
        TokenRecord(3, ".", ".", "PUNCT", ".", "", "punct", 2, 20, 21, True, False),
    ]
    concepts = [ConceptRecord("neural network", "Neural networks", 0, 2, 0, 15, "noun_chunk")]
    result = replace_document_index("doc1", "en", "en_core_web_sm", "source.txt", "body.md",
                                    sentences, tokens, concepts)
    assert result == {"sentences": 1, "tokens": 4, "concepts": 1}
    assert search_sentences('"Neural networks"')[0]["document_id"] == "doc1"
    assert search_concepts('"neural network"')[0]["canonical_form"] == "neural network"
    token_result = search_tokens("network")
    assert token_result[0]["lemma"] == "network"
    assert token_result[0]["morph"] == "Number=Plur"
    with db.connect() as conn:
        occurrence = conn.execute("SELECT char_start,char_end,surface_form FROM concept_occurrences").fetchone()
        punct = conn.execute("SELECT is_punct FROM tokens WHERE text='.'").fetchone()
    assert tuple(occurrence) == (0, 15, "Neural networks")
    assert punct["is_punct"] == 1
    with db.connect() as conn:
        morphology = conn.execute("SELECT morph FROM tokens WHERE text='networks'").fetchone()["morph"]
    assert morphology == "Number=Plur"


def test_rtf_converter_uses_striprtf(tmp_path, monkeypatch):
    import sys
    import types
    from sciintel.linguistic.converters import convert_to_markdown

    package = types.ModuleType("striprtf")
    module = types.ModuleType("striprtf.striprtf")
    module.rtf_to_text = lambda payload, encoding="cp1252": "RTF title\\n\\nConverted paragraph."
    package.striprtf = module
    monkeypatch.setitem(sys.modules, "striprtf", package)
    monkeypatch.setitem(sys.modules, "striprtf.striprtf", module)
    source = tmp_path / "paper.rtf"
    source.write_bytes(b"{\\rtf1\\ansi Test}")

    markdown, metadata = convert_to_markdown(source)
    assert "Converted paragraph." in markdown
    assert metadata["source_format"] == "rtf"


def test_pdf_converter_extracts_text_layer(tmp_path):
    import pymupdf
    from sciintel.linguistic.converters import convert_to_markdown

    source = tmp_path / "paper.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "A test scientific paper.")
    doc.save(source)
    doc.close()

    markdown, metadata = convert_to_markdown(source)
    assert "A test scientific paper." in markdown
    assert metadata["page_count"] == 1
