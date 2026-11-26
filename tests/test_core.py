from pathlib import Path

from sciintel.utils import clean_text, extract_abstract, extract_doi, extract_title, extract_year, embedding_text, infer_domain


def test_clean_text_normalizes_whitespace():
    assert clean_text("A\t B\n\n\nC") == "A B\n\nC"


def test_extract_abstract_until_introduction():
    text = "Title\n\nAbstract\nWe present a useful method for testing a research system with several carefully chosen examples and clear results.\n\n1 Introduction\nThe full paper starts here."
    assert "We present a useful method" in extract_abstract(text)


def test_extract_title_and_identifiers():
    text = "# A Research Paper About Fast Graph Search\n\nAbstract\nThis is a sufficiently long abstract for testing."
    assert extract_title(text).startswith("A Research Paper")
    assert extract_doi("doi: 10.1234/ABC.DEF.") == "10.1234/abc.def"
    assert extract_year("Published in 2024. Other text") == 2024


def test_embedding_input_uses_title_and_abstract():
    actual = embedding_text("Graph Search", "A short abstract.", "Ignored body text")
    assert actual == "Graph Search [SEP] A short abstract."


def test_infer_domain(tmp_path: Path):
    inbox = tmp_path / "inbox"
    file = inbox / "machine_learning" / "paper.pdf"
    assert infer_domain(file, inbox) == "machine_learning"
