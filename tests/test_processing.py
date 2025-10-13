import types
import pytest
from app import build_chunks_from_pages, extract_text_from_pdf


class DummyPage:
    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


class DummyPdf:
    def __init__(self, pages):
        self.pages = [DummyPage(p) for p in pages]


def test_extract_text_from_pdf_monkeypatch(monkeypatch):
    # monkeypatch PdfReader to return our DummyPdf
    dummy = DummyPdf(["Page one text.", "Second page text.", ""])  # third page empty

    def fake_reader(file):
        return dummy

    monkeypatch.setattr("app.PdfReader", fake_reader)

    class FakeFile:
        def __init__(self):
            self.name = "dummy.pdf"

    pages = extract_text_from_pdf(FakeFile())
    assert isinstance(pages, dict)
    assert pages[1].startswith("Page one text")
    assert pages[2].startswith("Second page text")
    assert pages[3] == ""


def test_build_chunks_from_pages_simple():
    page_texts = {1: "Hello world.", 2: "Another page text."}
    chunks = build_chunks_from_pages("doc.pdf", page_texts)
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    # each chunk should have text and metadata
    for c in chunks:
        assert "text" in c and "metadata" in c
        assert c["metadata"].get("source") == "doc.pdf"


def test_build_chunks_header_parsing():
    # ensure chunk header parsing extracts source and page properly
    page_texts = {1: "A short page."}
    chunks = build_chunks_from_pages("mydoc.pdf", page_texts)
    assert len(chunks) >= 1
    for c in chunks:
        md = c["metadata"]
        assert md.get("source") == "mydoc.pdf"
        assert isinstance(md.get("page"), int)


def test_build_chunks_empty_pages():
    page_texts = {1: "", 2: "\n   \n", 3: "Valid text here."}
    chunks = build_chunks_from_pages("emptytest.pdf", page_texts)
    # should skip empty pages and still produce chunks for the valid page
    assert any(c["metadata"]["source"] == "emptytest.pdf" for c in chunks)


def test_build_chunks_large_page():
    # simulate a large page that will be split into multiple chunks
    long_text = "\n".join([f"Line {i} - sample text." for i in range(500)])
    page_texts = {1: long_text}
    chunks = build_chunks_from_pages("large.pdf", page_texts)
    assert len(chunks) > 1
    for c in chunks:
        assert c["metadata"]["source"] == "large.pdf"