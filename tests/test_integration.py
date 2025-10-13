import os
from pathlib import Path
from reportlab.pdfgen import canvas
from app import extract_text_from_pdf, build_chunks_from_pages


def make_sample_pdf(path: Path):
    c = canvas.Canvas(str(path))
    c.drawString(100, 750, "Hello integration test page 1")
    c.showPage()
    c.drawString(100, 750, "Second page text for integration test")
    c.save()


def test_integration_pdf_chunking(tmp_path):
    pdf_path = tmp_path / "sample_integration.pdf"
    make_sample_pdf(pdf_path)

    # Use pypdf PdfReader reading from the file
    with open(pdf_path, "rb") as f:
        pages = extract_text_from_pdf(f)
    assert isinstance(pages, dict)
    chunks = build_chunks_from_pages(pdf_path.name, pages)
    assert isinstance(chunks, list)
    assert len(chunks) > 0
