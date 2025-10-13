"""Quick smoke script to run simple processing functions locally without Streamlit.
This script imports the functions from `app.py` and runs them with small sample input.
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path so imports work when running this script
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import build_chunks_from_pages, extract_text_from_pdf


def main():
    page_texts = {1: "This is a test page.", 2: "Second page has more text."}
    chunks = build_chunks_from_pages("sample.pdf", page_texts)
    print(f"Generated {len(chunks)} chunks")
    for ch in chunks[:3]:
        print(ch['metadata'], ch['text'][:60])


if __name__ == '__main__':
    main()
