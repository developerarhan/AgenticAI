import pymupdf
from typing import List, Dict


def extract_pdf_pages(file_path: str) -> List[Dict[str, any]]:
    """Extracts text per page, tracking metadata."""

    doc = pymupdf.open(file_path)
    pages_data = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()
        if text:
            pages_data.append({
                "page_number": page_num + 1,
                "text": text
            })

    return pages_data


def recursive_chunk_text(
    pages_data: List[Dict[str, any]],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[Dict[str, any]]:
    """Chunks text while maintaining page metadata and semantic continuity."""
    chunks = []
    chunks_index = 0

    for page in pages_data:
        text = page["text"]
        words = text.split()
        start = 0

        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start: end]
            chunk_str = " ".join(chunk_words)

            chunks.append({
                "chunk_index": chunks_index,
                "content": chunk_str,
                "page_number": page["page_number"],
            })

            chunks_index += 1

            # Slide window forward minus overlap
            start += (chunk_size - chunk_overlap)

    return chunks
