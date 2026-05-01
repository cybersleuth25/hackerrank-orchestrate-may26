"""
Chunker — Splits documents into overlapping text chunks while
preserving metadata for retrieval.
"""

from dataclasses import dataclass, field
from typing import List

from corpus_loader import Document
from config import CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class Chunk:
    """A text chunk derived from a Document, with inherited metadata."""
    text: str
    doc_title: str = ""
    ecosystem: str = ""
    product_area: str = ""
    source_url: str = ""
    file_path: str = ""
    chunk_index: int = 0


def chunk_document(doc: Document, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    """
    Split a document's content into overlapping chunks.
    Uses character-based splitting with paragraph-aware boundaries.
    """
    text = doc.content.strip()
    if not text:
        return []
    
    # If the document is short enough, return as a single chunk
    if len(text) <= chunk_size:
        return [Chunk(
            text=text,
            doc_title=doc.title,
            ecosystem=doc.ecosystem,
            product_area=doc.product_area,
            source_url=doc.source_url,
            file_path=doc.file_path,
            chunk_index=0,
        )]
    
    chunks = []
    start = 0
    chunk_idx = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # If we're not at the end, try to break at a paragraph or sentence boundary
        if end < len(text):
            # Look for paragraph break (double newline)
            para_break = text.rfind("\n\n", start + overlap, end)
            if para_break > start + overlap:
                end = para_break
            else:
                # Look for single newline
                line_break = text.rfind("\n", start + overlap, end)
                if line_break > start + overlap:
                    end = line_break
                else:
                    # Look for sentence end
                    sent_break = text.rfind(". ", start + overlap, end)
                    if sent_break > start + overlap:
                        end = sent_break + 1  # include the period
        
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(
                text=chunk_text,
                doc_title=doc.title,
                ecosystem=doc.ecosystem,
                product_area=doc.product_area,
                source_url=doc.source_url,
                file_path=doc.file_path,
                chunk_index=chunk_idx,
            ))
            chunk_idx += 1
        
        # Move start forward, accounting for overlap
        start = end - overlap if end < len(text) else len(text)
    
    return chunks


def chunk_corpus(documents: List[Document]) -> List[Chunk]:
    """Chunk all documents in the corpus."""
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc))
    return all_chunks


if __name__ == "__main__":
    from corpus_loader import load_corpus
    docs = load_corpus()
    chunks = chunk_corpus(docs)
    print(f"Created {len(chunks)} chunks from {len(docs)} documents")
    print(f"Average chunk length: {sum(len(c.text) for c in chunks) / len(chunks):.0f} chars")
