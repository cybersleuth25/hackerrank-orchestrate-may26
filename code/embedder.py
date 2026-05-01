"""
Embedder — Creates and caches a FAISS vector index from corpus chunks
using sentence-transformers (runs 100% locally, no API key needed).
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import List

from config import EMBEDDINGS_DIR, EMBEDDING_MODEL, RANDOM_SEED
from chunker import Chunk


# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)


_model_cache = None


def _get_model():
    """Lazy-load and cache the sentence-transformers model (singleton)."""
    global _model_cache
    if _model_cache is None:
        from sentence_transformers import SentenceTransformer
        _model_cache = SentenceTransformer(EMBEDDING_MODEL)
    return _model_cache


def build_index(chunks: List[Chunk], force_rebuild: bool = False) -> tuple:
    """
    Build a FAISS index from chunks. Caches to data/embeddings/ for reuse.
    
    Returns:
        (faiss_index, chunks_metadata) where chunks_metadata is a list of
        dicts with the metadata for each chunk (parallel to the index vectors).
    """
    import faiss
    
    index_path = EMBEDDINGS_DIR / "faiss.index"
    meta_path = EMBEDDINGS_DIR / "chunks_meta.pkl"
    
    # Check cache
    if not force_rebuild and index_path.exists() and meta_path.exists():
        from rich.console import Console
        console = Console()
        console.print("[green]Loading cached FAISS index...[/green]")
        index = faiss.read_index(str(index_path))
        with open(meta_path, "rb") as f:
            chunks_meta = pickle.load(f)
        console.print(f"[green]Loaded {index.ntotal} vectors from cache[/green]")
        return index, chunks_meta
    
    # Build fresh
    from rich.console import Console
    from rich.progress import Progress
    console = Console()
    console.print(f"[yellow]Building FAISS index from {len(chunks)} chunks...[/yellow]")
    
    model = _get_model()
    
    # Prepare texts for embedding
    texts = [c.text for c in chunks]
    
    # Batch embed with progress
    batch_size = 64
    all_embeddings = []
    
    with Progress() as progress:
        task = progress.add_task("Embedding chunks...", total=len(texts))
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
            all_embeddings.append(embeddings)
            progress.update(task, advance=len(batch))
    
    embeddings_matrix = np.vstack(all_embeddings).astype("float32")
    
    # Build FAISS index (Inner Product since we normalized)
    dimension = embeddings_matrix.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_matrix)
    
    # Save metadata
    chunks_meta = []
    for c in chunks:
        chunks_meta.append({
            "text": c.text,
            "doc_title": c.doc_title,
            "ecosystem": c.ecosystem,
            "product_area": c.product_area,
            "source_url": c.source_url,
            "file_path": c.file_path,
            "chunk_index": c.chunk_index,
        })
    
    # Cache to disk
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with open(meta_path, "wb") as f:
        pickle.dump(chunks_meta, f)
    
    console.print(f"[green]Built and cached index: {index.ntotal} vectors, {dimension}d[/green]")
    return index, chunks_meta


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string. Returns a (1, dim) float32 array."""
    model = _get_model()
    embedding = model.encode([query], normalize_embeddings=True)
    return embedding.astype("float32")


if __name__ == "__main__":
    from corpus_loader import load_corpus
    from chunker import chunk_corpus
    
    docs = load_corpus()
    chunks = chunk_corpus(docs)
    index, meta = build_index(chunks, force_rebuild=True)
    print(f"Index has {index.ntotal} vectors")
