"""
Retriever — Performs semantic search over the FAISS index with
optional ecosystem filtering and score boosting.
"""

from typing import List, Optional
from dataclasses import dataclass

import numpy as np

from config import TOP_K, ECOSYSTEM_BOOST
from embedder import embed_query


@dataclass
class RetrievalResult:
    """A single retrieval result with text, metadata, and score."""
    text: str
    score: float
    doc_title: str
    ecosystem: str
    product_area: str
    source_url: str
    file_path: str
    chunk_index: int


class Retriever:
    """Semantic retriever over the FAISS index."""
    
    def __init__(self, index, chunks_meta: List[dict]):
        self.index = index
        self.chunks_meta = chunks_meta
    
    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        ecosystem_filter: Optional[str] = None,
        ecosystem_boost: float = ECOSYSTEM_BOOST,
    ) -> List[RetrievalResult]:
        """
        Search for the most relevant chunks given a query.
        
        Args:
            query: The search query (ticket issue + subject)
            top_k: Number of results to return
            ecosystem_filter: If set, boost results from this ecosystem
            ecosystem_boost: Score multiplier for matching ecosystem
        
        Returns:
            List of RetrievalResult objects, sorted by relevance (descending)
        """
        # Embed the query
        query_vec = embed_query(query)
        
        # Search more than needed so we can filter/rerank
        search_k = min(top_k * 4, self.index.ntotal)
        scores, indices = self.index.search(query_vec, search_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            
            meta = self.chunks_meta[idx]
            
            # Apply ecosystem boost
            adjusted_score = float(score)
            if ecosystem_filter and meta["ecosystem"] == ecosystem_filter.lower():
                adjusted_score *= ecosystem_boost
            
            results.append(RetrievalResult(
                text=meta["text"],
                score=adjusted_score,
                doc_title=meta["doc_title"],
                ecosystem=meta["ecosystem"],
                product_area=meta["product_area"],
                source_url=meta["source_url"],
                file_path=meta["file_path"],
                chunk_index=meta["chunk_index"],
            ))
        
        # Sort by adjusted score descending
        results.sort(key=lambda r: r.score, reverse=True)
        
        # If ecosystem filter is set, ensure at least some results from that ecosystem
        if ecosystem_filter:
            eco_results = [r for r in results if r.ecosystem == ecosystem_filter.lower()]
            other_results = [r for r in results if r.ecosystem != ecosystem_filter.lower()]
            
            # Take primarily from the target ecosystem, but allow some cross-domain
            min_eco = min(top_k - 1, len(eco_results))
            final = eco_results[:min_eco]
            remaining_slots = top_k - len(final)
            final.extend(other_results[:remaining_slots])
            final.sort(key=lambda r: r.score, reverse=True)
            return final[:top_k]
        
        return results[:top_k]
    
    def format_context(self, results: List[RetrievalResult]) -> str:
        """Format retrieval results into a context string for the LLM."""
        if not results:
            return "No relevant support documentation found."
        
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(
                f"[Source {i}] ({r.ecosystem.upper()} — {r.product_area})\n"
                f"Title: {r.doc_title}\n"
                f"Content: {r.text}\n"
            )
        return "\n---\n".join(parts)
