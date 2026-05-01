# Support Triage Agent

A terminal-based AI agent that triages support tickets across **HackerRank**, **Claude**, and **Visa** ecosystems using RAG (Retrieval-Augmented Generation).

## Architecture

```
Ticket → Retriever (FAISS + sentence-transformers) → Gemini Flash → Structured Output
```

- **Embeddings**: `sentence-transformers` (all-MiniLM-L6-v2) — runs locally, no API key needed
- **Vector Store**: FAISS-cpu for fast similarity search
- **LLM**: Google Gemini 2.5 Flash (free tier) for classification + response generation
- **Corpus**: 774 markdown articles from HackerRank, Claude, and Visa support centers

## Setup

```bash
# 1. Install dependencies
pip install -r code/requirements.txt

# 2. Set up your API key
cp code/.env.example code/.env
# Edit code/.env and add your Gemini API key from https://aistudio.google.com/apikey
```

## Usage

```bash
# Run on the full support_tickets.csv
python code/main.py

# Run on sample tickets (for testing/development)
python code/main.py --sample

# Force rebuild the embedding index
python code/main.py --rebuild-index
```

## Output

Results are written to `support_tickets/output.csv` with columns:
- `issue`, `subject`, `company` — copied from input
- `status` — `replied` or `escalated`
- `request_type` — `product_issue`, `feature_request`, `bug`, or `invalid`
- `product_area` — most relevant support category
- `response` — user-facing answer grounded in the corpus
- `justification` — concise explanation of the triage decision

## Design Decisions

1. **Single LLM call per ticket**: Combines classification + response generation to minimize API calls (important for free tier).
2. **Ecosystem-aware retrieval**: When the company is known, retrieval results from that ecosystem are boosted.
3. **Explicit escalation rules**: High-risk patterns (billing, fraud, outages, score disputes) are flagged for escalation.
4. **Cached embeddings**: FAISS index is saved to `data/embeddings/` so subsequent runs skip re-embedding.
5. **Deterministic output**: Low temperature (0.1) and seeded random operations for reproducibility.
