"""
Centralized configuration for the Support Triage Agent.
Reads secrets from environment variables, defines paths and constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the code/ directory
_code_dir = Path(__file__).resolve().parent
load_dotenv(_code_dir / ".env")

# ── Repo root (hackathon/) ──────────────────────────────────────────
REPO_ROOT = _code_dir.parent

# ── Paths ───────────────────────────────────────────────────────────
DATA_DIR = REPO_ROOT / "data"
CORPUS_DIRS = {
    "hackerrank": DATA_DIR / "hackerrank",
    "claude": DATA_DIR / "claude",
    "visa": DATA_DIR / "visa",
}
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
SUPPORT_TICKETS_DIR = REPO_ROOT / "support_tickets"
INPUT_CSV = SUPPORT_TICKETS_DIR / "support_tickets.csv"
SAMPLE_CSV = SUPPORT_TICKETS_DIR / "sample_support_tickets.csv"
OUTPUT_CSV = SUPPORT_TICKETS_DIR / "output.csv"

# ── API Keys ────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ── Model Config ────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # sentence-transformers, 384-dim, runs locally
GEMINI_MODEL = "gemini-3.1-flash-lite-preview"  # Free tier model with available quota
GEMINI_FALLBACK_MODELS = ["gemini-flash-lite-latest", "gemini-2.5-flash-lite", "gemini-2.0-flash-lite"]
GEMINI_TEMPERATURE = 0.1  # Low temperature for deterministic outputs
GEMINI_MAX_TOKENS = 2048

# ── Chunking ────────────────────────────────────────────────────────
CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 100    # overlap between consecutive chunks

# ── Retrieval ───────────────────────────────────────────────────────
TOP_K = 5              # number of chunks to retrieve per query
ECOSYSTEM_BOOST = 1.5  # score multiplier for matching ecosystem

# ── Determinism ─────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Logging (AGENTS.md compliance) ──────────────────────────────────
LOG_DIR = Path(os.environ.get("USERPROFILE", os.environ.get("HOME", "~"))) / "hackerrank_orchestrate"
LOG_FILE = LOG_DIR / "log.txt"

# ── Allowed output values ──────────────────────────────────────────
ALLOWED_STATUS = ["replied", "escalated"]
ALLOWED_REQUEST_TYPES = ["product_issue", "feature_request", "bug", "invalid"]
