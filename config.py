"""
Central configuration for Multi-Agent Legal Reasoning with GraphRAG.
Edit values here or override via environment variables / .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ──────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")   # fast & cheap
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DIM: int = 384  # all-MiniLM-L6-v2 output dimension

# ── Datasets ──────────────────────────────────────────────────────────────────
# Primary dataset (per proposal §2.4)
CASEHOLDER_SUBSET_SIZE: int = int(os.getenv("CASEHOLDER_SUBSET_SIZE", "500"))
# Optional ECtHR dataset
ECTHR_SUBSET_SIZE: int = int(os.getenv("ECTHR_SUBSET_SIZE", "200"))
# Optional LEDGAR extension
LEDGAR_SUBSET_SIZE: int = int(os.getenv("LEDGAR_SUBSET_SIZE", "200"))

DATA_CACHE_DIR: str = os.getenv("DATA_CACHE_DIR", "./cache/datasets")
GRAPH_CACHE_DIR: str = os.getenv("GRAPH_CACHE_DIR", "./cache/graphs")

# ── Graph RAG ─────────────────────────────────────────────────────────────────
# Minimum cosine similarity for edges between document chunks
GRAPH_EDGE_THRESHOLD: float = float(os.getenv("GRAPH_EDGE_THRESHOLD", "0.5"))
# Max hops when traversing the graph for retrieval
GRAPH_MAX_HOPS: int = int(os.getenv("GRAPH_MAX_HOPS", "2"))
# Number of top-k seed nodes to start graph traversal from
GRAPH_SEED_K: int = int(os.getenv("GRAPH_SEED_K", "3"))

# ── Vector RAG ────────────────────────────────────────────────────────────────
VECTOR_TOP_K: int = int(os.getenv("VECTOR_TOP_K", "5"))

# ── Multi-Agent ───────────────────────────────────────────────────────────────
MAX_AGENT_ROUNDS: int = int(os.getenv("MAX_AGENT_ROUNDS", "3"))

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))

# ── Evaluation ────────────────────────────────────────────────────────────────
RESULTS_DIR: str = os.getenv("RESULTS_DIR", "./results")

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
