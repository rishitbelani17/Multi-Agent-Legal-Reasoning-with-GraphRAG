"""
Central configuration for Multi-Agent Legal Reasoning with GraphRAG.
Edit values here or override via environment variables / .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ──────────────────────────────────────────────────────────────────────
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PASTE YOUR ANTHROPIC API KEY BELOW (between the quotes).                ║
# ║  Get one at: https://console.anthropic.com/settings/keys                 ║
# ║  This is the ONLY place you need to set the key for the whole project.  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
ANTHROPIC_API_KEY: str = ""

# (Optional) allow an env var / .env file to override the key above.
# Useful for CI or shared machines. Safe to ignore for normal use.
if os.getenv("ANTHROPIC_API_KEY"):
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
LLM_MODEL: str = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")   # fast & cheap
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# Judge model: the Judge is a *discriminator* — it synthesises 2 rounds of
# debate (~12k input tokens) and picks the right label from 5 near-identical
# candidates. Haiku empirically underperforms in this role; Sonnet is much
# better at long-context discrimination at modest extra cost (the Judge runs
# once per query while workers run 5+ times). Set LLM_MODEL_JUDGE='' to fall
# back to LLM_MODEL when minimising cost.
LLM_MODEL_JUDGE: str = os.getenv("LLM_MODEL_JUDGE", "claude-sonnet-4-6")
LLM_MAX_TOKENS_JUDGE: int = int(os.getenv("LLM_MAX_TOKENS_JUDGE", "2048"))

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

# Bumped from v1 → v2 when we removed the candidate holdings from the
# CaseHOLD retrieval corpus (loader.py / preprocessor.py corpus_text split).
# Any pickled graph built before that fix contained candidate-text chunks
# and would silently leak the answer label into retrieval. Bumping the
# version guarantees the next run rebuilds from the clean corpus_text.
CORPUS_VERSION: str = "v2"

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
# Number of (Plaintiff ↔ Defense) rebuttal rounds in the DebateOrchestrator.
# 1 = single-pass (legacy behaviour). 2+ = true back-and-forth debate.
N_DEBATE_ROUNDS: int = int(os.getenv("N_DEBATE_ROUNDS", "2"))

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))

# ── Evaluation ────────────────────────────────────────────────────────────────
RESULTS_DIR: str = os.getenv("RESULTS_DIR", "./results")

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
