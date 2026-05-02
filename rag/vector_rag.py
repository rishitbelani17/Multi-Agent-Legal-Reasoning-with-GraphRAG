"""
Standard vector RAG baseline (Pipeline 1).

Implements flat FAISS-based nearest-neighbour retrieval over chunk embeddings,
followed by a single LLM call to produce an answer.  No graph structure used.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import config
from data.preprocessor import Chunk

logger = logging.getLogger(__name__)


@dataclass
class VectorRetrievedChunk:
    chunk_id: str
    doc_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorIndex:
    """
    Lightweight cosine-similarity index over chunk embeddings.
    Uses numpy for portability (no GPU required); swap to faiss for speed.
    """

    def __init__(self, embedder):
        self.embedder = embedder
        self._chunk_ids: list[str] = []
        self._doc_ids: list[str] = []
        self._texts: list[str] = []
        self._metadata: list[dict] = []
        self._matrix: np.ndarray | None = None  # shape (N, D), L2-normalised

    def build(self, chunks: list[Chunk]) -> None:
        logger.info("Building vector index from %d chunks …", len(chunks))
        texts = [c.text for c in chunks]
        embs: np.ndarray = self.embedder.encode(
            texts, show_progress_bar=True, convert_to_numpy=True
        )
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
        self._matrix = embs / norms

        self._chunk_ids = [c.chunk_id for c in chunks]
        self._doc_ids   = [c.doc_id   for c in chunks]
        self._texts     = [c.text     for c in chunks]
        self._metadata  = [c.metadata for c in chunks]
        logger.info("Vector index ready (%d vectors, dim=%d)", len(chunks), embs.shape[1])

    def search(self, query: str, top_k: int = config.VECTOR_TOP_K) -> list[VectorRetrievedChunk]:
        if self._matrix is None:
            raise RuntimeError("Call build() before search().")
        q_emb = self.embedder.encode([query], convert_to_numpy=True)[0]
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-10)
        sims = self._matrix @ q_emb  # (N,)
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [
            VectorRetrievedChunk(
                chunk_id=self._chunk_ids[i],
                doc_id=self._doc_ids[i],
                text=self._texts[i],
                score=float(sims[i]),
                metadata=self._metadata[i],
            )
            for i in top_idx
        ]


class VectorRAG:
    """
    Pipeline 1 – Baseline vector RAG.

    retrieve → format context → LLM answer
    """

    def __init__(self, index: VectorIndex, llm_client, top_k: int = config.VECTOR_TOP_K):
        self.index = index
        self.llm = llm_client
        self.top_k = top_k

    def run(self, query: str, dataset: str = "ledgar") -> dict[str, Any]:
        t0 = time.perf_counter()

        # ── Retrieve ──────────────────────────────────────────────────────────
        chunks = self.index.search(query, top_k=self.top_k)
        context = "\n\n".join(
            f"[{i+1}] (score={c.score:.3f})\n{c.text}" for i, c in enumerate(chunks)
        )

        # ── Prompt ────────────────────────────────────────────────────────────
        system_prompt = _legal_system_prompt(dataset)
        user_prompt = _rag_user_prompt(query, context)

        # ── LLM call ──────────────────────────────────────────────────────────
        response = self.llm.call(system_prompt, user_prompt)

        elapsed = time.perf_counter() - t0
        return {
            "pipeline": "vector_rag",
            "query": query,
            "answer": response["content"],
            "retrieved_chunks": [
                {"chunk_id": c.chunk_id, "score": c.score, "text": c.text}
                for c in chunks
            ],
            "latency_s": elapsed,
            "input_tokens": response["input_tokens"],
            "output_tokens": response["output_tokens"],
        }


# ── Shared prompt helpers (reused by all pipelines) ───────────────────────────

def _legal_system_prompt(dataset: str) -> str:
    dataset_context = {
        "ledgar": (
            "You are a legal AI assistant specialising in contract clause classification. "
            "Given retrieved contract clauses and a query, classify the clause type and "
            "justify your answer with explicit references to the retrieved text."
        ),
        "caseholder": (
            "You are a legal AI assistant specialising in case-law analysis. "
            "Given a citing prompt and 5 candidate holdings (numbered 0-4), select the "
            "correct holding and explain your reasoning step by step."
        ),
        "ecthr": (
            "You are a legal AI assistant specialising in human rights law. "
            "Given facts from an ECtHR case and retrieved precedents, predict whether "
            "a Convention article was violated and explain your reasoning."
        ),
    }
    return dataset_context.get(dataset, dataset_context["ledgar"]) + "\n\n" + answer_format_instructions(dataset)


def answer_format_instructions(dataset: str) -> str:
    """
    Tells the LLM the EXACT label string it must emit on the ANSWER line.

    These match the ``label_str`` produced by data/loader.py, which is what
    evaluation.metrics.parse_answer searches for. Without these instructions
    the parser's fuzzy match is unreliable.
    """
    if dataset == "caseholder":
        return (
            "ANSWER FORMAT: Your final 'ANSWER:' line MUST be exactly one of:\n"
            "  holding_0, holding_1, holding_2, holding_3, holding_4\n"
            "These correspond to candidate holdings (0)-(4) in the question. "
            "Do not write 'Holding A', 'choice 2', or the holding text itself. "
            "Output only the literal token (e.g. 'ANSWER: holding_2')."
        )
    if dataset == "ecthr":
        return (
            "ANSWER FORMAT: Your final 'ANSWER:' line MUST be exactly one of:\n"
            "  violated, not_violated\n"
            "Do not paraphrase."
        )
    # LEDGAR: clause-type labels are dataset-specific; the parser handles fuzzy
    # matching, but we still ask the model to repeat the clause type verbatim.
    return (
        "ANSWER FORMAT: Your final 'ANSWER:' line should be the clause-type "
        "label exactly as it appears in standard LEDGAR taxonomy (e.g. "
        "'Indemnification', 'Governing Law'). Do not paraphrase."
    )


def _rag_user_prompt(query: str, context: str) -> str:
    return (
        f"## Query\n{query}\n\n"
        f"## Retrieved Context\n{context}\n\n"
        "## Instructions\n"
        "Using ONLY the retrieved context above, answer the query. "
        "Cite specific passages to support your answer. "
        "If the context is insufficient, state that clearly."
    )
