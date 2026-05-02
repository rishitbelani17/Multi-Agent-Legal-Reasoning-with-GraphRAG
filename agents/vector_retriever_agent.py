"""
Vector-RAG variant of the Retriever Agent.

Mirrors RetrieverAgent's interface (sub-query decomposition + summary)
but uses a flat VectorIndex instead of the GraphRetriever. This lets the
Multi-Agent + Vector-RAG pipeline reuse the rest of the debate stack
unchanged, isolating retrieval mode as the only ablated variable.
"""

from __future__ import annotations

from typing import Any

from agents.base_agent import AgentMessage, BaseAgent
from rag.vector_rag import VectorIndex


class VectorRetrieverAgent(BaseAgent):
    """
    Retriever agent backed by a flat vector index.

    Output schema matches RetrieverAgent so downstream agents (Plaintiff,
    Defense, Judge) don't need to know which retrieval mode was used.
    """

    def __init__(self, llm_client, index: VectorIndex, dataset: str = "caseholder", top_k: int = 5):
        super().__init__(
            name="RetrieverAgent",   # same name → downstream agents look it up
            llm_client=llm_client,
            role_description="Retrieves relevant legal text via vector similarity.",
        )
        self.index = index
        self.dataset = dataset
        self.top_k = top_k

    @property
    def system_prompt(self) -> str:
        return (
            "You are a legal research assistant specialized in evidence retrieval. "
            "Decompose the query into focused sub-queries, then summarize the "
            "retrieved passages. Cite passage numbers."
        )

    def run(
        self,
        query: str,
        context: str = "",
        history: list[AgentMessage] | None = None,
        **kwargs: Any,
    ) -> AgentMessage:
        # Step 1 — decompose
        decompose_prompt = (
            f"## Original Query\n{query}\n\n"
            "Identify 1–3 focused sub-queries for evidence retrieval. "
            "Return them as a numbered list."
        )
        decompose_msg = self._call_llm(decompose_prompt)
        sub_queries = _parse_sub_queries(decompose_msg.content, query)

        # Step 2 — retrieve via flat vector index
        all_chunks = []
        seen = set()
        for sq in sub_queries:
            for c in self.index.search(sq, top_k=self.top_k):
                if c.chunk_id not in seen:
                    seen.add(c.chunk_id)
                    all_chunks.append(c)

        all_chunks.sort(key=lambda c: c.score, reverse=True)
        top_chunks = all_chunks[: self.top_k]

        # Step 3 — build raw context (matches RetrieverAgent format)
        context_blocks = []
        for i, c in enumerate(top_chunks):
            context_blocks.append(
                f"[{i+1}] score={c.score:.3f} | doc={c.doc_id}\n\n{c.text}"
            )
        raw_context = "\n\n---\n\n".join(context_blocks)

        # Step 4 — summarize
        summary_prompt = (
            f"## Original Query\n{query}\n\n"
            f"## Retrieved Passages\n{raw_context}\n\n"
            "Provide a concise structured summary:\n"
            "- Key legal concepts found\n"
            "- Relevant statutes / cases cited\n"
            "- Potential gaps in the retrieved evidence\n"
            "Cite passage numbers."
        )
        summary_msg = self._call_llm(summary_prompt)

        return AgentMessage(
            sender=self.name,
            content=summary_msg.content,
            metadata={
                "sub_queries": sub_queries,
                "raw_context": raw_context,
                "chunks": [
                    {"chunk_id": c.chunk_id, "score": c.score, "text": c.text, "doc_id": c.doc_id}
                    for c in top_chunks
                ],
                "retrieval_mode": "vector",
            },
            input_tokens=decompose_msg.input_tokens + summary_msg.input_tokens,
            output_tokens=decompose_msg.output_tokens + summary_msg.output_tokens,
            latency_s=decompose_msg.latency_s + summary_msg.latency_s,
        )


def _parse_sub_queries(llm_output: str, fallback: str) -> list[str]:
    import re
    sub_queries = []
    for line in llm_output.strip().split("\n"):
        m = re.match(r"^\s*\d+[\.\)]\s*(.+)", line)
        if m:
            sq = m.group(1).strip()
            if len(sq) > 5:
                sub_queries.append(sq)
    return sub_queries if sub_queries else [fallback]
