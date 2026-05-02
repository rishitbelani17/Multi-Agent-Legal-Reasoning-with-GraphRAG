"""
Retriever Agent – responsible for formulating effective sub-queries and
selecting the most relevant graph-retrieved chunks for downstream reasoning.

In the multi-agent pipeline the orchestrator calls this agent first.
"""

from __future__ import annotations

from typing import Any

from agents.base_agent import AgentMessage, BaseAgent
from graph.retriever import GraphRetriever


class RetrieverAgent(BaseAgent):
    """
    Agent that:
    1. Analyses the original query and decomposes it into sub-queries if needed.
    2. Runs each sub-query through the GraphRetriever.
    3. Summarises the retrieved evidence for downstream agents.
    """

    def __init__(self, llm_client, retriever: GraphRetriever, dataset: str = "ledgar"):
        super().__init__(
            name="RetrieverAgent",
            llm_client=llm_client,
            role_description="Retrieves relevant legal text from the knowledge graph.",
        )
        self.retriever = retriever
        self.dataset = dataset

    @property
    def system_prompt(self) -> str:
        return (
            "You are a legal research assistant specialised in evidence retrieval. "
            "Your role is to:\n"
            "1. Analyse the query and identify the key legal concepts, entities, and relationships.\n"
            "2. List any relevant sub-queries that would help retrieve comprehensive evidence.\n"
            "3. Summarise the retrieved passages, highlighting cross-document connections.\n\n"
            "Be precise and focus only on legally relevant information. "
            "Always cite the passage numbers from the context."
        )

    def run(
        self,
        query: str,
        context: str = "",
        history: list[AgentMessage] | None = None,
        **kwargs: Any,
    ) -> AgentMessage:
        # Step 1: Ask LLM to decompose the query
        decompose_prompt = (
            f"## Original Query\n{query}\n\n"
            "Identify 1-3 focused sub-queries that would best retrieve evidence "
            "for this legal question. Return them as a numbered list."
        )
        decompose_msg = self._call_llm(decompose_prompt)
        sub_queries = _parse_sub_queries(decompose_msg.content, query)

        # Step 2: Retrieve chunks for each sub-query
        all_chunks: list = []
        seen_ids: set[str] = set()
        for sq in sub_queries:
            result = self.retriever.retrieve_with_context(sq)
            for rc in result["chunks"]:
                if rc.chunk_id not in seen_ids:
                    seen_ids.add(rc.chunk_id)
                    all_chunks.append(rc)

        # Rank by score and keep top-N
        all_chunks.sort(key=lambda x: x.score, reverse=True)
        top_chunks = all_chunks[:self.retriever.top_n]

        # Step 3: Build structured context string
        context_blocks = []
        for i, rc in enumerate(top_chunks):
            path_str = " → ".join(rc.path[-3:]) if rc.path else rc.chunk_id
            edge_str = ", ".join(rc.edge_types[-2:]) if rc.edge_types else "direct"
            context_blocks.append(
                f"[{i+1}] score={rc.score:.3f} | hop={rc.hop_distance} | "
                f"edge=[{edge_str}] | doc={rc.doc_id}\n"
                f"Path: {path_str}\n\n{rc.text}"
            )
        raw_context = "\n\n---\n\n".join(context_blocks)

        # Step 4: Summarise for downstream agents
        summary_prompt = (
            f"## Original Query\n{query}\n\n"
            f"## Graph-Retrieved Passages\n{raw_context}\n\n"
            "Provide a structured summary of the retrieved evidence:\n"
            "- Key legal concepts found\n"
            "- Relevant statutes / cases cited\n"
            "- Cross-document connections (from provenance paths)\n"
            "- Potential gaps in the retrieved evidence\n\n"
            "Be concise. Cite passage numbers."
        )
        summary_msg = self._call_llm(summary_prompt)

        # Attach raw retrieval data as metadata
        msg = AgentMessage(
            sender=self.name,
            content=summary_msg.content,
            metadata={
                "sub_queries": sub_queries,
                "raw_context": raw_context,
                "chunks": [
                    {
                        "chunk_id": rc.chunk_id,
                        "score": rc.score,
                        "hop_distance": rc.hop_distance,
                        "path": rc.path,
                        "edge_types": rc.edge_types,
                        "text": rc.text,
                    }
                    for rc in top_chunks
                ],
            },
            input_tokens=decompose_msg.input_tokens + summary_msg.input_tokens,
            output_tokens=decompose_msg.output_tokens + summary_msg.output_tokens,
            latency_s=decompose_msg.latency_s + summary_msg.latency_s,
        )
        return msg


def _parse_sub_queries(llm_output: str, fallback: str) -> list[str]:
    """Extract numbered sub-queries from LLM output, falling back to original query."""
    import re
    lines = llm_output.strip().split("\n")
    sub_queries = []
    for line in lines:
        m = re.match(r"^\s*\d+[\.\)]\s*(.+)", line)
        if m:
            sq = m.group(1).strip()
            if len(sq) > 5:
                sub_queries.append(sq)
    return sub_queries if sub_queries else [fallback]
