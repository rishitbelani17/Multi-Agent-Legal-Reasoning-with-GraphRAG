"""
GraphRAG retriever.

Given a query, it:
  1. Embeds the query.
  2. Finds the top-k semantically similar seed nodes.
  3. Expands outward via BFS/DFS up to max_hops hops.
  4. Scores and returns the top-N retrieved chunks with provenance paths.

The provenance paths enable interpretability – we can show the chain of
reasoning from query → seed → related chunks, which is a key hypothesis
in the proposal (does GraphRAG provide clearer evidence chains?).
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np

import config

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A chunk returned by the GraphRAG retriever."""
    chunk_id: str
    doc_id: str
    text: str
    score: float                              # combined relevance score
    hop_distance: int                         # 0 = seed node
    path: list[str] = field(default_factory=list)   # traversal path from seed
    metadata: dict[str, Any] = field(default_factory=dict)
    edge_types: list[str] = field(default_factory=list)   # edge types along path


class GraphRetriever:
    """
    GraphRAG retrieval from a pre-built NetworkX graph.

    Parameters
    ----------
    graph:        NetworkX Graph built by GraphBuilder.
    embedder:     Embedder with .encode() – same model used to build the graph.
    seed_k:       Number of seed nodes (highest cosine similarity to query).
    max_hops:     Maximum BFS hops from each seed node.
    top_n:        Final number of chunks to return.
    hop_decay:    Score multiplier applied per hop (penalises distant nodes).
    """

    def __init__(
        self,
        graph: nx.Graph,
        embedder,
        seed_k: int = config.GRAPH_SEED_K,
        max_hops: int = config.GRAPH_MAX_HOPS,
        top_n: int = config.VECTOR_TOP_K,
        hop_decay: float = 0.8,
    ):
        self.G = graph
        self.embedder = embedder
        self.seed_k = seed_k
        self.max_hops = max_hops
        self.top_n = top_n
        self.hop_decay = hop_decay

        # Pre-stack chunk-node embeddings for fast nearest-neighbor seeding.
        # Typed legal-knowledge nodes (case/issue/fact/ruling) have no
        # embedding attribute and are reachable only via traversal from
        # chunk seeds, so we exclude them from the seed pool here.
        self._node_ids: list[str] = [
            nid for nid in graph.nodes()
            if "embedding" in graph.nodes[nid]
        ]
        embeddings = [graph.nodes[nid]["embedding"] for nid in self._node_ids]
        mat = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-10
        self._normed_embeddings = mat / norms

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """
        Main retrieval method.
        Returns top-N RetrievedChunk objects sorted by score (descending).
        """
        query_emb = self._embed_query(query)
        seeds = self._find_seeds(query_emb)
        candidates = self._bfs_expand(seeds, query_emb)
        return candidates[: self.top_n]

    def retrieve_with_context(self, query: str) -> dict[str, Any]:
        """
        Retrieve chunks and return a structured context dict suitable for
        passing to an LLM prompt.
        """
        chunks = self.retrieve(query)
        context_blocks = []
        for i, rc in enumerate(chunks):
            block = (
                f"[{i+1}] (score={rc.score:.3f}, hop={rc.hop_distance}, "
                f"doc={rc.doc_id})\n{rc.text}"
            )
            context_blocks.append(block)

        return {
            "query": query,
            "context": "\n\n".join(context_blocks),
            "chunks": chunks,
            "num_chunks": len(chunks),
            "provenance": [
                {
                    "chunk_id": rc.chunk_id,
                    "path": rc.path,
                    "edge_types": rc.edge_types,
                    "hop_distance": rc.hop_distance,
                }
                for rc in chunks
            ],
        }

    # ── Internals ─────────────────────────────────────────────────────────────

    def _embed_query(self, query: str) -> np.ndarray:
        emb = self.embedder.encode([query], convert_to_numpy=True)[0]
        norm = np.linalg.norm(emb) + 1e-10
        return emb / norm

    def _find_seeds(self, query_emb: np.ndarray) -> list[tuple[str, float]]:
        """Return top-k (node_id, cosine_sim) pairs."""
        sims = self._normed_embeddings @ query_emb  # (N,)
        top_idx = np.argsort(sims)[::-1][: self.seed_k]
        return [(self._node_ids[i], float(sims[i])) for i in top_idx]

    def _bfs_expand(
        self,
        seeds: list[tuple[str, float]],
        query_emb: np.ndarray,
    ) -> list[RetrievedChunk]:
        """
        BFS from each seed up to max_hops.
        Score = seed_sim * hop_decay^hop * edge_weight_product_along_path.
        Deduplicate by chunk_id; keep best score.
        """
        best: dict[str, RetrievedChunk] = {}

        for seed_id, seed_sim in seeds:
            # BFS state: (node_id, hop, path_nodes, path_edge_types, path_score)
            queue: deque = deque()
            queue.append((seed_id, 0, [seed_id], [], seed_sim))
            visited: set[str] = set()

            while queue:
                node_id, hop, path, edge_types, score = queue.popleft()
                if node_id in visited:
                    continue
                visited.add(node_id)

                node_data = self.G.nodes[node_id]
                rc = RetrievedChunk(
                    chunk_id=node_id,
                    doc_id=node_data.get("doc_id", ""),
                    text=node_data.get("text", ""),
                    score=score,
                    hop_distance=hop,
                    path=list(path),
                    metadata=node_data.get("metadata", {}),
                    edge_types=list(edge_types),
                )

                # Keep best scoring occurrence
                if node_id not in best or best[node_id].score < score:
                    best[node_id] = rc

                if hop < self.max_hops:
                    for neighbor in self.G.neighbors(node_id):
                        if neighbor not in visited:
                            edge_data = self.G[node_id][neighbor]
                            ew = edge_data.get("weight", 0.5)
                            et = edge_data.get("edge_type", "unknown")
                            new_score = score * self.hop_decay * ew
                            queue.append(
                                (
                                    neighbor,
                                    hop + 1,
                                    path + [neighbor],
                                    edge_types + [et],
                                    new_score,
                                )
                            )

        results = sorted(best.values(), key=lambda x: x.score, reverse=True)
        return results
