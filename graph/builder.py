"""
Builds a legal knowledge graph from document chunks using NetworkX.

Two node families coexist in the same graph (per proposal §2.3):

1. Chunk nodes  (node_type="chunk")
       Text-chunk nodes carrying embeddings; connected by semantic-similarity
       and entity-co-occurrence edges. Used by the GraphRetriever for
       embedding-based traversal.

2. Typed legal-entity nodes  (node_type ∈ {"case","issue","fact","ruling"})
       Lightweight legal-knowledge-graph layer with TYPED edges
       (cites, supports, applies_to, contradicts) extracted from chunk text
       via regex/heuristic rules. This is the proposal's "lightweight legal
       knowledge graph"; it is overlaid on the chunk graph so a single
       NetworkX graph powers both retrieval styles.

Each chunk node is linked to the entity nodes mentioned in its text via
``mentions`` edges, enabling 1–2 hop subgraph retrieval over typed entities.

The resulting graph can be serialised and reloaded via pickle so expensive
embedding passes don't need to be repeated across experiments.
"""

from __future__ import annotations

import logging
import os
import pickle
import re
from typing import Any

import networkx as nx
import numpy as np

import config
from data.preprocessor import Chunk, Preprocessor

logger = logging.getLogger(__name__)


# ── Typed-edge / typed-node schema (per proposal §2.3) ───────────────────────
LEGAL_NODE_TYPES = ("case", "issue", "fact", "ruling")
LEGAL_EDGE_TYPES = ("cites", "supports", "applies_to", "contradicts", "mentions")


# Heuristic cue phrases for typed-edge extraction. These are intentionally
# lightweight – the proposal calls for a "lightweight" legal KG, not full
# information extraction. They can be swapped for an LLM-based extractor later.
_CITES_CUES        = (r"see\s+", r"cf\.?\s+", r"citing\s+", r"\bv\.\s+")
_SUPPORTS_CUES     = (r"holds?\s+that", r"affirm(?:s|ed)?", r"supports?\s+",
                       r"consistent\s+with")
_APPLIES_TO_CUES   = (r"applies?\s+to", r"governs?", r"under\s+", r"pursuant\s+to")
_CONTRADICTS_CUES  = (r"overrul(?:e|ed|es)", r"reject(?:s|ed)?", r"distinguish(?:es|ed)",
                       r"contradicts?")

_RULING_CUE_RE = re.compile(
    r"\b(?:held|holding|holds?|ruled?|conclude[ds]?|affirmed|reversed)\b",
    re.IGNORECASE,
)
_ISSUE_CUE_RE = re.compile(
    r"\b(?:whether|the\s+question|the\s+issue|at\s+issue)\b",
    re.IGNORECASE,
)


class GraphBuilder:
    """
    Constructs a document-chunk graph.

    Usage
    -----
    builder = GraphBuilder(embedder)
    graph   = builder.build(chunks)
    builder.save(graph, path)
    graph   = GraphBuilder.load(path)
    """

    def __init__(
        self,
        embedder,                              # any object with .encode(texts) -> np.ndarray
        edge_threshold: float = config.GRAPH_EDGE_THRESHOLD,
        add_entity_edges: bool = True,
        add_typed_legal_layer: bool = True,
    ):
        self.embedder = embedder
        self.edge_threshold = edge_threshold
        self.add_entity_edges = add_entity_edges
        self.add_typed_legal_layer = add_typed_legal_layer

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, chunks: list[Chunk]) -> nx.Graph:
        """
        Build and return an undirected weighted graph.

        Node attributes
        ---------------
        text, doc_id, embedding, metadata
        """
        logger.info("Building graph from %d chunks …", len(chunks))
        G = nx.Graph()

        # ── 1. Add nodes ──────────────────────────────────────────────────────
        texts = [c.text for c in chunks]
        logger.info("Embedding %d chunks …", len(texts))
        embeddings: np.ndarray = self.embedder.encode(
            texts, show_progress_bar=True, convert_to_numpy=True
        )

        for chunk, emb in zip(chunks, embeddings):
            G.add_node(
                chunk.chunk_id,
                node_type="chunk",
                text=chunk.text,
                doc_id=chunk.doc_id,
                embedding=emb,
                metadata=chunk.metadata,
            )

        # ── 2. Semantic similarity edges ──────────────────────────────────────
        logger.info("Computing pairwise cosine similarities …")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
        normed = embeddings / norms
        sim_matrix = normed @ normed.T  # (N, N)

        node_ids = [c.chunk_id for c in chunks]
        n = len(node_ids)
        edge_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(sim_matrix[i, j])
                if sim >= self.edge_threshold:
                    G.add_edge(
                        node_ids[i],
                        node_ids[j],
                        weight=sim,
                        edge_type="semantic",
                    )
                    edge_count += 1
        logger.info("Added %d semantic edges (threshold=%.2f)", edge_count, self.edge_threshold)

        # ── 3. Entity co-occurrence edges ─────────────────────────────────────
        if self.add_entity_edges:
            entity_count = self._add_entity_edges(G, chunks)
            logger.info("Added %d entity-co-occurrence edges", entity_count)

        # ── 4. Typed legal-knowledge layer (case/issue/fact/ruling nodes) ────
        if self.add_typed_legal_layer:
            typed_nodes, typed_edges = self._add_typed_legal_layer(G, chunks)
            logger.info(
                "Added typed legal layer: %d typed nodes, %d typed edges "
                "(types: %s)",
                typed_nodes, typed_edges, list(LEGAL_EDGE_TYPES),
            )

        logger.info(
            "Graph complete: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges()
        )
        return G

    # ── Entity edges ──────────────────────────────────────────────────────────

    @staticmethod
    def _add_entity_edges(G: nx.Graph, chunks: list[Chunk]) -> int:
        """
        Connect chunks that share at least one legal entity (case citation,
        statute, or article).  Edge weight = Jaccard similarity of entity sets.
        """
        entity_map: dict[str, list[str]] = {}  # entity_str → [chunk_ids]
        for chunk in chunks:
            ents = Preprocessor.extract_entities(chunk.text)
            for etype, matches in ents.items():
                for m in matches:
                    key = f"{etype}::{m.lower().strip()}"
                    entity_map.setdefault(key, []).append(chunk.chunk_id)

        added = 0
        for ent_key, cids in entity_map.items():
            if len(cids) < 2:
                continue
            for i in range(len(cids)):
                for j in range(i + 1, len(cids)):
                    u, v = cids[i], cids[j]
                    if G.has_edge(u, v):
                        # Upgrade existing edge
                        G[u][v]["edge_type"] = "semantic+entity"
                    else:
                        G.add_edge(u, v, weight=0.6, edge_type="entity")
                        added += 1
        return added

    # ── Typed legal-knowledge layer ───────────────────────────────────────────

    def _add_typed_legal_layer(
        self,
        G: nx.Graph,
        chunks: list[Chunk],
    ) -> tuple[int, int]:
        """
        Overlay typed legal-entity nodes and typed edges onto the chunk graph.

        Node types: case, issue, fact, ruling
        Edge types: cites, supports, applies_to, contradicts, mentions

        Returns
        -------
        (n_typed_nodes_added, n_typed_edges_added)
        """
        n_nodes_before = G.number_of_nodes()
        n_edges_before = G.number_of_edges()

        # Per-chunk: extract case-citation entities, issue/fact/ruling spans,
        # and link them with mentions/typed edges.
        case_node_ids: dict[str, str] = {}    # canonical_name → node_id

        for chunk in chunks:
            ents = Preprocessor.extract_entities(chunk.text)
            citations = ents.get("case_citation", [])

            # 1. Add case nodes
            chunk_case_nodes: list[str] = []
            for cite in citations:
                key = cite.lower().strip()
                if key not in case_node_ids:
                    nid = f"case::{key}"
                    G.add_node(nid, node_type="case", label=cite, source_chunks=[chunk.chunk_id])
                    case_node_ids[key] = nid
                else:
                    nid = case_node_ids[key]
                    G.nodes[nid].setdefault("source_chunks", []).append(chunk.chunk_id)
                chunk_case_nodes.append(nid)
                # Mentions edge
                G.add_edge(chunk.chunk_id, nid, edge_type="mentions", weight=0.5)

            # 2. Detect ruling and issue spans → derived nodes (one per chunk)
            ruling_nid = None
            if _RULING_CUE_RE.search(chunk.text):
                ruling_nid = f"ruling::{chunk.chunk_id}"
                G.add_node(
                    ruling_nid, node_type="ruling",
                    text=chunk.text[:240], source_chunk=chunk.chunk_id,
                )
                G.add_edge(chunk.chunk_id, ruling_nid, edge_type="mentions", weight=0.5)

            issue_nid = None
            if _ISSUE_CUE_RE.search(chunk.text):
                issue_nid = f"issue::{chunk.chunk_id}"
                G.add_node(
                    issue_nid, node_type="issue",
                    text=chunk.text[:240], source_chunk=chunk.chunk_id,
                )
                G.add_edge(chunk.chunk_id, issue_nid, edge_type="mentions", weight=0.5)

            # 3. Fact node (default for any chunk lacking ruling/issue cues)
            fact_nid = None
            if ruling_nid is None and issue_nid is None and len(chunk.text) >= 50:
                fact_nid = f"fact::{chunk.chunk_id}"
                G.add_node(
                    fact_nid, node_type="fact",
                    text=chunk.text[:240], source_chunk=chunk.chunk_id,
                )
                G.add_edge(chunk.chunk_id, fact_nid, edge_type="mentions", weight=0.4)

            # 4. Typed legal edges from case ↔ ruling/issue/fact
            target_legal_node = ruling_nid or issue_nid or fact_nid
            if target_legal_node and chunk_case_nodes:
                for case_nid in chunk_case_nodes:
                    et = self._classify_typed_edge(chunk.text)
                    G.add_edge(case_nid, target_legal_node, edge_type=et, weight=0.7)

            # 5. cites edges between co-occurring cases
            for i in range(len(chunk_case_nodes)):
                for j in range(i + 1, len(chunk_case_nodes)):
                    u, v = chunk_case_nodes[i], chunk_case_nodes[j]
                    if not G.has_edge(u, v):
                        G.add_edge(u, v, edge_type="cites", weight=0.6)

        n_nodes_added = G.number_of_nodes() - n_nodes_before
        n_edges_added = G.number_of_edges() - n_edges_before
        return n_nodes_added, n_edges_added

    @staticmethod
    def _classify_typed_edge(text: str) -> str:
        """
        Pick a typed legal edge label by scanning the chunk text for
        cue phrases. Falls back to 'applies_to' as the default relation.
        """
        lowered = text.lower()
        for pat in _CONTRADICTS_CUES:
            if re.search(pat, lowered):
                return "contradicts"
        for pat in _SUPPORTS_CUES:
            if re.search(pat, lowered):
                return "supports"
        for pat in _CITES_CUES:
            if re.search(pat, lowered):
                return "cites"
        for pat in _APPLIES_TO_CUES:
            if re.search(pat, lowered):
                return "applies_to"
        return "applies_to"

    # ── Persistence ───────────────────────────────────────────────────────────

    @staticmethod
    def save(G: nx.Graph, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(G, f)
        logger.info("Graph saved to %s", path)

    @staticmethod
    def load(path: str) -> nx.Graph:
        with open(path, "rb") as f:
            G = pickle.load(f)
        logger.info(
            "Graph loaded from %s (%d nodes, %d edges)",
            path, G.number_of_nodes(), G.number_of_edges(),
        )
        return G

    # ── Graph statistics ──────────────────────────────────────────────────────

    @staticmethod
    def stats(G: nx.Graph) -> dict[str, Any]:
        degrees = [d for _, d in G.degree()]
        weights = [d.get("weight", 0) for _, _, d in G.edges(data=True)]
        return {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "avg_degree": float(np.mean(degrees)) if degrees else 0.0,
            "max_degree": int(max(degrees)) if degrees else 0,
            "avg_edge_weight": float(np.mean(weights)) if weights else 0.0,
            "connected_components": nx.number_connected_components(G),
            "density": nx.density(G),
        }
