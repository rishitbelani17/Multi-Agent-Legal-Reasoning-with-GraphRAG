"""
Retrieval audit on the clean (post-leakage-fix) corpus.

Builds the graph from the new ``corpus_text`` (citing prompt only), runs
GraphRetriever on each query, reports Hit@5 / Precision / Recall / MRR
per query plus the aggregate. Use to double-check that the graph
retriever still finds the gold doc after the v2 corpus rebuild.

Usage
-----
  python3 -m tools.audit_retrieval --subset 10
  python3 -m tools.audit_retrieval --subset 25 --pipeline vector
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)


def main() -> int:
    p = argparse.ArgumentParser(description="Audit retrieval quality on the clean corpus.")
    p.add_argument("--subset", type=int, default=10,
                   help="Number of CaseHOLD examples to audit (default 10)")
    p.add_argument("--pipeline", choices=["graph", "vector"], default="graph",
                   help="Retrieval mode (default graph)")
    args = p.parse_args()

    from data.loader import load_caseholder
    from data.preprocessor import Preprocessor
    from sentence_transformers import SentenceTransformer
    from evaluation.retrieval_metrics import (
        chunks_per_doc, per_query_retrieval_metrics,
    )
    import config

    docs = load_caseholder(subset_size=args.subset)
    chunks = Preprocessor().process_batch(docs)

    # Confirm the fix: candidate-holding text MUST NOT be in the corpus
    corpus = " || ".join(c.text for c in chunks)
    leaks = sum(
        1
        for d in docs
        for cand in d["metadata"]["choices"]
        if cand and len(cand.strip()) >= 30 and cand.strip()[:80] in corpus
    )
    if leaks:
        print(f"FAIL — leakage detected on {leaks} candidates. "
              "Check data/loader.py corpus_text construction.")
        return 1
    print(f"OK    — corpus is clean ({len(docs)} docs, {len(chunks)} chunks, 0 leaks)")

    embedder = SentenceTransformer(config.EMBEDDING_MODEL)

    if args.pipeline == "graph":
        from graph.builder import GraphBuilder
        from graph.retriever import GraphRetriever
        graph = GraphBuilder(embedder).build(chunks)
        retriever = GraphRetriever(graph=graph, embedder=embedder)

        def get_chunks(query):
            res = retriever.retrieve_with_context(query)
            return [
                {"chunk_id": rc.chunk_id, "doc_id": rc.doc_id, "score": rc.score}
                for rc in res["chunks"]
            ]
    else:
        from rag.vector_rag import VectorIndex
        idx = VectorIndex(embedder); idx.build(chunks)

        def get_chunks(query):
            return [
                {"chunk_id": vrc.chunk_id, "doc_id": vrc.doc_id, "score": vrc.score}
                for vrc in idx.search(query, k=config.VECTOR_TOP_K)
            ]

    cpd = chunks_per_doc(chunks)
    print()
    print(f"{'doc_id':<22}{'Hit':>5}{'P':>8}{'R':>8}{'MRR':>8}")
    print("-" * 51)
    sums = [0.0, 0.0, 0.0, 0.0]
    for d in docs:
        q = d["text"][:4000]
        out = get_chunks(q)
        m = per_query_retrieval_metrics(out, d["id"], cpd.get(d["id"], 0))
        sums[0] += m["hit@k"]; sums[1] += m["precision"]
        sums[2] += m["recall"]; sums[3] += m["mrr"]
        print(f"{d['id']:<22}{int(m['hit@k']):>5}"
              f"{m['precision']:>8.2f}{m['recall']:>8.2f}{m['mrr']:>8.2f}")
    n = len(docs)
    print("-" * 51)
    print(f"{'mean':<22}{sums[0]/n:>5.2f}"
          f"{sums[1]/n:>8.2f}{sums[2]/n:>8.2f}{sums[3]/n:>8.2f}")
    print()
    print("Reading the numbers:")
    print("  Hit@K        — gold doc retrieved at all in top-K")
    print("  Precision    — fraction of retrieved chunks from the gold doc")
    print("  Recall       — fraction of gold-doc chunks in top-K")
    print("  MRR          — mean reciprocal rank of first gold-doc chunk")
    return 0


if __name__ == "__main__":
    sys.exit(main())
