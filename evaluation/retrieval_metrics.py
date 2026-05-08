"""
Retrieval quality metrics.

For each query in CaseHOLD/ECtHR/LEDGAR, the "gold" relevant document is the
source doc the query was drawn from (the citing prompt is taken from one
specific case). A retrieval system is good if it surfaces chunks from that
source doc.

We report two views:

1. **Document-level**: did *any* of the top-K retrieved chunks come from the
   gold doc? (`hit@k`)
2. **Chunk-level**: of the top-K retrieved chunks, what fraction came from the
   gold doc? (`precision`)
3. **Recall@k**: of all chunks belonging to the gold doc in the corpus, what
   fraction made it into the top-K? (`recall`)
4. **Mean reciprocal rank (MRR)**: 1 / rank of the first gold-doc chunk.

Aggregating across queries gives mean precision, mean recall, hit-rate, MRR.

Note on chunk_id parsing
------------------------
Chunks are created by ``data/preprocessor.py`` with ids of the form
``"{doc_id}_c{N}"``. We use ``_doc_id_from_chunk_id`` to recover the doc_id
without needing the pipelines to thread it through every result dict.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def _doc_id_from_chunk_id(chunk_id: str) -> str:
    """
    Recover the source doc_id from a chunk_id of the form '{doc_id}_c{N}'.
    Falls back to the full chunk_id if the pattern doesn't match.
    """
    if "_c" not in chunk_id:
        return chunk_id
    head, _, tail = chunk_id.rpartition("_c")
    return head if tail.isdigit() else chunk_id


def _retrieved_doc_ids(retrieved_chunks: list[dict]) -> list[str]:
    """Extract doc_ids in retrieval order from a list of chunk dicts."""
    out = []
    for c in retrieved_chunks:
        if "doc_id" in c and c["doc_id"]:
            out.append(c["doc_id"])
        elif "chunk_id" in c and c["chunk_id"]:
            out.append(_doc_id_from_chunk_id(c["chunk_id"]))
        else:
            out.append("")
    return out


def per_query_retrieval_metrics(
    retrieved_chunks: list[dict],
    gold_doc_id: str,
    n_gold_chunks_in_corpus: int | None = None,
) -> dict[str, float]:
    """
    Compute retrieval metrics for a single query.

    Parameters
    ----------
    retrieved_chunks:
        List of chunk dicts (each must have 'chunk_id' or 'doc_id').
    gold_doc_id:
        The id of the source document the query came from.
    n_gold_chunks_in_corpus:
        Total number of chunks belonging to gold_doc_id in the chunk pool.
        If None, recall is reported as the conservative `hit@k` (1 if any chunk
        retrieved, 0 otherwise).
    """
    if not retrieved_chunks:
        return {
            "hit@k": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "mrr": 0.0,
            "n_retrieved": 0,
            "n_gold_in_retrieved": 0,
        }

    doc_ids = _retrieved_doc_ids(retrieved_chunks)
    k = len(doc_ids)
    n_gold_in_retrieved = sum(1 for d in doc_ids if d == gold_doc_id)
    hit = 1.0 if n_gold_in_retrieved > 0 else 0.0

    # MRR
    mrr = 0.0
    for rank, d in enumerate(doc_ids, start=1):
        if d == gold_doc_id:
            mrr = 1.0 / rank
            break

    precision = n_gold_in_retrieved / k

    if n_gold_chunks_in_corpus and n_gold_chunks_in_corpus > 0:
        recall = min(1.0, n_gold_in_retrieved / n_gold_chunks_in_corpus)
    else:
        recall = hit  # conservative fallback

    return {
        "hit@k": hit,
        "precision": precision,
        "recall": recall,
        "mrr": mrr,
        "n_retrieved": k,
        "n_gold_in_retrieved": n_gold_in_retrieved,
    }


def aggregate_retrieval_metrics(
    pipeline_results: dict[str, list[dict]],
    chunks_by_doc: dict[str, int] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Aggregate retrieval metrics across all queries in each pipeline.

    Parameters
    ----------
    pipeline_results:
        ``{pipeline_name: [result_dict, ...]}`` from ExperimentRunner.
        Each result_dict must include ``doc_id`` (the source doc / gold) and
        ``retrieved_chunks`` (list of chunk dicts).
    chunks_by_doc:
        Optional ``{doc_id: n_chunks}`` map for proper recall computation.
    """
    summary: dict[str, dict[str, float]] = {}

    for pname, results in pipeline_results.items():
        if not results:
            continue
        # Pipelines without retrieval (single_llm) won't have retrieved_chunks
        per_query_records = []
        for r in results:
            chunks = r.get("retrieved_chunks", [])
            gold = r.get("doc_id", r.get("true_doc_id", ""))
            if not chunks or not gold:
                continue
            n_gold = (chunks_by_doc or {}).get(gold)
            per_query_records.append(
                per_query_retrieval_metrics(chunks, gold, n_gold)
            )

        if not per_query_records:
            continue

        n = len(per_query_records)
        summary[pname] = {
            "mean_hit@k":     sum(r["hit@k"]     for r in per_query_records) / n,
            "mean_precision": sum(r["precision"] for r in per_query_records) / n,
            "mean_recall":    sum(r["recall"]    for r in per_query_records) / n,
            "mean_mrr":       sum(r["mrr"]       for r in per_query_records) / n,
            "n_queries_with_retrieval": n,
        }

    return summary


def chunks_per_doc(chunks: list) -> dict[str, int]:
    """Build the ``{doc_id: n_chunks}`` map from a list of Chunk objects."""
    counts: dict[str, int] = defaultdict(int)
    for c in chunks:
        counts[c.doc_id] += 1
    return dict(counts)
