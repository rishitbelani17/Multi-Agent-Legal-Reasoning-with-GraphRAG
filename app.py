"""
Streamlit chat UI for Multi-Agent Legal Reasoning with GraphRAG.

Run:
    streamlit run app.py

Features
--------
- Pick any of the four ablation pipelines.
- Pick a CaseHOLD example from the dataset (or paste your own query).
- Watch agents talk in real time as chat bubbles:
      Retriever → Plaintiff (R1) → Defense (R1) → ... → Judge
- See retrieved chunks with their source doc + retrieval score.
- See the final prediction vs ground truth, latency, token usage and cost.

This file is intentionally self-contained and does not need ExperimentRunner.
It calls the same pipeline functions used by main.py, so behaviour matches.
"""

from __future__ import annotations

import os
import time
from typing import Any

import streamlit as st

import config
from agents.base_agent import AgentMessage
from data.loader import load_caseholder
from data.preprocessor import Preprocessor
from evaluation.metrics import parse_answer
from evaluation.retrieval_metrics import per_query_retrieval_metrics, chunks_per_doc
from graph.builder import GraphBuilder
from graph.retriever import GraphRetriever
from pipelines.baseline_rag import run_baseline_rag
from pipelines.multi_agent import run_multi_agent
from pipelines.multi_agent_vector import run_multi_agent_vector
from pipelines.single_llm import run_single_llm
from rag.vector_rag import VectorIndex
from utils.embedder import get_embedder
from utils.llm_client import LLMClient


# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Multi-Agent Legal Reasoning · GraphRAG",
    page_icon="⚖️",
    layout="wide",
)


# ─── Cached resources (load once per session) ────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model…")
def _load_embedder():
    return get_embedder()


@st.cache_resource(show_spinner="Loading CaseHOLD dataset (cached after first call)…")
def _load_dataset(subset_size: int):
    docs = load_caseholder(subset_size=subset_size)
    preprocessor = Preprocessor()
    chunks = preprocessor.process_batch(docs)
    return docs, chunks


@st.cache_resource(show_spinner="Building vector index…")
def _build_vector_index(_chunks_tuple):
    embedder = _load_embedder()
    index = VectorIndex(embedder)
    index.build(list(_chunks_tuple))
    return index


@st.cache_resource(show_spinner="Building / loading knowledge graph…")
def _build_graph_retriever(_chunks_tuple):
    embedder = _load_embedder()
    # Cache filename pinned to CORPUS_VERSION so the leaky pre-v2 graph
    # is automatically bypassed and rebuilt from the clean corpus_text.
    cache_path = os.path.join(
        config.GRAPH_CACHE_DIR,
        f"caseholder_graph_{config.CORPUS_VERSION}.pkl",
    )
    if os.path.exists(cache_path):
        graph = GraphBuilder.load(cache_path)
    else:
        os.makedirs(config.GRAPH_CACHE_DIR, exist_ok=True)
        builder = GraphBuilder(embedder)
        graph = builder.build(list(_chunks_tuple))
        GraphBuilder.save(graph, cache_path)
    return GraphRetriever(graph=graph, embedder=embedder)


@st.cache_resource
def _llm_client(_model: str):
    return LLMClient(model=_model)


# ─── Agent → chat-bubble styling ─────────────────────────────────────────────

AGENT_STYLE = {
    "RetrieverAgent": {"emoji": "🔎", "label": "Retriever",  "color": "#4B5563"},
    "PlaintiffAgent": {"emoji": "⚖️", "label": "Plaintiff",  "color": "#1E40AF"},
    "DefenseAgent":   {"emoji": "🛡️", "label": "Defense",    "color": "#B91C1C"},
    "JudgeAgent":     {"emoji": "👨‍⚖️", "label": "Judge",      "color": "#065F46"},
}


def _render_agent_message(stage: str, msg: AgentMessage, container) -> None:
    """Render one agent turn as a chat bubble in the given container."""
    style = AGENT_STYLE.get(msg.sender, {"emoji": "🤖", "label": msg.sender})
    with container.chat_message("assistant", avatar=style["emoji"]):
        round_label = ""
        if "_round_" in stage:
            r = stage.split("_round_")[-1]
            round_label = f" · Round {r}"
        st.markdown(f"**{style['label']}{round_label}**")
        st.markdown(msg.content)
        with st.expander("token usage / latency", expanded=False):
            st.caption(
                f"input={msg.input_tokens}  output={msg.output_tokens}  "
                f"latency={msg.latency_s:.2f}s"
            )


# ─── Sidebar: configuration ──────────────────────────────────────────────────

st.sidebar.title("⚙️ Configuration")

api_ok = bool(config.ANTHROPIC_API_KEY) and config.ANTHROPIC_API_KEY != "PASTE-YOUR-KEY-HERE"
if api_ok:
    st.sidebar.success(f"API key loaded ({config.ANTHROPIC_API_KEY[:12]}…)")
else:
    st.sidebar.error(
        "ANTHROPIC_API_KEY not set. Edit `config.py` line 17 and restart the app."
    )

pipeline_choice = st.sidebar.selectbox(
    "Pipeline",
    options=[
        "single_llm",
        "vector_rag",
        "multi_agent_vector",
        "multi_agent",
    ],
    format_func=lambda p: {
        "single_llm":         "C1 · Single LLM (no retrieval)",
        "vector_rag":         "C2 · Single-Agent + Vector RAG",
        "multi_agent_vector": "C3 · Multi-Agent debate + Vector RAG",
        "multi_agent":        "C4 · Multi-Agent debate + GraphRAG (full system)",
    }.get(p, p),
    index=3,
)

n_debate_rounds = st.sidebar.slider(
    "Debate rounds (Plaintiff↔Defense)",
    min_value=1,
    max_value=4,
    value=config.N_DEBATE_ROUNDS,
    help=(
        "Number of back-and-forth rounds before Judge synthesizes. "
        "1 = single-pass; 2+ = true debate. Cost scales linearly."
    ),
    disabled=pipeline_choice not in {"multi_agent_vector", "multi_agent"},
)

subset_size = st.sidebar.slider(
    "Dataset subset size",
    min_value=5,
    max_value=200,
    value=20,
    step=5,
    help="How many CaseHOLD examples to load into the corpus (affects retrieval pool).",
)

model_name = st.sidebar.selectbox(
    "Worker model (Plaintiff, Defense, Retriever)",
    options=[config.LLM_MODEL, "claude-haiku-4-5-20251001", "claude-sonnet-4-6"],
    index=0,
    help="Model used by the verbose generator agents.",
)

judge_model_options = [
    config.LLM_MODEL_JUDGE or config.LLM_MODEL,
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
]
# Deduplicate while preserving order
judge_model_options = list(dict.fromkeys(judge_model_options))
judge_model_name = st.sidebar.selectbox(
    "Judge model (final discriminator)",
    options=judge_model_options,
    index=0,
    help=(
        "The Judge is a discriminator over a long debate transcript. "
        "Stronger models (Sonnet) are usually worth the small extra cost — "
        "the Judge runs once per query while workers run 5+ times."
    ),
    disabled=pipeline_choice not in {"multi_agent_vector", "multi_agent"},
)


# ─── Main: query selection + run ─────────────────────────────────────────────

st.title("⚖️ Multi-Agent Legal Reasoning with GraphRAG")
st.caption(
    "Watch four specialised agents — Retriever, Plaintiff, Defense, Judge — debate "
    "and resolve a CaseHOLD multiple-choice question."
)

if not api_ok:
    st.stop()

# Load dataset (cached)
docs, chunks = _load_dataset(subset_size)

# Query selector
left, right = st.columns([2, 1])

with left:
    example_idx = st.selectbox(
        f"Pick a CaseHOLD example (out of {len(docs)} loaded)",
        options=list(range(len(docs))),
        format_func=lambda i: f"{docs[i]['id']}  ·  true holding = {docs[i]['label_str']}",
    )
    chosen_doc = docs[example_idx]
    with st.expander("Show full query (citing prompt + 5 candidate holdings)", expanded=False):
        st.text(chosen_doc["text"])

with right:
    st.metric("Ground truth", chosen_doc["label_str"])
    st.metric("Pipeline", pipeline_choice)
    if pipeline_choice in {"multi_agent_vector", "multi_agent"}:
        n_calls = 2 + 2 * n_debate_rounds + 1  # retriever(2) + (P+D)*rounds + judge
        st.metric("LLM calls (est.)", n_calls)

run_button = st.button("▶ Run debate", type="primary", use_container_width=True)


# ─── Run the pipeline ─────────────────────────────────────────────────────────

if run_button:
    llm = _llm_client(model_name)
    # Build a dedicated Judge client only when the Judge model differs from
    # the worker model (otherwise reuse the worker client to save a connection).
    judge_llm = (
        _llm_client(judge_model_name) if judge_model_name != model_name else llm
    )
    QUERY_CAP = 4000
    query = chosen_doc["text"][:QUERY_CAP]

    # Container for streaming agent messages
    chat_section = st.container()
    chat_section.subheader("🗣️ Agent Conversation")

    # Streaming callback for multi-agent pipelines
    def on_msg(stage: str, msg: AgentMessage):
        _render_agent_message(stage, msg, chat_section)

    t0 = time.perf_counter()
    raw: dict[str, Any] = {}

    try:
        if pipeline_choice == "single_llm":
            with chat_section.chat_message("assistant", avatar="🤖"):
                with st.spinner("Single LLM thinking…"):
                    raw = run_single_llm(query, llm, "caseholder")
                st.markdown("**Single LLM**")
                st.markdown(raw.get("answer", ""))

        elif pipeline_choice == "vector_rag":
            vector_index = _build_vector_index(tuple(chunks))
            with chat_section.chat_message("assistant", avatar="🔎"):
                with st.spinner("Retrieving + answering…"):
                    raw = run_baseline_rag(query, vector_index, llm, "caseholder")
                st.markdown("**Vector RAG (single agent)**")
                st.markdown(raw.get("answer", ""))

        elif pipeline_choice == "multi_agent_vector":
            vector_index = _build_vector_index(tuple(chunks))
            raw = run_multi_agent_vector(
                query, vector_index, llm, "caseholder",
                n_debate_rounds=n_debate_rounds,
                on_message=on_msg,
                judge_llm_client=judge_llm,
            )

        elif pipeline_choice == "multi_agent":
            graph_retriever = _build_graph_retriever(tuple(chunks))
            raw = run_multi_agent(
                query, graph_retriever, llm, "caseholder",
                n_debate_rounds=n_debate_rounds,
                on_message=on_msg,
                judge_llm_client=judge_llm,
            )

    except Exception as e:
        st.error(f"Pipeline failed: {e}")
        st.exception(e)
        st.stop()

    elapsed_total = time.perf_counter() - t0

    # ── Result panel ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Result — this query")
    st.caption(
        "All numbers below are **per-query** values for the single example you "
        "just ran. Aggregate accuracy / F1 / consistency across the full subset "
        "live in `results/<run>_metrics.json` from `python3 main.py`."
    )

    pred_label = parse_answer(
        raw.get("answer", ""),
        list({d["label_str"] for d in docs}),
    )
    correct = (pred_label == chosen_doc["label_str"])

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Prediction", pred_label, delta="✓ correct" if correct else "✗ wrong")
    c2.metric("Ground truth", chosen_doc["label_str"])
    c3.metric("Latency (this query)", f"{elapsed_total:.2f}s")
    c4.metric("Input tokens (this query)", raw.get("input_tokens", 0))
    c5.metric("Output tokens (this query)", raw.get("output_tokens", 0))

    # Cost estimate (Haiku 4.5 pricing as of May 2026: $1/MTok in, $5/MTok out)
    in_tok = raw.get("input_tokens", 0)
    out_tok = raw.get("output_tokens", 0)
    est_cost = in_tok * 1e-6 + out_tok * 5e-6
    st.caption(f"Estimated cost for **this single query**: **${est_cost:.4f}**")

    # ── Retrieved chunks (if applicable) ──────────────────────────────────
    retrieved = raw.get("retrieved_chunks", [])
    if retrieved:
        st.subheader(f"📚 Retrieved Chunks ({len(retrieved)}) — this query")

        # Retrieval metrics for this single query
        n_gold_in_corpus = chunks_per_doc(chunks).get(chosen_doc["id"], 0)
        rmetrics = per_query_retrieval_metrics(
            retrieved, chosen_doc["id"], n_gold_in_corpus,
        )
        st.caption(
            "Retrieval quality on **this query only**. Aggregate Hit@K / "
            "Precision / Recall / MRR across the full subset are reported "
            "by `experiments/run_experiments.py`, not here."
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Hit @ K (this query)",   "✓" if rmetrics["hit@k"] else "✗")
        m2.metric("Precision (this query)", f"{rmetrics['precision']:.2f}")
        m3.metric("Recall (this query)",    f"{rmetrics['recall']:.2f}")
        m4.metric("MRR (this query)",       f"{rmetrics['mrr']:.2f}")

        for i, ch in enumerate(retrieved):
            doc_id = ch.get("doc_id") or (
                ch.get("chunk_id", "").rpartition("_c")[0]
            )
            is_gold = doc_id == chosen_doc["id"]
            badge = "🟢 from gold doc" if is_gold else "⚪ other doc"
            with st.expander(
                f"#{i+1} · score={ch.get('score', 0):.3f} · {badge}",
                expanded=False,
            ):
                st.caption(f"chunk_id: `{ch.get('chunk_id', '?')}`  ·  doc_id: `{doc_id}`")
                st.text(ch.get("text", "")[:1500])

    # ── Sub-queries (if multi-agent) ──────────────────────────────────────
    sub_queries = raw.get("sub_queries", [])
    if sub_queries:
        with st.expander("🧩 Retriever's sub-queries", expanded=False):
            for sq in sub_queries:
                st.markdown(f"- {sq}")

    # ── Raw judge answer ──────────────────────────────────────────────────
    with st.expander("🧾 Raw final answer (Judge / LLM output)", expanded=False):
        st.text(raw.get("answer", ""))


# ─── Aggregate-results panel (auto-loads the latest eval run) ───────────────

def _load_latest_metrics() -> dict | None:
    """Find the most recent caseholder_*_metrics.json in ./results and return
    its parsed contents, or None if no eval run has been performed yet."""
    import glob
    import json as _json
    pattern = os.path.join(config.RESULTS_DIR, "caseholder_*_metrics.json")
    # Prefer the rescored variant (post-parser-fix) if present
    rescored = sorted(glob.glob(pattern.replace("metrics.json", "metrics_rescored.json")))
    primary  = sorted(glob.glob(pattern))
    paths    = rescored + [p for p in primary if "_rescored" not in p]
    if not paths:
        return None
    latest = paths[-1]
    try:
        with open(latest) as f:
            return {"path": latest, "metrics": _json.load(f)}
    except Exception:
        return None


with st.expander(
    "📈 Aggregate evaluation results (latest `python3 main.py` run)",
    expanded=False,
):
    agg = _load_latest_metrics()
    if agg is None:
        st.info(
            "No eval run found yet. Run "
            "`python3 main.py --dataset caseholder --subset 25 --n-runs 1` "
            "to populate this panel with headline accuracy / F1 / "
            "retrieval-quality numbers across the full subset."
        )
    else:
        st.caption(f"Source: `{agg['path']}`")
        rows = []
        for pipeline_name, m in agg["metrics"].items():
            rows.append({
                "pipeline":      pipeline_name,
                "accuracy":      m.get("accuracy", 0.0),
                "F1 (macro)":    m.get("f1_macro", 0.0),
                "F1 (weighted)": m.get("f1_weighted", 0.0),
                "n":             m.get("n_samples", 0),
                "avg latency s": m.get("avg_latency_s", 0.0),
            })
        st.dataframe(rows, use_container_width=True)
        # Significance vs baseline (if reported)
        sig_rows = []
        for name, m in agg["metrics"].items():
            sig = m.get("significance")
            if sig:
                sig_rows.append({
                    "pipeline": name,
                    "vs baseline":   sig.get("baseline", ""),
                    "Δacc":          sig.get("delta_accuracy", 0.0),
                    "95% CI low":    sig.get("ci_low", 0.0),
                    "95% CI high":   sig.get("ci_high", 0.0),
                    "bootstrap p":   sig.get("p_value", 1.0),
                    "McNemar p":     sig.get("mcnemar_p", 1.0),
                })
        if sig_rows:
            st.markdown("**Paired significance vs baseline**")
            st.dataframe(sig_rows, use_container_width=True)


# ─── Footer ───────────────────────────────────────────────────────────────────

st.sidebar.divider()
st.sidebar.caption(
    "**About** — This UI is the front-end for the multi-agent debate system "
    "described in the proposal. The Plaintiff and Defense actually rebut each "
    "other across rounds; the Judge synthesises a final ANSWER. Switch between "
    "the four ablation pipelines to compare GraphRAG vs vector RAG vs no retrieval. "
    "**Per-query metrics** in the main panel are for the single example just run; "
    "**aggregate metrics** appear in the expander above once an eval run has been "
    "executed via `python3 main.py`."
)
