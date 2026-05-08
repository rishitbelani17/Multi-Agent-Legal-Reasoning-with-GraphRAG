# Multi-Agent Legal Reasoning with GraphRAG

**Rishit Belani & Sushmita Korishetty — Columbia University, Spring 2026**

## Overview

This project investigates whether combining **multi-agent LLM collaboration**
with **Graph-based Retrieval-Augmented Generation (GraphRAG)** improves legal
reasoning over standard approaches. The implementation follows the proposal's
controlled-ablation design: agent count and prompt structure are held constant
while only one variable changes between adjacent conditions.

### Quickstart

```bash
# 1. Configure: paste your Anthropic API key into config.py line 17
# 2. Install: pip3 install -r requirements.txt
# 3. Smoke test (≈30 s, < $0.01):
python3 main.py --dataset caseholder --pipelines single_llm --subset 5

# 4. Full proposal-aligned ablation (≈30 min, ≈$2):
python3 main.py --dataset caseholder --subset 100 --n-runs 3 --n-debate-rounds 2

# 5. Live agent-debate UI:
python3 -m streamlit run app.py
```

### What's new in this revision

- **True multi-agent debate** — Plaintiff and Defense now rebut each other
  across multiple rounds (not a single-pass cascade). Configurable via
  `--n-debate-rounds` or `config.N_DEBATE_ROUNDS`.
- **Inter-run consistency metric** wired up via `--n-runs N`; reports pairwise
  / majority / full agreement per pipeline.
- **Retrieval quality metrics** — precision, recall, MRR, hit@k against the
  source (gold) document for every retrieval-using pipeline.
- **Statistical significance** — paired-bootstrap accuracy CI + McNemar's
  exact test against a baseline (default: `vector_rag`).
- **Confusion-matrix + Δ-accuracy plots** auto-generated alongside the existing
  metric plots.
- **Streamlit chat UI (`app.py`)** — watch the four agents debate any CaseHOLD
  example in real time, with retrieved chunks annotated against the gold doc.
- **Automated error-analysis tool** — classifies multi-agent failures into
  `retrieval_miss / plaintiff_wrong / defense_flipped / judge_overrode /
  parser_glitch` and emits a markdown report.

### Research Questions

1. Does a multi-agent system outperform a single LLM on legal reasoning benchmarks?
2. Does GraphRAG improve over standard vector RAG, especially for relationship-heavy queries?
3. Which contributes more — multi-agent collaboration or structured retrieval?
4. Does GraphRAG improve interpretability via clearer evidence chains and lower hallucinations?
5. What are the computational tradeoffs (tokens, latency) as system complexity increases?

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   main.py (CLI)                     │
└──────────────────────┬──────────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │   ExperimentRunner      │
          │  (run_experiments.py)   │
          └────────────┬────────────┘
                       │
   ┌───────────┬───────┴────────┬────────────────────┐
   ▼           ▼                ▼                    ▼
┌────────┐ ┌─────────┐ ┌─────────────────┐ ┌──────────────────┐
│ C1:    │ │ C2:     │ │ C3:             │ │ C4:              │
│ Single │ │ Vector  │ │ Multi-Agent     │ │ Multi-Agent      │
│  LLM   │ │  RAG    │ │  + Vector RAG   │ │  + GraphRAG      │
│ (no    │ │(single  │ │  (debate)       │ │  (full system)   │
│  retr.)│ │ agent)  │ │                 │ │                  │
└────────┘ └─────────┘ └────────┬────────┘ └────────┬─────────┘
                                │                   │
                                ▼                   ▼
                       ┌────────────────────────────────────┐
                       │    Sequential Debate Pipeline      │
                       │                                    │
                       │  Retriever → Plaintiff → Defense   │
                       │              → Judge               │
                       └────────────────────────────────────┘
```

### 4 Experimental Conditions (proposal §2.1)

| #  | CLI name              | Retrieval     | Agents | Role                       |
|----|-----------------------|---------------|--------|----------------------------|
| C1 | `single_llm`          | None          | 1      | Baseline                   |
| C2 | `vector_rag`          | Standard RAG  | 1      | RAG baseline               |
| C3 | `multi_agent_vector`  | Standard RAG  | 4      | Multi-agent baseline       |
| C4 | `multi_agent`         | GraphRAG      | 4      | Full proposed system       |

Two diagnostic pipelines are also available (`graph_rag`, `single_agent`) for
isolating retrieval-only or single-agent-on-graph effects.

### Multi-Agent Debate Pipeline (proposal §2.2)

Sequential, role-specialized agents:

- **Retriever** — fetches relevant evidence (graph subgraph or vector top-k).
- **Plaintiff** — constructs the strongest affirmative legal argument.
- **Defense** — identifies weak reasoning, missing evidence, and counter-arguments.
- **Judge** — synthesizes both positions and produces the final reasoned answer.

### Knowledge Graph Schema (proposal §2.3)

Two layers coexist in a single NetworkX graph:

1. **Chunk layer** — text-chunk nodes with embeddings; semantic-similarity and
   entity-co-occurrence edges. Used for embedding-based seed search.
2. **Typed legal layer** — typed nodes (`case`, `issue`, `fact`, `ruling`) and
   typed edges (`cites`, `supports`, `applies_to`, `contradicts`, `mentions`).
   Used for relational 1–2 hop reasoning over legal entities.

---

## Datasets

| Dataset       | Role                              | Task                              | Source               |
|---------------|-----------------------------------|-----------------------------------|----------------------|
| **CaseHOLD**  | **Primary benchmark** (proposal)  | Multiple-choice (5 holdings)      | `casehold/casehold`  |
| ECtHR         | Optional                          | Binary violation prediction       | `lex_glue/ecthr_a`   |
| LEDGAR        | Optional extension                | Multi-class clause classification | `lex_glue/ledgar`    |

All datasets are loaded via HuggingFace `datasets` and cached locally.

---

## Project Structure

```
ProjGenAI/
├── main.py                          # CLI entry point
├── config.py                        # Central configuration
├── requirements.txt
├── .env.example                     # Copy → .env, add your API key
│
├── data/
│   ├── loader.py                    # CaseHOLD, ECtHR, LEDGAR loaders
│   └── preprocessor.py              # Chunking + entity extraction
│
├── graph/
│   ├── builder.py                   # NetworkX chunk graph + typed legal layer
│   └── retriever.py                 # GraphRAG retrieval with BFS traversal
│
├── rag/
│   ├── vector_rag.py                # Flat vector retrieval + single-call RAG
│   └── graph_rag.py                 # Graph-aware retrieval (no agents)
│
├── agents/
│   ├── base_agent.py                # Base agent + AgentMessage
│   ├── retriever_agent.py           # Sub-query decomposition + GraphRAG
│   ├── vector_retriever_agent.py    # Sub-query decomposition + vector RAG
│   ├── plaintiff_agent.py           # Affirmative legal argument
│   ├── defense_agent.py             # Opposing counter-argument
│   ├── judge_agent.py               # Final synthesis + ANSWER
│   ├── reasoner_agent.py            # (legacy) reasoning agent
│   ├── critic_agent.py              # (legacy) critic + revision agent
│   └── orchestrator.py              # DebateOrchestrator (+ legacy MultiAgentOrchestrator)
│
├── pipelines/
│   ├── single_llm.py                # C1: no-retrieval baseline
│   ├── baseline_rag.py              # C2: vector RAG (single agent)
│   ├── multi_agent_vector.py        # C3: 4-agent debate + vector RAG
│   ├── multi_agent.py               # C4: 4-agent debate + GraphRAG
│   ├── graph_rag.py                 # diagnostic: GraphRAG, no agents
│   └── single_agent.py              # diagnostic: 1 agent + GraphRAG
│
├── evaluation/
│   ├── metrics.py                   # Accuracy, F1, consistency, parse_answer
│   ├── retrieval_metrics.py         # Precision / Recall / MRR vs gold doc
│   ├── significance.py              # Paired bootstrap + McNemar's exact test
│   ├── cost_tracker.py              # Token + USD cost
│   └── error_taxonomy.py            # 9-category automatic error classification
│
├── experiments/
│   ├── run_experiments.py           # ExperimentRunner orchestrator
│   └── visualize.py                 # Plots (metrics, confusion, significance)
│
├── tools/
│   └── analyze_errors.py            # python3 -m tools.analyze_errors PREFIX
│
├── app.py                           # Streamlit chat UI
└── utils/
    ├── llm_client.py                # Anthropic API wrapper with retries
    ├── embedder.py                  # Cached SentenceTransformer loader
    └── helpers.py                   # Logging, JSON I/O, dirs
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API key

Paste your Anthropic API key into `config.py` line 17 (between the quotes):

```python
ANTHROPIC_API_KEY: str = "sk-ant-api03-…your-key…"
```

This is the **single source of truth** for the key for the whole project. The
optional fallback to an env-var / `.env` is kept for CI use only.

### 3. Run experiments

```bash
# Default: all 4 ablation conditions on CaseHOLD (the primary benchmark)
python3 main.py

# Quick smoke-test (5 examples, baseline only — ≈30 s, < $0.01)
python3 main.py --dataset caseholder --pipelines single_llm --subset 5

# Run only the GraphRAG ablation pair on CaseHOLD
python3 main.py --dataset caseholder --pipelines multi_agent_vector multi_agent

# Full proposal-aligned ablation with consistency and debate enabled
# (≈30 min on a Mac, ≈$2 on Haiku 4.5)
python3 main.py --dataset caseholder --subset 100 --n-runs 3 --n-debate-rounds 2
```

### 4. Live agent-debate UI

```bash
python3 -m streamlit run app.py
```
Opens at `http://localhost:8501`. Pick any CaseHOLD example, watch the four
agents debate it in real-time chat bubbles, and inspect retrieved chunks with
gold-doc badges and per-query retrieval P/R/MRR.

### 5. CLI flags reference

| Flag                   | Default            | Purpose                                                   |
|------------------------|--------------------|-----------------------------------------------------------|
| `--dataset`            | `caseholder`       | `caseholder` / `ledgar` / `ecthr` / `all`                |
| `--pipelines`          | all 4 ablations    | Subset of pipelines to run                                |
| `--subset N`           | dataset default    | Override the per-dataset subset size                      |
| `--n-runs N`           | `1`                | Repeat each pipeline N times (enables consistency metric) |
| `--n-debate-rounds N`  | `2`                | Plaintiff↔Defense rebuttal rounds (1 = single-pass)       |
| `--model`              | `claude-haiku-4-5` | Anthropic model id                                        |
| `--no-plots`           | off                | Skip plot generation                                      |

### 6. After-the-fact error analysis

```bash
# Replace the prefix with a real one printed at the end of your run
python3 -m tools.analyze_errors ./results/caseholder_20260505_154054 \
        --pipeline multi_agent --top 5
```
Writes `<prefix>_error_analysis.md` containing per-failure classification
(`retrieval_miss / plaintiff_wrong / defense_flipped / judge_overrode /
parser_glitch`) and a recommendations section based on the dominant failure mode.

---

## Evaluation Metrics

| Metric                | Description                                               |
|-----------------------|-----------------------------------------------------------|
| **Accuracy**              | Exact match of predicted vs. true label                                     |
| **F1 (Macro)**            | Unweighted mean F1 across all classes                                       |
| **F1 (Weighted)**         | Class-frequency-weighted F1                                                 |
| **Inter-run consistency** | Pairwise / majority / full agreement across N repeated runs (`--n-runs N`) |
| **Retrieval Precision**   | Fraction of top-K retrieved chunks that come from the gold (source) doc     |
| **Retrieval Recall**      | Fraction of the gold doc's chunks that made it into the top-K               |
| **MRR (retrieval)**       | Mean reciprocal rank of the first gold-doc chunk in the retrieval ranking   |
| **Hit@K**                 | Whether *any* gold-doc chunk was in the top-K                               |
| **ΔAccuracy + bootstrap CI** | Paired-bootstrap accuracy difference vs the chosen baseline              |
| **McNemar p-value**       | Exact paired test on disagreements (b vs. c)                                |
| **Avg Latency**           | Wall-clock time per query (seconds)                                         |
| **Token Usage**           | Input + output tokens per query                                             |
| **Cost (USD)**            | Estimated API cost (per pipeline + total)                                   |

### Error Taxonomy

Two complementary classifications are produced:

1. **Outcome taxonomy** (auto, all pipelines) — 9 categories:
   `correct`, `retrieval_miss`, `reasoning_gap`, `wrong_label`, `overconfident`,
   `underconfident`, `citation_hallucination`, `context_overflow`, `ambiguous_answer`.

2. **Multi-agent failure taxonomy** (`tools/analyze_errors.py`) — for any
   `multi_agent*` pipeline, each failure is classified as one of:
   - `retrieval_miss` — gold doc absent from retrieved chunks
   - `plaintiff_wrong` — Plaintiff opened with the wrong holding
   - `defense_flipped` — Plaintiff was right but Defense's counter became the final answer
   - `judge_overrode` — Plaintiff was right, Defense raised a wrong counter, and the Judge picked yet a third holding
   - `parser_glitch` — the Judge's reasoning favoured the gold answer but the parsed `FINAL ANSWER` line said something else

   The tool surfaces the dominant failure mode and recommends the corresponding
   fix (e.g. raise `GRAPH_MAX_HOPS`, lower `N_DEBATE_ROUNDS`, tighten Judge prompt).

---

## Reproducibility

- All random sampling uses `RANDOM_SEED=42` (configurable via `.env`)
- Dataset subsets are deterministic given the seed
- Model temperature is `0.0` for greedy decoding
- Graph construction uses the same embedding model (`all-MiniLM-L6-v2`) across runs

---

## Key Hyperparameters

| Parameter              | Default                       | Description                              |
|------------------------|-------------------------------|------------------------------------------|
| `GRAPH_EDGE_THRESHOLD` | 0.5                           | Min cosine similarity for semantic edges |
| `GRAPH_MAX_HOPS`       | 2                             | BFS traversal depth from seed nodes      |
| `GRAPH_SEED_K`         | 3                             | Number of seed nodes per query           |
| `VECTOR_TOP_K`         | 5                             | Top-K chunks returned by retriever       |
| `MAX_AGENT_ROUNDS`     | 3                             | (Legacy critic loop only)                |
| `N_DEBATE_ROUNDS`      | 2                             | Plaintiff↔Defense rebuttal rounds (multi-agent debate) |
| `LLM_MODEL`            | `claude-haiku-4-5-20251001`   | Anthropic model                          |

---

## Citation

```
Belani, R. & Korishetty, S. (2026). Multi-Agent Legal Reasoning with GraphRAG.
Columbia University IEOR / GenAI Course Project, Spring 2026.
```
