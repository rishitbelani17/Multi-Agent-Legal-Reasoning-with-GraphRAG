# Multi-Agent Legal Reasoning with GraphRAG

**Rishit Belani & Sushmita Korishetty — Columbia University, Spring 2026**

## Overview

This project investigates whether combining **multi-agent LLM collaboration**
with **Graph-based Retrieval-Augmented Generation (GraphRAG)** improves legal
reasoning over standard approaches. The implementation follows the proposal's
controlled-ablation design: agent count and prompt structure are held constant
while only one variable changes between adjacent conditions.

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
│   ├── metrics.py                   # Accuracy, F1, consistency
│   ├── cost_tracker.py              # Token + USD cost
│   └── error_taxonomy.py            # 9-category error classification
│
├── experiments/
│   ├── run_experiments.py           # ExperimentRunner orchestrator
│   └── visualize.py                 # Plots
│
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

```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=your_key_here
```

### 3. Run experiments

```bash
# Default: all 4 ablation conditions on CaseHOLD (the primary benchmark)
python3 main.py

# All datasets, all conditions
python3 main.py --dataset all

# Quick smoke-test (5 examples, baseline only)
python3 main.py --dataset caseholder --pipelines single_llm --subset 5

# Run only the GraphRAG ablation pair on CaseHOLD
python3 main.py --dataset caseholder --pipelines multi_agent_vector multi_agent
```

---

## Evaluation Metrics

| Metric                | Description                                               |
|-----------------------|-----------------------------------------------------------|
| **Accuracy**          | Exact match of predicted vs. true label                   |
| **F1 (Macro)**        | Unweighted mean F1 across all classes                     |
| **F1 (Weighted)**     | Class-frequency-weighted F1                               |
| **Inter-run consistency** | Agreement across N repeated runs (`consistency_score`) |
| **Avg Latency**       | Wall-clock time per query (seconds)                       |
| **Token Usage**       | Input + output tokens per query                           |
| **Cost (USD)**        | Estimated API cost                                        |

### Error Taxonomy (9 categories)

`correct`, `retrieval_miss`, `reasoning_gap`, `wrong_label`, `overconfident`,
`underconfident`, `citation_hallucination`, `context_overflow`, `ambiguous_answer`.

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
| `LLM_MODEL`            | `claude-haiku-4-5-20251001`   | Anthropic model                          |

---

## Citation

```
Belani, R. & Korishetty, S. (2026). Multi-Agent Legal Reasoning with GraphRAG.
Columbia University IEOR / GenAI Course Project, Spring 2026.
```
