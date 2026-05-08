# Defense script — 32-slide deck (Untitled presentation.pdf)

**Speaking time:** ~10 min · **Total slides:** 32 · **Visual slides:** 20 (S6, S9, S11–S30)

---

## Pre-flight (T-30 min)

1. Streamlit on `localhost:8501`, pre-tested CaseHOLD example loaded, subset = 200.
2. Two terminal tabs: one at `~/ProjGenAI`, one tailing Streamlit logs.
3. 90-s screencast backup open in QuickTime.
4. Manual PDF on a second monitor at §8.1 (headline table).
5. **Two slide edits done:** S3/S31 say "25 queries (post-leakage-fix v2 corpus)", and S5 H3 has a small footnote "see S32 — H3 not supported at small-corpus scale".

---

## Time map

| Block | Slides | Target | Cumulative |
|---|---|---|---|
| Open | S1 | 0:25 | 0:25 |
| Motivation | S2 | 0:50 | 1:15 |
| Context | S3 | 0:50 | 2:05 |
| Audience | S4 | 0:15 | 2:20 |
| Hypothesis | S5 | 0:50 | 3:10 |
| Visual bridge | S6 | 0:20 | 3:30 |
| Use cases | S7 | 0:15 | 3:45 |
| RQs | S8 | 0:30 | 4:15 |
| Visual bridge | S9 | 0:20 | 4:35 |
| Innovation | S10 | 0:50 | 5:25 |
| **Architecture & demo block** | **S11–S30** | **3:30** | **8:55** |
| Limits | S31 | 0:35 | 9:30 |
| Conclusion | S32 | 0:30 | 10:00 |

---

# THE SCRIPT

## S1 · Title (0:00 – 0:25)

> *(Stand. Title slide visible.)*

**Hi everyone. I'm Rishit, this is Sushmita. Project: Multi-Agent Legal Reasoning with GraphRAG. In one sentence — we built a four-agent debate system over a structured legal knowledge graph, ran a clean four-condition ablation on CaseHOLD, and have one statistically significant result and one honest negative finding to report. Ten minutes: motivation, design, demo, results, what worked, what didn't.**

---

## S2 · Motivation (0:25 – 1:15)

> *(Anchor. Three blocks visible plus "The Gap".)*

**Three things make legal-holding identification hard for LLMs. Stakes are high — a wrong holding misinforms a brief or skews a ruling. LLMs hallucinate citations without retrieval — they invent plausible case names. And single-pass reasoning is brittle — chain-of-thought rarely challenges its first answer, but legal reasoning *requires* stress-testing every candidate.**

> *(Point at "The Gap" callout.)*

**The gap: no prior pipeline simultaneously combines structured-graph retrieval, adversarial multi-agent debate, and an evidence-anchored Judge — with proper ablation isolating each contribution. That's the hole we filled.**

---

## S3 · Context: Current Limitation (1:15 – 2:05)

> *(Anchor. Three columns: CaseHOLD, What is a Holding, Prior Work.)*

**Benchmark is CaseHOLD: 53,000 examples from the Harvard Caselaw Access Project, five-way multiple choice. Each example is a citing prompt plus five candidate holdings — one real, four real-but-from-other-cases. We evaluate on a label-balanced 25-query subset on the leak-free v2 corpus.**

> *(NOTE if your slide still says "100 queries" — say "25" out loud and trust the eval.)*

**What's a holding? The specific legal rule a court applied — not the dicta, not procedural history, not policy rationale. Picking from near-identical candidates demands deep substantive reasoning.**

**Prior work in three categories. LegalBERT and fine-tuned LLMs — strong on CaseHOLD but opaque, no audit trail. RAG pipelines — reduce hallucination but miss graph-level evidence. Multi-agent debate from Du et al. — domain-agnostic, no legal scaffolding. We sit at the intersection.**

---

## S4 · Target Audience (2:05 – 2:20)

> *(Fast skim. Don't dwell.)*

**Six audiences in one breath: attorneys want accuracy plus an audit trail; judges want explainability; researchers want ablation rigour; legal educators want interpretable transcripts; LegalTech startups want a reusable backbone; ML engineers want a production template. The same artefact — an evidence-anchored debate transcript — serves all six.**

---

## S5 · Hypothesis (2:20 – 3:10)

> *(Anchor. Read the central hypothesis, then hit each H briefly.)*

**Central hypothesis: a multi-agent debate over a graph-structured retrieval system significantly outperforms both a single-LLM and a standard RAG baseline on CaseHOLD, while keeping per-query cost under ten cents.**

**Four sub-hypotheses, each tied to one ablation. H1: retrieval is necessary. H2: debate is the biggest single driver. H3: graph adds targeted recall on multi-hop inference. H4: asymmetric Haiku-plus-Sonnet is Pareto-optimal.**

**One thing I'll flag now rather than later — H3 is the one our data did not support at the corpus size we tested. I'll come back to it on the conclusion slide. The system reports honestly which hypotheses survived contact with data.**

---

## S6 · [Visual] (3:10 – 3:30)

> *(Visual slide — likely a hypothesis-mapping diagram or transition. Adapt the line below to what's actually on the slide.)*

**[If this is an architecture diagram / hypothesis-to-pipeline map:] Quick visual of how each hypothesis maps to a specific ablation cell — H1 to C2 vs C1, H2 to C3 vs C2, and so on. Every hypothesis has a clean comparison.**

**[If this is a transition slide:] With the hypotheses on the table, let me walk through what we actually built.**

---

## S7 · Use Cases (3:30 – 3:45)

> *(Fast.)*

**Six concrete use cases — precedent research, motion brief support, due-diligence review, academic benchmarking, judicial drafting, legal-AI curriculum. Common thread: every prediction comes with a debate transcript a human can read.**

---

## S8 · Research Questions (3:45 – 4:15)

> *(Anchor. Four RQs.)*

**Four research questions, each answered by one pairwise comparison. RQ1: how much does retrieval help — C2 vs C1. RQ2: how much does debate help, retrieval held constant — C3 vs C2. RQ3: does graph structure add over flat retrieval — C4 vs C3. RQ4: total system contribution and is the 25× cost premium worth it — C4 vs C1. Each pair holds one variable constant so the comparison is clean.**

---

## S9 · [Visual] (4:15 – 4:35)

> *(Visual — probably the 2×2 ablation matrix or a flow diagram. Adapt to content.)*

**[If 2×2 matrix:] Two-by-two ablation: rows are retrieval strategies, columns are reasoning styles. C1 single LLM, no retrieval. C2 vector RAG. C3 multi-agent + vector. C4 multi-agent + GraphRAG, our full system. Each cell isolates one variable.**

**[If pipeline flow:] One-line summary: every pipeline routes through a single LLM client and a single answer parser, so the comparisons are byte-paired.**

---

## S10 · Technical Innovation (4:35 – 5:25)

> *(Anchor. Three blocks plus the bonus.)*

**Three original contributions. One: the four-condition ablation matrix itself — first legal-AI work I've seen run all four pairwise comparisons with paired statistical tests. Two: the legal knowledge graph — semantic edges plus a typed entity overlay with case, issue, fact, and ruling nodes connected by typed edges including cites, supports, and contradicts. Three: the asymmetric model split — Haiku workers, Sonnet Judge — a Pareto improvement over single-model pipelines that we discovered through a truncation regression.**

> *(Point at the bonus line.)*

**Bonus: a seven-stage robust answer parser. When our Judge's output got truncated mid-project, the parser's body-vote fallback rescued accuracy from 0.06 to 0.25 — with zero new API calls. That tool ships in the repo as `tools/rescore.py`.**

---

# S11 – S30 · ARCHITECTURE & DEMO BLOCK (5:25 – 8:55, ~3:30)

> *(20 visual slides. Without seeing them I can't write per-slide narration; instead, here's a four-segment bridge that hits every "why" in the right order. Pace at ~10 s per slide on average. Slow on the demo, fast on the architecture diagrams.)*

### Segment A — Architecture (~75 s, ≈ 5–8 slides)

**The system has four agents: Retriever, Plaintiff, Defense, Judge. Sequential pipeline — Retriever first, two debate rounds of Plaintiff-then-Defense, Judge synthesises. Seven Anthropic API calls per query, around seventy seconds, five cents.**

**Why four agents and not three or five?** **Four is the minimum that supports a real two-sided debate with a neutral arbiter and an explicit retrieval step. Removing the Defense gives you chain-of-thought with extra steps; adding a Critic introduces revision dynamics that complicate termination.**

**Why two debate rounds?** **We tested one, two, and three. Round three changes the ruling on fewer than ten percent of cases at fifty percent more cost. Two is the knee.**

**Why GraphRAG over flat retrieval?** **Vector RAG retrieves chunks independently — it can't follow a citation, can't prefer evidence that grounds the same legal entity across documents. Our graph has a chunk layer with semantic-similarity edges and an entity overlay with typed legal nodes. The retriever does seed-and-expand: top-3 nearest seeds, then BFS up to 2 hops along any edge.**

### Segment B — Models, prompts, parser (~45 s, ≈ 3–4 slides)

**Why the asymmetric model split?** **Generators and discriminators are different jobs. Plaintiff and Defense write a lot of structured prose — Haiku handles that. The Judge synthesises a 12,000-token transcript and picks one of five near-identical candidates — long-context discrimination, exactly where Sonnet outperforms. Judge runs once per query, workers run five times. Asymmetric gives us Sonnet quality where it matters without paying Sonnet rates everywhere.**

**Why structured-output contracts on every agent?** **Plaintiff outputs five required fields — POSITION, ARGUMENT, EVIDENCE_CITED, STRONGEST_POINT, ANSWER. The Judge leads with FINAL ANSWER on line one. The Judge-first format isn't cosmetic — earlier in the project a 1,024-token output cap truncated 52 of 66 saved transcripts before they reached the answer line. We moved FINAL ANSWER to the front, bumped the cap to 2,048, and added a body-vote parser fallback. Three layers of recovery.**

**Why temperature zero?** **Reproducibility. Same query, same answer across reruns. Paired bootstrap and McNemar tests are valid. We give up the small bump from self-consistency voting in exchange for a defensible result.**

### Segment C — Live demo (~75 s, ≈ 4–5 slides if screenshots, or 1 if it's the live tab)

> *(Switch to Streamlit if doing live, or narrate over the screenshot slides.)*

**This is the system live. Pipeline C4 — Multi-Agent + GraphRAG. Subset 200. I'm picking a pre-tested CaseHOLD example, true holding is holding-3.**

> *(Click Run debate — or move to the next screenshot.)*

**Retriever pulls evidence and explicitly flags retrieval *gaps*, so the agents argue from absence rather than fabricate. Plaintiff Round 1 commits to a position with five required fields. Defense Round 1 — point at one specific objection here — raises that the Plaintiff conflated two doctrinal bases. That's the multi-agent value proposition: a single-pass reasoner doesn't talk itself out of an early commitment; debate forces articulation of the strongest case for some other holding before deciding.**

**Round 2 — Plaintiff addresses Defense's objection directly. The orchestrator passes the full prior conversation to every agent — actual back-and-forth, not parallel monologues.**

**Judge ruling: FINAL ANSWER holding-3. Ground truth holding-3. Green check. Hit@K yes. Recall 1.0 — both gold chunks retrieved. MRR 1.0 — gold chunk ranked first. Precision 0.4 is the theoretical max on this corpus shape. Five cents. Sixty-eight seconds. Seven LLM calls.**

### Segment D — Results & findings (~45 s, ≈ 2–3 slides)

> *(Switch to results table slide.)*

**Across 25 queries on the leak-free v2 corpus, total run cost two dollars: C1 single LLM 0.64. C2 vector RAG 0.68. C3 multi-agent vector 0.84. C4 multi-agent GraphRAG 0.76.**

**Headline win: multi-agent debate beats vector RAG by 16 accuracy points. Paired bootstrap p equals 0.025; 95% CI is plus 0.04 to plus 0.32, excluding zero. Statistically significant.**

**Honest negative finding: GraphRAG plus debate landed at 0.76 — plus 0.08 over baseline, p of 0.44, not significant. Reason: vector RAG already hits the gold doc on every query at this corpus size — Hit@K is 1.0. Graph expansion can only dilute precision when recall is saturated. We treat this as corpus-size-dependent: at 1,000 documents we expect it to flip; at 25, it doesn't.**

**Two bugs we caught and fixed mid-project. Truncation regression — Judge cap at 1,024 tokens — fixed three ways, recovered 0.06 to 0.25 from saved transcripts with no new API spend. Corpus leakage — our loader was bundling candidate holdings into the same field the chunker indexed — caught by reading agents' transcripts, fixed by splitting the LLM-query field from the retrieval-only corpus field, verified by `tools/audit_retrieval.py` showing zero candidates leak. The fix dropped pre-fix numbers about four points. Honest reporting, not regression.**

---

## S31 · Limitations (8:55 – 9:30)

> *(Anchor. Six tiles visible — call out three, skim the rest.)*

**Six limits, each with a mitigation. Three I want to surface explicitly. One: evaluation N — 25 queries gives wide CIs; bootstrap is significant, McNemar is underpowered at that N — that's expected, bootstrap is the more informative statistic at small N. Two: regex entity extraction misses implicit references like "the court" and "the petitioner" — replacing with a one-shot LLM extractor is the highest-leverage change for retrieval recall. Three: no calibrated confidence — the Judge's High/Medium/Low isn't Brier-validated, so it can't drive triage in production.**

**The other three — API cost at scale, single domain, no self-consistency voting — are flagged with mitigations on the slide.**

---

## S32 · Conclusion & Future Work (9:30 – 10:00)

> *(Anchor. Final slide. Memorise this.)*

**What we proved. Multi-agent debate is the single largest driver of accuracy on CaseHOLD: plus 16 points, p equals 0.025. Structured output contracts plus an evidence-anchored Judge eliminate rhetoric-driven errors. The asymmetric Haiku-Sonnet split achieves Sonnet accuracy at Haiku-class cost. GraphRAG did not gain over flat retrieval at our corpus size — H3 not supported, framed honestly as a saturation effect that should flip at larger scale. And a failed run we diagnosed and rescued from saved transcripts is engineering rigour, not a weakness.**

**Future work in three tiers — near-term: LLM entity extractor and a 500-query confirmation run. Mid-term: cross-dataset on ECtHR and LEDGAR, self-consistency voting, query-level retrieval cache. Long-term: ReAct comparison, full-corpus deployment with practising-attorney evaluation.**

**Bottom line: every prediction is evidence-anchored, auditable, and statistically validated. Happy to take questions.**

> *(Stop. Don't fill silence. Wait.)*

---

# Backup contingency plans

## If the live demo's API call hangs > 30 s
**"While that runs — yes, this is slow; seven sequential LLM calls plus a Sonnet Judge over a 12,000-token transcript. The latency is the trade we made for interpretability and a stronger discriminator. Three optimisations would cut it without changing the architecture: parallelise the Retriever's sub-queries, cache retrieval per query, and add a fast-path that runs only Round 1 with an early-exit on agreement."**
At 60 s, switch to the screencast.

## If the API errors out
**"Looks like an API error — let me play the 90-second screencast instead."** *(QuickTime.)*

## If a panellist interrupts during a slide
Pause politely, answer, return to the slide. Do *not* skip slides to make up time.

## If the live prediction is wrong
**"That's actually informative. The Judge sided with the Defense's counter-position — our error-analysis tool calls this `defense_flipped`. Across the eval run it happens about 8% of the time, and it's the single category that future work — Brier-calibrated confidence and self-consistency voting — would target. Showing you a failure case live is more honest than cherry-picking."**

---

# Statistics to memorize cold

| Metric | Value |
|---|---|
| C3 multi_agent_vector accuracy | **0.840** |
| C2 vector_rag baseline | **0.680** |
| C3 vs C2 Δacc | **+0.160** |
| C3 vs C2 95% CI | **[+0.04, +0.32]** |
| C3 vs C2 bootstrap p | **0.025** |
| C4 multi_agent (graph) accuracy | **0.760** |
| C4 vs C2 bootstrap p | **0.440** (not sig.) |
| Headline run cost | **$2.01** |
| Per-query cost (C4) | **~$0.05** |
| Per-query latency (C4) | **~70 s at subset 200** |
| Eval N (post-fix v2 corpus) | **25 queries** |
| Leakage audit | **0 / 48 candidates leak** |
| Truncation rescue | **0.06 → 0.25**, no API spend |

---

# Six panel questions you'll get (30 s answers)

**Q: Is N=25 enough?**
Bootstrap CI excludes zero, p=0.025. McNemar (p=0.125) is underpowered at this N — bootstrap is the more informative statistic. A 100-query confirmation run is the obvious next step at ~$8.

**Q: Why did GraphRAG underperform?**
Hit@K saturation. Flat retrieval finds the gold doc on every query at 25 docs. Two-hop expansion can only dilute precision when recall is at the ceiling. Expected to flip at 1,000+ docs.

**Q: Did your eval set leak the answer?**
Pre-fix yes — caught it post-demo by reading agents' transcripts. Fixed (split `text` from `corpus_text`), verified by `tools/audit_retrieval.py`. Pre-fix accuracy dropped about 4 points after the fix; that's honest reporting.

**Q: Why Anthropic specifically?**
Practical fit for course budget; Claude's instruction-following on structured-output contracts. One line in `LLMClient` swaps to OpenAI or Gemini.

**Q: Smallest change that would most increase accuracy?**
LLM-based entity extractor for the typed-graph layer. Catches implicit references the regex misses; addresses our residual `retrieval_miss` failures.

**Q: Is this novel?**
Components individually published. Novel is the *combination* applied to legal MCQ with proper ablation, the asymmetric model split, and the honest negative finding for graph structure. Engineering integration with measurable ablation, not new fundamental method.

---

# Final rehearsal advice

- Read this script aloud, end to end, three times. The third with a stopwatch.
- For the visual block (S11–S30), once you know what each slide actually shows, write a one-line narration on a sticky note for each. Average 10 s per slide.
- Memorise S1 (open) and S32 (close) word-for-word. The middle can be loose.
- If you go over 10 minutes, cut from the architecture sub-segment of the visual block (Segment A). Don't cut from the demo or the results.
- Smile when you say "two methodological bugs we caught, fixed, and verified." That framing is the strongest moment in the whole talk.
