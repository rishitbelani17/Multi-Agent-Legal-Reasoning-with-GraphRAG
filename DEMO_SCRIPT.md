# Multi-Agent Legal Reasoning with GraphRAG — Defense Script

**Speaking time:** ~10 min · **Authors:** Rishit Belani & Sushmita Korishetty
**Course:** GenAI / LLMs · Spring 2026 · Columbia University

---

## How to use this document

- **Bold = what you say.** Italic = stage direction (where to click, what to point at).
- Each segment has a target time at the top — don't go over.
- Speaker notes between segments cover the "why" if a panellist interrupts.
- The pre-flight checklist at the top is non-negotiable. Run it 30 minutes before the talk.
- The Q&A bank in the manual (§11) is your safety net — review it the night before, not during.

---

## Pre-flight checklist (T-30 min)

1. Streamlit running locally on `localhost:8501`. Test once with a known-good example.
2. Pre-tested CaseHOLD example loaded in the dropdown. Recommendation: an example you've already verified C4 gets right at `subset=200`. Note its doc ID.
3. Two terminal tabs open: one at `~/ProjGenAI` (for `cat results/...`), one streaming Streamlit logs (so you can see if it fails silently).
4. Backup screencast (90 s) of a successful run open in QuickTime, ready to play if wifi or the API fails.
5. Manual PDF open on a second monitor at §8.1 (the headline table) — you'll cite it.
6. Mute Slack, email, calendar notifications. Browser tab tree pruned.
7. Water on the desk. The Sonnet Judge takes ~70 s per query — you'll need it.

---

# THE SCRIPT

## 0:00 – 1:00 · Opening hook + frame

> *(Stand. Slide 1: title.)*

**Hi everyone. I'm Rishit, and this is Sushmita. Our project is Multi-Agent Legal Reasoning with GraphRAG.**

**Here's the problem in one sentence: when you hand a large language model a citation in a legal opinion and ask it to pick the correct holding from five candidates, today's frontier models get it right about 65 to 70 percent of the time. Random is 20 percent, so it's well above chance — but in legal settings a wrong holding is the kind of mistake that gets you fired, or sued. Our question was: can a multi-agent debate over a structured graph of evidence reliably beat single-LLM retrieval-augmented generation?**

**In the next ten minutes I'll walk you through what we proposed, what we built, what we measured, what worked, what didn't, and two methodological bugs we caught and fixed in the process.**

> *(Click to next slide: agenda — 4 pipelines, 2 findings, live demo. Or skip if no slides.)*

---

## 1:00 – 2:30 · The problem and the proposal

> *(Slide 2: CaseHOLD example. Or pull up a CaseHOLD JSON sample on screen.)*

**The benchmark is CaseHOLD. Each example is a citing prompt — a passage from one opinion that quotes another — and five candidate holdings, where one is the real holding from the cited case and four are real holdings from other cases. The model has to pick the correct one. Fifty-three thousand examples; we evaluate on a label-balanced 25-example subset that costs us about two dollars to run.**

**Why CaseHOLD specifically? Three reasons. First, multiple choice removes evaluation ambiguity — there's one right answer and the parser is deterministic. Second, the distractors are real legal holdings, so wrong answers are domain-realistic, not toy errors. Third, each example points to a known source document, which lets us compute proper retrieval metrics — Hit@K, MRR, precision, recall. ECtHR and LEDGAR don't have that structure.**

**What makes this hard for a single LLM? Three failure modes recur. Substantive versus procedural confusion — the candidates often include both a substantive doctrine and a related procedural rule, and a one-shot LLM picks the wrong one. Citation hallucination — without retrieval the model invents case names. And long-context fragility — when you bundle the citing prompt plus retrieved passages plus the candidate set, you're pushing 12,000 tokens, and smaller models drift.**

**Our proposal was a four-condition ablation that decomposes performance into raw LLM prior, retrieval, multi-agent debate, and graph structure. Four pipelines, four cells in a two-by-two:**

> *(Slide 3: 2x2 ablation table.)*

**Single LLM with no retrieval is the floor. Vector RAG is the standard baseline every paper publishes against. Multi-agent debate over flat retrieval isolates the contribution of debate. Multi-agent debate over GraphRAG is the full proposed system.**

---

## 2:30 – 4:30 · Architecture: every "why" we made

> *(Slide 4: agent flow diagram. Or open the Streamlit "About" panel.)*

**The proposed system has four agents. A Retriever decomposes the query into one to three sub-queries and runs each through a knowledge graph. A Plaintiff argues affirmatively for one holding, citing specific passages. A Defense rebuts, raises specific weaknesses, and commits to a counter-position. They go two rounds. Then a Judge synthesises and rules.**

**Four explicit "why" decisions in this design — let me hit each one.**

**Why four agents and not three or five?** **Four is the minimum that supports a real two-sided debate with a neutral arbiter and an explicit retrieval step. Adding a Critic introduces revision dynamics that complicate termination. Removing the Defense gives you a chain-of-thought reasoner with extra steps.**

**Why two debate rounds, not one or three?** **We tested. Round three changes the ruling on fewer than ten percent of cases at fifty percent more cost. Round two is the knee.**

**Why GraphRAG over flat vector RAG?** **Vector RAG retrieves chunks independently — it can't follow a citation, can't prefer evidence that grounds the same legal entity across documents. The graph has two layers: a chunk graph with semantic-similarity edges thresholded at cosine 0.5, and an entity overlay with typed nodes — case, issue, fact, ruling — connected by typed edges — cites, supports, applies-to, contradicts. The retriever does seed-and-expand: top-3 nearest chunks, then BFS up to 2 hops.**

**Why the asymmetric model split — Haiku workers, Sonnet Judge?** **Generators and discriminators are different jobs. Plaintiff and Defense write a lot of structured prose; Haiku is good at that. The Judge synthesises a 12,000-token transcript and picks one of five near-identical candidates — long-context discrimination, exactly where Sonnet outperforms. The Judge runs once per query; the workers run five times. Asymmetric models give us Sonnet quality where it matters without paying Sonnet rates everywhere.**

**Why temperature zero?** **Reproducibility. Same query, same answer across reruns, so paired bootstrap and McNemar tests are valid. We don't get the small bump from self-consistency voting, but we get a defensible result.**

**Why all-MiniLM-L6 for embeddings instead of a legal-domain encoder?** **Free at the margin, sub-millisecond latency, and Hit@5 was already saturating on our corpus — we profiled BGE-base, it gained under two points of recall for three times the build cost. Not worth it.**

---

## 4:30 – 7:00 · LIVE DEMO (or 90-second backup)

> *(Switch to Streamlit tab. Pipeline = C4 — Multi-Agent + GraphRAG. Subset slider = 200. Pre-tested CaseHOLD example loaded.)*

**This is the system live. Haiku workers, Sonnet Judge, graph built on 200 docs. I'm picking a pre-tested CaseHOLD example — true holding is holding-3.**

> *(Click "Run debate".)*

**You'll see a Retriever pull evidence, a Plaintiff open, a Defense rebut, them go a second round, and a Judge rule. Seven Anthropic API calls per query, about seventy seconds.**

> *(Wait ~10 s, point at the Retriever output.)*

**The Retriever doesn't just return raw passages — it summarises by legal concept and explicitly flags retrieval *gaps*, so the Plaintiff and Defense can argue from absence rather than fabricate.**

> *(Plaintiff Round 1 streams. Point at the structured fields.)*

**Plaintiff outputs five required fields — POSITION, ARGUMENT, EVIDENCE_CITED, STRONGEST_POINT, ANSWER. Structured contract enforced by prompt; Haiku at temperature zero respects it about 95% of the time, and the parser degrades gracefully when it doesn't.**

> *(Defense Round 1 streams. Read one specific objection aloud.)*

**Look — the Defense raised a specific objection that the Plaintiff conflated two doctrinal bases. That's the multi-agent value proposition. A single-pass chain-of-thought reasoner doesn't talk itself out of an early commitment; debate forces the system to articulate the strongest case for *some other* holding before deciding.**

> *(Round 2 streams. Skip detailed narration; let it run.)*

**Round 2 is where Plaintiff addresses the Defense's objection directly. The orchestrator passes the full prior conversation to every agent — actual back-and-forth, not parallel monologues.**

> *(Judge streams. Point at FINAL ANSWER.)*

**The Judge leads with FINAL ANSWER on line one. That's deliberate — earlier the Judge hit a 1,024-token cap and got truncated before reaching the answer. Putting it first guarantees recovery under truncation. We bumped the cap to 2,048 and added a body-vote parser fallback as belt and suspenders.**

> *(Scroll to result panel. Point at each tile.)*

**Prediction holding-3, ground truth holding-3, green check. Hit@K yes. Recall 1.00 — both gold chunks retrieved. MRR 1.00 — gold chunk ranked first. Precision 0.40 is the theoretical max on this corpus shape. Five cents, 68 seconds, seven LLM calls.**

> *(Open "Raw final answer" expander briefly.)*

**Strict format: FINAL ANSWER, CONFIDENCE, CITATIONS, REASONING_SUMMARY. The parser isn't guessing — it's reading a contract.**

---

## 7:00 – 8:45 · Results & two findings

> *(Switch to terminal or to §8.1 in the manual.)*

**That was one query. Across 25 queries on a leak-free corpus, total run cost two dollars:**

> *(Show the headline table.)*

**Single LLM 0.64. Vector RAG 0.68. Multi-agent vector RAG — 0.84. Multi-agent GraphRAG — 0.76.**

**Two findings, one positive and one negative — both worth surfacing.**

**The positive headline: multi-agent debate over flat retrieval beats vector RAG by 16 accuracy points. Paired bootstrap p equals 0.025; 95% CI is plus 0.04 to plus 0.32 — entire interval on the positive side of zero. Statistically significant gain.**

**The harder finding: multi-agent over GraphRAG — the full proposed system — landed at 0.76. Plus 0.08 over baseline, bootstrap p of 0.44. Not significant.**

**Why didn't graph retrieval pay off? Vector RAG hits the gold doc on every query at this corpus size — Hit@K is 1.00, MRR is 1.00. Nothing left for graph expansion to *find*. Two-hop expansion can only do one of two things: surface more gold chunks (impossible at saturation) or surface non-gold chunks (drags precision). We treat this as a corpus-size-dependent result. At 1,000 documents it's expected to flip; at 25, it doesn't.**

**Two bugs we caught and fixed mid-project. First, a truncation regression: the Haiku Judge hit a 1,024-token output cap on 52 of 66 saved transcripts. We fixed it three ways — moved FINAL ANSWER to the first line, bumped the cap, added a body-vote parser fallback. Re-parsing the broken transcripts with no new API calls lifted accuracy from 0.06 to 0.25.**

**Second, a corpus-leakage bug. Caught it after the first demo by reading the agents' transcripts. Our loader was bundling the five candidate holdings into the same field the chunker indexed — so the retriever could find the answer text. We split the LLM query field from the retrieval-only corpus field, bumped the corpus cache version, and shipped `tools/audit_retrieval.py` to verify zero candidates leak into the v2 corpus. The fix dropped our numbers about four points. That's honest reporting, not a regression.**

---

## 8:45 – 10:00 · Honest seams, future work, close

> *(Final slide: "Defensible findings & open questions.")*

**Three seams I want to surface before you ask.**

**One: N equals 25 is small. Bootstrap is significant; McNemar is underpowered — that's expected at small N, bootstrap is the more informative statistic here. A 100-query confirmation run is the obvious next step.**

**Two: our proposed headline — GraphRAG plus multi-agent — didn't beat baseline at p&lt;0.05. We deliver multi-agent debate as the validated contribution and GraphRAG as a corpus-size-dependent extension we expect to pay off at larger scale.**

**Three: the demo runs at 200 documents; the eval ran at 25. Intentional — the demo shows scaling behaviour, the eval reports the small-corpus headline. Both numbers in the manual.**

**With more budget we'd swap the regex entity extractor for an LLM-based one — single highest-leverage change for retrieval recall. We'd add Brier-calibrated confidence and a self-consistency vote at temperature 0.3.**

**To close — what we proposed was a multi-agent legal reasoning system over a graph-structured corpus. What we deliver is a working four-pipeline ablation, a statistically significant +16 point gain from multi-agent debate, a clean negative finding for GraphRAG at small scale that we explain mechanistically, and two methodological bugs we caught, fixed, verified, and documented. End-to-end reproducible from a single `python3 main.py` command at $2 of API spend. Happy to take questions.**

> *(Stop. Don't fill silence. Wait for the first question.)*

---

# Backup contingency plans

## If the live demo's API call hangs > 30 s
**"While that runs, let me pre-empt the obvious question — yes, this is slow; seven sequential LLM calls plus a Sonnet Judge over a 12,000-token transcript. The latency budget is the trade we made for interpretability and a stronger discriminator. Three optimisations would cut this without changing the architecture: parallelise the Retriever's sub-queries, cache retrieval per query, and add a fast-path mode that runs only Round 1 with an early-exit threshold."**

If still hanging at 60 s, switch to the screencast.

## If the API errors out
**"Looks like we hit an API error. I have a 90-second screencast of a successful run — let me play it."** *(Cut to QuickTime.)*

## If a panellist interrupts during the demo
Pause the narration politely. Answer the question. *Don't* abandon the demo — return to it after.

## If the prediction is wrong on the live demo
**"That's actually informative. The system got this one wrong — the Judge sided with the Defense's counter-position. This is the failure mode our error-analysis tool calls 'defense_flipped': Plaintiff was right but Defense's rebuttal was rhetorically stronger. Across the eval run, this happens about 8% of the time, and it's the single category that future work — Brier-calibrated confidence, self-consistency voting — would target. Showing you a failure case live is more honest than cherry-picking."**

That answer is *better* than the cherry-picked happy-path. Don't fear a wrong prediction live.

---

# Statistics to memorize cold

| Metric | Value |
|---|---|
| C3 (multi_agent_vector) accuracy | **0.840** |
| C2 (vector_rag) baseline accuracy | **0.680** |
| C3 vs C2 Δacc | **+0.160** |
| C3 vs C2 95% CI | **[+0.04, +0.32]** |
| C3 vs C2 bootstrap p | **0.025** |
| C4 (multi_agent + GraphRAG) accuracy | **0.760** |
| C4 vs C2 bootstrap p | **0.440** (not significant) |
| Headline run cost | **$2.01** |
| Per-query cost (C4) | **~$0.05** |
| Per-query latency (C4) | **~70 s at subset=200** |
| Eval N | **25 queries × 4 pipelines = 100 calls × 7 LLM calls** |
| Corpus version | **v2** (post-leakage-fix) |
| Leakage audit | **0 / 48 candidates leak** |
| Truncation rescue | **0.06 → 0.25** with no API spend |

---

# Most likely panel questions (have these answers ready in 30 sec)

**Q: Is N=25 enough?**
A: Bootstrap CI excludes zero, p=0.025. McNemar p=0.125 is underpowered at this N — bootstrap is the more informative statistic at small N. A 100-query confirmation run is the obvious next step and would cost ~$8.

**Q: Why did GraphRAG underperform?**
A: Hit@K saturation. Flat retrieval already finds the gold doc on every query at 25 docs, so graph expansion can only dilute precision. Expected to flip at 1,000+ docs.

**Q: Did your evaluation set leak the answer into the corpus?**
A: Pre-fix, yes — caught it after the first demo, in the agents' own transcripts. Fixed, verified by `tools/audit_retrieval.py`, accuracy drops are documented in §9.4. The post-fix numbers are honest.

**Q: Why Anthropic specifically?**
A: Practical fit for the course budget and Claude's instruction-following on structured-output contracts. Nothing in the architecture is Anthropic-specific; one line in `LLMClient` swaps to OpenAI or Gemini.

**Q: What's the one thing you'd change?**
A: LLM-based entity extractor for the typed-graph layer, replacing the regex. Single highest-leverage change for retrieval recall on cases where implicit references matter.

**Q: Show me a failure case.**
A: Open `results/caseholder_20260507_002505_raw_results.json`, point at a `defense_flipped` example, walk through the Judge's reasoning. Or run `python3 -m tools.analyze_errors` and point at the top-K most illustrative failures.

**Q: Is this novel?**
A: Components are individually published — multi-agent debate, GraphRAG, structured-output contracts, evidence-anchored Judges. Novel is the *combination* applied to legal-domain MCQ with proper ablation, the asymmetric model split we converged on after the truncation regression, and the honest negative finding for graph structure at small scale. We frame it as engineering integration with measurable ablation, not new fundamental method.

---

# Final rehearsal advice

- Read this script aloud, end to end, three times. The third time, with a stopwatch.
- Practise the live demo five times. The agent panel does not stream the same way every time; you need to know what to skip narrating if a stage runs long.
- The opening 60 seconds and the closing 60 seconds are the most important. Memorise them word for word. The middle can be loose.
- Smile when you say "we caught and fixed two methodological bugs." The panel respects that more than a clean-looking result.
- If you go over 10 minutes, cut from §2:30 (architecture deep-dive). Don't cut from the demo or the results.
