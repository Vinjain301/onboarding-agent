# Architecture and Design Decisions

This document explains the reasoning behind the major technical decisions in this
project, not just what was built. It is intended for a reader evaluating engineering
judgment, not only the end result.

## Problem framing

An onboarding assistant that answers HR questions carries real downside risk if it
gets things wrong: incorrect statements about PTO accrual, benefits eligibility, or
compliance training deadlines can create liability and erode trust in the tool. The
design priority throughout this project was therefore not "generate a plausible
answer," but "generate an answer that is traceable to a source document, and refuse
to answer when it cannot be traced." Every architectural choice below follows from
that priority.

## System overview

```
                    +----------------------+
                    |  Mock Workday API    |
                    |  (employee profile,  |
                    |   role, milestones)  |
                    +----------+-----------+
                               |
                               v
User question ---> +----------------------+       +----------------------+
                    |   RAG retrieval      |------>|  Confidence scoring  |
                    |  (FAISS + local      |       |  (cosine similarity) |
                    |   embeddings)        |       +----------+-----------+
                    +----------------------+                  |
                                                    low        |      high
                                                confidence     |  confidence
                                                     v          v
                                          +------------------+   +------------------+
                                          |  HR escalation   |   |  Chat model       |
                                          |  ticket logged   |   |  (OpenRouter,     |
                                          |  (no LLM call)   |   |   grounded answer |
                                          +------------------+   |   with citations) |
                                                                  +------------------+
```

## Key decisions

### 1. Retrieval confidence gates generation, rather than the model self-reporting uncertainty

A large language model asked "are you sure?" will often say yes even when it is
wrong; asking the model to grade its own confidence does not reliably reduce
hallucination. Instead, confidence here is computed independently of the model, from
the retrieval step: how semantically close is the best-matching document to the
question. If the closest match is not close enough (see `CONFIDENCE_THRESHOLD` in
`app/config.py`), the system never calls the chat model at all and instead logs an
HR escalation ticket (`app/escalation.py`). This makes "I don't know" a deterministic,
testable code path rather than something the model has to volunteer correctly.

### 2. Confidence score: direct cosine similarity, not the library default

`app/rag_pipeline.py` computes confidence as cosine similarity derived directly from
FAISS's L2 distance output, rather than using LangChain's built-in
`similarity_search_with_relevance_scores`. This was not the original implementation.

During integration testing against the local embedding model, a clearly answerable
question ("How much PTO do I accrue?") scored 0.43 under LangChain's default
relevance-score formula, just under the 0.45 confidence threshold, which would have
caused a legitimate, answerable question to be escalated to HR incorrectly.
Investigation showed the default formula (`1 - distance / sqrt(2)`, a linear
rescaling) is a reasonable general-purpose approximation but does not track true
cosine similarity for this embedding model's output distribution. Computing
`1 - (distance ** 2) / 2` directly is the mathematically correct conversion from L2
distance to cosine similarity for unit-normalized vectors, and it produces well
separated scores in practice (0.59-0.67 for on-topic questions, 0.0 for an
unrelated question in manual testing). Embeddings are explicitly normalized in
`get_embeddings()` so that this formula's assumption holds.

The broader point: a confidence threshold is only as trustworthy as the scoring
function behind it, and that function was benchmarked against real queries before
being trusted as a safety gate, not assumed correct from the library default.

#### Confidence threshold calibration

Fixing the scoring formula (above) was necessary but not sufficient: the threshold
value itself also needed to be evaluated against real questions rather than left at
an arbitrary default. Manual evaluation against the live retrieval pipeline produced
the following scores:

| Question | Expected outcome | Confidence score |
|---|---|---|
| "Can I bring my dog to the office?" | Not covered, should escalate | 0.00 |
| "What training do I need in my first two weeks?" (loosely phrased) | Covered, should answer | 0.27 |
| "What is my stock option vesting schedule?" | Not covered, should escalate | 0.32 |
| "What training do I need in my first 14 days?" (matches document wording) | Covered, should answer | 0.38 |
| "What systems will I get access to as a Data Analyst?" | Covered, should answer | 0.59 |
| "How much PTO do I accrue, and when can I use it?" | Covered, should answer | 0.67 |

This data shows an overlapping zone, roughly 0.27 to 0.38, where a genuinely
answerable question phrased loosely (0.27) scores lower than a genuinely
unanswerable question that is merely topically adjacent to a real document (0.32).
With this embedding model, no single threshold value classifies every case in this
zone correctly: a threshold set low enough to accept the loosely phrased answerable
question would also accept the unanswerable one.

Given the problem framing above, an incorrect escalation (a valid question is
unnecessarily routed to HR) is a usability cost, while an incorrect answer (an
unanswerable question is confidently answered anyway) is a correctness and trust
cost. Those two error types are not equally expensive for an HR assistant, so the
threshold (`CONFIDENCE_THRESHOLD`, default `0.35`) is set to resolve the ambiguous
zone in favor of escalation rather than in favor of coverage. This is a deliberate,
data-informed tradeoff, not a tuning oversight, and it is revisited below as a
known limitation rather than treated as fully solved.

### 3. Mock Workday API as a boundary, not a shortcut

`mock_workday/api.py` is written as if it were a thin client over a real Workday
integration: it returns plain dictionaries with a fixed shape
(`MockWorkdayClient.get_onboarding_context`), and every downstream component
(`app/agent.py`, `app/plan_generator.py`, `streamlit_app.py`) depends only on that
shape, never on the fact that the data is hardcoded. This means replacing the mock
with a real Workday RaaS report or REST integration is a change confined to
`mock_workday/`, with no changes required in the RAG, agent, or UI layers. This
mirrors how the integration would realistically be staged in a production project:
build and test against a contract first, wire up the real system second.

### 4. Local embeddings and OpenRouter, with explicit fallback

OpenRouter provides an OpenAI-compatible chat completions API but does not host an
embeddings endpoint. Embeddings therefore run locally via a HuggingFace
sentence-transformer (`all-MiniLM-L6-v2`), which also has the practical benefit of
making the retrieval and confidence-scoring path fully testable without any network
call or API key (see `tests/test_rag_pipeline.py`).

Free-tier OpenRouter models share a public capacity pool and are, in practice,
frequently rate-limited or retired without notice; this was observed directly during
development, not anticipated in the abstract. `app/llm.py` addresses this with an
explicit fallback list (`CHAT_MODEL`, then `CHAT_MODEL_FALLBACKS` in order) rather
than assuming a single configured model will remain available. This is a deliberate
tradeoff of the free-tier constraint: a paid, dedicated model endpoint would not need
this, but given the constraint, failing over automatically is preferable to the
application going down whenever one upstream provider is temporarily saturated.

### 5. Structured output via prompted JSON and parsing, not function calling

`app/plan_generator.py` requests the 30/60/90 plan as JSON text, validated against a
Pydantic schema (`PydanticOutputParser`), rather than using tool-calling based
structured output (`with_structured_output`). This was a deliberate downgrade from a
more convenient API: many free OpenRouter models have inconsistent or absent
function-calling support, so a tool-calling approach would work with some models and
silently fail with others. Prompted JSON with explicit format instructions works
uniformly across chat models regardless of function-calling support, at the cost of
needing an explicit parse-and-repair step (`_repair_json` in `app/plan_generator.py`)
for the cases where a model's output is not quite valid JSON.

### 6. Citations are structural, not model-generated

The chat model is never asked to cite its sources. Instead, `app/citations.py`
formats the retrieved chunks with their source file paths before they are ever sent
to the model, and the same chunk metadata is rendered back to the user directly from
retrieval, independent of what the model's response text says. This avoids a known
failure mode where a model fabricates a plausible-looking but incorrect citation.

## Known limitations

- Free-tier OpenRouter model availability changes over time. If both the primary and
  fallback models configured in `.env` become unavailable, chat and plan generation
  will fail; the current free-model catalog can be checked at
  `https://openrouter.ai/models?max_price=0`.
- As documented under "Confidence threshold calibration" above, there is an
  overlapping score range (approximately 0.27 to 0.38) in which loosely phrased
  answerable questions and topically adjacent unanswerable questions cannot be
  reliably distinguished by this embedding model. The current threshold resolves
  that ambiguity in favor of escalation, which means some genuinely answerable
  questions phrased loosely will be escalated rather than answered. A larger or
  higher-resolution embedding model, or a query-rewriting step before retrieval,
  would be the next things to evaluate to narrow this ambiguous range, rather than
  further threshold tuning alone.
- The confidence threshold was calibrated against the sample HR documents included
  in this repository and the local embedding model. Swapping either the source
  documents or the embedding model would warrant re-running the calibration
  exercise above against a representative set of answerable and unanswerable
  questions before trusting the existing default.
- All HR content in `data/` and employee records in `mock_workday/mock_data.py` are
  fictional and created for this project. See the Data Notice in `README.md`.
