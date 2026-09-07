# Intelligent New Hire Onboarding Agent

Live demo: not yet deployed. See [Deployment](#deployment) below for how this
project deploys to Render; once deployed, the live URL will be linked here.

An AI-assisted onboarding tool that answers new-hire questions and generates
personalized 30/60/90-day onboarding plans. It combines a retrieval-augmented
generation (RAG) pipeline over HR policies, benefits guides, standard operating
procedures, and role-specific onboarding guides with a mock Workday-style API
that personalizes guidance by role, department, location, and start date.

The design priority is reliability over convenience: every answer is grounded in a
retrieved source document, every answer is accompanied by a citation back to that
document, and questions that cannot be answered with sufficient confidence are
routed to a human (HR) escalation path instead of being guessed. The reasoning
behind these choices, including a case where a naive implementation was found to
be miscalibrated during testing and corrected, is documented in
[ARCHITECTURE.md](ARCHITECTURE.md).

## Key capabilities

- Question answering grounded in HR policies, benefits guides, SOPs, and role
  guides, using FAISS for retrieval and a local embedding model.
- A mock Workday-style API (`mock_workday/`) that returns employee profile, role
  metadata, and derived onboarding milestones, built as a swappable boundary so a
  real Workday integration can be substituted without changing the retrieval,
  generation, or UI layers.
- Personalized 30/60/90-day onboarding plans, checklists, required training, and
  recommended learning paths, generated as schema-validated JSON grounded only in
  retrieved context.
- Source citations on every answer, generated from retrieval metadata rather than
  from the language model, so citations cannot be fabricated.
- A confidence threshold on retrieval quality that gates generation: when the best
  available match is not a close enough match to the question, the system logs an
  HR escalation ticket instead of producing an answer.
- A Streamlit interface with an employee selector, a chat assistant, and a
  plan/checklist/training-path view.

## Technology stack

| Layer | Technology |
|---|---|
| Orchestration | LangChain |
| Vector store | FAISS (local, in-process) |
| Embeddings | HuggingFace sentence-transformers (local, no external API) |
| Chat generation | OpenRouter (OpenAI-compatible API), configurable model |
| Structured output | Pydantic |
| Interface | Streamlit |
| Testing | pytest |
| Linting and formatting | ruff |

## Repository structure

```
data/                      HR policies, benefits guides, SOPs, and role-specific
                            onboarding guides (Markdown with metadata frontmatter)
mock_workday/               Mock Workday-style HCM API: employee directory,
                            role profiles, and onboarding milestone calculation
app/
  rag_pipeline.py          Document loading, chunking, FAISS index management,
                            and confidence-scored retrieval
  llm.py                   Chat model access with automatic fallback across
                            configured OpenRouter models
  citations.py             Formats retrieved chunks into model context and
                            into user-facing citations
  escalation.py            HR escalation ticket logging and messaging
  plan_generator.py        30/60/90 plan, checklist, and learning path
                            generation with schema validation and JSON repair
  agent.py                 Retrieval-grounded question answering with a
                            confidence gate and escalation fallback
scripts/build_index.py     Command-line entry point to rebuild the FAISS index
streamlit_app.py           Streamlit user interface
tests/                     Unit tests for the mock API and document pipeline
ARCHITECTURE.md            Design rationale, tradeoffs, and known limitations
```

## Prerequisites

- Python 3.11 or later
- An OpenRouter API key (free tier is sufficient): https://openrouter.ai/keys

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
copy .env.example .env
```

On macOS or Linux, replace the activation and copy commands with:

```bash
source .venv/bin/activate
cp .env.example .env
```

`requirements-dev.txt` installs the runtime dependencies plus pytest and ruff.
For a runtime-only installation (for example, in a deployment image), use
`pip install -r requirements.txt` instead.

## Configuration

Edit `.env` and set `OPENROUTER_API_KEY` to a key from
https://openrouter.ai/keys. All other values in `.env.example` have working
defaults and are documented inline, including:

- `CHAT_MODEL` and `CHAT_MODEL_FALLBACKS`: free OpenRouter models are shared
  across all users of the platform and are occasionally rate-limited or retired
  without notice. The application tries `CHAT_MODEL` first and falls back to
  each model in `CHAT_MODEL_FALLBACKS` in order. If both become unavailable,
  check current free models at https://openrouter.ai/models?max_price=0 and
  update `.env`.
- `EMBEDDING_MODEL`: runs locally, so it requires no API key and incurs no
  per-request cost. The model weights (approximately 80 MB) are downloaded once
  on first use and cached locally.
- `CONFIDENCE_THRESHOLD`: the minimum retrieval confidence required before the
  system will generate an answer. See ARCHITECTURE.md for how this value was
  derived and validated.

## Running the application

```bash
streamlit run streamlit_app.py
```

The FAISS index is built automatically on first run and cached to
`faiss_index/`. To rebuild it after editing the documents in `data/`, run:

```bash
python scripts/build_index.py
```

## Deployment

This repository includes `render.yaml`, a Render Blueprint that defines the web
service declaratively so it can be deployed without manual dashboard
configuration.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Vinjain301/onboarding-agent)

1. Click the button above, or in the Render dashboard choose **New > Blueprint**
   and select this repository directly.
2. Render will read `render.yaml` and propose the service configuration
   automatically.
3. When prompted, set `OPENROUTER_API_KEY` to your OpenRouter key. This value is
   deliberately excluded from `render.yaml` (`sync: false`) so it is entered
   directly in the dashboard rather than stored in source control.
4. Deploy. The build step installs `requirements.txt`; the start command runs
   Streamlit bound to the port Render assigns.
5. Once the deploy finishes, Render provides a public URL. Add that URL to the
   "Live demo" line at the top of this README.

To configure the service manually instead of using the Blueprint, use these
settings:

| Setting | Value |
|---|---|
| Runtime | Python 3.11.9 (see `.python-version`) |
| Build command | `pip install -r requirements.txt` |
| Start command | `streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false` |
| Environment variables | Same keys as `.env.example` |

### Resource considerations on Render's free tier

`sentence-transformers` depends on PyTorch, which has a meaningful memory
footprint before the application even starts serving requests. Combined with
FAISS, LangChain, and Streamlit, this may approach or exceed the RAM ceiling on
Render's free tier; check Render's current free-tier limits before assuming it
fits. The free tier also spins the service down after a period of inactivity,
so the first request after idle time will be slow while the service restarts
and the FAISS index rebuilds from `data/`. If memory becomes a problem, the
practical options are: upgrade to a paid Render plan with more RAM, or replace
the local sentence-transformer embedding model with a smaller model or a
hosted embeddings API (this would require updating `get_embeddings()` in
`app/rag_pipeline.py` and is not implemented in this repository).

## Running tests

```bash
pytest
```

The test suite covers the mock Workday API and the document loading and
chunking pipeline. It does not call OpenRouter or the local embedding model, so
it runs without any API key or network access, and is safe to run in CI.

## Code quality

```bash
ruff check .
ruff format --check .
```

A GitHub Actions workflow (`.github/workflows/ci.yml`) runs both of the above,
plus the test suite, on every push and pull request to `main`.

## Known limitations

See the "Known limitations" section of [ARCHITECTURE.md](ARCHITECTURE.md) for a
full discussion, including free-tier model availability and the scope of the
validated confidence threshold.

## Data and privacy notice

All HR policy content in `data/` and all employee records in
`mock_workday/mock_data.py` are fictional and were created solely for this
project. No real employee, benefits, or organizational data is used or
represented anywhere in this repository.

## License

Released under the MIT License. See [LICENSE](LICENSE) for the full text.
