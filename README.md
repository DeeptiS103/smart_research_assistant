# Smart Research Assistant (Agent + NLP Project)

A research assistant that searches the web, summarizes multiple sources,
and synthesizes a single cited answer to a question — with memory and a
chat UI.

Built in stages to demonstrate core AI Engineering concepts: NLP
(summarization, semantic ranking), agentic behavior (planning, tool use,
reflection), evaluation, and memory.

## Status: Step 5 complete — Memory + UI Added

- **Short-term (conversation) memory**: follow-up questions like "what
  about the cost side?" are rewritten into standalone questions using
  recent conversation history (query rewriting / condensing)
- **Long-term memory**: every research result is cached to
  `memory_store.json`, indexed by embedding. A similar future question
  reuses the cached answer instantly instead of repeating all the work
- **Streamlit UI**: a chat interface showing the answer, sources with
  relevance scores, and whether an answer came from cache or fresh research

## Roadmap

- [x] **Step 1**: Manual pipeline (search → summarize → synthesize)
- [x] **Step 2**: Agent behavior — planning sub-queries + deciding when to stop
- [x] **Step 3**: Semantic ranking + filtering + deduplication using embeddings
- [x] **Step 4**: Evaluation harness — citation validity, faithfulness, efficiency, coverage
- [x] **Step 5**: Short-term + long-term memory, wrapped in a Streamlit UI

## Project structure

```
research_agent/
├── step1_research_pipeline.py   # v1: fixed pipeline (search → summarize → synthesize)
├── step2_agent_pipeline.py      # v2: + planning + reflection (agentic loop)
├── step3_semantic_ranking.py    # v3: + embedding-based ranking/dedup
├── step4_evaluation.py          # eval harness (citation validity, faithfulness, coverage)
├── agent_core.py                # v5: full agent + short-term & long-term memory (no UI)
├── app.py                       # Streamlit UI — the app to actually run/demo
├── test_questions.json          # fixed test set used by step4_evaluation.py
├── requirements.txt
├── .env.example
└── .gitignore
```

Earlier `stepN_*.py` files are kept in the repo intentionally — they
show the project's progression and are useful to walk an interviewer
through, one concept at a time.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# then edit .env and add your real API keys
```

Get free API keys:
- **Groq** (LLM inference): https://console.groq.com
- **Tavily** (search): https://tavily.com

## Usage

**Run the app (recommended):**
```bash
streamlit run app.py
```

**Or run evaluation:**
```bash
python step4_evaluation.py
```

**Or run any individual stage from the terminal:**
```bash
python step3_semantic_ranking.py
python agent_core.py   # includes a quick memory/follow-up demo
```

## Why this project

Demonstrates practical AI Engineering skills: agent orchestration
(planning, reflection, tool use), core NLP techniques (query-focused
summarization, semantic ranking via embeddings), rigorous evaluation
(citation validity, LLM-as-judge faithfulness checks), and memory design
(short-term conversational context + long-term caching) — wrapped in a
usable interface.

## Tech stack

- **LLM**: Groq (Llama 3.1 8B Instant)
- **Search**: Tavily API
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2, runs locally)
- **UI**: Streamlit
- **Language**: Python

## Honest limitations (worth knowing, and worth saying out loud in interviews)

- The small, fast LLM (Llama 3.1 8B) occasionally makes planning/reflection
  decisions that a larger model wouldn't — a known tradeoff of prioritizing
  free/fast inference
- Faithfulness checking uses LLM-as-judge, which is itself imperfect —
  it's a practical proxy for hallucination detection, not a ground-truth measure
- Long-term memory reuse is based on question similarity only — it doesn't
  account for information going stale over time (e.g. reusing a cached
  answer about "current AI models" months later)
