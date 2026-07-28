# Smart Research Assistant (Agent + NLP Project)

A research assistant that searches the web, summarizes multiple sources,
and synthesizes a single cited answer to a question.

This project is being built in stages to demonstrate core AI Engineering
concepts: NLP (summarization, semantic ranking), agentic behavior
(planning, tool use, multi-step reasoning), and evaluation.

## Status: Step 3 complete — Semantic Ranking Added

Step 1 was a fixed pipeline. Step 2 added agentic behavior (planning +
reflection). Step 3 adds embedding-based relevance ranking on top:

- **Ranking**: every search result is scored for true semantic relevance
  to the question (via cosine similarity of embeddings), not just trusted
  in raw search-engine order
- **Filtering**: results below a relevance threshold are dropped before
  ever reaching the LLM, saving API calls and avoiding off-topic noise
  in the final answer
- **Deduplication**: near-identical sources (common when multiple sites
  cover the same story) are collapsed to one, avoiding redundant summaries

This runs entirely locally via `sentence-transformers` — no API calls,
no cost — and it's the same core technique (embeddings + cosine similarity)
used in every RAG and vector-search system.

## Roadmap

- [x] **Step 1**: Manual pipeline (search → summarize → synthesize)
- [x] **Step 2**: Agent behavior — planning sub-queries + deciding when to stop
- [x] **Step 3**: Semantic ranking + filtering + deduplication using embeddings
- [ ] **Step 4**: Evaluation harness — a test set of questions with
      expected answers, tracking citation accuracy and hallucination rate
- [ ] **Step 5**: Memory + simple UI (Streamlit)

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

```bash
python step3_semantic_ranking.py
```

(Earlier stages are kept in the repo too — `step1_research_pipeline.py`
and `step2_agent_pipeline.py` — to show project progression.)

Edit the `test_question` variable at the bottom of the script to try
your own questions.

## Why this project

Built to demonstrate practical AI Engineering skills: agent orchestration
(planning, reflection, tool use), core NLP techniques (query-focused
summarization, semantic ranking via embeddings), and multi-source
synthesis with citations — the same core pattern used in production
RAG and agent systems.

## Tech stack

- **LLM**: Groq (Llama 3.1 8B Instant)
- **Search**: Tavily API
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2, runs locally)
- **Language**: Python
