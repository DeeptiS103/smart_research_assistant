# Smart Research Assistant (Agent + NLP Project)

A research assistant that searches the web, summarizes multiple sources,
and synthesizes a single cited answer to a question.

This project is being built in stages to demonstrate core AI Engineering
concepts: NLP (summarization, query-focused text processing), agentic
behavior (planning, tool use, multi-step reasoning), and evaluation.

## Status: Step 2 complete — Agent Behavior Added

Step 1 was a fixed pipeline. Step 2 adds real agentic behavior:
- **Planning**: the LLM decomposes the question into 2-3 targeted sub-queries
  instead of searching the raw question once
- **Reflection**: after each round of searching, the LLM decides whether it
  has enough information or should search again (up to a safety limit of
  3 rounds)

This means the LLM's own judgment now controls the program's flow — a
core distinction between "using an LLM" and "building an agent."

## Roadmap

- [x] **Step 1**: Manual pipeline (search → summarize → synthesize)
- [x] **Step 2**: Agent behavior — planning sub-queries + deciding when to stop
- [ ] **Step 3**: Semantic ranking of search results using embeddings
      (`sentence-transformers`) instead of trusting raw search order
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
python step1_research_pipeline.py
```

Edit the `test_question` variable at the bottom of the script to try
your own questions.

## Why this project

Built to demonstrate practical AI Engineering skills: orchestrating
LLM calls, query-focused summarization (an NLP task), and multi-source
synthesis with citations — the same core pattern used in production
RAG and agent systems.

## Tech stack

- **LLM**: Groq (Llama 3.1 8B Instant)
- **Search**: Tavily API
- **Language**: Python
