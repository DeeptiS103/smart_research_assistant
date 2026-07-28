# Smart Research Assistant (Agent + NLP Project)

A research assistant that searches the web, summarizes multiple sources,
and synthesizes a single cited answer to a question.

This project is being built in stages to demonstrate core AI Engineering
concepts: NLP (summarization, semantic ranking), agentic behavior
(planning, tool use, multi-step reasoning), and evaluation.

## Status: Step 4 complete — Evaluation Harness Added

Step 4 adds an evaluation harness that runs the agent against a fixed
test set (`test_questions.json`) and scores it on:

- **Citation validity**: does every `[Source N]` in the answer refer to
  a source that actually exists?
- **Faithfulness (hallucination check)**: an LLM-as-judge compares the
  final answer against the gathered source summaries and flags any
  claims not actually supported by them
- **Efficiency**: search rounds, LLM calls, and wall-clock time per question
- **Topic coverage**: whether the answer touched the topics expected
  for that question

Run `python step4_evaluation.py` to produce a console report and an
`eval_results.json` file with full per-question detail — useful evidence
for a portfolio README or interview discussion.

## Roadmap

- [x] **Step 1**: Manual pipeline (search → summarize → synthesize)
- [x] **Step 2**: Agent behavior — planning sub-queries + deciding when to stop
- [x] **Step 3**: Semantic ranking + filtering + deduplication using embeddings
- [x] **Step 4**: Evaluation harness — citation validity, faithfulness, efficiency, coverage
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
