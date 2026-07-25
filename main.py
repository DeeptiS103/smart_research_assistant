
# Smart Research Assistant - Step 2: Adding Agent Behavior
# ==========================================================
# Step 1 was a FIXED pipeline: 1 search -> summarize -> synthesize. Always.

# Step 2 introduces two agentic capabilities:
#   1. PLANNING    - the LLM decomposes your question into multiple targeted
#                    sub-queries, instead of searching the raw question once.
#   2. REFLECTION  - after each round of searching, the LLM judges whether it
#                    has enough information, or whether it should search again
#                    (up to a max number of rounds, so it can't loop forever).

# This is the key difference from Step 1: the LLM's output now CONTROLS
# the flow of your program (how many times the loop runs). That's what
# makes this agentic instead of a fixed pipeline.

# Setup: same as Step 1 - pip install -r requirements.txt, fill in .env


import os
import json
from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

LLM_MODEL = "llama-3.1-8b-instant"
MAX_SEARCH_ROUNDS = 3  # safety limit - agents can loop forever without this


# ---------------------------------------------------------------------------
# REUSED FROM STEP 1: search + summarize + synthesize (unchanged)
# ---------------------------------------------------------------------------
def search_web(query: str, max_results: int = 3) -> list[dict]:
    print(f"    🔍 Searching: {query}")
    response = tavily_client.search(query=query, max_results=max_results, search_depth="basic")
    return response.get("results", [])


def summarize_source(content: str, question: str) -> str:
    prompt = f"""Summarize the following text in 3-4 sentences, focusing ONLY on information relevant to this question:

    Question: {question}

    Text to summarize: {content[:3000]}

    Give ONLY the summary, no preamble."""
    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=150,
    )
    return response.choices[0].message.content.strip()


def synthesize_answer(question: str, summarized_sources: list[dict]) -> str:
    sources_text = "\n\n".join(
        f"[Source {i+1}: {s['title']}]\n{s['summary']}"
        for i, s in enumerate(summarized_sources)
    )
    prompt = f"""Based on the following sources, write a clear, well-organized answer to this question. Cite sources using [Source N] notation after each claim.

    Question: {question}

    Sources: {sources_text}

    Write a comprehensive answer in 10-15 sentences with citations."""
    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=700,
    )
    if response.choices[0].finish_reason == "length":
        print("\nWarning: answer may have been truncated by max_tokens")

    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# NEW - AGENT CAPABILITY 1: PLANNING
# ---------------------------------------------------------------------------
def plan_searches(question: str, already_known: str = "") -> list[str]:
    
    # Instead of searching the raw question, ask the LLM to break it into
    # 2-3 targeted sub-queries. This is the PLANNING step of the agent.

    # If `already_known` is provided (from a previous round), the LLM plans
    # searches that fill GAPS rather than repeating what it already has -
    # this is what makes round 2+ smarter than just "search again."

    # We ask for JSON output because we need to parse this programmatically -
    # unlike the summarization prompts, this output feeds into CODE, not
    # just into another prompt.
    
    context_note = (
        f"\n\nYou already have this information:\n{already_known}\n\n"
        "Plan searches that fill in what's MISSING - don't repeat what you already know."
        if already_known else ""
    )

    prompt = f"""You are planning research for this question:

    Question: {question}
    {context_note}

    Break this into 2-3 specific, targeted search queries that together would give a complete answer. Respond with ONLY a JSON array of strings, nothing else.

    Example format: ["query one", "query two", "query three"]"""

    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=200,
    )
    raw = response.choices[0].message.content.strip()

    # LLMs sometimes wrap JSON in ```json fences despite instructions -
    # strip those defensively rather than assuming perfect compliance
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        queries = json.loads(raw)
        if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
            return queries
    except json.JSONDecodeError:
        pass

    # Fallback: if parsing fails, just use the original question as one query.
    # Agents need graceful fallbacks - never let a parsing failure crash the pipeline.
    print("\nCould not parse planned queries, falling back to original question")
    return [question]


# ---------------------------------------------------------------------------
# NEW - AGENT CAPABILITY 2: REFLECTION (deciding whether to stop)
# ---------------------------------------------------------------------------
def enough_info(question: str, summaries_so_far: list[dict]) -> bool:
    
    # This is the core agentic decision: after gathering some information,
    # should we STOP (we have enough) or CONTINUE (search again)?

    # This is what separates an agent from a pipeline - the LLM's judgment
    # directly controls whether your while-loop keeps running.
    
    combined = "\n".join(f"- {s['summary']}" for s in summaries_so_far)

    prompt = f"""Question: {question}

    Information gathered so far: {combined}

    Is this enough information to write a complete, accurate answer to the question? Respond with ONLY one word: YES or NO."""

    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,  # 0 = deterministic, we want a reliable YES/NO, not creativity
        max_tokens=5,
    )
    decision = response.choices[0].message.content.strip().upper()
    return "YES" in decision


# ---------------------------------------------------------------------------
# MAIN AGENT LOOP
# ---------------------------------------------------------------------------
def research_agent(question: str):
    print(f"\nQuestion: {question}\n")

    all_summaries = []
    round_num = 0

    while round_num < MAX_SEARCH_ROUNDS:
        round_num += 1
        print(f"Round {round_num}")

        # PLAN: decide what to search for (aware of what we already have)
        already_known = "\n".join(s["summary"] for s in all_summaries)
        queries = plan_searches(question, already_known)
        print(f"Planned queries: {queries}")

        # ACT: run each planned search and summarize results
        for query in queries:
            results = search_web(query)
            for result in results:
                summary = summarize_source(result["content"], question)
                all_summaries.append(
                    {"title": result["title"], "url": result["url"], "summary": summary}
                )

        # REFLECT: does the agent think it has enough now?
        if enough_info(question, all_summaries):
            print(f"Agent decided: enough information gathered after round {round_num}\n")
            break
        else:
            print(f"Agent decided: needs more information, continuing...\n")

    if round_num == MAX_SEARCH_ROUNDS:
        print(f"Hit max rounds ({MAX_SEARCH_ROUNDS}) - stopping to avoid infinite loop\n")

    # SYNTHESIZE final answer from everything gathered across all rounds
    print("Synthesizing final answer...\n")
    final_answer = synthesize_answer(question, all_summaries)

    print("=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print(final_answer)
    print("\n" + "=" * 70)
    print(f"SOURCES ({len(all_summaries)} total, across {round_num} round(s))")
    print("=" * 70)
    for i, s in enumerate(all_summaries):
        print(f"[Source {i+1}] {s['title']}\n{s['url']}\n")

    return final_answer, all_summaries, round_num


# ---------------------------------------------------------------------------
# RUN IT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # test_question = "What are the main differences between RAG and fine-tuning for LLMs?"
    test_question = input("Please Enter a Question : ")
    research_agent(test_question)
