
# Smart Research Assistant - Step 3: Semantic Ranking with Embeddings
# ======================================================================
# Step 2 gave you an agent that plans searches and decides when to stop.
# But it blindly trusts search results in whatever order Tavily returns them,
# and summarizes EVERY result - even ones that are only loosely related.

# Step 3 fixes this with EMBEDDINGS - turning text into vectors of numbers
# that capture MEANING, so we can mathematically measure how relevant each
# search result actually is to the question, and:
#   1. RANK results by true relevance (not just search engine order)
#   2. FILTER OUT results below a relevance threshold (don't waste LLM
#      calls summarizing irrelevant junk)
#   3. DEDUPLICATE near-identical sources (common when multiple sites
#      report the same info)

# This is a foundational NLP/RAG concept - the same technique powers
# vector databases, semantic search, and retrieval in RAG systems.

# Setup: pip install -r requirements.txt (now includes sentence-transformers)


import os
import json
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient
from sentence_transformers import SentenceTransformer

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

LLM_MODEL = "llama-3.1-8b-instant"
MAX_SEARCH_ROUNDS = 3

# This downloads a small, fast embedding model the first time you run it
# (~80MB, cached locally after that - no API calls, runs on your own CPU)
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

RELEVANCE_THRESHOLD = 0.3   # results below this similarity score get dropped
DUPLICATE_THRESHOLD = 0.92  # results above this similarity to EACH OTHER = duplicates


# ---------------------------------------------------------------------------
# NEW - EMBEDDINGS: turning text into vectors that capture meaning
# ---------------------------------------------------------------------------
def embed_text(text: str) -> np.ndarray:
    
    # Converts text into a vector (list of numbers) that represents its
    # MEANING. Texts with similar meaning end up with similar vectors,
    # even if they use completely different words.

    # Example: "car" and "automobile" get similar vectors even though they
    # share zero letters - because embeddings capture semantic meaning,
    # not just word overlap. This is what makes them more powerful than
    # simple keyword matching.
    
    return EMBED_MODEL.encode(text, convert_to_numpy=True)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    
    # Measures how similar two vectors are, from -1 (opposite) to 1 (identical).
    # In practice for text embeddings, scores usually fall between 0 and 1.

    # This is THE standard way to compare embeddings - you'll see this exact
    # calculation in every RAG/semantic search system.
    
    return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))


# ---------------------------------------------------------------------------
# NEW - RANK + FILTER: use embeddings to keep only genuinely relevant results
# ---------------------------------------------------------------------------
def rank_and_filter_results(results: list[dict], question: str) -> list[dict]:
    
    # Takes raw search results and:
    # 1. Embeds the question once
    # 2. Embeds each result's content
    # 3. Scores each result by similarity to the question
    # 4. Drops anything below RELEVANCE_THRESHOLD (too off-topic to bother with)
    # 5. Sorts remaining results by relevance, best first

    # This means the LLM only ever sees/summarizes content that's ACTUALLY
    # relevant - saving API calls and improving answer quality.
    
    if not results:
        return []

    question_vec = embed_text(question)

    scored = []
    for r in results:
        # Embed a chunk of the content (embedding the whole page is
        # unnecessary and slower - the first ~500 chars usually capture
        # the gist for ranking purposes)
        content_vec = embed_text(r["content"][:500])
        score = cosine_similarity(question_vec, content_vec)
        scored.append({**r, "relevance_score": score})

    # Filter out low-relevance results
    relevant = [r for r in scored if r["relevance_score"] >= RELEVANCE_THRESHOLD]

    # Sort best-first
    relevant.sort(key=lambda r: r["relevance_score"], reverse=True)

    dropped = len(results) - len(relevant)
    if dropped > 0:
        print(f"Filtered out {dropped} low-relevance result(s)")

    return relevant


# ---------------------------------------------------------------------------
# NEW - DEDUPLICATE: multiple sources often say the same thing
# ---------------------------------------------------------------------------
def deduplicate_results(results: list[dict]) -> list[dict]:
    
    # Compares every result against results we've already kept. If a new
    # result is too similar (near-duplicate content, common when multiple
    # news sites cover the same story), skip it - it wastes an LLM call
    # and adds no new information to the final answer.
    
    if not results:
        return []

    kept = [results[0]]
    kept_vecs = [embed_text(results[0]["content"][:500])]

    for r in results[1:]:
        r_vec = embed_text(r["content"][:500])
        is_duplicate = any(
            cosine_similarity(r_vec, kept_vec) >= DUPLICATE_THRESHOLD
            for kept_vec in kept_vecs
        )
        if not is_duplicate:
            kept.append(r)
            kept_vecs.append(r_vec)

    removed = len(results) - len(kept)
    if removed > 0:
        print(f"Removed {removed} near-duplicate result(s)")

    return kept


# ---------------------------------------------------------------------------
# REUSED FROM STEP 1/2 (unchanged)
# ---------------------------------------------------------------------------
def search_web(query: str, max_results: int = 4) -> list[dict]:
    print(f"Searching: {query}")
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
        print("Warning: answer may have been truncated by max_tokens")
    return response.choices[0].message.content.strip()


def plan_searches(question: str, already_known: str = "") -> list[str]:
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
    raw = response.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
    try:
        queries = json.loads(raw)
        if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
            return queries
    except json.JSONDecodeError:
        pass
    print("    ⚠️  Could not parse planned queries, falling back to original question")
    return [question]


def enough_info(question: str, summaries_so_far: list[dict]) -> bool:
    combined = "\n".join(f"- {s['summary']}" for s in summaries_so_far)
    prompt = f"""Question: {question}

    Information gathered so far: {combined}

    Is this enough information to write a complete, accurate answer to the question? Respond with ONLY one word: YES or NO."""
    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=5,
    )
    return "YES" in response.choices[0].message.content.strip().upper()


# ---------------------------------------------------------------------------
# MAIN AGENT LOOP - now with semantic ranking + dedup inserted
# ---------------------------------------------------------------------------
def research_agent(question: str):
    print(f"\nQuestion: {question}\n")

    all_summaries = []
    round_num = 0

    while round_num < MAX_SEARCH_ROUNDS:
        round_num += 1
        print(f"Round {round_num}")

        already_known = "\n".join(s["summary"] for s in all_summaries)
        queries = plan_searches(question, already_known)
        print(f"Planned queries: {queries}")

        # Gather raw results from all planned queries for this round
        raw_results = []
        for query in queries:
            raw_results.extend(search_web(query))

        # NEW: rank by true relevance and drop off-topic results
        ranked_results = rank_and_filter_results(raw_results, question)

        # NEW: remove near-duplicate content before wasting LLM calls on them
        unique_results = deduplicate_results(ranked_results)

        print(f"{len(raw_results)} raw → {len(unique_results)} after ranking/dedup")

        # Only NOW do we spend LLM calls summarizing - on filtered, ranked,
        # deduplicated results. This is more efficient AND more accurate.
        for result in unique_results:
            summary = summarize_source(result["content"], question)
            all_summaries.append(
                {
                    "title": result["title"],
                    "url": result["url"],
                    "summary": summary,
                    "relevance_score": round(result["relevance_score"], 3),
                }
            )

        if enough_info(question, all_summaries):
            print(f"Agent decided: enough information gathered after round {round_num}\n")
            break
        else:
            print(f"Agent decided: needs more information, continuing...\n")

    if round_num == MAX_SEARCH_ROUNDS:
        print(f"Hit max rounds ({MAX_SEARCH_ROUNDS}) - stopping to avoid infinite loop\n")

    print("Synthesizing final answer...\n")
    final_answer = synthesize_answer(question, all_summaries)

    print("=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print(final_answer)
    print("\n" + "=" * 70)
    print(f"SOURCES ({len(all_summaries)} total, across {round_num} round(s))")
    print("=" * 70)
    # Show sources sorted by relevance so you can see the ranking worked
    for i, s in enumerate(sorted(all_summaries, key=lambda x: x["relevance_score"], reverse=True)):
        print(f"[Source {i+1}] (relevance: {s['relevance_score']}) {s['title']}\n{s['url']}\n")

    return final_answer, all_summaries, round_num


if __name__ == "__main__":
    # test_question = "What are the main differences between RAG and fine-tuning for LLMs?"
    test_question = input("Please Enter a Question : ")
    research_agent(test_question)
