
# Smart Research Assistant - Step 5: Memory
# =============================================
# Two kinds of memory, matching how memory works in most real agent products:

#   1. SHORT-TERM (conversation) memory
#      Keeps track of the current session's question/answer history so a
#      follow-up like "what about the cost side?" can be understood in
#      context of what was just discussed - without this, every question
#      is treated as if it's the first one ever asked.

#   2. LONG-TERM memory (persistent cache across runs)
#      Saves every research result to disk (memory_store.json), indexed by
#      embedding. Before doing fresh research, we check: "have I answered
#      something very similar to this before?" If yes, reuse it instantly
#      instead of burning API calls repeating the same searches.

# This file contains the AGENT + MEMORY logic only (no UI) so it can be
# imported cleanly by both a CLI and the Streamlit app.


import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient
from sentence_transformers import SentenceTransformer
import numpy as np

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# LLM_MODEL = "llama-3.1-8b-instant"
LLM_MODEL = "openai/gpt-oss-20b"
MAX_SEARCH_ROUNDS = 3
RELEVANCE_THRESHOLD = 0.3
DUPLICATE_THRESHOLD = 0.92

MEMORY_FILE = "memory_store.json"
MEMORY_REUSE_THRESHOLD = 0.85  # how similar a past question must be to reuse its answer


# ---------------------------------------------------------------------------
# LONG-TERM MEMORY: persistent cache of past research, indexed by embedding
# ---------------------------------------------------------------------------
def load_memory() -> list[dict]:
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE) as f:
            return json.load(f)
    return []


def save_to_memory(question: str, answer: str, sources: list[dict], question_vec: np.ndarray):
    memory = load_memory()
    memory.append({
        "question": question,
        "answer": answer,
        "sources": sources,
        "embedding": question_vec.tolist(),  # numpy arrays aren't JSON-serializable
        "timestamp": datetime.now().isoformat(),
    })
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)


def find_similar_past_question(question: str) -> dict | None:
    
    # Checks long-term memory for a past question similar enough to reuse.
    # This is the same embedding + cosine similarity technique from Step 3,
    # now applied to QUESTIONS instead of search results - same core idea,
    # different use case, which is a good thing to point out in interviews:
    # one technique, multiple applications.
    
    memory = load_memory()
    if not memory:
        return None

    q_vec = EMBED_MODEL.encode(question, convert_to_numpy=True)
    best_match, best_score = None, 0.0

    for entry in memory:
        past_vec = np.array(entry["embedding"])
        score = float(np.dot(q_vec, past_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(past_vec)))
        if score > best_score:
            best_score, best_match = score, entry

    if best_match and best_score >= MEMORY_REUSE_THRESHOLD:
        return {**best_match, "similarity": round(best_score, 3)}
    return None


# ---------------------------------------------------------------------------
# SHORT-TERM (conversation) MEMORY: rewriting follow-up questions with context
# ---------------------------------------------------------------------------
def contextualize_question(question: str, conversation_history: list[dict]) -> str:
    
    # Takes a possibly-ambiguous follow-up ("what about the cost side?") and
    # rewrites it into a standalone question using recent conversation history.
    # This is a well-known real technique (often called "query rewriting" or
    # "condensing question" in RAG chat systems) - without it, follow-up
    # questions silently fail because the search step has no idea what
    # "it" or "that" refers to.
    
    if not conversation_history:
        return question  # nothing to contextualize against yet

    recent = conversation_history[-3:]  # last few turns is usually enough
    history_text = "\n".join(f"Q: {h['question']}\nA: {h['answer'][:200]}..." for h in recent)

    prompt = f"""Given this recent conversation history:

    {history_text}

    The user's new message is: "{question}"

    If this new message is a follow-up that depends on the conversation above (e.g. uses "it", "that", "what about...", or is otherwise ambiguous alone), rewrite it as a complete, standalone question. If it's already a complete standalone question, return it UNCHANGED.

    Respond with ONLY the resulting question, nothing else."""

    response = groq_client.chat.completions.create(
        model=LLM_MODEL, messages=[{"role": "user", "content": prompt}],
        temperature=0, max_tokens=100,
    )
    return response.choices[0].message.content.strip().strip('"')


# ---------------------------------------------------------------------------
# CORE AGENT (same as Step 3/4)
# ---------------------------------------------------------------------------
def embed_text(text: str) -> np.ndarray:
    return EMBED_MODEL.encode(text, convert_to_numpy=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def search_web(query: str, max_results: int = 4) -> list[dict]:
    response = tavily_client.search(query=query, max_results=max_results, search_depth="basic")
    return response.get("results", [])


def rank_and_filter_results(results: list[dict], question: str) -> list[dict]:
    if not results:
        return []
    q_vec = embed_text(question)
    scored = [{**r, "relevance_score": cosine_similarity(q_vec, embed_text(r["content"][:500]))} for r in results]
    relevant = [r for r in scored if r["relevance_score"] >= RELEVANCE_THRESHOLD]
    relevant.sort(key=lambda r: r["relevance_score"], reverse=True)
    return relevant


def deduplicate_results(results: list[dict]) -> list[dict]:
    if not results:
        return []
    kept, kept_vecs = [results[0]], [embed_text(results[0]["content"][:500])]
    for r in results[1:]:
        r_vec = embed_text(r["content"][:500])
        if not any(cosine_similarity(r_vec, kv) >= DUPLICATE_THRESHOLD for kv in kept_vecs):
            kept.append(r)
            kept_vecs.append(r_vec)
    return kept


def summarize_source(content: str, question: str) -> str:
    prompt = f"""Summarize the following text in 2-3 sentences, focusing ONLY on information relevant to this question:

    Question: {question}

    Text to summarize:
    {content[:3000]}

    Give ONLY the summary, no preamble."""
    response = groq_client.chat.completions.create(
        model=LLM_MODEL, messages=[{"role": "user", "content": prompt}],
        temperature=0.3, max_tokens=150,
    )
    return response.choices[0].message.content.strip()


def synthesize_answer(question: str, summarized_sources: list[dict]) -> str:
    sources_text = "\n\n".join(f"[Source {i+1}: {s['title']}]\n{s['summary']}" for i, s in enumerate(summarized_sources))
    prompt = f"""Based on the following sources, write a clear, well-organized answer to this question. Cite sources using [Source N] notation after each claim.

    Question: {question}

    Sources:
    {sources_text}

    Write a comprehensive answer in 5-8 sentences with citations."""
    response = groq_client.chat.completions.create(
        model=LLM_MODEL, messages=[{"role": "user", "content": prompt}],
        temperature=0.3, max_tokens=700,
    )
    return response.choices[0].message.content.strip()


def plan_searches(question: str, already_known: str = "") -> list[str]:
    context_note = (
        f"\n\nYou already have this information:\n{already_known}\n\nPlan searches that fill in what's MISSING - don't repeat what you already know."
        if already_known else ""
    )
    prompt = f"""You are planning research for this question:

    Question: {question}
    {context_note}

    Break this into 2-3 specific, targeted search queries that together would give a complete answer. Respond with ONLY a JSON array of strings, nothing else.

    Example format: ["query one", "query two", "query three"]"""
    response = groq_client.chat.completions.create(
        model=LLM_MODEL, messages=[{"role": "user", "content": prompt}],
        temperature=0.3, max_tokens=200,
    )
    raw = response.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
    try:
        queries = json.loads(raw)
        if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
            return queries
    except json.JSONDecodeError:
        pass
    return [question]


def enough_info(question: str, summaries_so_far: list[dict]) -> bool:
    combined = "\n".join(f"- {s['summary']}" for s in summaries_so_far)
    prompt = f"""Question: {question}

    Information gathered so far:
    {combined}

    Is this enough information to write a complete, accurate answer to the question? Respond with ONLY one word: YES or NO."""
    response = groq_client.chat.completions.create(
        model=LLM_MODEL, messages=[{"role": "user", "content": prompt}],
        temperature=0, max_tokens=5,
    )
    return "YES" in response.choices[0].message.content.strip().upper()


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT - now memory-aware
# ---------------------------------------------------------------------------
def research_agent_with_memory(question: str, conversation_history: list[dict] = None) -> dict:
    
    # The full pipeline with both memory types wired in:
    #   1. Rewrite the question using conversation history (short-term memory)
    #   2. Check long-term memory for a similar past question - reuse if found
    #   3. Otherwise, run the full agent pipeline and save the result for next time
    
    conversation_history = conversation_history or []
    start_time = time.time()

    # SHORT-TERM: resolve follow-up questions like "what about X?" into
    # standalone questions using recent conversation context
    standalone_question = contextualize_question(question, conversation_history)

    # LONG-TERM: check if we've essentially answered this before
    cached = find_similar_past_question(standalone_question)
    if cached:
        return {
            "answer": cached["answer"],
            "sources": cached["sources"],
            "standalone_question": standalone_question,
            "from_cache": True,
            "cache_similarity": cached["similarity"],
            "elapsed_seconds": round(time.time() - start_time, 2),
        }

    # No cache hit - run the full agent pipeline (Step 2-3 logic)
    all_summaries = []
    round_num = 0
    while round_num < MAX_SEARCH_ROUNDS:
        round_num += 1
        already_known = "\n".join(s["summary"] for s in all_summaries)
        queries = plan_searches(standalone_question, already_known)

        raw_results = []
        for q in queries:
            raw_results.extend(search_web(q))

        unique = deduplicate_results(rank_and_filter_results(raw_results, standalone_question))
        for result in unique:
            summary = summarize_source(result["content"], standalone_question)
            all_summaries.append({
                "title": result["title"], "url": result["url"],
                "summary": summary, "relevance_score": round(result["relevance_score"], 3),
            })

        if enough_info(standalone_question, all_summaries):
            break

    final_answer = synthesize_answer(standalone_question, all_summaries)

    # Save this result to long-term memory for future reuse
    q_vec = embed_text(standalone_question)
    save_to_memory(standalone_question, final_answer, all_summaries, q_vec)

    return {
        "answer": final_answer,
        "sources": all_summaries,
        "standalone_question": standalone_question,
        "from_cache": False,
        "rounds_used": round_num,
        "elapsed_seconds": round(time.time() - start_time, 2),
    }


if __name__ == "__main__":
    # Quick CLI test of memory behavior: ask a question, then a follow-up
    history = []

    q1 = "What are the main differences between RAG and fine-tuning for LLMs?"
    result1 = research_agent_with_memory(q1, history)
    print(f"\nQ: {q1}\nA: {result1['answer']}\n(from cache: {result1['from_cache']})")
    history.append({"question": q1, "answer": result1["answer"]})

    q2 = "What about the cost side specifically?"
    result2 = research_agent_with_memory(q2, history)
    print(f"\nQ: {q2}")
    print(f"(rewritten as: {result2['standalone_question']})")
    print(f"A: {result2['answer']}\n(from cache: {result2['from_cache']})")
