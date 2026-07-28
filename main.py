
# Smart Research Assistant - Step 4: Evaluation Harness
# =========================================================
# Every step so far has been about building the agent. This step is about
# PROVING it works - and being honest about where it doesn't.

# Most people skip this. It's exactly why interviewers ask about it: it
# shows you think about AI systems as things that need to be MEASURED,
# not just built and shipped on faith.

# We evaluate 4 things per question, run over a fixed test set:

#   1. CITATION VALIDITY - does every [Source N] in the answer refer to
#      a source that actually exists? (catches a common LLM failure mode:
#      inventing a citation number that doesn't exist)

#   2. FAITHFULNESS (hallucination check) - does the answer only contain
#      claims that are actually supported by the gathered sources? We use
#      an LLM-as-judge for this (the model checking its own output against
#      the evidence) - a standard technique in production eval pipelines.

#   3. EFFICIENCY - how many search rounds, search calls, and LLM calls did
#      it take? Useful for spotting an agent that's "getting there" but
#      wastefully (e.g. always using max rounds).

#   4. TOPIC COVERAGE - did the answer touch on the topics we expected,
#      based on a lightweight keyword/topic check against test_questions.json

# Setup: same as before. Also needs test_questions.json in the same folder.
# Run: python step4_evaluation.py
# Output: prints a report AND saves eval_results.json for your README/portfolio.


import os
import re
import json
import time
from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient
from sentence_transformers import SentenceTransformer
import numpy as np

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

LLM_MODEL = "llama-3.1-8b-instant"
MAX_SEARCH_ROUNDS = 3
RELEVANCE_THRESHOLD = 0.3
DUPLICATE_THRESHOLD = 0.92


# ---------------------------------------------------------------------------
# AGENT CODE (same as Step 3 - copied here so this file is self-contained
# and easy to hand to someone reviewing just the eval logic)
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
    scored = []
    for r in results:
        c_vec = embed_text(r["content"][:500])
        scored.append({**r, "relevance_score": cosine_similarity(q_vec, c_vec)})
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
    sources_text = "\n\n".join(
        f"[Source {i+1}: {s['title']}]\n{s['summary']}"
        for i, s in enumerate(summarized_sources)
    )
    prompt = f"""Based on the following sources, write a clear, well-organized answer to this question. Cite sources using [Source N] notation after each claim.

    Question: {question}

    Sources: {sources_text}

    Write a comprehensive answer in 5-8 sentences with citations."""
    response = groq_client.chat.completions.create(
        model=LLM_MODEL, messages=[{"role": "user", "content": prompt}],
        temperature=0.3, max_tokens=700,
    )
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

    Information gathered so far: {combined}

    Is this enough information to write a complete, accurate answer to the question? Respond with ONLY one word: YES or NO."""
    response = groq_client.chat.completions.create(
        model=LLM_MODEL, messages=[{"role": "user", "content": prompt}],
        temperature=0, max_tokens=5,
    )
    return "YES" in response.choices[0].message.content.strip().upper()


def research_agent(question: str) -> dict:
    
    # Same agent loop as Step 3, but now returns a dict with everything the
    # evaluator needs: the answer, sources, and metadata about HOW it got there
    # (rounds, call counts, timing) - not just the final text.
    
    all_summaries = []
    round_num = 0
    search_call_count = 0
    llm_call_count = 0
    start_time = time.time()

    while round_num < MAX_SEARCH_ROUNDS:
        round_num += 1
        already_known = "\n".join(s["summary"] for s in all_summaries)
        queries = plan_searches(question, already_known)
        llm_call_count += 1

        raw_results = []
        for query in queries:
            raw_results.extend(search_web(query))
            search_call_count += 1

        ranked = rank_and_filter_results(raw_results, question)
        unique = deduplicate_results(ranked)

        for result in unique:
            summary = summarize_source(result["content"], question)
            llm_call_count += 1
            all_summaries.append({
                "title": result["title"], "url": result["url"],
                "summary": summary, "relevance_score": round(result["relevance_score"], 3),
            })

        decision = enough_info(question, all_summaries)
        llm_call_count += 1
        if decision:
            break

    final_answer = synthesize_answer(question, all_summaries)
    llm_call_count += 1
    elapsed = time.time() - start_time

    return {
        "answer": final_answer,
        "sources": all_summaries,
        "rounds_used": round_num,
        "search_calls": search_call_count,
        "llm_calls": llm_call_count,
        "elapsed_seconds": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# EVALUATION METRICS - the new part for Step 4
# ---------------------------------------------------------------------------
def check_citation_validity(answer: str, num_sources: int) -> dict:
    
    # Metric 1: Extracts every [Source N] reference from the answer and checks
    # whether N actually corresponds to a real source. Catches a real, common
    # LLM failure: citing "[Source 7]" when only 4 sources exist.
    
    cited_numbers = [int(n) for n in re.findall(r"\[Source (\d+)\]", answer)]
    if not cited_numbers:
        return {"valid": False, "reason": "No citations found in answer", "invalid_citations": []}

    invalid = [n for n in cited_numbers if n < 1 or n > num_sources]
    return {
        "valid": len(invalid) == 0,
        "total_citations": len(cited_numbers),
        "invalid_citations": invalid,
    }


def check_faithfulness(answer: str, sources: list[dict]) -> dict:
    
    # Metric 2: LLM-as-judge for hallucination. We show the judge model the
    # answer AND the source summaries, and ask it to flag any claims that
    # AREN'T actually supported. This is imperfect (the judge is itself an
    # LLM, and can make mistakes) but it's the standard practical approach
    # used in production eval pipelines when you don't have human graders.
    
    sources_text = "\n".join(f"- {s['summary']}" for s in sources)
    prompt = f"""You are a fact-checker. Below is an ANSWER and the SOURCE information it was supposed to be based on.

    SOURCES: {sources_text}

    ANSWER: {answer}

    Does the answer contain any claims NOT supported by the sources (i.e. information the sources don't mention)? Respond with ONLY valid JSON:
    {{"faithful": true or false, "unsupported_claims": ["claim1", ...]}}
    No preamble, no markdown fences."""

    response = groq_client.chat.completions.create(
        model=LLM_MODEL, messages=[{"role": "user", "content": prompt}],
        temperature=0, max_tokens=300,
    )
    raw = response.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"faithful": None, "unsupported_claims": [], "note": "judge output unparseable"}


def check_topic_coverage(answer: str, expected_topics: list[str]) -> dict:
    
    # Metric 3: lightweight check - did the answer touch on the topics we
    # expected for this question? This isn't a strict grading rubric, just a
    # signal for "did it go completely off the rails."
    
    answer_lower = answer.lower()
    covered = [t for t in expected_topics if t.lower() in answer_lower]
    return {
        "covered": covered,
        "coverage_ratio": round(len(covered) / len(expected_topics), 2) if expected_topics else None,
    }


# ---------------------------------------------------------------------------
# RUN THE FULL EVALUATION
# ---------------------------------------------------------------------------
def run_evaluation(test_file: str = "test_questions.json"):
    with open(test_file) as f:
        test_cases = json.load(f)

    results = []
    for case in test_cases:
        print(f"\n{'='*70}\nEvaluating: {case['id']} - {case['question']}\n{'='*70}")

        agent_output = research_agent(case["question"])
        citation_check = check_citation_validity(agent_output["answer"], len(agent_output["sources"]))
        faithfulness_check = check_faithfulness(agent_output["answer"], agent_output["sources"])
        coverage_check = check_topic_coverage(agent_output["answer"], case.get("expected_topics", []))

        result = {
            "id": case["id"],
            "question": case["question"],
            "answer": agent_output["answer"],
            "num_sources": len(agent_output["sources"]),
            "rounds_used": agent_output["rounds_used"],
            "search_calls": agent_output["search_calls"],
            "llm_calls": agent_output["llm_calls"],
            "elapsed_seconds": agent_output["elapsed_seconds"],
            "citation_check": citation_check,
            "faithfulness_check": faithfulness_check,
            "coverage_check": coverage_check,
        }
        results.append(result)

        print(f"  Sources used: {result['num_sources']} | Rounds: {result['rounds_used']} | "
              f"LLM calls: {result['llm_calls']} | Time: {result['elapsed_seconds']}s")
        print(f"  Citations valid: {citation_check['valid']}")
        print(f"  Faithful (no hallucination): {faithfulness_check.get('faithful')}")
        print(f"  Topic coverage: {coverage_check['coverage_ratio']}")

    # ---- Summary report ----
    print(f"\n\n{'='*70}\nSUMMARY REPORT ({len(results)} questions)\n{'='*70}")
    avg_rounds = sum(r["rounds_used"] for r in results) / len(results)
    avg_llm_calls = sum(r["llm_calls"] for r in results) / len(results)
    avg_time = sum(r["elapsed_seconds"] for r in results) / len(results)
    citation_pass_rate = sum(r["citation_check"]["valid"] for r in results) / len(results)
    faithful_results = [r for r in results if r["faithfulness_check"].get("faithful") is not None]
    faithfulness_rate = (
        sum(r["faithfulness_check"]["faithful"] for r in faithful_results) / len(faithful_results)
        if faithful_results else None
    )
    avg_coverage = sum(
        r["coverage_check"]["coverage_ratio"] for r in results if r["coverage_check"]["coverage_ratio"] is not None
    ) / len(results)

    print(f"Average search rounds per question: {avg_rounds:.1f}")
    print(f"Average LLM calls per question:     {avg_llm_calls:.1f}")
    print(f"Average time per question:          {avg_time:.1f}s")
    print(f"Citation validity rate:             {citation_pass_rate*100:.0f}%")
    print(f"Faithfulness rate (no hallucination): {faithfulness_rate*100:.0f}%" if faithfulness_rate is not None else "Faithfulness rate: N/A")
    print(f"Average topic coverage:             {avg_coverage*100:.0f}%")

    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to eval_results.json")

    return results


if __name__ == "__main__":
    run_evaluation()
