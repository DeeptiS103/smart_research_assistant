
# Smart Research Assistant - Step 1: Manual Pipeline
# =====================================================
# This is the FOUNDATION of your agent project. No "agent" logic yet -
# just a straight-line pipeline: Question -> Search -> Summarize -> Combine.

# Once this works, Step 2 will make the LLM DECIDE what to search for,
# turning this into a real agent.

# Setup:
# 1. pip install groq tavily-python python-dotenv
# 2. Create a file called `.env` in this same folder with:
#      GROQ_API_KEY=your_key_here
#      TAVILY_API_KEY=your_key_here
# 3. Run: python step1_research_pipeline.py

import os
from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient

# ---------------------------------------------------------------------------
# 1. SETUP: Load API keys from .env file (never hardcode keys in your script!)
# ---------------------------------------------------------------------------
load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Groq's free tier gives you access to several open models.
# llama-3.1-8b-instant is fast and good enough for this project.
LLM_MODEL = "llama-3.1-8b-instant"


# ---------------------------------------------------------------------------
# 2. SEARCH: Get raw information from the web
# ---------------------------------------------------------------------------
def search_web(query: str, max_results: int = 4) -> list[dict]:
    
    # Calls Tavily's search API and returns a list of results.
    # Each result looks like: {"title": ..., "url": ..., "content": ...}
    
    print(f"Searching for: {query}")
    response = tavily_client.search(
        query=query,
        max_results=max_results,
        search_depth="basic",  # "advanced" costs more of your free quota
    )
    return response.get("results", [])


# ---------------------------------------------------------------------------
# 3. SUMMARIZE: This is your first real NLP task - condensing text with an LLM
# ---------------------------------------------------------------------------
def summarize_source(content: str, question: str) -> str:
    # """
    # Takes raw text from one search result and summarizes ONLY the parts
    # relevant to the user's original question. This is a targeted summary,
    # not a generic one - an important NLP concept (query-focused summarization).
    # """
    prompt = f"""You are helping research a question. Summarize the following text in 1-2 sentences, focusing ONLY on information relevant to this question:

    Question: {question}

    Text to summarize: {content[:3000]}

    Give ONLY the summary, no preamble."""

    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,  # low temperature = more focused, less creative
        max_tokens=150,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# 4. SYNTHESIZE: Combine all summaries into one coherent, cited final answer
# ---------------------------------------------------------------------------
def synthesize_answer(question: str, summarized_sources: list[dict]) -> str:
    # """
    # Takes all the per-source summaries and asks the LLM to write ONE
    # coherent answer that cites which source each claim came from.
    # """
    sources_text = "\n\n".join(
        f"[Source {i+1}: {s['title']}]\n{s['summary']}"
        for i, s in enumerate(summarized_sources)
    )

    prompt = f"""Based on the following sources, write a clear, well-organized answer to this question. Cite sources using [Source N] notation after each claim.

    Question: {question}

    Sources: {sources_text}

    Write a comprehensive detail answer (10-15 sentences) with citations."""

    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=400,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# 5. MAIN PIPELINE: Wire everything together
# ---------------------------------------------------------------------------
def research(question: str):
    print(f"\n Question: {question}\n")

    # Step A: Search
    results = search_web(question)
    if not results:
        print("No search results found.")
        return

    print(f" Found {len(results)} sources\n")

    # Step B: Summarize each source individually
    summarized_sources = []
    for i, result in enumerate(results):
        print(f"Summarizing source {i+1}: {result['title'][:60]}...")
        summary = summarize_source(result["content"], question)
        summarized_sources.append(
            {
                "title": result["title"],
                "url": result["url"],
                "summary": summary,
            }
        )

    # Step C: Synthesize final answer
    print("\nSynthesizing final answer...\n")
    final_answer = synthesize_answer(question, summarized_sources)

    # Step D: Print results nicely
    print("=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print(final_answer)
    print("\n" + "=" * 70)
    print("SOURCES")
    print("=" * 70)
    for i, s in enumerate(summarized_sources):
        print(f"[Source {i+1}] {s['title']}\n{s['url']}\n")

    return final_answer, summarized_sources


# ---------------------------------------------------------------------------
# RUN IT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Try this with your own question once it's working!
    # test_question = "What are the main differences between RAG and fine-tuning for LLMs?"
    test_question = input("Enter your question : ")
    research(test_question)
