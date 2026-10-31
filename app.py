
# Smart Research Assistant - Step 5: Streamlit UI
# ====================================================
# A simple chat-style interface wrapping agent_core.py. This is the final
# piece that turns your project from "a script I run in a terminal" into
# "a product I can demo in 30 seconds" - genuinely useful for interviews
# and for showing non-technical people what you built.


import streamlit as st
from agent_core import research_agent_with_memory

st.set_page_config(page_title="Smart Research Assistant", page_icon="🔎", layout="wide")

st.title("🔎 Smart Research Assistant")
st.caption("An agent that plans searches, reasons about when it has enough information, "
           "and remembers past conversations. Built with Groq + Tavily + sentence-transformers.")

# Short-term memory lives in Streamlit's session state - it resets each
# time you restart the app, which is the correct behavior for a
# per-conversation memory (as opposed to long-term memory, which persists
# to memory_store.json regardless of session).
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# ---------------------------------------------------------------------------
# Sidebar: show what's happening under the hood - genuinely useful for a
# portfolio demo, since it makes the agent's reasoning visible rather than
# a black box
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("How it works")
    st.markdown("""
    1. **Contextualize** — rewrites follow-up questions using conversation history
    2. **Check memory** — reuses a past answer if a similar question was asked before
    3. **Plan** — breaks the question into targeted search queries
    4. **Search + rank** — searches the web, ranks results by semantic relevance, removes duplicates
    5. **Summarize + synthesize** — summarizes each source, combines into one cited answer
    6. **Reflect** — decides if it has enough info, or repeats steps 3-6
    """)
    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.conversation_history = []
        st.rerun()

# ---------------------------------------------------------------------------
# Display conversation history (short-term memory, rendered as chat)
# ---------------------------------------------------------------------------
for turn in st.session_state.conversation_history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        if turn.get("from_cache"):
            st.caption(f"💾 Reused from memory (similarity: {turn['cache_similarity']})")
        with st.expander(f"📚 {len(turn.get('sources', []))} sources"):
            for i, s in enumerate(turn.get("sources", [])):
                st.markdown(f"**[{i+1}] {s['title']}** (relevance: {s.get('relevance_score', '—')})")
                st.markdown(f"{s['summary']}")
                st.markdown(f"[{s['url']}]({s['url']})")
                st.divider()

# ---------------------------------------------------------------------------
# Input box + agent call
# ---------------------------------------------------------------------------
user_question = st.chat_input("Ask a research question...")

if user_question:
    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Researching..."):
            result = research_agent_with_memory(
                user_question, st.session_state.conversation_history
            )
        st.write(result["answer"])

        if result.get("from_cache"):
            st.caption(f"💾 Reused from memory (similarity: {result['cache_similarity']})")
        else:
            st.caption(
                f"🔄 {result.get('rounds_used', '—')} search round(s) · "
                f"⏱️ {result['elapsed_seconds']}s"
            )

        with st.expander(f"📚 {len(result['sources'])} sources"):
            for i, s in enumerate(result["sources"]):
                st.markdown(f"**[{i+1}] {s['title']}** (relevance: {s.get('relevance_score', '—')})")
                st.markdown(f"{s['summary']}")
                st.markdown(f"[{s['url']}]({s['url']})")
                st.divider()

    # Save this turn into short-term (session) memory for future context
    st.session_state.conversation_history.append({
        "question": user_question,
        "answer": result["answer"],
        "sources": result["sources"],
        "from_cache": result.get("from_cache", False),
        "cache_similarity": result.get("cache_similarity"),
    })
