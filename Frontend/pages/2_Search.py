# import streamlit as st
# import sys
# import os

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Backend')))

# from src.search import RAGSearch

# st.set_page_config(page_title="Search — RAGnify", page_icon="💬", layout="wide")

# # Guard
# if not st.session_state.get("authenticated"):
#     st.warning("Please sign in first.")
#     st.stop()

# PERSIST_DIR = os.path.abspath(
#     os.path.join(os.path.dirname(__file__), '..', '..', 'Backend', 'faiss_store')
# )

# # Sidebar
# with st.sidebar:
#     st.markdown("## 🔍 RAGnify")
#     st.markdown(f"**User:** {st.session_state.username}")
#     st.divider()
#     st.page_link("app.py",            label="🏠 Home")
#     st.page_link("pages/1_Upload.py", label="📤 Upload PDFs")
#     st.page_link("pages/2_Search.py", label="💬 Search")
#     st.divider()

#     st.markdown("**Settings**")
#     top_k = st.slider("Top-K results", min_value=1, max_value=10, value=5,
#                       help="How many document chunks to retrieve before summarising.")
#     model_choice = st.selectbox("LLM model", [
#         "llama-3.1-8b-instant",
#         "llama-3.3-70b-versatile",
#         "mixtral-8x7b-32768",
#     ])

#     st.divider()
#     if st.button("🗑️ Clear chat", use_container_width=True):
#         st.session_state.chat_history = []
#         st.rerun()

#     from auth import logout
#     if st.button("🚪 Logout", use_container_width=True):
#         logout()
#         st.rerun()

# # ── Load RAG engine (cached per session) ──────────────────────────────────────

# @st.cache_resource(show_spinner=False)
# def load_rag(persist_dir: str, llm_model: str) -> RAGSearch:
#     return RAGSearch(persist_dir=persist_dir, llm_model=llm_model)

# faiss_exists = os.path.exists(os.path.join(PERSIST_DIR, "faiss.index"))

# # ── Page ──────────────────────────────────────────────────────────────────────
# st.title("💬 Search your Documents")

# if not faiss_exists:
#     st.warning("No vector store found. Please upload some PDFs first.", icon="⚠️")
#     if st.button("Go to Upload →"):
#         st.switch_page("pages/1_Upload.py")
#     st.stop()

# # Init chat history
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # Render existing messages
# for msg in st.session_state.chat_history:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])
#         if msg.get("sources"):
#             with st.expander("📄 Source context", expanded=False):
#                 for i, src in enumerate(msg["sources"], 1):
#                     st.markdown(f"**Chunk {i}:**")
#                     st.markdown(f"> {src[:600]}{'...' if len(src) > 600 else ''}")

# # Chat input
# if prompt := st.chat_input("Ask anything about your documents…"):
#     # Show user message
#     st.session_state.chat_history.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     # Generate answer
#     with st.chat_message("assistant"):
#         with st.spinner("Searching documents and generating answer..."):
#             try:
#                 rag = load_rag(PERSIST_DIR, model_choice)

#                 # Get retrieved chunks for sources panel
#                 raw_results = rag.vectorstore.query(prompt, top_k=top_k)
#                 sources = [r["metadata"].get("text", "") for r in raw_results if r.get("metadata")]

#                 # Get summarised answer
#                 answer = rag.search_and_summarize(prompt, top_k=top_k)

#             except Exception as e:
#                 answer = f"❌ Error: {e}"
#                 sources = []

#         st.markdown(answer)
#         if sources:
#             with st.expander("📄 Source context", expanded=False):
#                 for i, src in enumerate(sources, 1):
#                     st.markdown(f"**Chunk {i}:**")
#                     st.markdown(f"> {src[:600]}{'...' if len(src) > 600 else ''}")

#     st.session_state.chat_history.append({
#         "role": "assistant",
#         "content": answer,
#         "sources": sources,
#     })
    
    
    


import streamlit as st
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Backend')))

from src.search import RAGSearch

st.set_page_config(page_title="Search — RAGnify", page_icon="💬", layout="wide")

# Guard
if not st.session_state.get("authenticated"):
    st.warning("Please sign in first.")
    st.stop()

PERSIST_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'Backend', 'faiss_store')
)

# Sidebar
with st.sidebar:
    st.markdown("## 🔍 RAGnify")
    st.markdown(f"**User:** {st.session_state.username}")
    st.divider()
    st.page_link("app.py",            label="🏠 Home")
    st.page_link("pages/1_Upload.py", label="📤 Upload PDFs")
    st.page_link("pages/2_Search.py", label="💬 Search")
    st.divider()

    st.markdown("**Settings**")
    top_k = st.slider("Top-K results", min_value=1, max_value=10, value=5,
                      help="How many document chunks to retrieve before summarising.")
    model_choice = st.selectbox("LLM model", [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
    ])

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    from auth import logout
    if st.button("🚪 Logout", use_container_width=True):
        logout()
        st.rerun()

# ── Preconditions ───────────────────────────────────────────────────────────
if not os.getenv("GROQ_API_KEY"):
    st.error("⚠️ `GROQ_API_KEY` is not set. Add it to your `.env` file at the project root and restart the app.")
    st.stop()

faiss_exists = os.path.exists(os.path.join(PERSIST_DIR, "faiss.index"))

st.title("💬 Search your Documents")

if not faiss_exists:
    st.warning("No vector store found. Please upload some PDFs first.", icon="⚠️")
    if st.button("Go to Upload →"):
        st.switch_page("pages/1_Upload.py")
    st.stop()

# ── Load RAG engine (cached per model choice) ──────────────────────────────
@st.cache_resource(show_spinner="Loading RAG engine...")
def load_rag(persist_dir: str, llm_model: str) -> RAGSearch:
    return RAGSearch(persist_dir=persist_dir, llm_model=llm_model)

try:
    rag = load_rag(PERSIST_DIR, model_choice)
except Exception as e:
    st.error(f"Failed to initialize the RAG engine: {e}")
    st.stop()

# Init chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Render existing messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Source context", expanded=False):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.markdown(f"> {src[:600]}{'...' if len(src) > 600 else ''}")

# Chat input
if prompt := st.chat_input("Ask anything about your documents…"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            try:
                raw_results = rag.vectorstore.query(prompt, top_k=top_k)
                sources = [r["metadata"].get("text", "") for r in raw_results if r.get("metadata")]
                answer = rag.search_and_summarize(prompt, top_k=top_k)
            except Exception as e:
                answer = f"❌ Error: {e}"
                sources = []

        st.markdown(answer)
        if sources:
            with st.expander("📄 Source context", expanded=False):
                for i, src in enumerate(sources, 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.markdown(f"> {src[:600]}{'...' if len(src) > 600 else ''}")

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
    
