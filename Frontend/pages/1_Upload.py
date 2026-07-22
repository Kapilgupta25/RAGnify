# import streamlit as st
# import sys
# import os
# import tempfile
# from pathlib import Path

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Backend')))

# from src.vectorStore import FaissVectorStore
# from src.data_loader import load_all_documents

# st.set_page_config(page_title="Upload — RAGnify", page_icon="📤", layout="wide")

# # Guard: redirect if not logged in
# if not st.session_state.get("authenticated"):
#     st.warning("Please sign in first.")
#     st.stop()

# # Sidebar
# with st.sidebar:
#     st.markdown("## 🔍 RAGnify")
#     st.markdown(f"**User:** {st.session_state.username}")
#     st.divider()
#     st.page_link("app.py",            label="🏠 Home")
#     st.page_link("pages/1_Upload.py", label="📤 Upload PDFs")
#     st.page_link("pages/2_Search.py", label="💬 Search")
#     st.divider()
#     from auth import logout
#     if st.button("🚪 Logout", use_container_width=True):
#         logout()
#         st.rerun()

# # ── Page ──────────────────────────────────────────────────────────────────────
# st.title("📤 Upload PDFs")
# st.markdown("Upload one or more PDF files. RAGnify will chunk them, create embeddings, and build your personal vector store.")

# PERSIST_DIR = os.path.abspath(
#     os.path.join(os.path.dirname(__file__), '..', '..', 'Backend', 'faiss_store')
# )

# # Status indicator
# faiss_exists = os.path.exists(os.path.join(PERSIST_DIR, "faiss.index"))
# if faiss_exists:
#     st.success("✅ A vector store is already loaded. You can upload more files to expand it or rebuild.")
# else:
#     st.info("No vector store found yet. Upload PDFs below to get started.", icon="ℹ️")

# st.divider()

# uploaded_files = st.file_uploader(
#     "Drag & drop PDFs here",
#     type=["pdf"],
#     accept_multiple_files=True,
#     help="Supports multiple PDF files at once.",
# )

# col_a, col_b = st.columns([1, 3])
# rebuild = col_a.checkbox("Rebuild from scratch", value=not faiss_exists,
#                          help="Uncheck to append to the existing store.")

# if uploaded_files:
#     st.markdown(f"**{len(uploaded_files)} file(s) selected:**")
#     for f in uploaded_files:
#         st.markdown(f"- `{f.name}` ({f.size / 1024:.1f} KB)")

#     if st.button("🚀 Process & Ingest", type="primary", use_container_width=False):
#         with st.spinner("Saving files..."):
#             # Write to a temp dir then load via LangChain loader
#             with tempfile.TemporaryDirectory() as tmp_dir:
#                 for uf in uploaded_files:
#                     dest = Path(tmp_dir) / uf.name
#                     dest.write_bytes(uf.read())

#                 progress = st.progress(0, text="Loading documents...")
#                 docs = load_all_documents(tmp_dir)
#                 progress.progress(30, text=f"Loaded {len(docs)} document pages...")

#                 if not docs:
#                     st.error("No documents could be loaded. Make sure the PDFs contain extractable text.")
#                     st.stop()

#                 progress.progress(50, text="Building vector store (this may take a minute)...")

#                 store = FaissVectorStore(PERSIST_DIR)

#                 if not rebuild and faiss_exists:
#                     store.load()
#                     # Build embeddings for new docs only and add
#                     from src.embedding import EmbeddingPipeline
#                     pipe = EmbeddingPipeline()
#                     chunks = pipe.chunk_documents(docs)
#                     embeddings = pipe.embed_chunks(chunks)
#                     import numpy as np
#                     metadatas = [{"text": c.page_content} for c in chunks]
#                     store.add_embeddings(embeddings.astype('float32'), metadatas)
#                     store.save()
#                 else:
#                     store.build_from_documents(docs)

#                 progress.progress(100, text="Done!")
#                 st.session_state["vector_store"] = store

#         st.success(f"✅ Ingested {len(uploaded_files)} file(s) — {len(docs)} pages processed.")
#         st.balloons()
# else:
#     st.markdown("*No files selected yet.*")

# st.divider()
# st.markdown("Once ingestion is complete, head over to **[💬 Search](./Search)** to start asking questions.")




import streamlit as st
import sys
import os
import shutil
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Backend')))

from src.vectorStore import FaissVectorStore
from src.data_loader import load_all_documents

st.set_page_config(page_title="Upload — RAGnify", page_icon="📤", layout="wide")

# Guard: redirect if not logged in
if not st.session_state.get("authenticated"):
    st.warning("Please sign in first.")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("## 🔍 RAGnify")
    st.markdown(f"**User:** {st.session_state.username}")
    st.divider()
    st.page_link("app.py",            label="🏠 Home")
    st.page_link("pages/1_Upload.py", label="📤 Upload PDFs")
    st.page_link("pages/2_Search.py", label="💬 Search")
    st.divider()
    from auth import logout
    if st.button("🚪 Logout", use_container_width=True):
        logout()
        st.rerun()

# ── Page ──────────────────────────────────────────────────────────────────────
st.title("📤 Upload PDFs")
st.markdown("Upload one or more PDF files. RAGnify will chunk them, create embeddings, and build your personal vector store.")

PERSIST_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'Backend', 'faiss_store')
)

# Status indicator
faiss_exists = os.path.exists(os.path.join(PERSIST_DIR, "faiss.index"))

col_status, col_reset = st.columns([4, 1])
with col_status:
    if faiss_exists:
        try:
            import pickle
            with open(os.path.join(PERSIST_DIR, "metadata.pkl"), "rb") as f:
                meta = pickle.load(f)
            sources = sorted({m.get("source", "unknown") for m in meta if m})
            st.success(f"✅ Vector store loaded — **{len(meta)} chunks** from **{len(sources)} file(s)**: "
                       + ", ".join(sources) if sources else f"✅ Vector store loaded — {len(meta)} chunks.")
        except Exception:
            st.success("✅ A vector store is already loaded.")
    else:
        st.info("No vector store found yet. Upload PDFs below to get started.", icon="ℹ️")

with col_reset:
    if faiss_exists and st.button("🗑️ Reset Store", use_container_width=True):
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)
        st.session_state.pop("vector_store", None)
        st.success("Vector store cleared.")
        st.rerun()

st.divider()

uploaded_files = st.file_uploader(
    "Drag & drop PDFs here",
    type=["pdf"],
    accept_multiple_files=True,
    help="Supports multiple PDF files at once.",
)

col_a, col_b = st.columns([1, 3])
rebuild = col_a.checkbox("Rebuild from scratch", value=not faiss_exists,
                         help="Uncheck to append to the existing store.")

if uploaded_files:
    st.markdown(f"**{len(uploaded_files)} file(s) selected:**")
    for f in uploaded_files:
        st.markdown(f"- `{f.name}` ({f.size / 1024:.1f} KB)")

    if st.button("🚀 Process & Ingest", type="primary", use_container_width=False):
        with st.spinner("Saving files..."):
            with tempfile.TemporaryDirectory() as tmp_dir:
                for uf in uploaded_files:
                    dest = Path(tmp_dir) / uf.name
                    dest.write_bytes(uf.getbuffer())

                progress = st.progress(0, text="Loading documents...")
                try:
                    docs = load_all_documents(tmp_dir)
                except Exception as e:
                    st.error(f"Failed to load documents: {e}")
                    st.stop()

                progress.progress(30, text=f"Loaded {len(docs)} document pages...")

                if not docs:
                    st.error("No documents could be loaded. Make sure the PDFs contain extractable text.")
                    st.stop()

                progress.progress(50, text="Building vector store (this may take a minute)...")

                try:
                    store = FaissVectorStore(PERSIST_DIR)

                    if not rebuild and faiss_exists:
                        store.load()
                        from src.embedding import EmbeddingPipeline
                        pipe = EmbeddingPipeline()
                        chunks = pipe.chunk_documents(docs)
                        embeddings = pipe.embed_chunks(chunks)
                        metadatas = [{"text": c.page_content, "source": c.metadata.get("source", "unknown")}
                                     for c in chunks]
                        store.add_embeddings(embeddings.astype('float32'), metadatas)
                        store.save()
                    else:
                        store.build_from_documents(docs)

                    progress.progress(100, text="Done!")
                    st.session_state["vector_store"] = store
                except Exception as e:
                    st.error(f"Failed to build vector store: {e}")
                    st.stop()

        st.success(f"✅ Ingested {len(uploaded_files)} file(s) — {len(docs)} pages processed.")
        st.balloons()
else:
    st.markdown("*No files selected yet.*")

st.divider()
st.markdown("Once ingestion is complete, head over to **[💬 Search](./Search)** to start asking questions.")