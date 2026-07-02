import streamlit as st
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Backend')))

from auth import login_page, signup_page, logout

st.set_page_config(
    page_title="RAGnify",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #1a1a2e; }
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s ease;
}
.main-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.subtitle { color: #888; font-size: 1.1rem; margin-bottom: 2rem; }
.card {
    background: #f8f9fa;
    border-radius: 12px;
    padding: 1.5rem;
    border: 1px solid #e9ecef;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# --- Session defaults ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "page" not in st.session_state:
    st.session_state.page = "login"

# --- Sidebar (only when logged in) ---
if st.session_state.authenticated:
    with st.sidebar:
        st.markdown("## 🔍 RAGnify")
        st.markdown(f"**User:** {st.session_state.username}")
        st.divider()
        st.page_link("app.py",           label="🏠 Home",   icon=None)
        st.page_link("pages/1_Upload.py", label="📤 Upload PDFs")
        st.page_link("pages/2_Search.py", label="💬 Search")
        st.divider()
        if st.button("🚪 Logout", use_container_width=True):
            logout()
            st.rerun()

# --- Main area ---
if not st.session_state.authenticated:
    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        st.markdown('<p class="main-title">RAGnify</p>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">AI-powered document search & QA</p>', unsafe_allow_html=True)

        tab_login, tab_signup = st.tabs(["Sign In", "Create Account"])
        with tab_login:
            login_page()
        with tab_signup:
            signup_page()
else:
    st.markdown('<p class="main-title">Welcome to RAGnify 🔍</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="subtitle">Hello, **{st.session_state.username}**! Choose an action below.</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card">
            <h3>📤 Upload Documents</h3>
            <p>Upload your PDF files to train RAGnify. Documents are chunked, embedded, and stored in a FAISS vector store.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Upload →", key="upload_btn", use_container_width=True):
            st.switch_page("pages/1_Upload.py")

    with col2:
        st.markdown("""
        <div class="card">
            <h3>💬 Search & Ask</h3>
            <p>Ask questions about your uploaded documents. RAGnify finds the most relevant context and generates precise answers.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Search →", key="search_btn", use_container_width=True):
            st.switch_page("pages/2_Search.py")
            
            