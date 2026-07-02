import streamlit as st
import hashlib
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Simple file-based user store (swap for a real DB in production)
USERS_FILE = Path(__file__).parent / ".users.json"


def _load_users() -> dict:
    if USERS_FILE.exists():
        with open(USERS_FILE) as f:
            return json.load(f)
    return {}


def _save_users(users: dict):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


# ── Login ─────────────────────────────────────────────────────────────────────

def login_page():
    with st.form("login_form"):
        st.markdown("### Sign in to your account")
        username = st.text_input("Username", placeholder="you@example.com")
        password = st.text_input("Password", type="password", placeholder="••••••••")
        submitted = st.form_submit_button("Sign In", use_container_width=True, type="primary")

    if submitted:
        _do_login(username, password)

    st.divider()
    st.markdown("**Or continue with:**")
    if st.button("🔵  Sign in with Google", use_container_width=True):
        _google_oauth()


def _do_login(username: str, password: str):
    users = _load_users()
    hashed = _hash_password(password)
    user = users.get(username)
    
    if user and user.get("password") is None:
        st.error("This account uses Google Sign-In. Please use the Google button.")
        return
    
    if user and user["password"] == hashed:
        st.session_state.authenticated = True
        st.session_state.username = username
        st.success("Logged in!")
        st.rerun()
    else:
        st.error("Invalid username or password.")
        
        
# ── Sign Up ───────────────────────────────────────────────────────────────────

def signup_page():
    with st.form("signup_form"):
        st.markdown("### Create a new account")
        username   = st.text_input("Username",        placeholder="you@example.com")
        password   = st.text_input("Password",        type="password", placeholder="Min 8 characters")
        password2  = st.text_input("Confirm password", type="password", placeholder="Repeat password")
        submitted  = st.form_submit_button("Create Account", use_container_width=True, type="primary")

    if submitted:
        _do_signup(username, password, password2)

    st.divider()
    st.markdown("**Or continue with:**")
    if st.button("🔵  Sign up with Google", use_container_width=True, key="google_signup"):
        _google_oauth()


def _do_signup(username: str, password: str, password2: str):
    if not username or not password:
        st.error("Username and password are required.")
        return
    if len(password) < 8:
        st.error("Password must be at least 8 characters.")
        return
    if password != password2:
        st.error("Passwords do not match.")
        return

    users = _load_users()
    if username in users:
        st.error("Username already exists.")
        return

    users[username] = {"password": _hash_password(password)}
    _save_users(users)
    st.success("Account created! Please sign in.")


# ── Google OAuth (stub — wire up streamlit-oauth or authlib) ──────────────────

def _google_oauth():
    from streamlit_oauth import OAuth2Component
    
    oauth = OAuth2Component(
        client_id=os.getenv("GOOGLE_CLIENT_ID"),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
        authorize_endpoint="https://accounts.google.com/o/oauth2/auth",
        token_endpoint="https://oauth2.googleapis.com/token",
        refresh_token_endpoint="https://oauth2.googleapis.com/token",
        revoke_token_endpoint="https://oauth2.googleapis.com/revoke",
    )

    result = oauth.authorize_button(
        name="Continue with Google",
        redirect_uri="http://localhost:8501",
        scope="openid email profile",
        key="google_oauth_btn",
        use_container_width=True,
    )

    if result and "token" in result:
        userinfo = result["token"].get("userinfo", {})
        email = userinfo.get("email", "")
        if email:
            # Auto-register Google users if first time
            users = _load_users()
            if email not in users:
                users[email] = {"password": None, "oauth": "google"}
                _save_users(users)
            st.session_state.authenticated = True
            st.session_state.username = email
            st.rerun()
        else:
            st.error("Could not retrieve email from Google.")
            
            
# ── Logout ────────────────────────────────────────────────────────────────────

def logout():
    for key in ["authenticated", "username", "vector_store", "chat_history"]:
        st.session_state.pop(key, None)
        
        