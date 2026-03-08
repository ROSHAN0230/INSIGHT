import streamlit as st
import pandas as pd
from utils import extract_content, build_tfidf_index, generate_suggestions, auto_generate_pulse

st.set_page_config(page_title="InsightX AI", page_icon="🚀", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #E0E0E0; }
    .main-header { font-size: 3rem; font-weight: 800; color: #00FFAA; margin-bottom: 0px; }
    .sub-header { color: #888; margin-bottom: 2rem; }
</style>
""", unsafe_allow_safe=True)

st.markdown('<p class="main-header">InsightX AI</p>', unsafe_allow_safe=True)
st.markdown('<p class="sub-header">Universal RAG & Data Analytics Engine</p>', unsafe_allow_safe=True)

# Shared state initialization
if "uploaded_df" not in st.session_state: st.session_state.uploaded_df = None
if "uploaded_text" not in st.session_state: st.session_state.uploaded_text = None
if "filename" not in st.session_state: st.session_state.filename = ""
if "chunks" not in st.session_state: st.session_state.chunks = []

# Sidebar uploader
st.sidebar.header("📂 Upload Center")
uploaded_file = st.sidebar.file_uploader("Upload content (CSV, PDF, Images...)", type=None)

if uploaded_file:
    if st.session_state.filename != uploaded_file.name:
        with st.spinner(f"Reading {uploaded_file.name}..."):
            text, df = extract_content(uploaded_file)
            st.session_state.uploaded_text = text
            st.session_state.uploaded_df = df
            st.session_state.filename = uploaded_file.name
            
            # Re-index
            v, m, c = build_tfidf_index(text)
            st.session_state.vectorizer = v
            st.session_state.tfidf_matrix = m
            st.session_state.chunks = c
            
            st.session_state.pulse = auto_generate_pulse(df, c)
            st.session_state.suggestions = generate_suggestions(text, uploaded_file.name, df is not None)
            st.rerun()

if not st.session_state.filename:
    st.info("👋 Welcome! Please upload a file in the sidebar to begin.")
else:
    st.success(f"✅ Loaded: **{st.session_state.filename}**")
    st.markdown("### 📊 Data Pulse")
    st.info(st.session_state.get("pulse", "Check back in a moment..."))
    
    st.markdown("---")
    st.markdown("### 🧭 Navigation")
    c1, c2 = st.columns(2)
    with c1:
        # UPDATED to match your GitHub filename
        if st.button("Go to Chat"): st.switch_page("pages/ask_ai_.py")
    with c2:
        if st.session_state.uploaded_df is not None:
            # UPDATED to match your GitHub filename
            if st.button("Go to Dashboard"): st.switch_page("pages/leadership_dashboard.py")
