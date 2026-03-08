import streamlit as st
import pandas as pd
from utils import extract_content, build_tfidf_index, generate_suggestions, auto_generate_pulse

# 1. Page Configuration
st.set_page_config(page_title="InsightX AI", page_icon="🚀", layout="wide")

# 2. Custom CSS for a Premium Dark Aesthetic
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #E0E0E0; }
    .main-header { font-size: 3rem; font-weight: 800; color: #00FFAA; margin-bottom: 0px; }
    .sub-header { color: #888; margin-bottom: 2rem; }
    div[data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">InsightX AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Universal RAG & Data Analytics Engine</p>', unsafe_allow_html=True)

# 3. Shared Session State Initialization
if "uploaded_df" not in st.session_state: st.session_state.uploaded_df = None
if "uploaded_text" not in st.session_state: st.session_state.uploaded_text = None
if "filename" not in st.session_state: st.session_state.filename = ""
if "vectorizer" not in st.session_state: st.session_state.vectorizer = None
if "tfidf_matrix" not in st.session_state: st.session_state.tfidf_matrix = None
if "chunks" not in st.session_state: st.session_state.chunks = []

# 4. Sidebar File Uploader (Global State)
st.sidebar.header("📂 Upload Center")
uploaded_file = st.sidebar.file_uploader("Upload content (CSV, PDF, Images...)", type=None)

if uploaded_file:
    if st.session_state.filename != uploaded_file.name:
        with st.spinner(f"Reading {uploaded_file.name}..."):
            # Use central utils for extraction
            text, df = extract_content(uploaded_file)
            st.session_state.uploaded_text = text
            st.session_state.uploaded_df = df
            st.session_state.filename = uploaded_file.name
            
            # Re-index for RAG
            v, m, c = build_tfidf_index(text)
            st.session_state.vectorizer = v
            st.session_state.tfidf_matrix = m
            st.session_state.chunks = c
            
            # Generate AI Pulse and Suggestions
            st.session_state.pulse = auto_generate_pulse(df, c)
            st.session_state.suggestions = generate_suggestions(text, uploaded_file.name, df is not None)
            st.rerun()

# 5. Conditional Landing Page Content
if not st.session_state.filename:
    st.info("👋 Welcome! Please upload a file in the sidebar to begin your analysis.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 💬 Semantic Chat")
        st.write("Talk to your documents. Ask for summaries, interpretations, or trends.")
    with col2:
        st.markdown("### ⚡ Precise Math")
        st.write("Crunches numbers with 100% accuracy using generated Python code.")
    with col3:
        st.markdown("### 📊 Smart Dashboards")
        st.write("Auto-detects columns and generates visual insights instantly.")
else:
    st.success(f"✅ Successfully loaded: **{st.session_state.filename}**")
    st.markdown("### 📊 Data Pulse")
    st.info(st.session_state.get("pulse", "Generating executive observations..."))
    
    st.markdown("---")
    st.markdown("### 🧭 Navigation")
    col1, col2 = st.columns(2)
    with col1:
        # Standardized path matching GitHub: pages/ask_ai.py
        if st.button("Go to Chat", use_container_width=True): 
            st.switch_page("pages/ask_ai.py")
    with col2:
        if st.session_state.uploaded_df is not None:
            # Standardized path matching GitHub: pages/dashboard.py
            if st.button("Go to Dashboard", use_container_width=True): 
                st.switch_page("pages/dashboard.py")
        else:
            st.info("Upload a CSV/Excel to enable the visual Dashboard tab.")
