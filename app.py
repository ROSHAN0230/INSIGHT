import streamlit as st
import pandas as pd
from utils import extract_content, build_tfidf_index, generate_suggestions, auto_generate_pulse, inject_lumina_css

# 1. Initialize Premium UI
st.set_page_config(page_title="Lumina AI", page_icon="💎", layout="wide")
inject_lumina_css()

# 2. Hero Section
st.markdown('<div style="text-align: center; padding-top: 20px;">', unsafe_allow_html=True)
# Embed the 3D Orb Asset
try:
    # Relative path for repository/deployment
    orb_path = "assets/hero_orb.png"
    st.image(orb_path, width=400)
except:
    st.markdown('<h1 style="font-size: 100px;">💎</h1>', unsafe_allow_html=True)

st.markdown('<p class="hero-text">LUMINA AI</p>', unsafe_allow_html=True)
st.markdown('<p style="color: #888; font-size: 1.2rem; margin-top: -10px;">The Supreme RAG & Data Analytics Intelligence</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Shared state initialization
if "uploaded_df" not in st.session_state: st.session_state.uploaded_df = None
if "uploaded_text" not in st.session_state: st.session_state.uploaded_text = None
if "filename" not in st.session_state: st.session_state.filename = ""
if "vectorizer" not in st.session_state: st.session_state.vectorizer = None
if "tfidf_matrix" not in st.session_state: st.session_state.tfidf_matrix = None
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
            
            # Pulse
            st.session_state.pulse = auto_generate_pulse(df, c)
            st.session_state.suggestions = generate_suggestions(text, uploaded_file.name, df is not None)
            st.rerun()

# 3. Dynamic Content Hub
if not st.session_state.filename:
    st.markdown('<div style="height: 40px;"></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="lumina-card">', unsafe_allow_html=True)
        st.markdown('<h3>💬 Semantic Chat</h3><p style="color: #888;">Talk to your documents. Ask for summaries, interpretations, or trends.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="lumina-card">', unsafe_allow_html=True)
        st.markdown('<h3>⚡ Precise Math</h3><p style="color: #888;">Crunches numbers with 100% accuracy using generated Python code.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="lumina-card">', unsafe_allow_html=True)
        st.markdown('<h3>📊 Smart Dashboards</h3><p style="color: #888;">Auto-detects columns and generates visual insights instantly.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    st.info("👋 Welcome! Please upload a file in the sidebar to begin your analysis.")

else:
    # Data Pulse Card
    st.markdown('<div class="lumina-card" style="border-left: 4px solid #BC00FF;">', unsafe_allow_html=True)
    st.markdown('<h3>🔮 Intelligence Pulse</h3>', unsafe_allow_html=True)
    st.write(st.session_state.get("pulse", "Generating elite insights..."))
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:#BC00FF; font-weight:600; text-transform:uppercase; letter-spacing:2px;">Execute Intelligence</p>', unsafe_allow_html=True)
    
    # Navigation Cards
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)
    
    with nav_col1:
        st.markdown('<div class="lumina-card">', unsafe_allow_html=True)
        st.markdown('<p style="font-size:1.5rem;">💬</p><h4>Ask AI</h4><p style="font-size:0.8rem; color:#888;">Knowledge RAG</p>', unsafe_allow_html=True)
        if st.button("Open Chat", use_container_width=True): st.switch_page("pages/ask_ai.py")
        st.markdown('</div>', unsafe_allow_html=True)

    with nav_col2:
        st.markdown('<div class="lumina-card">', unsafe_allow_html=True)
        st.markdown('<p style="font-size:1.5rem;">📊</p><h4>Dashboard</h4><p style="font-size:0.8rem; color:#888;">Visual Analytics</p>', unsafe_allow_html=True)
        if st.button("Insights", use_container_width=True): st.switch_page("pages/dashboard.py")
        st.markdown('</div>', unsafe_allow_html=True)

    with nav_col3:
        st.markdown('<div class="lumina-card">', unsafe_allow_html=True)
        st.markdown('<p style="font-size:1.5rem;">⚡</p><h4>Logic</h4><p style="font-size:0.8rem; color:#888;">Code Engine</p>', unsafe_allow_html=True)
        if st.button("Run Engine", use_container_width=True): st.switch_page("pages/calculator.py")
        st.markdown('</div>', unsafe_allow_html=True)

    with nav_col4:
        st.markdown('<div class="lumina-card">', unsafe_allow_html=True)
        st.markdown('<p style="font-size:1.5rem;">👀</p><h4>Preview</h4><p style="font-size:0.8rem; color:#888;">Raw Data Hub</p>', unsafe_allow_html=True)
        if st.button("Explore", use_container_width=True): st.switch_page("pages/preview.py")
        st.markdown('</div>', unsafe_allow_html=True)
