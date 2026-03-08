import streamlit as st
import pandas as pd
from utils import extract_content, build_tfidf_index, generate_suggestions, auto_generate_pulse, inject_lumina_css

# 1. Initialize Premium UI
st.set_page_config(page_title="Lumina AI", page_icon="💎", layout="wide")
inject_lumina_css()

# 2. Hero Section (Elite Desktop Landing)
st.markdown('<div style="text-align: center; margin-top: -100px; padding-bottom: 60px;">', unsafe_allow_html=True)
st.markdown('<div class="hero-text">LUMINA AI</div>', unsafe_allow_html=True)
st.markdown('<p class="lumina-subtext">Supreme Intelligence Suite</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Shared state initialization
if "uploaded_df" not in st.session_state: st.session_state.uploaded_df = None
if "uploaded_text" not in st.session_state: st.session_state.uploaded_text = None
if "filename" not in st.session_state: st.session_state.filename = ""
if "vectorizer" not in st.session_state: st.session_state.vectorizer = None
if "tfidf_matrix" not in st.session_state: st.session_state.tfidf_matrix = None
if "chunks" not in st.session_state: st.session_state.chunks = []

# Sidebar - Minimalist Upload Center
with st.sidebar:
    st.markdown('<div style="display:flex; align-items:center; gap:12px; margin-bottom:40px; background: rgba(187,0,255,0.05); padding:15px; border-radius:15px; border: 1px solid rgba(187,0,255,0.1);">'
                '<div style="width:45px; height:45px; border-radius:12px; background:linear-gradient(135deg, #bb00ff 0%, #7a00ff 100%); display:flex; align-items:center; justify-content:center; color:white; box-shadow:0 0 15px rgba(187,0,255,0.3);">'
                '<span class="material-symbols-outlined">blur_on</span></div>'
                '<div><h2 style="margin:0; font-size:1rem; color:white; font-family:\"Space Grotesk\";">Lumina OS</h2>'
                '<p style="margin:0; font-size:0.6rem; color:#888; text-transform:uppercase; letter-spacing:1px;">Core V2.4</p></div></div>', unsafe_allow_html=True)

    st.markdown('<p style="color:#888; font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:2px; margin-bottom:15px;">Neural Injection</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drop high-dimensional datasets", type=None, help="Support for CSV, PDF, XLSX, Images...")

    if uploaded_file:
        if st.session_state.filename != uploaded_file.name:
            with st.spinner(f"Synthesizing {uploaded_file.name}..."):
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
    st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
    
    # Feature Grid
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="lumina-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:2.5rem; margin-bottom:15px; filter: drop-shadow(0 0 10px rgba(187,0,255,0.4));">💬</div>'
                    '<h3 style="color:#fff; font-family:\'Space Grotesk\'; font-weight:700;">Neural Chat</h3>'
                    '<p style="color: #888; font-size:0.85rem;">Conversational RAG for deep knowledge synthesis.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="lumina-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:2.5rem; margin-bottom:15px; filter: drop-shadow(0 0 10px rgba(187,0,255,0.4));">⚡</div>'
                    '<h3 style="color:#fff; font-family:\'Space Grotesk\'; font-weight:700;">Logic Engine</h3>'
                    '<p style="color: #888; font-size:0.85rem;">High-precision mathematical code execution.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="lumina-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:2.5rem; margin-bottom:15px; filter: drop-shadow(0 0 10px rgba(187,0,255,0.4));">📊</div>'
                    '<h3 style="color:#fff; font-family:\'Space Grotesk\'; font-weight:700;">Visual Intel</h3>'
                    '<p style="color: #888; font-size:0.85rem;">Automated visual insights and KPI detection.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    st.markdown('<div style="text-align:center; margin-top:40px;">'
                '<p style="color:#555; font-size:0.8rem; letter-spacing:1px;">SYSTEM STATUS: AWAITING NEURAL INJECTION (UPLOAD FILE)</p></div>', unsafe_allow_html=True)

else:
    # Intelligence Pulse Card (Stitch Inspired)
    st.markdown('<div class="glass-card" style="border-left: 2px solid #bb00ff; text-align:left; background:rgba(187,0,255,0.03); padding:40px;">', unsafe_allow_html=True)
    st.markdown('<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">'
                '<h3 style="color:#bb00ff; margin:0; font-family:\'Space Grotesk\'; letter-spacing:2px; font-weight:700;">🔮 INTELLIGENCE PULSE</h3>'
                '<div class="badge-validated">Active Neural Link</div></div>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:1.1rem; line-height:1.7; color:#ccc; font-weight:400;">{st.session_state.get("pulse", "Generating elite insights...")}</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:#bb00ff; font-weight:700; text-transform:uppercase; letter-spacing:5px; font-size:0.7rem; margin-bottom:30px; opacity:0.6;">Select Operating Module</p>', unsafe_allow_html=True)
    
    # Navigation Cards (Elite Stitch layout)
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)
    
    with nav_col1:
        st.markdown('<div class="lumina-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:2rem; margin-bottom:10px;">💬</div><h4 style="font-family:\'Space Grotesk\';">Ask AI</h4><p style="font-size:0.7rem; color:#666; margin-bottom:20px; text-transform:uppercase; letter-spacing:1px;">Neural RAG</p>', unsafe_allow_html=True)
        if st.button("Initialize Chat", key="nav_chat", use_container_width=True): st.switch_page("pages/ask_ai.py")
        st.markdown('</div>', unsafe_allow_html=True)

    with nav_col2:
        st.markdown('<div class="lumina-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:2rem; margin-bottom:10px;">📊</div><h4 style="font-family:\'Space Grotesk\';">Analytics</h4><p style="font-size:0.7rem; color:#666; margin-bottom:20px; text-transform:uppercase; letter-spacing:1px;">Visual Intel</p>', unsafe_allow_html=True)
        if st.button("View Console", key="nav_dash", use_container_width=True): st.switch_page("pages/dashboard.py")
        st.markdown('</div>', unsafe_allow_html=True)

    with nav_col3:
        st.markdown('<div class="lumina-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:2rem; margin-bottom:10px;">⚡</div><h4 style="font-family:\'Space Grotesk\';">Logic</h4><p style="font-size:0.7rem; color:#666; margin-bottom:20px; text-transform:uppercase; letter-spacing:1px;">Code Engine</p>', unsafe_allow_html=True)
        if st.button("Run Processor", key="nav_calc", use_container_width=True): st.switch_page("pages/calculator.py")
        st.markdown('</div>', unsafe_allow_html=True)

    with nav_col4:
        st.markdown('<div class="lumina-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:2rem; margin-bottom:10px;">👀</div><h4 style="font-family:\'Space Grotesk\';">Audit</h4><p style="font-size:0.7rem; color:#666; margin-bottom:20px; text-transform:uppercase; letter-spacing:1px;">Raw Dataset</p>', unsafe_allow_html=True)
        if st.button("Explore Data", key="nav_prev", use_container_width=True): st.switch_page("pages/preview.py")
        st.markdown('</div>', unsafe_allow_html=True)
