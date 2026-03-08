import streamlit as st
import pandas as pd
from utils import extract_content, build_tfidf_index, generate_suggestions, auto_generate_pulse, inject_lumina_css, load_template

# 1. Initialize Premium UI
st.set_page_config(page_title="Lumina AI", page_icon="💎", layout="wide")
inject_lumina_css()

# 2. Hero Section (Elite Desktop Landing Overlay)
st.markdown('<div style="text-align: center; margin-top: -100px; padding-bottom: 60px;">', unsafe_allow_html=True)
st.markdown('<div class="hero-text" style="font-size: 8rem; font-weight: 900; background: linear-gradient(135deg, #bc00ff 0%, #7a00ff 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">LUMINA AI</div>', unsafe_allow_html=True)
st.markdown('<p style="color: rgba(255,255,255,0.4); letter-spacing: 0.5em; text-transform: uppercase;">Supreme Intelligence Suite</p>', unsafe_allow_html=True)
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
    st.markdown("""
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:40px; background: rgba(187,0,255,0.05); padding:15px; border-radius:15px; border: 1px solid rgba(187,0,255,0.1);">
            <div style="width:45px; height:45px; border-radius:12px; background:linear-gradient(135deg, #bb00ff 0%, #7a00ff 100%); display:flex; align-items:center; justify-content:center; color:white;">
                <span class="material-symbols-outlined">blur_on</span>
            </div>
            <div>
                <h2 style="margin:0; font-size:1rem; color:white; font-family:'Space Grotesk';">Lumina OS</h2>
                <p style="margin:0; font-size:0.6rem; color:#888; text-transform:uppercase;">Core V2.4</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Drop high-dimensional datasets", type=None)

    if uploaded_file:
        if st.session_state.filename != uploaded_file.name:
            with st.spinner(f"Synthesizing {uploaded_file.name}..."):
                text, df = extract_content(uploaded_file)
                st.session_state.uploaded_text = text
                st.session_state.uploaded_df = df
                st.session_state.filename = uploaded_file.name
                
                v, m, c = build_tfidf_index(text)
                st.session_state.vectorizer = v
                st.session_state.tfidf_matrix = m
                st.session_state.chunks = c
                
                st.session_state.pulse = auto_generate_pulse(df, c)
                st.session_state.suggestions = generate_suggestions(text, uploaded_file.name, df is not None)
                st.rerun()

# 3. Dynamic Content Hub
if not st.session_state.filename:
    html = load_template("landing")
    st.markdown(html, unsafe_allow_html=True)
    
    st.markdown('<div style="margin-top:-150px;"></div>', unsafe_allow_html=True) 
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.button("Initialize Chat", key="btn_chat", on_click=lambda: st.switch_page("pages/ask_ai.py"), use_container_width=True)
    with c2: st.button("Open Explorer", key="btn_prev", on_click=lambda: st.switch_page("pages/preview.py"), use_container_width=True)
    with c3: st.button("Execute Engine", key="btn_logic", on_click=lambda: st.switch_page("pages/calculator.py"), use_container_width=True)
    with c4: st.button("View Metrics", key="btn_dash", on_click=lambda: st.switch_page("pages/dashboard.py"), use_container_width=True)

else:
    st.markdown('<div class="glass-card" style="border-left: 2px solid #bb00ff; padding:40px;">', unsafe_allow_html=True)
    st.markdown(f'<h3 style="color:#bb00ff; margin-bottom:15px;">🔮 INTELLIGENCE PULSE</h3>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:1.1rem; color:#ccc;">{st.session_state.get("pulse", "Generating elite insights...")}</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)
    
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)
    with nav_col1:
        if st.button("💬 Ask AI", key="nav_chat", use_container_width=True): st.switch_page("pages/ask_ai.py")
    with nav_col2:
        if st.button("📊 Analytics", key="nav_dash", use_container_width=True): st.switch_page("pages/dashboard.py")
    with nav_col3:
        if st.button("⚡ Logic", key="nav_calc", use_container_width=True): st.switch_page("pages/calculator.py")
    with nav_col4:
        if st.button("👀 Audit", key="nav_prev", use_container_width=True): st.switch_page("pages/preview.py")
