import streamlit as st
from utils import run_code, inject_lumina_css

st.set_page_config(page_title="Logic Engine | Lumina AI", page_icon="⚡", layout="wide")
inject_lumina_css()

if st.session_state.get("uploaded_df") is None:
    st.markdown('<div class="lumina-card" style="border-left: 4px solid #FF4B4B;">', unsafe_allow_html=True)
    st.warning("Please upload a CSV or Excel file first on the Home page.")
    if st.button("Go Home"): st.switch_page("app.py")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

st.markdown(f'<div class="hero-text" style="font-size: 3rem; margin-top:-40px; text-align:center;">LOGIC ENGINE</div>', unsafe_allow_html=True)
st.markdown(f'<p style="color: #888; margin-top: -10px; margin-bottom: 40px; text-align:center; letter-spacing:2px;">CODE-DRIVEN INTELLIGENCE: {st.session_state.filename}</p>', unsafe_allow_html=True)

if "calc_history" not in st.session_state: st.session_state.calc_history = []

# Main Layout: Code Workspace
editor_col, result_col = st.columns([1.8, 1])

with editor_col:
    st.markdown('<div class="glass-card" style="padding:40px; background:rgba(0,0,0,0.2);">', unsafe_allow_html=True)
    st.markdown('<div style="display:flex; align-items:center; gap:10px; margin-bottom:20px;">'
                '<div style="width:12px; height:12px; border-radius:50%; background:#ff5f56;"></div>'
                '<div style="width:12px; height:12px; border-radius:50%; background:#ffbd2e;"></div>'
                '<div style="width:12px; height:12px; border-radius:50%; background:#27c93f;"></div>'
                '<span style="margin-left:10px; font-size:0.7rem; color:#666; font-family:\'Space Grotesk\';">python_engine_v2.py</span></div>', unsafe_allow_html=True)
    
    q = st.text_input("Injection Prompt (e.g., 'sum amount by territory')", placeholder="Enter logic parameters...")
    
    if st.button("EXECUTE NEURAL LOGIC", key="run_logic", use_container_width=True):
        if q:
            with st.spinner("Processing High-Dimensional Calculation..."):
                code, err = run_code(q, st.session_state.uploaded_df, st.session_state.calc_history)
                if err:
                    st.error(f"Execution Error: {err}")
                else:
                    st.session_state.calc_history.append({"question": q, "code": code})
                    st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with result_col:
    st.markdown('<h4 style="color:#bb00ff; font-family:\'Space Grotesk\'; letter-spacing:1px; margin-bottom:20px;">⚡ EXECUTION CONSOLE</h4>', unsafe_allow_html=True)
    
    if not st.session_state.calc_history:
        st.markdown('<div class="glass-card" style="padding:40px; text-align:center; color:#555;">'
                    '<p style="font-size:0.8rem; text-transform:uppercase; letter-spacing:1px;">Awaiting Injection...</p></div>', unsafe_allow_html=True)
    
    for calc in reversed(st.session_state.calc_history):
        with st.container():
            st.markdown(f'<div class="glass-card" style="padding:20px; border-left: 2px solid #bb00ff; text-align:left;">'
                        f'<p style="font-size:0.75rem; color:#888; font-weight:700; margin-bottom:5px;">Q: {calc["question"]}</p>'
                        f'</div>', unsafe_allow_html=True)
            with st.expander("VIEW SOURCE CODE"):
                st.code(calc["code"], language="python")
