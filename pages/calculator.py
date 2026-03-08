import streamlit as st
from utils import run_code, inject_lumina_css, load_template

st.set_page_config(page_title="Logic Engine", layout="wide")
inject_lumina_css()

st.markdown(load_template("logic_engine", {"CODE_CONTENT": "", "TERMINAL_LOG": "Ready."}), unsafe_allow_html=True)

q = st.chat_input("Logic Command...")
if q:
    run_code(q, st.session_state.uploaded_df, [])
    st.rerun()
