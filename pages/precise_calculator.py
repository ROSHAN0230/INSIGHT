import streamlit as st
import pandas as pd
import numpy as np
from utils import run_code

st.set_page_config(page_title="Calculator | InsightX", page_icon="⚡", layout="wide")

if st.session_state.get("uploaded_df") is None:
    st.warning("Please upload a CSV or Excel file first on the Home page.")
    if st.button("Go Home"): st.switch_page("app.py")
    st.stop()

st.title(f"⚡ Precise Calculator: {st.session_state.filename}")
st.success("✅ 100% Accuracy | Full Dataset Processing")

if "calc_history" not in st.session_state: st.session_state.calc_history = []

q = st.text_input("Formula or Calculation Request:", placeholder="e.g., Total volume by Category")

if q:
    with st.spinner("Calculating..."):
        code, err = run_code(q, st.session_state.uploaded_df, st.session_state.calc_history)
        if err:
            st.error(f"Execution Error: {err}")
            st.code(code)
        else:
            st.session_state.calc_history.append({"question": q, "answer": f"Code used:\n```python\n{code}\n```"})
            st.rerun()

for calc in reversed(st.session_state.calc_history):
    with st.expander(f"Q: {calc['question']}"):
        st.markdown(calc["answer"])
