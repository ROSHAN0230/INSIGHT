import streamlit as st
import pandas as pd
from utils import get_df_stats, get_csv_download

st.set_page_config(page_title="Preview | InsightX", page_icon="👀", layout="wide")

if not st.session_state.get("filename"):
    st.warning("No file loaded.")
    st.stop()

st.title(f"👀 Audit: {st.session_state.filename}")

if st.session_state.uploaded_df is not None:
    miss, dup, stats = get_df_stats(st.session_state.uploaded_df)
    c1, c2 = st.columns(2)
    c1.metric("Duplicates", dup)
    c2.metric("Missing Values", sum(miss.values()))
    
    st.write("### Statistical Summary")
    st.dataframe(stats)
    
    st.write("### Raw Sample")
    st.dataframe(st.session_state.uploaded_df.head(100))
    
    st.download_button("Download Processed CSV", get_csv_download(st.session_state.uploaded_df), "data.csv")
else:
    st.write("### Extracted Text Preview")
    st.text_area("Content", st.session_state.uploaded_text[:5000], height=500)
