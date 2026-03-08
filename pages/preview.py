import streamlit as st
from utils import get_df_stats, get_csv_download, inject_lumina_css, load_template

st.set_page_config(page_title="Audit | Lumina AI", layout="wide")
inject_lumina_css()

if not st.session_state.get("filename"):
    st.error("Audit Access Denied.")
    st.stop()

stats_str = "Neural Stream Ready"
table_html = ""

if st.session_state.uploaded_df is not None:
    miss, dup, s_df = get_df_stats(st.session_state.uploaded_df)
    stats_str = f"Rows: {len(st.session_state.uploaded_df)} | Dups: {dup}"
    table_html = st.session_state.uploaded_df.head(100).to_html(classes="min-w-full text-sm", index=False)
else:
    table_html = f"<pre>{st.session_state.uploaded_text[:5000]}</pre>"

html = load_template("preview", {"STATS": stats_str, "TABLE_CONTENT": table_html})
st.markdown(html, unsafe_allow_html=True)

if st.session_state.uploaded_df is not None:
    st.download_button("📥 EXPORT", get_csv_download(st.session_state.uploaded_df), "export.csv")
