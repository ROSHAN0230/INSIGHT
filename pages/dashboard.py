import streamlit as st
import pandas as pd
from utils import find_col

st.set_page_config(page_title="Dashboard | InsightX", page_icon="📊", layout="wide")

if st.session_state.get("uploaded_df") is None:
    st.warning("Please upload a tabular file first.")
    st.stop()

df = st.session_state.uploaded_df
st.title("🏆 Leadership Insights Dashboard")

# Metrics
c1, c2, c3 = st.columns(3)
amt_col = find_col(df, ['amount', 'amt', 'value', 'price', 'total'])
if amt_col:
    c1.metric("Total Volume", f"₹{df[amt_col].sum():,.0f}")
    c2.metric("Avg Transaction", f"₹{df[amt_col].mean():,.2f}")

cat_col = find_col(df, ['category', 'cat', 'type', 'industry'])
if cat_col:
    c3.metric("Top Category", df[cat_col].value_counts().idxmax())
    st.bar_chart(df[cat_col].value_counts().head(5))

# Comparison
comp_col = find_col(df, ['device', 'mode', 'channel'])
if comp_col and amt_col:
    st.write(f"**Value by {comp_col}**")
    st.bar_chart(df.groupby(comp_col)[amt_col].mean())
