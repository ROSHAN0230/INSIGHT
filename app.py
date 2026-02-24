import streamlit as st
import pandas as pd
import re
import os
from groq import Groq

# PAGE SETUP
st.set_page_config(page_title="InsightX | Universal AI", layout="wide")
st.title("🚀 InsightX: Universal Data Engine")

# API KEY SETUP
# >>> PASTE YOUR GROQ API KEY BETWEEN THE QUOTES BELOW <<<
HARDCODED_KEY = "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"

api_key = ""
try:
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

if not api_key:
    api_key = os.environ.get("GROQ_API_KEY", "")

if not api_key:
    api_key = HARDCODED_KEY

if not api_key or api_key == "PASTE_YOUR_GROQ_API_KEY_HERE":
    st.error("Please paste your Groq API key into line 13 of app.py")
    st.stop()

client = Groq(api_key=api_key)

# DATA LOADING
@st.cache_data
def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        for col in df.columns:
            if any(k in col.lower() for k in ["amount", "price", "value", "salary"]):
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(r"[^\d.]", "", regex=True),
                    errors="coerce",
                )
        for col in df.select_dtypes("object"):
            if df[col].nunique() < 100:
                df[col] = df[col].astype("category")
        return df
    except Exception as e:
        st.error(f"Upload Error: {e}")
        return None

# SIDEBAR
st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload any CSV (Up to 1GB)", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)

    if df is not None:
        st.sidebar.success(f"Data Ready: {len(df):,} rows")

        st.subheader("📊 Fast Analysis")
        preset_options = [
            "Custom Query...",
            "Show me a bar chart of the top 5 categories by total amount",
            "What is the average transaction value?",
            "Show a line chart of transactions over time",
            "Which state has the highest number of transactions?",
        ]
        selected_preset = st.selectbox(
            "Choose a quick insight or type your own below:", preset_options
        )

        user_input = st.text_input(
            "Describe the analysis you want:",
            placeholder="e.g., Compare total spend by gender",
        )

        final_query = user_input if user_input else (
            selected_preset if selected_preset != "Custom Query..." else None
        )

        if final_query:
            with st.spinner("AI is generating your insights..."):
                columns_list = str(list(df.columns))
                system_prompt = (
                    "You are a Data Scientist. Dataset columns: "
                    + columns_list
                    + ". Write Python code using streamlit and pandas. "
                    "The dataframe variable is called df. "
                    "Use st.bar_chart, st.line_chart, or st.write for output. "
                    "Return ONLY raw executable Python code. "
                    "No backticks, no markdown, no explanations."
                )

                raw_code = ""
                clean_code = ""
                try:
                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": final_query},
                        ],
                    )

                    raw_code = (
                        response.choices[0].message.content
                        .replace("```python", "")
                        .replace("```", "")
                        .strip()
                    )

                    cleaned_lines = []
                    for line in raw_code.splitlines():
                        stripped = line.strip()
                        if re.match(r"^\d{1,2}:\d{2}\s*(AM|PM)?$", stripped, re.IGNORECASE):
                            continue
                        cleaned_lines.append(line)
                    clean_code = "\n".join(cleaned_lines).strip()

                    exec(clean_code, {"df": df, "st": st, "pd": pd})

                except Exception as e:
                    st.error(f"Execution Error: {e}")
                    if clean_code:
                        st.code(clean_code)
                    elif raw_code:
                        st.code(raw_code)

        with st.expander("👀 View Raw Data Preview"):
            st.dataframe(df.head(100))

else:
    st.info("👋 Welcome to InsightX! Upload your CSV file in the sidebar to begin.")
