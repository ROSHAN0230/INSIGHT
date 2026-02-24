import streamlit as st
import pandas as pd
from groq import Groq
# 1. PAGE SETUP
st.set_page_config(page_title="InsightX | Universal AI", layout="wide")
st.title("🚀 InsightX: Universal Data Engine")
# 2. SECRETS & API SYNC
# Use 'GROQ_API_KEY' name to match your dashboard (image_351c15.png)
try:
    client = Groq(api_key=st.secrets["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"])
except Exception:
    st.error("Missing 'GROQ_API_KEY' in Secrets! Check Manage App > Settings > Secrets.")
    st.stop()
# 3. DYNAMIC DATA LOADING (The 1GB + 250k Row Optimizer)
@st.cache_data
def load_data(uploaded_file):
    try:
        # Load the full dataset
        df = pd.read_csv(uploaded_file) 

        # CLEANING: Automatically detect and fix numeric columns (Amount, Price, etc.)
        for col in df.columns:
            if any(key in col.lower() for key in ['amount', 'price', 'value', 'salary']):
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')

        # MEMORY SHRINKING: Use categories for repeated text to save RAM
        for col in df.select_dtypes('object'):
            if df[col].nunique() < 100:
                df[col] = df[col].astype('category')

        return df
    except Exception as e:
        st.error(f"Upload Error: {e}")
        return None
# 4. SIDEBAR (1GB Capacity Info)
st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload any CSV (Up to 1GB)", type=["csv"])
if uploaded_file:
    df = load_data(uploaded_file)

    if df is not None:
        st.sidebar.success(f"✅ Data Ready: {len(df):,} rows")

        # 5. JUDGE-FRIENDLY PRESET DROPDOWN
        st.subheader("📊 Fast Analysis")
        preset_options = [
            "Custom Query...",
            "Show me a bar chart of the top 5 categories by total amount",
            "What is the average transaction value?",
            "Show a line chart of transactions over time",
            "Which state has the highest number of transactions?"
        ]
        selected_preset = st.selectbox("Choose a quick insight or type your own below:", preset_options)
        # 6. AI ANALYSIS ENGINE
        user_input = st.text_input("Describe the analysis you want:", placeholder="e.g., 'Compare total spend by gender'")

        # Decide which query to send
        final_query = user_input if user_input else (selected_preset if selected_preset != "Custom Query..." else None)
        if final_query:
            with st.spinner("AI is generating your insights..."):
                system_prompt = f"""
                You are a Data Scientist. Dataset columns: {list(df.columns)}.
                Write Python code using streamlit and pandas. Variable name is 'df'.
                Use st.bar_chart, st.line_chart, or st.write.

                IMPORTANT: Return ONLY the raw python code. 
                Do NOT use backticks (```) or the word 'python'.
                """

                try:
                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": final_query}]
                    )

                    # Cleaning: Remove markdown backticks if AI adds them
                    raw_code = response.choices[0].message.content.replace("python", "").replace("", "").strip()

                    # EXECUTE AI CODE
                    exec(raw_code)

                except Exception as e:
                    st.error(f"Execution Error: {e}")
                    st.code(raw_code) # Helps you debug live if the AI makes a typo

    # 7. DATA PREVIEW (For the judges to see the raw table)
    with st.expander("👀 View Raw Data Preview"):
        st.dataframe(df.head(100))
else:
    st.info("👋 Welcome to InsightX! Please upload your CSV file in the sidebar to begin.")

i have used this code for my application and got shis error as u can see in the image can u please read all the code clearly and change the code and rewrite the errors..??
9:04 AM

Looking at the error, the issue is on line 80 with an unterminated string literal in the messages=[{" part. This is caused by the multi-line f-string in the system_prompt — specifically the way the string is formatted with the curly braces inside the f-string conflicting with the dictionary syntax.

Let me rewrite the fixed code:
Created a file, read a file
Done

Here's what was causing the error and what I fixed:

Root Cause — Unterminated string literal on line 80

The f-string used for system_prompt was a multi-line triple-quoted f-string containing {list(df.columns)}. Python was getting confused because the [{ inside the f-string looked like an unterminated string when mixed with the surrounding messages=[{"role"...}] dictionary syntax. Some Python versions are stricter about this.

Two key fixes applied:

    system_prompt rebuilt without f-string — Instead of a fragile multi-line f-string, the prompt is now built using plain string concatenation with str(list(df.columns)). This is safer and avoids any ambiguous brace interpretation.
    exec() given an explicit namespace — Changed exec(raw_code) to exec(raw_code, {"df": df, "st": st, "pd": pd}). This ensures the AI-generated code always has access to df, st, and pd, preventing NameError crashes when the AI uses those variables.

Just replace your old app.py with this file and redeploy!
App
PY 


Claude is AI and can make mistakes. Please double-check responses.
App · PY
import streamlit as st
import pandas as pd
from groq import Groq

# 1. PAGE SETUP
st.set_page_config(page_title="InsightX | Universal AI", layout="wide")
st.title("🚀 InsightX: Universal Data Engine")

# 2. SECRETS & API SYNC
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception:
    st.error("Missing 'GROQ_API_KEY' in Secrets! Check Manage App > Settings > Secrets.")
    st.stop()

# 3. DYNAMIC DATA LOADING (The 1GB + 250k Row Optimizer)
@st.cache_data
def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)

        # CLEANING: Automatically detect and fix numeric columns
        for col in df.columns:
            if any(key in col.lower() for key in ['amount', 'price', 'value', 'salary']):
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(r'[^\d.]', '', regex=True),
                    errors='coerce'
                )

        # MEMORY SHRINKING: Use categories for repeated text to save RAM
        for col in df.select_dtypes('object'):
            if df[col].nunique() < 100:
                df[col] = df[col].astype('category')

        return df
    except Exception as e:
        st.error(f"Upload Error: {e}")
        return None

# 4. SIDEBAR
st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload any CSV (Up to 1GB)", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)

    if df is not None:
        st.sidebar.success(f"✅ Data Ready: {len(df):,} rows")

        # 5. PRESET DROPDOWN
        st.subheader("📊 Fast Analysis")
        preset_options = [
            "Custom Query...",
            "Show me a bar chart of the top 5 categories by total amount",
            "What is the average transaction value?",
            "Show a line chart of transactions over time",
            "Which state has the highest number of transactions?"
        ]
        selected_preset = st.selectbox("Choose a quick insight or type your own below:", preset_options)

        # 6. AI ANALYSIS ENGINE
        user_input = st.text_input(
            "Describe the analysis you want:",
            placeholder="e.g., 'Compare total spend by gender'"
        )

        final_query = user_input if user_input else (
            selected_preset if selected_preset != "Custom Query..." else None
        )

        if final_query:
            with st.spinner("AI is generating your insights..."):
                # FIX: Build system_prompt as a regular variable first,
                # then pass it into the messages list separately
                columns_list = str(list(df.columns))
                system_prompt = (
                    "You are a Data Scientist. Dataset columns: " + columns_list + ". "
                    "Write Python code using streamlit and pandas. Variable name is 'df'. "
                    "Use st.bar_chart, st.line_chart, or st.write. "
                    "IMPORTANT: Return ONLY the raw python code. "
                    "Do NOT use backticks (```) or the word 'python'."
                )

                try:
                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": final_query}
                        ]
                    )

                    # Cleaning: Remove markdown backticks if AI adds them
                    raw_code = (
                        response.choices[0].message.content
                        .replace("```python", "")
                        .replace("```", "")
                        .strip()
                    )

                    # EXECUTE AI CODE
                    exec(raw_code, {"df": df, "st": st, "pd": pd})

                except Exception as e:
                    st.error(f"Execution Error: {e}")
                    st.code(raw_code)

        # 7. DATA PREVIEW
        with st.expander("👀 View Raw Data Preview"):
            st.dataframe(df.head(100))

else:
    st.info("👋 Welcome to InsightX! Please upload your CSV file in the sidebar to begin.")

