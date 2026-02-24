import streamlit as st
import pandas as pd
from groq import Groq

# 1. PAGE SETUP
st.set_page_config(page_title="InsightX | Universal AI", layout="wide")
st.title("🚀 InsightX: Universal Data Engine")

# 2. SECRETS CHECK & CLIENT INITIALIZATION
# This MUST match the name 'GROQ_API_KEY' you saved in your dashboard
try:
    client = Groq(api_key=st.secrets["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"])
except Exception:
    st.error("Missing 'GROQ_API_KEY' in Streamlit Secrets! Check Manage App > Settings > Secrets.")
    st.stop()

# 3. DYNAMIC DATA LOADING (Memory Optimized)
@st.cache_data
def load_data(uploaded_file):
    try:
        # Load the full dataset (Handles your 250k+ rows)
        df = pd.read_csv(uploaded_file) 
        
        # CLEANING: Automatically fix currency/numeric columns
        for col in df.columns:
            if any(key in col.lower() for key in ['amount', 'price', 'value', 'salary']):
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
        
        # SHRINKING: Use categories for repeated text to save RAM
        for col in df.select_dtypes('object'):
            if df[col].nunique() < 100:
                df[col] = df[col].astype('category')
        return df
    except Exception as e:
        st.error(f"Upload Error: {e}")
        return None

# 4. SIDEBAR & FILE UPLOAD
uploaded_file = st.sidebar.file_uploader("Upload any CSV (Up to 1GB)", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)
    
    if df is not None:
        st.sidebar.success(f"✅ Data Ready: {len(df):,} rows")
        
        # 5. FAST ANALYSIS OPTIONS
        st.subheader("📊 Fast Analysis")
        preset_options = [
            "Custom Query...",
            "Show me a bar chart of the top 5 categories by total amount",
            "What is the average transaction value?",
            "Show a line chart of transactions over time",
            "Which state has the highest number of transactions?"
        ]
        selected_preset = st.selectbox("Choose a quick insight or type your own below:", preset_options)

        # 6. CHAT SYSTEM
        user_input = st.text_input("Describe the analysis you want:", placeholder="e.g., 'Compare total spend by gender'")
        
        final_query = user_input if user_input else (selected_preset if selected_preset != "Custom Query..." else None)

        if final_query:
            with st.spinner("AI is generating your insights..."):
                system_prompt = f"""
                You are a Data Scientist. Dataset columns: {list(df.columns)}.
                Write Python code using streamlit and pandas. Variable name is 'df'.
                Return ONLY the raw python code. Do NOT use backticks (```) or the word 'python'.
                """
                
                try:
                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": final_query}]
                    )
                    
                    # CLEANING: Strip backticks to prevent SyntaxErrors seen in image_3494f4.png
                    raw_code = response.choices[0].message.content.replace("```python", "").replace("```", "").strip()
                    
                    # RUN THE GENERATED CODE
                    exec(raw_code)
                except Exception as e:
                    st.error(f"Execution Error: {e}")
                    st.code(raw_code)
    
    # 7. DATA PREVIEW
    with st.expander("👀 View Raw Data Preview"):
        st.dataframe(df.head(100))
else:
    st.info("👋 Welcome! Please upload your CSV file in the sidebar to start.")
