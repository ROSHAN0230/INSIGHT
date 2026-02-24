import streamlit as st
import pandas as pd
from groq import Groq

# 1. PAGE SETUP
st.set_page_config(page_title="InsightX | Universal AI", layout="wide")
st.title("🚀 InsightX: Universal Data Engine")

# 2. SECRETS CHECK
if "GROQ_API_KEY" not in st.secrets:
    st.error("Please add GROQ_API_KEY to your Streamlit Secrets!")
    st.stop()

client = Groq(api_key="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")
# 3. DYNAMIC DATA LOADING (Memory Optimized)
@st.cache_data
def load_data(uploaded_file):
    try:
        # SAFETY VALVE: Only load first 100k rows to stay within 1GB RAM limit
        df = pd.read_csv(uploaded_file, nrows=100000) 
        
        # CLEANING: Detect and fix numeric columns
        for col in df.columns:
            if any(key in col.lower() for key in ['amount', 'price', 'value', 'salary']):
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
        
        # SHRINKING: Use categories for repeated text (States, Cities, etc.)
        for col in df.select_dtypes('object'):
            if df[col].nunique() < 100:
                df[col] = df[col].astype('category')
                
        return df
    except Exception as e:
        st.error(f"Upload Error: {e}")
        return None

# SIDEBAR
uploaded_file = st.sidebar.file_uploader("Upload any CSV (Up to 1GB)", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)
    st.sidebar.success(f"✅ Data Ready: {len(df):,} rows")
    
    # 4. CHAT SYSTEM
    user_query = st.text_input("Analyze your data (e.g., 'Chart the average amount by category')")

    if user_query:
        with st.spinner("Analyzing..."):
            system_prompt = f"""
            You are a Data Scientist. Dataset columns: {list(df.columns)}.
            Write Python code using streamlit and pandas. Variable name is 'df'.
            Use st.bar_chart, st.line_chart, or st.write.
            Only return the code. No backticks or 'python' text.
            """
            
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_query}]
            )
            
            # RUN THE CODE
            exec(response.choices[0].message.content)

    with st.expander("Preview Data"):
        st.dataframe(df.head(50))
else:
    st.info("👋 Upload a CSV file to begin.")
