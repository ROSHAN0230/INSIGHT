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
                        messages=[{"
