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

# Use the secret name for security, or keep your direct key if you prefer
client = Groq(api_key=st.secrets["AIzaSyDBLWRjQPhdwKWIPU39eJa9921jJAFg7m4"])

# 3. DYNAMIC DATA LOADING (Memory Optimized)
@st.cache_data
def load_data(uploaded_file):
    try:
        # Load full file (Handling your 250k rows)
        df = pd.read_csv(uploaded_file) 
        
        # CLEANING: Detect and fix numeric columns
        for col in df.columns:
            if any(key in col.lower() for key in ['amount', 'price', 'value', 'salary']):
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
        
        # SHRINKING: Use categories for repeated text
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
    # Use the optimized load_data function
    df = load_data(uploaded_file)
    
    if df is not None:
        st.sidebar.success(f"✅ Data Ready: {len(df):,} rows")
        
        # 5. CHAT SYSTEM (Inside the 'if uploaded_file' block)
        user_query = st.text_input("Analyze your data (e.g., 'Chart the average amount by category')")

        if user_query:
            with st.spinner("Analyzing..."):
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
                        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_query}]
                    )
                    
                    # Clean the response to strip any backticks the AI might add
                    raw_code = response.choices[0].message.content.replace("```python", "").replace("```", "").strip()
                    
                    # RUN THE CODE
                    exec(raw_code)
                    
                except Exception as e:
                    st.error(f"Execution Error: {e}")
                    # If it fails, show the code so you can debug during the demo
                    st.code(raw_code) 
    
    # Optional: Preview the data
    with st.expander("Preview Dataset"):
        st.dataframe(df.head(100))

else:
    st.info("👋 Welcome! Please upload a CSV file in the sidebar to begin your analysis.")
