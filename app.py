import streamlit as st
import pandas as pd
from groq import Groq
import io
import re
from contextlib import redirect_stdout

# 1. SETUP & CLIENT
st.set_page_config(page_title="InsightX Groq Edition", layout="wide")
st.title("⚡ InsightX: High-Speed Analytics")

# Replace with your key - No 404s here!
GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

# 2. DATA LOADING
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("transactions.csv")
        # Quick clean for currency/nan
        for col in df.columns:
            if 'amount' in col.lower() or 'value' in col.lower():
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0)
        return df
    except:
        st.error("Please ensure 'transactions.csv' is in the folder!")
        return None

df = load_data()

# 3. ANALYSIS FUNCTION
# 3. ANALYSIS FUNCTION (Upgraded for Visuals)
def groq_analytic_agent(query, columns):
    # System instruction forces the AI to use Streamlit's visual capabilities
    system_instruction = """
    You are a expert data analyst. 
    1. If the user's request can be visualized (trends, comparisons, distributions), ALWAYS use Streamlit charts like st.bar_chart(), st.line_chart(), or st.area_chart().
    2. Use st.write() to explain the data.
    3. If it's a simple calculation, use print() as a fallback.
    4. The dataframe is already loaded as 'df'.
    5. ONLY output the Python code inside a single ```python block.
    """
    
    prompt = f"{system_instruction}\n\nData columns: {columns}\nUser Question: {query}"
    
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    
    code_text = completion.choices[0].message.content
    match = re.search(r"```python\n(.*?)\n```", code_text, re.DOTALL)
    return match.group(1) if match else code_text

# 4. UI
# 4. UI (Updated to handle direct Streamlit rendering)
if user_input := st.chat_input("Ask about your data..."):
    with st.chat_message("user"): st.write(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("InsightX is generating visuals..."):
            code = groq_analytic_agent(user_input, df.columns.tolist())
            
            # Create a buffer for any text that uses print()
            f = io.StringIO()
            with redirect_stdout(f):
                try:
                    # Executing the code. If the code contains st.bar_chart(), 
                    # it will appear right here in the UI automatically.
                    exec(code) 
                    
                    # If the AI also used print() for a summary, show it here:
                    output = f.getvalue()
                    if output:
                        st.info(output)
                        
                except Exception as e:
                    st.error(f"Execution Error: {e}")
                    st.code(code)