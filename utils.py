import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import re
import json
import time
import zipfile
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIG ---
MAX_WORDS = 30000
TOP_K = 3
CONTEXT_CAP = 4000

# --- API SETUP ---
def get_client():
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if api_key: return Groq(api_key=api_key)
    return None

client = get_client()

# --- UI ENGINE ---
def load_template(name, mappings=None):
    path = os.path.join(os.path.dirname(__file__), "templates", f"{name}.html")
    if not os.path.exists(path): return f"<p>Template {name} not found.</p>"
    with open(path, "r", encoding="utf-8") as f: html = f.read()
    if mappings:
        for k, v in mappings.items(): html = html.replace(f"{{{{ {k} }}}}", str(v))
    return html

def inject_lumina_css():
    st.markdown(\"\"\"
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0" />
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
        .stApp { background-color: #0a0a0c !important; color: #f1f5f9 !important; }
        header, footer { visibility: hidden; }
        #MainMenu { visibility: hidden; }
        .hero-text { font-family: 'Space Grotesk', sans-serif !important; }
    </style>
    \"\"\", unsafe_allow_html=True)

def find_col(df, keywords):
    for col in df.columns:
        if any(k in col.lower() for k in keywords): return col
    return df.columns[0]

# --- EXTRACTORS ---
def extract_content(uploaded_file):
    name = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()
    if name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes), low_memory=True)
        return df.to_string(), df
    if name.endswith(".pdf"):
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        text = "\\n".join([p.extract_text() or "" for p in reader.pages])
        return text, None
    return file_bytes.decode("utf-8", errors="ignore"), None

# --- AI & RAG ---
@st.cache_resource
def build_tfidf_index(text):
    if not text: return None, None, []
    words = text.split()[:MAX_WORDS]
    chunks = [" ".join(words[i:i + 400]) for i in range(0, len(words), 300)]
    v = TfidfVectorizer(max_features=2000, stop_words="english")
    m = v.fit_transform(chunks)
    return v, m, chunks

def retrieve_chunks(question, vectorizer, tfidf_matrix, chunks):
    if not vectorizer: return chunks[:TOP_K]
    q_vec = vectorizer.transform([question])
    sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
    indices = sims.argsort()[-TOP_K:][::-1]
    return [chunks[i] for i in indices if sims[i] > 0]

def ask_ai(question, chunks, history):
    if not client: return "API Key Missing"
    context = "\\n---\\n".join(chunks)[:CONTEXT_CAP]
    messages = [{"role": "system", "content": f"You are Lumina AI. Use context: {context}"}]
    messages.append({"role": "user", "content": question})
    r = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=messages)
    return r.choices[0].message.content.strip()

def run_code(question, df, history, execute=True):
    if not client: return "", "Error"
    sys_prompt = f"Generate Python for 'df' (cols: {list(df.columns)}). Output ONLY raw code."
    r = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role":"system","content":sys_prompt}, {"role":"user","content":question}])
    code = r.choices[0].message.content.replace("```python", "").replace("```", "").strip()
    if execute: exec(code, {"df": df, "st": st, "pd": pd, "np": np})
    return code, None

def auto_generate_pulse(df, chunks):
    if not client: return "Analysis ready."
    data_summary = df.head(2).to_string() if df is not None else chunks[0][:500]
    r = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role":"user","content":f"Summarize this data in 1 sentence: {data_summary}"}])
    return r.choices[0].message.content.strip()

def generate_suggestions(text, name, is_tab):
    return ["Summarize key trends", "Analyze structure", "Detect anomalies"]

def get_df_stats(df):
    return df.isnull().sum().to_dict(), df.duplicated().sum(), df.describe(include='all').transpose()

def get_csv_download(df):
    return df.to_csv(index=False).encode('utf-8')
