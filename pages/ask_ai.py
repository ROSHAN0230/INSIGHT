import streamlit as st
from utils import ask_ai, retrieve_chunks, inject_lumina_css, load_template

st.set_page_config(page_title="Ask AI", layout="wide")
inject_lumina_css()

if "chat_history" not in st.session_state: st.session_state.chat_history = []
chat_html = ""
for c in st.session_state.chat_history:
    chat_html += f'<div style="margin:10px; padding:15px; background:rgba(255,255,255,0.05); border-radius:15px;"><b>User:</b> {c["question"]}<br><b>AI:</b> {c["answer"]}</div>'

st.markdown(load_template("ask_ai", {"FILENAME": st.session_state.filename, "CHAT_HISTORY": chat_html}), unsafe_allow_html=True)

q = st.chat_input("Ask Lumina...")
if q:
    chunks = retrieve_chunks(q, st.session_state.vectorizer, st.session_state.tfidf_matrix, st.session_state.chunks)
    ans = ask_ai(q, chunks, st.session_state.chat_history)
    st.session_state.chat_history.append({"question": q, "answer": ans})
    st.rerun()
