import streamlit as st
from utils import ask_ai, retrieve_chunks

st.set_page_config(page_title="Ask AI | InsightX", page_icon="💬", layout="wide")

if not st.session_state.get("filename"):
    st.warning("Please upload a file first on the Home page.")
    if st.button("Go Home"): st.switch_page("app.py")
    st.stop()

st.title(f"💬 Conversational Intel: {st.session_state.filename}")

if "chat_history" not in st.session_state: st.session_state.chat_history = []

# Quick suggestions
suggestions = st.session_state.get("suggestions", [])
if suggestions:
    st.write("💡 Suggested questions:")
    cols = st.columns(len(suggestions[:3]))
    for i, q in enumerate(suggestions[:3]):
        if cols[i].button(q, key=f"sug_{i}"):
            st.session_state.pending_q = q

# Question input
q_input = st.chat_input("Ask about your data...")
final_q = q_input or st.session_state.get("pending_q")

if final_q:
    st.session_state.pending_q = None
    with st.spinner("Analyzing..."):
        chunks = retrieve_chunks(final_q, st.session_state.vectorizer, st.session_state.tfidf_matrix, st.session_state.chunks)
        ans = ask_ai(final_q, chunks, st.session_state.chat_history)
        st.session_state.chat_history.append({"question": final_q, "answer": ans})

# Display Chat
for chat in reversed(st.session_state.chat_history):
    with st.chat_message("user"): st.write(chat["question"])
    with st.chat_message("assistant"): st.write(chat["answer"])
