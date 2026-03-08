import streamlit as st
from utils import ask_ai, retrieve_chunks, inject_lumina_css

st.set_page_config(page_title="Ask AI | Lumina AI", page_icon="💬", layout="wide")
inject_lumina_css()

if not st.session_state.get("filename"):
    st.markdown('<div class="lumina-card" style="border-left: 4px solid #FF4B4B;">', unsafe_allow_html=True)
    st.warning("Please upload a file first on the Home page.")
    if st.button("Go Home"): st.switch_page("app.py")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

st.markdown(f'<div class="hero-text" style="font-size: 3rem; margin-top:-40px; text-align:center;">NEURAL CHAT</div>', unsafe_allow_html=True)
st.markdown(f'<p style="color: #888; margin-top: -10px; margin-bottom: 40px; text-align:center; letter-spacing:2px;">SYNTHESIZING: {st.session_state.filename}</p>', unsafe_allow_html=True)

if "chat_history" not in st.session_state: st.session_state.chat_history = []

# Main Layout: Centered Chat with Knowledge Sidebar
chat_col, side_col = st.columns([2.5, 1])

with side_col:
    st.markdown('<div class="glass-card" style="padding:25px; min-height:600px; background:rgba(187,0,255,0.01); border-left: 1px solid rgba(187,0,255,0.2);">', unsafe_allow_html=True)
    st.markdown('<h4 style="color:#bb00ff; font-family:\'Space Grotesk\'; letter-spacing:1px; margin-bottom:20px;">📦 KNOWLEDGE NODES</h4>', unsafe_allow_html=True)
    
    # Quick suggestions (moved to sidebar nodes)
    suggestions = st.session_state.get("suggestions", [])
    if suggestions:
        for i, q in enumerate(suggestions[:5]):
            if st.button(f"Node {(i+1):02d}: {q[:25]}...", key=f"sug_{i}", use_container_width=True):
                st.session_state.pending_q = q
                st.rerun()
    
    st.markdown('<div style="margin-top:40px; border-top:1px solid rgba(255,255,255,0.05); padding-top:20px;">'
                '<p style="font-size:0.6rem; color:#555; text-transform:uppercase;">Context Density</p>'
                '<div style="height:4px; background:rgba(187,0,255,0.1); border-radius:2px; margin-top:8px;">'
                '<div style="width:70%; height:100%; background:#bb00ff; border-radius:2px; box-shadow:0 0 10px #bb00ff;"></div></div>'
                '</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with chat_col:
    # Thinking Indicator logic
    final_q = st.chat_input("Inject Query to Lumina AI...") or st.session_state.get("pending_q")
    
    if final_q:
        st.session_state.pending_q = None
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("""
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:30px;">
            <div style="width:40px; height:40px; border-radius:50%; background:#1a1a1e; border:1px solid rgba(187,0,255,0.3); display:flex; align-items:center; justify-content:center; color:#bb00ff;">
                <span class="material-symbols-outlined" style="font-size:22px; animation: pulse 1.5s infinite alternate;">bolt</span>
            </div>
            <div class="glass-card" style="padding:10px 25px; border-radius:999px; display:flex; align-items:center; gap:12px; border:1px solid rgba(187,0,255,0.1);">
                <div class="badge-validated" style="background:transparent; border:none; box-shadow:none; padding:0;">SYNTHESIZING</div>
                <div style="display:flex; gap:5px;">
                    <span style="width:4px; height:4px; background:#bb00ff; border-radius:50%;"></span>
                    <span style="width:4px; height:4px; background:#bb00ff; border-radius:50%; opacity:0.5;"></span>
                    <span style="width:4px; height:4px; background:#bb00ff; border-radius:50%; opacity:0.2;"></span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        chunks = retrieve_chunks(final_q, st.session_state.vectorizer, st.session_state.tfidf_matrix, st.session_state.chunks)
        ans = ask_ai(final_q, chunks, st.session_state.chat_history)
        st.session_state.chat_history.append({"question": final_q, "answer": ans})
        thinking_placeholder.empty()

    # Reversed chat history for modern flow
    for chat in reversed(st.session_state.chat_history):
        # User
        st.markdown(f'<div class="user-bubble">{chat["question"]}</div>', unsafe_allow_html=True)
        
        # AI
        st.markdown(f"""
        <div style="display:flex; gap:20px; margin-bottom:35px;">
            <div style="width:45px; height:45px; border-radius:50%; background:linear-gradient(135deg, #bb00ff, #7a00ff); display:flex; align-items:center; justify-content:center; color:white; flex-shrink:0; box-shadow: 0 0 20px rgba(187,0,255,0.4);">
                <span class="material-symbols-outlined" style="font-size:22px;">bolt</span>
            </div>
            <div class="ai-bubble" style="flex:1;">
                <p style="margin:0;">{chat["answer"]}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
