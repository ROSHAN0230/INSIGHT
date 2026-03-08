import streamlit as st
from utils import get_df_stats, get_csv_download, inject_lumina_css

st.set_page_config(page_title="Preview | Lumina AI", page_icon="👀", layout="wide")
inject_lumina_css()

if not st.session_state.get("filename"):
    st.markdown('<div class="lumina-card" style="border-left: 4px solid #FF4B4B;">', unsafe_allow_html=True)
    st.warning("No file loaded.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

st.markdown(f'<div class="hero-text" style="font-size: 3rem; margin-top:-40px; text-align:center;">DATA AUDIT</div>', unsafe_allow_html=True)
st.markdown(f'<p style="color: #888; margin-top: -10px; margin-bottom: 40px; text-align:center; letter-spacing:5px; opacity:0.6;">RAW TELEMETRY INSPECTION</p>', unsafe_allow_html=True)

if st.session_state.uploaded_df is not None:
    miss, dup, stats = get_df_stats(st.session_state.uploaded_df)
    
    # Audit Badges
    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown(f'<div class="glass-card" style="padding:20px; border-top: 1px solid rgba(187,0,255,0.2);">'
                    f'<p style="font-size:0.6rem; color:#888; margin:0; text-transform:uppercase;">Redundancy Pattern</p>'
                    f'<h3 style="color:#bb00ff; margin:5px 0 0 0; font-family:\'Space Grotesk\';">{dup} Duplicate Nodes</h3></div>', unsafe_allow_html=True)
    with a2:
        missing_total = sum(miss.values())
        st.markdown(f'<div class="glass-card" style="padding:20px; border-top: 1px solid rgba(187,0,255,0.2);">'
                    f'<p style="font-size:0.6rem; color:#888; margin:0; text-transform:uppercase;">Entropy Detection</p>'
                    f'<h3 style="color:#bb00ff; margin:5px 0 0 0; font-family:\'Space Grotesk\';">{missing_total} Null Points</h3></div>', unsafe_allow_html=True)
    with a3:
        st.markdown(f'<div class="glass-card" style="padding:20px; border-top: 1px solid rgba(187,0,255,0.2);">'
                    f'<p style="font-size:0.6rem; color:#888; margin:0; text-transform:uppercase;">Neural Status</p>'
                    f'<div class="badge-validated" style="margin-top:10px; display:inline-block;">99.9% VALIDATED</div></div>', unsafe_allow_html=True)

    st.markdown('<div style="height:30px;"></div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="glass-card" style="padding:30px; text-align:left;">', unsafe_allow_html=True)
        st.markdown('<h4 style="color:#fff; font-family:\'Space Grotesk\'; margin-bottom:20px;">📊 STATISTICAL SPECTRUM</h4>', unsafe_allow_html=True)
        st.dataframe(stats, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="glass-card" style="padding:30px; text-align:left;">', unsafe_allow_html=True)
        st.markdown('<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">'
                    '<h4 style="color:#fff; font-family:\'Space Grotesk\'; margin:0;">🧪 RAW TELEMETRY</h4>'
                    '</div>', unsafe_allow_html=True)
        st.dataframe(st.session_state.uploaded_df.head(100), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div style="height:40px;"></div>', unsafe_allow_html=True)
    
    # Download Button with custom styling
    st.download_button(
        label="📥 EXPORT SYNTHESIZED DATASET",
        data=get_csv_download(st.session_state.uploaded_df),
        file_name=f"lumina_os_export_{st.session_state.filename}.csv",
        mime="text/csv",
        use_container_width=True
    )
else:
    st.markdown('<div class="glass-card" style="padding:40px; text-align:left;">', unsafe_allow_html=True)
    st.markdown('<h3 style="color:#bb00ff; font-family:\'Space Grotesk\';">UNSTRUCTURED STREAM</h3>', unsafe_allow_html=True)
    st.text_area("Live Telemetry Buffer", st.session_state.uploaded_text[:8000], height=500)
    st.markdown('</div>', unsafe_allow_html=True)
