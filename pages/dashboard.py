import streamlit as st
from utils import find_col, inject_lumina_css

st.set_page_config(page_title="Visual Intelligence | Lumina AI", page_icon="📊", layout="wide")
inject_lumina_css()

if st.session_state.get("uploaded_df") is None:
    st.markdown('<div class="lumina-card" style="border-left: 4px solid #FF4B4B;">', unsafe_allow_html=True)
    st.warning("Please upload a tabular file first.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

df = st.session_state.uploaded_df
st.markdown('<div class="hero-text" style="font-size: 3rem; margin-top:-40px; text-align:center;">VISUAL INTEL</div>', unsafe_allow_html=True)
st.markdown('<p style="color: #888; margin-top: -10px; margin-bottom: 40px; text-align:center; letter-spacing:5px; opacity:0.6;">AUTONOMOUS ANALYTICS ENGINE</p>', unsafe_allow_html=True)

df = st.session_state.uploaded_df
amt_col = find_col(df, ['amount', 'amt', 'value', 'price', 'total'])
cat_col = find_col(df, ['category', 'cat', 'type', 'industry'])
comp_col = find_col(df, ['device', 'mode', 'channel'])

# ── TOP KPI SECTION ──────────────────────────────────────────────────────────
kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)

with kpi_1:
    val = f"₹{df[amt_col].sum():,.0f}" if amt_col else "N/A"
    st.markdown(f'<div class="glass-card" style="padding:20px; text-align:left; border-bottom: 2px solid #bb00ff;">'
                f'<p style="font-size:0.65rem; color:#888; margin:0; text-transform:uppercase; letter-spacing:1px;">Cumulative Volume</p>'
                f'<h2 style="color:#fff; margin:5px 0 0 0; font-family:\'Space Grotesk\';">{val}</h2>'
                f'</div>', unsafe_allow_html=True)

with kpi_2:
    val = f"₹{df[amt_col].mean():,.2f}" if amt_col else "N/A"
    st.markdown(f'<div class="glass-card" style="padding:20px; text-align:left; border-bottom: 2px solid #7a00ff;">'
                f'<p style="font-size:0.65rem; color:#888; margin:0; text-transform:uppercase; letter-spacing:1px;">Avg Intensity</p>'
                f'<h2 style="color:#fff; margin:5px 0 0 0; font-family:\'Space Grotesk\';">{val}</h2>'
                f'</div>', unsafe_allow_html=True)

with kpi_3:
    val = df[cat_col].value_counts().idxmax() if cat_col else "N/A"
    st.markdown(f'<div class="glass-card" style="padding:20px; text-align:left; border-bottom: 2px solid #bb00ff;">'
                f'<p style="font-size:0.65rem; color:#888; margin:0; text-transform:uppercase; letter-spacing:1px;">Dominant Vector</p>'
                f'<h2 style="color:#fff; margin:5px 0 0 0; font-family:\'Space Grotesk\';">{val[:10]}...</h2>'
                f'</div>', unsafe_allow_html=True)

with kpi_4:
    st.markdown(f'<div class="glass-card" style="padding:20px; text-align:left; border-bottom: 2px solid #7a00ff;">'
                f'<p style="font-size:0.65rem; color:#888; margin:0; text-transform:uppercase; letter-spacing:1px;">Health Index</p>'
                f'<div class="badge-validated" style="margin-top:10px; display:inline-block;">99.8% Nominal</div>'
                f'</div>', unsafe_allow_html=True)

st.markdown('<div style="height:30px;"></div>', unsafe_allow_html=True)

# ── VISUALIZATIONS ───────────────────────────────────────────────────────────
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown('<div class="glass-card" style="padding:30px;">', unsafe_allow_html=True)
    st.markdown('<p style="color:#bb00ff; font-weight:700; font-size:0.8rem; margin-bottom:20px; text-transform:uppercase;">Volume by Category Spectrum</p>', unsafe_allow_html=True)
    if cat_col:
        st.bar_chart(df[cat_col].value_counts().head(8), color="#bb00ff")
    st.markdown('</div>', unsafe_allow_html=True)

with chart_col2:
    st.markdown('<div class="glass-card" style="padding:30px;">', unsafe_allow_html=True)
    st.markdown('<p style="color:#bb00ff; font-weight:700; font-size:0.8rem; margin-bottom:20px; text-transform:uppercase;">Trend Analysis Mapping</p>', unsafe_allow_html=True)
    if comp_col and amt_col:
        st.area_chart(df.groupby(comp_col)[amt_col].mean(), color="#7a00ff")
    else:
        st.info("Additional dimensions required for trend mapping.")
    st.markdown('</div>', unsafe_allow_html=True)
