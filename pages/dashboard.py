import streamlit as st
from utils import find_col, inject_lumina_css, load_template

st.set_page_config(page_title="Visual Intel | Lumina AI", layout="wide")
inject_lumina_css()

if st.session_state.get("uploaded_df") is None:
    st.error("Offline.")
    st.stop()

df = st.session_state.uploaded_df
amt_col = find_col(df, ['amount', 'amt', 'value'])
cat_col = find_col(df, ['category', 'cat', 'type'])

kpi_html = f'''
    <div class="bg-primary/5 p-6 rounded-2xl border border-primary/10">
        <p class="text-[10px] text-slate-500 font-bold uppercase">Volume</p>
        <p class="text-2xl font-black">₹{df[amt_col].sum():,.0f}</p>
    </div>
'''

html = load_template("dashboard", {"KPI_CARDS": kpi_html})
st.markdown(html, unsafe_allow_html=True)

st.markdown('<div style="margin-top:-300px;"></div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1: st.bar_chart(df[cat_col].value_counts().head(8), color="#bb00ff")
with c2: st.line_chart(df[amt_col].head(100), color="#7a00ff")
