import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

st.set_page_config(
    page_title="Ultimate Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

model = pickle.load(open("car_pipeline.pkl", "rb"))
df    = pd.read_csv("used_cars_dataset_v2.csv")

df['kmDriven'] = pd.to_numeric(
    df['kmDriven'].astype(str).str.replace(',','').str.extract(r'(\d+)')[0], errors='coerce')
df['AskPrice'] = pd.to_numeric(
    df['AskPrice'].astype(str).str.replace(',','').str.extract(r'(\d+)')[0], errors='coerce')
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df = df.dropna(subset=['AskPrice','kmDriven','Year'])
df['Age'] = (2026 - df['Year']).astype(int)
df = df[df['AskPrice'] < df['AskPrice'].quantile(0.99)]
df = df[(df['AskPrice'] > 0) & (df['kmDriven'] > 0)]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;600;700&family=Poppins:wght@400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Poppins', 'Segoe UI', sans-serif !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    min-height: 100vh;
    color: #333 !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 28px 20px 48px !important;
    max-width: 960px !important;
    margin: 0 auto !important;
}
.stSidebar { display: none !important; }
div[data-testid="column"] { padding: 0 0.6rem !important; }

/* ── WHITE CARD ── */
.main-card {
    background: #ffffff;
    border-radius: 20px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.18);
    overflow: hidden;
}

/* ── HEADER ── */
.app-header {
    background: linear-gradient(45deg, #ff6b6b, #feca57);
    padding: 36px 40px;
    text-align: center;
    color: white;
}
.app-header h1 {
    font-size: 2.6rem;
    font-weight: 800;
    margin-bottom: 8px;
    text-shadow: 0 2px 8px rgba(0,0,0,0.12);
    letter-spacing: -0.01em;
}
.app-header p {
    font-size: 1.05rem;
    opacity: 0.93;
    font-weight: 500;
}
.app-header p strong { font-weight: 700; }

/* ── FORM AREA ── */
.form-area {
    padding: 36px 40px 32px;
    background: #fff;
}
.section-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #333;
    margin-bottom: 22px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── RESULT BOX ── */
.result-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 30px 24px;
    border-radius: 16px;
    text-align: center;
    margin-bottom: 20px;
}
.result-label {
    font-size: 12px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    opacity: 0.85;
    margin-bottom: 6px;
}
.result-price {
    font-size: 3.2rem;
    font-weight: 800;
    line-height: 1;
    margin: 6px 0 4px;
    letter-spacing: -0.02em;
}
.result-range {
    font-size: 13px;
    opacity: 0.85;
    margin-bottom: 18px;
}
.stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-top: 4px;
}
.stat-card {
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(8px);
    border-radius: 10px;
    padding: 12px 8px;
    text-align: center;
}
.stat-val {
    font-size: 1.4rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 3px;
}
.stat-lbl {
    font-size: 11px;
    opacity: 0.8;
    letter-spacing: 0.04em;
}

/* ── EMPTY STATE ── */
.empty-state {
    background: #f8f9ff;
    border: 2px dashed #dde1f5;
    border-radius: 14px;
    padding: 52px 24px;
    text-align: center;
    color: #bbb;
}
.empty-state .icon { font-size: 3rem; margin-bottom: 12px; }
.empty-state p { font-size: 14px; font-weight: 500; color: #c0c0c8; line-height: 1.6; }

/* ── ANALYTICS ── */
.analytics-area {
    background: #f8f9ff;
    border-top: 1px solid #eef0f8;
    padding: 32px 40px 40px;
    border-radius: 0 0 20px 20px;
}
.analytics-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #333;
    margin-bottom: 4px;
}
.analytics-sub {
    font-size: 13px;
    color: #999;
    margin-bottom: 24px;
}

/* ── BUTTON ── */
.stButton > button {
    width: 100% !important;
    padding: 20px !important;
    background: linear-gradient(45deg, #ff6b6b, #feca57) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 1.3rem !important;
    font-weight: 700 !important;
    font-family: 'Poppins', sans-serif !important;
    letter-spacing: 0.06em !important;
    box-shadow: 0 6px 20px rgba(255,107,107,0.4) !important;
    transition: all 0.3s !important;
    cursor: pointer !important;
    margin-bottom: 20px !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 30px rgba(255,107,107,0.55) !important;
}

/* ── INPUTS ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: #fff !important;
    border: 2px solid #e1e5e9 !important;
    border-radius: 12px !important;
    color: #333 !important;
    font-family: 'Poppins', sans-serif !important;
    font-size: 15px !important;
    transition: all 0.3s !important;
}
.stSelectbox > div > div:focus-within,
.stNumberInput > div > div > input:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102,126,234,0.12) !important;
}
label, .stSelectbox label, .stNumberInput label {
    font-family: 'Poppins', sans-serif !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    color: #333 !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    margin-bottom: 6px !important;
}

/* ── FOOTER ── */
.app-footer {
    text-align: center;
    padding: 18px;
    color: rgba(255,255,255,0.65);
    font-size: 13px;
    letter-spacing: 0.06em;
}
.app-footer span { color: #feca57; font-weight: 700; font-size: 15px; }
</style>
""", unsafe_allow_html=True)

# ── OPEN CARD ─────────────────────────────────────────
st.markdown('<div class="main-card">', unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>🚗 Ultimate Car Price Predictor</h1>
    <p>R²: <strong>0.90+</strong> &nbsp;|&nbsp; RMSE: <strong>₹446K</strong> &nbsp;|&nbsp; Powered by Random Forest ML</p>
</div>
""", unsafe_allow_html=True)

# ── FORM + RESULT ─────────────────────────────────────
st.markdown('<div class="form-area">', unsafe_allow_html=True)

col_left, col_right = st.columns([1.05, 0.95])

with col_left:
    st.markdown('<div class="section-title">📝 Enter Car Details</div>', unsafe_allow_html=True)
    brand        = st.selectbox("Brand",        sorted(df["Brand"].dropna().unique()))
    model_name   = st.selectbox("Model",        sorted(df["model"].dropna().unique()))
    year         = st.number_input("Year",       min_value=1990, max_value=2024, value=2020)
    km_driven    = st.number_input("KM Driven",  min_value=0, max_value=500000, value=50000, step=1000)
    transmission = st.selectbox("Transmission",  sorted(df["Transmission"].dropna().unique()))
    owner        = st.selectbox("Owner",         sorted(df["Owner"].dropna().unique()))
    fuel         = st.selectbox("Fuel Type",     sorted(df["FuelType"].dropna().unique()))

with col_right:
    st.markdown('<div class="section-title">🎯 Prediction Results</div>', unsafe_allow_html=True)
    predict_btn = st.button("🔮 PREDICT PRICE")

    if predict_btn:
        age      = 2026 - year
        input_df = pd.DataFrame([[
            year, age, km_driven, brand, model_name, transmission, owner, fuel
        ]], columns=["Year","Age","kmDriven","Brand","model","Transmission","Owner","FuelType"])

        price    = float(model.predict(input_df)[0])
        low      = int(price * 0.92)
        high     = int(price * 1.08)
        kpy      = int(km_driven / (age + 1))
        orig_est = int(price * ((1.10) ** age))

        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">Estimated Market Price</div>
            <div class="result-price">₹{int(price):,}</div>
            <div class="result-range">Range: ₹{low:,} — ₹{high:,}</div>
            <div class="stat-grid">
                <div class="stat-card">
                    <div class="stat-val">90.1%</div>
                    <div class="stat-lbl">R² Score</div>
                </div>
                <div class="stat-card">
                    <div class="stat-val">{age}yr</div>
                    <div class="stat-lbl">Car Age</div>
                </div>
                <div class="stat-card">
                    <div class="stat-val">{kpy:,}</div>
                    <div class="stat-lbl">KM / Year</div>
                </div>
                <div class="stat-card">
                    <div class="stat-val">{fuel[:3].upper()}</div>
                    <div class="stat-lbl">Fuel Type</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── ORIGINAL vs CURRENT CHART ─────────────────
        fig = go.Figure(go.Bar(
            x=['Original Price\n(When New)', 'Current Market\nValue',
               'Low Estimate', 'High Estimate'],
            y=[orig_est, int(price), low, high],
            marker=dict(
                color=['#667eea', '#ff6b6b', '#feca57', '#48cae4'],
                line=dict(width=0),
                opacity=0.92
            ),
            text=[f"₹{v:,}" for v in [orig_est, int(price), low, high]],
            textposition='outside',
            textfont=dict(size=11, color='#333', family='Poppins, sans-serif'),
        ))
        fig.update_layout(
            title=dict(
                text="Original Price vs Current Market Value",
                font=dict(family="Poppins, sans-serif", size=13, color="#333"),
                x=0
            ),
            paper_bgcolor='#ffffff',
            plot_bgcolor='#ffffff',
            font=dict(family="Poppins, sans-serif", color="#999", size=11),
            margin=dict(l=8, r=8, t=44, b=8),
            xaxis=dict(gridcolor='#f0f0f8', linecolor='#e8eaf0',
                       tickfont=dict(size=10, color='#666')),
            yaxis=dict(gridcolor='#f0f0f8', linecolor='#e8eaf0',
                       tickprefix='₹', tickformat=',.0f',
                       tickfont=dict(size=10)),
            showlegend=False,
            height=280
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">🚗</div>
            <p>Fill in the car details on the left<br>and click <strong>Predict Price</strong> to get results</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── ANALYTICS ─────────────────────────────────────────
st.markdown('<div class="analytics-area">', unsafe_allow_html=True)
st.markdown("""
<div class="analytics-title">📊 Market Analytics</div>
<div class="analytics-sub">Insights from the Indian used car market dataset</div>
""", unsafe_allow_html=True)

def chart_theme(fig, title=""):
    fig.update_layout(
        title=dict(text=title,
                   font=dict(family="Poppins, sans-serif", size=13, color="#333"), x=0),
        paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
        font=dict(family="Poppins, sans-serif", color="#999", size=11),
        margin=dict(l=8, r=8, t=44, b=8),
        xaxis=dict(gridcolor='#f0f0f8', linecolor='#e8eaf0', zerolinecolor='#e8eaf0'),
        yaxis=dict(gridcolor='#f0f0f8', linecolor='#e8eaf0', zerolinecolor='#e8eaf0'),
        showlegend=False
    )
    return fig

g1, g2 = st.columns(2)
with g1:
    fig = go.Figure(go.Histogram(
        x=df["AskPrice"], nbinsx=60,
        marker=dict(color='#667eea', opacity=0.85, line=dict(width=0))
    ))
    chart_theme(fig, "Price Distribution")
    fig.update_xaxes(title="Price (₹)", tickprefix="₹", tickformat=",.0f")
    fig.update_yaxes(title="Count")
    st.plotly_chart(fig, use_container_width=True)

with g2:
    top_brands = (df.groupby("Brand")["AskPrice"]
                  .median().sort_values(ascending=False).head(10).reset_index())
    fig = go.Figure(go.Bar(
        x=top_brands["AskPrice"].tolist(), y=top_brands["Brand"].tolist(),
        orientation='h',
        marker=dict(color=top_brands["AskPrice"].tolist(),
                    colorscale=[[0,'#feca57'],[1,'#ff6b6b']], line=dict(width=0)),
        text=[f"₹{v:,.0f}" for v in top_brands["AskPrice"]],
        textposition='outside', textfont=dict(size=10)
    ))
    chart_theme(fig, "Median Price by Brand (Top 10)")
    fig.update_xaxes(title="₹", tickprefix="₹", tickformat=",.0f")
    fig.update_yaxes(title="")
    st.plotly_chart(fig, use_container_width=True)

g3, g4 = st.columns(2)
with g3:
    sample = df[['kmDriven','AskPrice']].dropna().sample(min(1500, len(df)), random_state=42)
    x_vals = sample["kmDriven"].values
    y_vals = sample["AskPrice"].values
    z = np.polyfit(x_vals, y_vals, 1); p = np.poly1d(z)
    x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals.tolist(), y=y_vals.tolist(), mode="markers",
                             marker=dict(color='#667eea', size=4, opacity=0.3)))
    fig.add_trace(go.Scatter(x=x_line.tolist(), y=p(x_line).tolist(), mode="lines",
                             line=dict(color="#ff6b6b", width=2, dash="dash")))
    chart_theme(fig, "Price vs KM Driven")
    fig.update_xaxes(title="KM Driven", tickformat=",.0f")
    fig.update_yaxes(title="Price (₹)", tickprefix="₹", tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True)

with g4:
    ap = (df[df['Age'].between(0,20)].groupby("Age")["AskPrice"]
          .median().reset_index().sort_values("Age").dropna())
    fig = go.Figure(go.Scatter(
        x=ap["Age"].tolist(), y=ap["AskPrice"].tolist(),
        mode="lines+markers",
        line=dict(color="#764ba2", width=3),
        marker=dict(color="#ff6b6b", size=8, line=dict(color="#fff", width=2)),
        fill="tozeroy", fillcolor="#667eea15"
    ))
    chart_theme(fig, "Price Depreciation by Age")
    fig.update_xaxes(title="Car Age (years)")
    fig.update_yaxes(title="Median Price (₹)", tickprefix="₹", tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True)

g5, g6 = st.columns(2)
with g5:
    fd = df["FuelType"].value_counts().reset_index()
    fd.columns = ["FuelType","Count"]
    fig = go.Figure(go.Pie(
        labels=fd["FuelType"].tolist(), values=fd["Count"].tolist(), hole=0.42,
        marker=dict(colors=['#667eea','#ff6b6b','#feca57','#48cae4','#06d6a0','#f72585'],
                    line=dict(color='#ffffff', width=2)),
        textfont=dict(size=12)
    ))
    chart_theme(fig, "Listings by Fuel Type")
    fig.update_layout(showlegend=True,
                      legend=dict(font=dict(color="#555"), bgcolor="#fff"))
    st.plotly_chart(fig, use_container_width=True)

with g6:
    op = (df.groupby("Owner")["AskPrice"].median().reset_index()
          .dropna().sort_values("AskPrice", ascending=False))
    fig = go.Figure(go.Bar(
        x=op["Owner"].tolist(), y=op["AskPrice"].tolist(),
        marker=dict(color=op["AskPrice"].tolist(),
                    colorscale=[[0,'#feca57'],[1,'#667eea']], line=dict(width=0)),
        text=[f"₹{v:,.0f}" for v in op["AskPrice"]],
        textposition='outside', textfont=dict(size=10)
    ))
    chart_theme(fig, "Median Price by Owner Type")
    fig.update_xaxes(title="Owner")
    fig.update_yaxes(title="Median Price (₹)", tickprefix="₹", tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    Ultimate Car Price Predictor &nbsp;·&nbsp; Random Forest ML &nbsp;·&nbsp; R² 90.1%
    &nbsp;&nbsp;|&nbsp;&nbsp; MADE BY <span>VELERAX</span>
</div>
""", unsafe_allow_html=True)