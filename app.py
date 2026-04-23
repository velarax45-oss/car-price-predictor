import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="VELERAX — Car Valuator",
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

# chart theme — warm cream
CB   = "#faf7f2"
GRID = "#ede8e0"
ACC  = "#e85d04"
DARK = "#1a1208"
MID  = "#8a7560"

def theme(fig, title=""):
    fig.update_layout(
        title=dict(text=title, font=dict(family="Playfair Display, serif", size=15, color=DARK), x=0),
        paper_bgcolor=CB, plot_bgcolor=CB,
        font=dict(family="DM Sans, sans-serif", color=MID, size=12),
        margin=dict(l=16, r=16, t=48, b=16),
        xaxis=dict(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID),
        yaxis=dict(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID),
        showlegend=False
    )
    return fig

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,700&family=DM+Sans:wght@300;400;500&family=Space+Mono:wght@400;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: #faf7f2 !important;
    color: #1a1208 !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
.stSidebar { display: none !important; }
div[data-testid="column"] { padding: 0 0.4rem !important; }

/* ── TOP BAR ── */
.topbar {
    background: #1a1208;
    padding: 12px 48px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.topbar-logo {
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 18px;
    color: #e85d04;
    letter-spacing: 0.1em;
}
.topbar-tag {
    font-size: 11px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #8a7560;
}

/* ── HERO ── */
.hero-wrap {
    background: #1a1208;
    padding: 60px 48px 56px;
    position: relative;
    overflow: hidden;
}
.hero-wrap::after {
    content: 'VALUATE';
    position: absolute;
    right: -20px; top: 10px;
    font-family: 'Playfair Display', serif;
    font-size: 140px;
    font-weight: 900;
    color: #ffffff08;
    pointer-events: none;
    line-height: 1;
}
.hero-eyebrow {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.25em;
    color: #e85d04;
    text-transform: uppercase;
    margin-bottom: 16px;
}
.hero-h1 {
    font-family: 'Playfair Display', serif;
    font-size: clamp(3rem, 6vw, 5.5rem);
    font-weight: 900;
    color: #faf7f2;
    line-height: 0.95;
    letter-spacing: -0.02em;
    margin-bottom: 20px;
}
.hero-h1 em { color: #e85d04; font-style: italic; }
.hero-sub {
    font-size: 15px;
    color: #8a7560;
    max-width: 420px;
    line-height: 1.7;
}

/* ── FORM SECTION ── */
.form-wrap { padding: 48px 48px 32px; background: #faf7f2; }
.form-section-label {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #c4b9a8;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.form-section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #ede8e0;
}

/* ── RESULT ── */
.result-wrap {
    background: #e85d04;
    padding: 40px 48px;
    position: relative;
    overflow: hidden;
}
.result-wrap::before {
    content: '₹';
    position: absolute;
    right: 40px; top: -20px;
    font-family: 'Playfair Display', serif;
    font-size: 200px;
    font-weight: 900;
    color: #ffffff10;
    pointer-events: none;
}
.result-inner {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    flex-wrap: wrap;
    gap: 24px;
}
.result-left {}
.result-eyebrow {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.25em;
    color: #f5c4a8;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.result-price {
    font-family: 'Playfair Display', serif;
    font-size: clamp(3rem, 7vw, 5.5rem);
    font-weight: 900;
    color: #fff;
    letter-spacing: -0.03em;
    line-height: 1;
}
.result-range {
    font-size: 13px;
    color: #f5c4a8;
    margin-top: 6px;
    font-family: 'Space Mono', monospace;
}
.result-chips {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}
.rchip {
    background: #ffffff18;
    border: 1px solid #ffffff30;
    border-radius: 6px;
    padding: 8px 14px;
    font-size: 12px;
    color: #fff;
    text-align: center;
    min-width: 80px;
}
.rchip strong {
    display: block;
    font-family: 'Space Mono', monospace;
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 2px;
}

/* ── ANALYTICS SECTION ── */
.analytics-wrap { padding: 48px 48px 32px; background: #faf7f2; }
.analytics-heading {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 900;
    color: #1a1208;
    letter-spacing: -0.02em;
    margin-bottom: 28px;
}
.analytics-heading span { color: #e85d04; font-style: italic; }

/* ── FOOTER ── */
.footer {
    background: #1a1208;
    padding: 24px 48px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
}
.footer-left {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: #4a3f30;
}
.footer-right {
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    color: #8a7560;
    letter-spacing: 0.1em;
}
.footer-right span {
    color: #e85d04;
    font-weight: 700;
    font-size: 14px;
}

/* ── STREAMLIT OVERRIDES ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: #fff !important;
    border: 2px solid #ede8e0 !important;
    border-radius: 8px !important;
    color: #1a1208 !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stSelectbox > div > div:focus-within,
.stNumberInput > div > div > input:focus {
    border-color: #e85d04 !important;
    box-shadow: 0 0 0 3px #e85d0418 !important;
}
label, .stSelectbox label, .stNumberInput label {
    font-family: 'Space Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    color: #c4b9a8 !important;
}
.stButton > button {
    width: 100% !important;
    background: #1a1208 !important;
    color: #faf7f2 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    letter-spacing: 0.12em !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.9rem !important;
    transition: all 0.2s !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    background: #e85d04 !important;
    color: #fff !important;
}
</style>
""", unsafe_allow_html=True)

# ── TOP BAR ───────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div class="topbar-logo">VELERAX</div>
    <div class="topbar-tag">Used Car Valuation Engine</div>
</div>
""", unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-eyebrow">// AI-Powered Market Valuation</div>
    <div class="hero-h1">What's your<br>car <em>worth?</em></div>
    <div class="hero-sub">
        Enter your car details and get an instant price estimate
        powered by machine learning trained on real Indian market data.
    </div>
</div>
""", unsafe_allow_html=True)

# ── FORM ──────────────────────────────────────────────
st.markdown('<div class="form-wrap">', unsafe_allow_html=True)
st.markdown('<div class="form-section-label">// Enter Car Details</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    brand      = st.selectbox("Brand",     sorted(df["Brand"].dropna().unique()))
    model_name = st.selectbox("Model",     sorted(df["model"].dropna().unique()))
    fuel       = st.selectbox("Fuel Type", sorted(df["FuelType"].dropna().unique()))

with c2:
    year      = st.number_input("Year",       min_value=1990, max_value=2024, value=2018)
    km_driven = st.number_input("KM Driven",  min_value=0, max_value=500000, value=50000, step=1000)
    age       = 2026 - year
    st.markdown(f"""
    <div style="background:#fff;border:2px solid #ede8e0;border-radius:8px;
                padding:10px 14px;margin-top:4px">
        <div style="font-family:'Space Mono',monospace;font-size:10px;
                    letter-spacing:0.2em;color:#c4b9a8;text-transform:uppercase;
                    margin-bottom:4px">Car Age</div>
        <div style="font-family:'Playfair Display',serif;font-size:22px;
                    font-weight:700;color:#e85d04">{age} years</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    transmission = st.selectbox("Transmission", sorted(df["Transmission"].dropna().unique()))
    owner        = st.selectbox("Owner",        sorted(df["Owner"].dropna().unique()))
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn  = st.button("→ ESTIMATE PRICE")

st.markdown('</div>', unsafe_allow_html=True)

# ── RESULT ────────────────────────────────────────────
if predict_btn:
    input_df = pd.DataFrame([[
        year, age, km_driven, brand, model_name, transmission, owner, fuel
    ]], columns=["Year","Age","kmDriven","Brand","model","Transmission","Owner","FuelType"])

    price = float(model.predict(input_df)[0])
    low   = int(price * 0.92)
    high  = int(price * 1.08)

    st.markdown(f"""
    <div class="result-wrap">
        <div class="result-inner">
            <div class="result-left">
                <div class="result-eyebrow">// Estimated Market Price</div>
                <div class="result-price">₹{int(price):,}</div>
                <div class="result-range">Range: ₹{low:,} — ₹{high:,} &nbsp;·&nbsp; ±8% confidence</div>
            </div>
            <div class="result-chips">
                <div class="rchip"><strong>{brand}</strong>Brand</div>
                <div class="rchip"><strong>{model_name}</strong>Model</div>
                <div class="rchip"><strong>{year}</strong>Year</div>
                <div class="rchip"><strong>{km_driven:,}</strong>KM</div>
                <div class="rchip"><strong>{age}yr</strong>Age</div>
                <div class="rchip"><strong>{transmission[:4]}</strong>Trans</div>
                <div class="rchip"><strong>{fuel[:3]}</strong>Fuel</div>
                <div class="rchip"><strong>90.1%</strong>R² Score</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── ANALYTICS ─────────────────────────────────────────
st.markdown('<div class="analytics-wrap">', unsafe_allow_html=True)
st.markdown('<div class="analytics-heading">Market <span>Analytics</span></div>', unsafe_allow_html=True)

g1, g2 = st.columns(2)

with g1:
    fig = px.histogram(df, x="AskPrice", nbins=60,
                       color_discrete_sequence=[ACC])
    fig.update_traces(marker_line_width=0, opacity=0.9)
    theme(fig, "Price Distribution")
    fig.update_xaxes(title="Price (₹)", tickprefix="₹", tickformat=",.0f")
    fig.update_yaxes(title="Count")
    st.plotly_chart(fig, use_container_width=True)

with g2:
    top_brands = (df.groupby("Brand")["AskPrice"]
                  .median().sort_values(ascending=False)
                  .head(12).reset_index())
    fig = px.bar(top_brands, x="AskPrice", y="Brand", orientation="h",
                 color="AskPrice",
                 color_continuous_scale=[[0, GRID],[1, ACC]])
    fig.update_traces(marker_line_width=0)
    fig.update_coloraxes(showscale=False)
    theme(fig, "Median Price by Brand (Top 12)")
    fig.update_xaxes(title="₹", tickprefix="₹", tickformat=",.0f")
    fig.update_yaxes(title="")
    st.plotly_chart(fig, use_container_width=True)

g3, g4 = st.columns(2)

with g3:
    sample = df[['kmDriven','AskPrice']].dropna().sample(min(1500, len(df)), random_state=42)
    x_vals = sample["kmDriven"].values
    y_vals = sample["AskPrice"].values
    z      = np.polyfit(x_vals, y_vals, 1)
    p      = np.poly1d(z)
    x_line = np.linspace(x_vals.min(), x_vals.max(), 100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals.tolist(), y=y_vals.tolist(), mode="markers",
        marker=dict(color=ACC, size=4, opacity=0.35)
    ))
    fig.add_trace(go.Scatter(
        x=x_line.tolist(), y=p(x_line).tolist(), mode="lines",
        line=dict(color="#1a1208", width=2, dash="dash")
    ))
    theme(fig, "Price vs KM Driven")
    fig.update_xaxes(title="KM Driven", tickformat=",.0f")
    fig.update_yaxes(title="Price (₹)", tickprefix="₹", tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True)

with g4:
    age_price = (df[df['Age'].between(0,20)]
                 .groupby("Age")["AskPrice"].median()
                 .reset_index().sort_values("Age").dropna())
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=age_price["Age"].tolist(),
        y=age_price["AskPrice"].tolist(),
        mode="lines+markers",
        line=dict(color=ACC, width=3),
        marker=dict(color=ACC, size=7),
        fill="tozeroy",
        fillcolor="#e85d0415"
    ))
    theme(fig, "Price Depreciation by Age")
    fig.update_xaxes(title="Age (years)")
    fig.update_yaxes(title="Median Price (₹)", tickprefix="₹", tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True)

g5, g6 = st.columns(2)

with g5:
    fuel_data = df["FuelType"].value_counts().reset_index()
    fuel_data.columns = ["FuelType", "Count"]
    fig = px.pie(fuel_data, names="FuelType", values="Count",
                 color_discrete_sequence=[
                     "#e85d04","#1a1208","#c4b9a8","#8a7560","#ede8e0","#f5c4a8"
                 ])
    fig.update_traces(
        textposition="outside",
        textfont=dict(color=DARK, size=12),
        marker=dict(line=dict(color=CB, width=2))
    )
    theme(fig, "Listings by Fuel Type")
    fig.update_layout(showlegend=True,
                      legend=dict(font=dict(color=DARK), bgcolor=CB))
    st.plotly_chart(fig, use_container_width=True)

with g6:
    owner_price = (df.groupby("Owner")["AskPrice"]
                   .median().reset_index().dropna()
                   .sort_values("AskPrice", ascending=False))
    fig = px.bar(owner_price, x="Owner", y="AskPrice",
                 color="AskPrice",
                 color_continuous_scale=[[0, GRID],[1, ACC]])
    fig.update_traces(marker_line_width=0)
    fig.update_coloraxes(showscale=False)
    theme(fig, "Median Price by Owner Type")
    fig.update_xaxes(title="Owner")
    fig.update_yaxes(title="Median Price (₹)", tickprefix="₹", tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="footer-left">
        CarValuate · Random Forest · R² 90.1%<br>
        Built with Streamlit · India Used Car Market
    </div>
    <div class="footer-right">
        MADE BY <span>VELERAX</span>
    </div>
</div>
""", unsafe_allow_html=True)