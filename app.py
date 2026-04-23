import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="CarValuate — AI Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# ── LOAD ──────────────────────────────────────────────
model = pickle.load(open("car_pipeline.pkl", "rb"))
df    = pd.read_csv("used_cars_dataset_v2.csv")

# ── CLEAN DATA FOR CHARTS ─────────────────────────────
df['kmDriven'] = pd.to_numeric(
    df['kmDriven'].astype(str).str.replace(',','').str.extract(r'(\d+)')[0],
    errors='coerce'
)
df['AskPrice'] = pd.to_numeric(
    df['AskPrice'].astype(str).str.replace(',','').str.extract(r'(\d+)')[0],
    errors='coerce'
)
df = df.dropna(subset=['AskPrice', 'kmDriven'])
df['Age'] = 2026 - pd.to_numeric(df['Year'], errors='coerce')
df = df[df['AskPrice'] < df['AskPrice'].quantile(0.99)]

# ── PLOTLY THEME ──────────────────────────────────────
CHART_BG    = "#080c10"
CARD_BG     = "#0d1117"
BORDER      = "#1e2328"
ACCENT      = "#c8f04a"
TEXT_DIM    = "#4a4e55"
TEXT_MAIN   = "#e8e6e0"

def chart_layout(fig, title=""):
    fig.update_layout(
        title=dict(text=title, font=dict(family="Syne", size=14, color=TEXT_MAIN), x=0),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(family="DM Sans", color=TEXT_DIM, size=12),
        margin=dict(l=16, r=16, t=40, b=16),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickcolor=BORDER),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickcolor=BORDER),
        showlegend=False
    )
    return fig

# ── CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #080c10 !important;
    color: #e8e6e0 !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem !important; max-width: 1300px !important; }

.hero { text-align: center; padding: 3rem 0 2rem; }
.hero-tag {
    display: inline-block; font-size: 11px; letter-spacing: 0.2em;
    text-transform: uppercase; color: #c8f04a;
    border: 1px solid #c8f04a44; padding: 4px 14px;
    border-radius: 20px; margin-bottom: 1.2rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.4rem, 5vw, 4rem);
    font-weight: 800; line-height: 1.05;
    letter-spacing: -0.03em; color: #ffffff; margin-bottom: 1rem;
}
.hero h1 span { color: #c8f04a; }
.hero p { font-size: 1rem; color: #7a7870; max-width: 480px; margin: 0 auto; line-height: 1.7; }

.divider-line { border: none; border-top: 1px solid #1e2328; margin: 2rem 0; }

.card { background: #0d1117; border: 1px solid #1e2328; border-radius: 16px; padding: 1.8rem; margin-bottom: 1rem; }
.card-title { font-family: 'Syne', sans-serif; font-size: 11px; letter-spacing: 0.15em; text-transform: uppercase; color: #4a4e55; margin-bottom: 1.2rem; }

.section-heading {
    font-family: 'Syne', sans-serif; font-size: 1.4rem;
    font-weight: 700; color: #ffffff;
    margin: 2.5rem 0 1.2rem; letter-spacing: -0.02em;
}
.section-heading span { color: #c8f04a; }

.result-box {
    background: linear-gradient(135deg, #0d1117 0%, #111820 100%);
    border: 1px solid #c8f04a33; border-radius: 16px;
    padding: 2.5rem; text-align: center; position: relative; overflow: hidden;
}
.result-box::before {
    content: ''; position: absolute; top: -60px; right: -60px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, #c8f04a18, transparent 70%);
    pointer-events: none;
}
.result-label { font-size: 11px; letter-spacing: 0.2em; text-transform: uppercase; color: #4a4e55; margin-bottom: 0.6rem; }
.result-price { font-family: 'Syne', sans-serif; font-size: clamp(2.4rem, 5vw, 3.6rem); font-weight: 800; color: #c8f04a; letter-spacing: -0.03em; line-height: 1; }
.result-sub { font-size: 13px; color: #4a4e55; margin-top: 0.6rem; }

.chips { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 1.4rem; }
.chip { background: #131820; border: 1px solid #1e2328; border-radius: 8px; padding: 8px 14px; font-size: 12px; color: #7a7870; flex: 1; text-align: center; min-width: 90px; }
.chip strong { display: block; font-family: 'Syne', sans-serif; font-size: 15px; font-weight: 700; color: #e8e6e0; margin-bottom: 2px; }

.pill-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 1rem; justify-content: center; }
.pill { font-size: 11px; padding: 5px 12px; border-radius: 20px; border: 1px solid #1e2328; color: #4a4e55; }
.pill.green  { border-color: #c8f04a44; color: #c8f04a; background: #c8f04a0d; }
.pill.yellow { border-color: #f0c84a44; color: #f0c84a; background: #f0c84a0d; }

.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: #0d1117 !important; border: 1px solid #1e2328 !important;
    border-radius: 10px !important; color: #e8e6e0 !important;
}
label, .stSelectbox label, .stNumberInput label {
    font-size: 12px !important; letter-spacing: 0.08em !important;
    text-transform: uppercase !important; color: #4a4e55 !important;
}
.stButton > button {
    width: 100%; background: #c8f04a !important; color: #080c10 !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    font-size: 15px !important; border: none !important;
    border-radius: 12px !important; padding: 0.85rem 2rem !important;
}
.stButton > button:hover { background: #d8ff5a !important; }
.stSidebar { display: none !important; }
div[data-testid="column"] { padding: 0 0.5rem !important; }
</style>
""", unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-tag">AI-Powered Valuation</div>
    <h1>Know Your Car's<br><span>True Value</span></h1>
    <p>Enter your car details and get an instant AI-powered price estimate with full market analytics.</p>
</div>
<hr class="divider-line">
""", unsafe_allow_html=True)

# ── INPUT FORM ────────────────────────────────────────
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown('<div class="card"><div class="card-title">Brand & Model</div>', unsafe_allow_html=True)
    brand      = st.selectbox("Brand",     sorted(df["Brand"].dropna().unique()))
    model_name = st.selectbox("Model",     sorted(df["model"].dropna().unique()))
    fuel       = st.selectbox("Fuel Type", sorted(df["FuelType"].dropna().unique()))
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="card"><div class="card-title">Usage Details</div>', unsafe_allow_html=True)
    year      = st.number_input("Year",        min_value=1990, max_value=2024, value=2018)
    age       = st.number_input("Age (years)", min_value=0,    max_value=40,   value=2026-2018)
    km_driven = st.number_input("KM Driven",   min_value=0,    max_value=500000, value=50000, step=1000)
    st.markdown('</div>', unsafe_allow_html=True)

with c3:
    st.markdown('<div class="card"><div class="card-title">Other Details</div>', unsafe_allow_html=True)
    transmission = st.selectbox("Transmission", sorted(df["Transmission"].dropna().unique()))
    owner        = st.selectbox("Owner Type",   sorted(df["Owner"].dropna().unique()))
    st.markdown('<br>', unsafe_allow_html=True)
    predict_btn  = st.button("Get Price Estimate →")
    st.markdown('</div>', unsafe_allow_html=True)

# ── PREDICTION RESULT ─────────────────────────────────
if predict_btn:
    input_data = pd.DataFrame([[
        year, age, km_driven, brand, model_name, transmission, owner, fuel
    ]], columns=["Year","Age","kmDriven","Brand","model","Transmission","Owner","FuelType"])

    price = model.predict(input_data)[0]
    low   = int(price * 0.92)
    high  = int(price * 1.08)

    st.markdown('<hr class="divider-line">', unsafe_allow_html=True)
    r1, r2 = st.columns([3, 2])

    with r1:
        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">Estimated Market Price</div>
            <div class="result-price">₹{int(price):,}</div>
            <div class="result-sub">Confidence range: ₹{low:,} — ₹{high:,}</div>
            <div class="chips">
                <div class="chip"><strong>{brand}</strong>Brand</div>
                <div class="chip"><strong>{model_name}</strong>Model</div>
                <div class="chip"><strong>{year}</strong>Year</div>
                <div class="chip"><strong>{km_driven:,}</strong>KM</div>
            </div>
            <div class="pill-row">
                <span class="pill green">{fuel}</span>
                <span class="pill yellow">{transmission}</span>
                <span class="pill">{owner} Owner</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        st.markdown(f"""
        <div class="card" style="height:100%">
            <div class="card-title">Valuation Summary</div>
            <p style="font-size:13px; color:#4a4e55; line-height:1.8; margin-bottom:1rem;">
                Based on <strong style="color:#e8e6e0">Random Forest ML</strong>
                trained on thousands of Indian used car listings.
            </p>
            <div style="border-top:1px solid #1e2328; padding-top:1rem">
                <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:8px">
                    <span style="color:#4a4e55">Car Age</span>
                    <span style="color:#e8e6e0;font-weight:500">{age} years</span>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:8px">
                    <span style="color:#4a4e55">KM / Year</span>
                    <span style="color:#e8e6e0;font-weight:500">{int(km_driven/(age+1)):,} km</span>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:8px">
                    <span style="color:#4a4e55">Model Accuracy</span>
                    <span style="color:#c8f04a;font-weight:500">R² 90.1%</span>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:13px">
                    <span style="color:#4a4e55">Price Range</span>
                    <span style="color:#e8e6e0;font-weight:500">±8%</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
#  GRAPHS SECTION
# ═══════════════════════════════════════════════════
st.markdown('<hr class="divider-line">', unsafe_allow_html=True)
st.markdown('<div class="section-heading">Market <span>Analytics</span></div>', unsafe_allow_html=True)

# ── ROW 1: Price Distribution + Avg Price by Brand ──
g1, g2 = st.columns(2)

with g1:
    fig = px.histogram(
        df, x="AskPrice", nbins=60,
        color_discrete_sequence=[ACCENT]
    )
    fig.update_traces(marker_line_width=0, opacity=0.85)
    chart_layout(fig, "Price Distribution")
    fig.update_xaxes(title="Price (₹)", tickprefix="₹", tickformat=",.0f")
    fig.update_yaxes(title="Count")
    st.plotly_chart(fig, use_container_width=True)

with g2:
    top_brands = (
        df.groupby("Brand")["AskPrice"]
        .median()
        .sort_values(ascending=False)
        .head(12)
        .reset_index()
    )
    fig = px.bar(
        top_brands, x="AskPrice", y="Brand",
        orientation="h",
        color="AskPrice",
        color_continuous_scale=[[0, "#1e2328"], [1, ACCENT]]
    )
    fig.update_traces(marker_line_width=0)
    fig.update_coloraxes(showscale=False)
    chart_layout(fig, "Median Price by Brand (Top 12)")
    fig.update_xaxes(title="Median Price (₹)", tickprefix="₹", tickformat=",.0f")
    fig.update_yaxes(title="")
    st.plotly_chart(fig, use_container_width=True)

# ── ROW 2: Price vs KM + Price vs Age ───────────────
g3, g4 = st.columns(2)

with g3:
    sample = df.sample(min(1500, len(df)), random_state=42)
    fig = px.scatter(
        sample, x="kmDriven", y="AskPrice",
        color_discrete_sequence=[ACCENT],
        opacity=0.45
    )
    # Trend line
    z   = np.polyfit(sample["kmDriven"].dropna(), sample["AskPrice"].dropna(), 1)
    p   = np.poly1d(z)
    x_line = np.linspace(sample["kmDriven"].min(), sample["kmDriven"].max(), 100)
    fig.add_trace(go.Scatter(
        x=x_line, y=p(x_line),
        mode="lines",
        line=dict(color="#ff6b6b", width=2, dash="dash"),
        name="Trend"
    ))
    chart_layout(fig, "Price vs KM Driven")
    fig.update_xaxes(title="KM Driven", tickformat=",.0f")
    fig.update_yaxes(title="Price (₹)", tickprefix="₹", tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True)

with g4:
    age_price = (
        df.groupby("Age")["AskPrice"]
        .median()
        .reset_index()
        .dropna()
        .sort_values("Age")
    )
    age_price = age_price[age_price["Age"].between(0, 20)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=age_price["Age"], y=age_price["AskPrice"],
        mode="lines+markers",
        line=dict(color=ACCENT, width=2.5),
        marker=dict(color=ACCENT, size=6),
        fill="tozeroy",
        fillcolor="#c8f04a12"
    ))
    chart_layout(fig, "How Price Drops with Age")
    fig.update_xaxes(title="Car Age (years)")
    fig.update_yaxes(title="Median Price (₹)", tickprefix="₹", tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True)

# ── ROW 3: Fuel Type Breakdown + Owner vs Price ──────
g5, g6 = st.columns(2)

with g5:
    fuel_counts = df["FuelType"].value_counts().reset_index()
    fuel_counts.columns = ["FuelType", "Count"]
    fig = px.pie(
        fuel_counts, names="FuelType", values="Count",
        color_discrete_sequence=["#c8f04a","#4af0c8","#f0c84a","#4a8cf0","#f04a8c","#8cf04a"]
    )
    fig.update_traces(
        textposition="outside",
        textfont=dict(color=TEXT_MAIN, size=12),
        marker=dict(line=dict(color=CARD_BG, width=2))
    )
    chart_layout(fig, "Listings by Fuel Type")
    fig.update_layout(showlegend=True, legend=dict(
        font=dict(color=TEXT_MAIN), bgcolor=CARD_BG
    ))
    st.plotly_chart(fig, use_container_width=True)

with g6:
    owner_price = (
        df.groupby("Owner")["AskPrice"]
        .median()
        .reset_index()
        .sort_values("AskPrice", ascending=False)
    )
    fig = px.bar(
        owner_price, x="Owner", y="AskPrice",
        color="AskPrice",
        color_continuous_scale=[[0, "#1e2328"], [1, ACCENT]]
    )
    fig.update_traces(marker_line_width=0)
    fig.update_coloraxes(showscale=False)
    chart_layout(fig, "Median Price by Owner Type")
    fig.update_xaxes(title="Owner Type")
    fig.update_yaxes(title="Median Price (₹)", tickprefix="₹", tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True)

# ── FOOTER ────────────────────────────────────────────
st.markdown("""
<hr class="divider-line">
<p style="text-align:center; color:#2a2e35; font-size:12px; padding-bottom:1rem;">
    CarValuate — ML-powered used car valuation · Random Forest · R² 90.1%
</p>
""", unsafe_allow_html=True)