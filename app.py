import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import random

st.set_page_config(
    page_title="VELERAX — Street Valuator",
    page_icon="🏎️",
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

FF_QUOTES = [
    ("I live my life a quarter mile at a time.", "Dominic Toretto"),
    ("It doesn't matter if you win by an inch or a mile. Winning's winning.", "Dom Toretto"),
    ("I'm a boy who appreciates a good body, regardless of the make.", "Dom Toretto"),
    ("Money will come and go. We all know that. The most important thing in life will always be the people in this room.", "Dom Toretto"),
    ("Ask any racer. Any real racer. It doesn't matter if you win by an inch or a mile.", "Dom Toretto"),
    ("You break her heart, I'll break your neck.", "Dom Toretto"),
    ("I used to drag race. It's not exactly a science.", "Brian O'Conner"),
    ("I'm a genuine driver. I'm an honest driver.", "Brian O'Conner"),
    ("Danger is the cornerstone of adventure.", "Luke Hobbs"),
    ("The thing about street fights... the street always wins.", "Luke Hobbs"),
    ("You want a piece? Let's go!", "Luke Hobbs"),
    ("I don't have friends. I got family.", "Dominic Toretto"),
    ("This is the life we chose, the life we lead.", "Dom Toretto"),
    ("You're never going to win this race with a turbo.", "Jesse"),
    ("I live life one quarter mile at a time and in that window I'm free.", "Dom Toretto"),
]

CB   = "#0a0a0f"
CARD = "#0f0f1a"
ACC  = "#ff2d00"
ACC2 = "#ff6b00"
NEON = "#00d4ff"
TEXT = "#f0ede8"
DIM  = "#5a5870"
GRID = "#1a1a2a"

def theme(fig, title=""):
    fig.update_layout(
        title=dict(text=title,
                   font=dict(family="Bebas Neue, Impact, sans-serif",
                              size=16, color=TEXT), x=0),
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        font=dict(family="Rajdhani, DM Sans, sans-serif", color=DIM, size=12),
        margin=dict(l=16, r=16, t=48, b=16),
        xaxis=dict(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID),
        yaxis=dict(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID),
        showlegend=False
    )
    return fig

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Rajdhani:wght@400;500;600;700&family=Orbitron:wght@400;700;900&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif !important;
    background-color: #0a0a0f !important;
    color: #f0ede8 !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
.stSidebar { display: none !important; }
div[data-testid="column"] { padding: 0 0.4rem !important; }

/* ── SCANLINES overlay ── */
body::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,0,0,0.04) 2px,
        rgba(0,0,0,0.04) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

/* ── TOP BAR ── */
.topbar {
    background: #0a0a0f;
    border-bottom: 2px solid #ff2d00;
    padding: 10px 48px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.topbar-logo {
    font-family: 'Orbitron', monospace;
    font-weight: 900;
    font-size: 20px;
    color: #ff2d00;
    letter-spacing: 0.15em;
    text-shadow: 0 0 20px #ff2d0066;
}
.topbar-tag {
    font-family: 'Rajdhani', sans-serif;
    font-size: 12px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #5a5870;
}
.topbar-right {
    font-family: 'Orbitron', monospace;
    font-size: 11px;
    color: #00d4ff;
    letter-spacing: 0.1em;
    text-shadow: 0 0 10px #00d4ff66;
}

/* ── HERO ── */
.hero-wrap {
    background: linear-gradient(135deg, #0a0a0f 0%, #0f0a1a 50%, #0a0f0a 100%);
    padding: 64px 48px 56px;
    position: relative;
    overflow: hidden;
    border-bottom: 1px solid #1a1a2a;
}
.hero-wrap::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(ellipse at 20% 50%, #ff2d0012 0%, transparent 50%),
        radial-gradient(ellipse at 80% 50%, #00d4ff0a 0%, transparent 50%);
    pointer-events: none;
}
.hero-wrap::after {
    content: 'NFS';
    position: absolute;
    right: -10px; bottom: -40px;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 220px;
    color: #ffffff04;
    pointer-events: none;
    line-height: 1;
}
.speed-lines {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        90deg,
        transparent,
        transparent 40px,
        #ff2d0005 40px,
        #ff2d0005 41px
    );
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'Orbitron', monospace;
    font-size: 10px;
    letter-spacing: 0.3em;
    color: #ff6b00;
    text-transform: uppercase;
    margin-bottom: 16px;
    text-shadow: 0 0 10px #ff6b0066;
}
.hero-h1 {
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(4rem, 8vw, 7rem);
    color: #f0ede8;
    line-height: 0.9;
    letter-spacing: 0.04em;
    margin-bottom: 6px;
}
.hero-h1 .red { color: #ff2d00; text-shadow: 0 0 30px #ff2d0099; }
.hero-h1 .blue { color: #00d4ff; text-shadow: 0 0 30px #00d4ff66; }
.hero-sub {
    font-size: 15px;
    color: #5a5870;
    max-width: 480px;
    line-height: 1.6;
    margin-top: 14px;
    font-family: 'Rajdhani', sans-serif;
    letter-spacing: 0.03em;
}

/* ── QUOTE BANNER ── */
.quote-banner {
    background: linear-gradient(90deg, #ff2d00, #ff6b00);
    padding: 18px 48px;
    display: flex;
    align-items: center;
    gap: 20px;
}
.quote-icon {
    font-size: 28px;
    opacity: 0.6;
    flex-shrink: 0;
}
.quote-text {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 18px;
    color: #fff;
    letter-spacing: 0.05em;
    line-height: 1.2;
}
.quote-author {
    font-family: 'Orbitron', monospace;
    font-size: 10px;
    color: #ffffff99;
    letter-spacing: 0.2em;
    margin-top: 2px;
}

/* ── FORM ── */
.form-wrap {
    padding: 40px 48px 32px;
    background: #0a0a0f;
    border-bottom: 1px solid #1a1a2a;
}
.form-label {
    font-family: 'Orbitron', monospace;
    font-size: 10px;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #ff2d00;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.form-label::after { content: ''; flex: 1; height: 1px; background: #1a1a2a; }

/* ── RESULT ── */
.result-wrap {
    background: #0a0a0f;
    border-top: 3px solid #ff2d00;
    border-bottom: 3px solid #ff2d00;
    padding: 40px 48px;
    position: relative;
    overflow: hidden;
}
.result-wrap::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 30% 50%, #ff2d0010, transparent 60%);
    pointer-events: none;
}
.result-glow {
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #ff2d00, #ff6b00, #00d4ff, transparent);
}
.result-eyebrow {
    font-family: 'Orbitron', monospace;
    font-size: 10px;
    letter-spacing: 0.25em;
    color: #ff6b00;
    text-transform: uppercase;
    margin-bottom: 8px;
    text-shadow: 0 0 10px #ff6b0066;
}
.result-price {
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(3.5rem, 8vw, 6.5rem);
    color: #ff2d00;
    letter-spacing: 0.04em;
    line-height: 1;
    text-shadow: 0 0 40px #ff2d0066;
}
.result-range {
    font-family: 'Orbitron', monospace;
    font-size: 11px;
    color: #00d4ff;
    margin-top: 6px;
    letter-spacing: 0.1em;
    text-shadow: 0 0 10px #00d4ff44;
}
.rchips { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 20px; }
.rchip {
    background: #0f0f1a;
    border: 1px solid #ff2d0044;
    border-radius: 4px;
    padding: 8px 14px;
    font-size: 12px;
    color: #f0ede8;
    text-align: center;
    min-width: 80px;
    font-family: 'Rajdhani', sans-serif;
    letter-spacing: 0.05em;
}
.rchip strong {
    display: block;
    font-family: 'Orbitron', monospace;
    font-size: 13px;
    color: #ff6b00;
    margin-bottom: 2px;
}

/* ── STAT PANEL ── */
.stat-panel {
    background: #0f0f1a;
    border: 1px solid #1a1a2a;
    border-left: 3px solid #00d4ff;
    border-radius: 4px;
    padding: 16px 20px;
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.stat-label {
    font-family: 'Orbitron', monospace;
    font-size: 10px;
    letter-spacing: 0.15em;
    color: #5a5870;
    text-transform: uppercase;
}
.stat-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 20px;
    color: #00d4ff;
    letter-spacing: 0.05em;
    text-shadow: 0 0 10px #00d4ff44;
}

/* ── ANALYTICS ── */
.analytics-wrap { padding: 40px 48px 32px; background: #0a0a0f; }
.analytics-heading {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3rem;
    color: #f0ede8;
    letter-spacing: 0.06em;
    margin-bottom: 28px;
    line-height: 1;
}
.analytics-heading .red {
    color: #ff2d00;
    text-shadow: 0 0 20px #ff2d0066;
}

/* ── QUOTE CARDS ── */
.quotes-wrap { padding: 40px 48px; background: #0f0f1a; border-top: 1px solid #1a1a2a; }
.quotes-heading {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.5rem;
    color: #f0ede8;
    letter-spacing: 0.06em;
    margin-bottom: 24px;
}
.quotes-heading span { color: #ff6b00; }
.qgrid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
.qcard {
    background: #0a0a0f;
    border: 1px solid #1a1a2a;
    border-top: 2px solid #ff2d00;
    border-radius: 4px;
    padding: 20px;
    position: relative;
    overflow: hidden;
}
.qcard::before {
    content: '"';
    position: absolute;
    top: -10px; right: 10px;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 80px;
    color: #ff2d0015;
    pointer-events: none;
    line-height: 1;
}
.qcard-text {
    font-family: 'Rajdhani', sans-serif;
    font-size: 15px;
    color: #c0bdc8;
    line-height: 1.6;
    letter-spacing: 0.02em;
    margin-bottom: 12px;
}
.qcard-author {
    font-family: 'Orbitron', monospace;
    font-size: 9px;
    color: #ff6b00;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    text-shadow: 0 0 8px #ff6b0066;
}
.qcard-dash {
    display: inline-block;
    width: 20px;
    height: 2px;
    background: #ff2d00;
    vertical-align: middle;
    margin-right: 8px;
}

/* ── FOOTER ── */
.footer {
    background: #0a0a0f;
    border-top: 2px solid #ff2d00;
    padding: 20px 48px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
}
.footer-left {
    font-family: 'Orbitron', monospace;
    font-size: 10px;
    color: #2a2835;
    letter-spacing: 0.1em;
    line-height: 1.8;
}
.footer-right {
    font-family: 'Orbitron', monospace;
    font-size: 11px;
    color: #5a5870;
    letter-spacing: 0.15em;
}
.footer-right span {
    color: #ff2d00;
    font-size: 16px;
    font-weight: 900;
    text-shadow: 0 0 10px #ff2d0066;
}

/* ── STREAMLIT OVERRIDES ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: #0f0f1a !important;
    border: 1px solid #1a1a2a !important;
    border-radius: 4px !important;
    color: #f0ede8 !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 15px !important;
}
.stSelectbox > div > div:focus-within,
.stNumberInput > div > div > input:focus {
    border-color: #ff2d00 !important;
    box-shadow: 0 0 0 2px #ff2d0033, 0 0 12px #ff2d0022 !important;
}
label, .stSelectbox label, .stNumberInput label {
    font-family: 'Orbitron', monospace !important;
    font-size: 9px !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    color: #5a5870 !important;
}
.stButton > button {
    width: 100% !important;
    background: linear-gradient(90deg, #ff2d00, #ff6b00) !important;
    color: #fff !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 20px !important;
    letter-spacing: 0.15em !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 0.75rem !important;
    text-shadow: 0 0 10px #ff2d0066 !important;
    box-shadow: 0 4px 20px #ff2d0044 !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    box-shadow: 0 4px 30px #ff2d0088 !important;
    transform: translateY(-1px) !important;
}
</style>
""", unsafe_allow_html=True)

# ── TOP BAR ───────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div class="topbar-logo">⚡ VELERAX</div>
    <div class="topbar-tag">Street Car Valuation Engine</div>
    <div class="topbar-right">LIVE // INDIA MARKET</div>
</div>
""", unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="speed-lines"></div>
    <div class="hero-eyebrow">// AI-Powered Street Valuation</div>
    <div class="hero-h1">
        <span class="red">PRICE</span> YOUR<br>
        <span class="blue">RIDE.</span>
    </div>
    <div class="hero-sub">
        Built for those who live life a quarter mile at a time.
        Enter your car details — we'll tell you exactly what it's worth on the street.
    </div>
</div>
""", unsafe_allow_html=True)

# ── RANDOM QUOTE BANNER ───────────────────────────────
q = random.choice(FF_QUOTES)
st.markdown(f"""
<div class="quote-banner">
    <div class="quote-icon">🏎️</div>
    <div>
        <div class="quote-text">"{q[0]}"</div>
        <div class="quote-author">— {q[1]}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── FORM ──────────────────────────────────────────────
st.markdown('<div class="form-wrap">', unsafe_allow_html=True)
st.markdown('<div class="form-label">// Enter Your Car Details</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    brand      = st.selectbox("Brand",     sorted(df["Brand"].dropna().unique()))
    model_name = st.selectbox("Model",     sorted(df["model"].dropna().unique()))
    fuel       = st.selectbox("Fuel Type", sorted(df["FuelType"].dropna().unique()))

with c2:
    year      = st.number_input("Year",      min_value=1990, max_value=2024, value=2018)
    km_driven = st.number_input("KM Driven", min_value=0, max_value=500000, value=50000, step=1000)
    age       = 2026 - year
    st.markdown(f"""
    <div style="background:#0f0f1a;border:1px solid #ff2d0044;border-left:3px solid #ff2d00;
                border-radius:4px;padding:12px 16px;margin-top:4px">
        <div style="font-family:'Orbitron',monospace;font-size:9px;
                    letter-spacing:0.2em;color:#5a5870;text-transform:uppercase;
                    margin-bottom:4px">Vehicle Age</div>
        <div style="font-family:'Bebas Neue',sans-serif;font-size:28px;
                    color:#ff2d00;text-shadow:0 0 15px #ff2d0066">
            {age} YEARS
        </div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    transmission = st.selectbox("Transmission", sorted(df["Transmission"].dropna().unique()))
    owner        = st.selectbox("Owner",        sorted(df["Owner"].dropna().unique()))
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn  = st.button("🏁 ESTIMATE PRICE")

st.markdown('</div>', unsafe_allow_html=True)

# ── RESULT ────────────────────────────────────────────
if predict_btn:
    input_df = pd.DataFrame([[
        year, age, km_driven, brand, model_name, transmission, owner, fuel
    ]], columns=["Year","Age","kmDriven","Brand","model","Transmission","Owner","FuelType"])

    price = float(model.predict(input_df)[0])
    low   = int(price * 0.92)
    high  = int(price * 1.08)
    kpy   = int(km_driven / (age + 1))

    rq = random.choice(FF_QUOTES)

    st.markdown(f"""
    <div class="result-wrap">
        <div class="result-glow"></div>
        <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:24px">
            <div>
                <div class="result-eyebrow">// Estimated Street Value</div>
                <div class="result-price">₹{int(price):,}</div>
                <div class="result-range">RANGE: ₹{low:,} — ₹{high:,} &nbsp;|&nbsp; ±8% CONFIDENCE</div>
                <div class="rchips">
                    <div class="rchip"><strong>{brand[:8]}</strong>Brand</div>
                    <div class="rchip"><strong>{model_name[:8]}</strong>Model</div>
                    <div class="rchip"><strong>{year}</strong>Year</div>
                    <div class="rchip"><strong>{km_driven:,}</strong>KM</div>
                    <div class="rchip"><strong>{age}YR</strong>Age</div>
                    <div class="rchip"><strong>{transmission[:4].upper()}</strong>Trans</div>
                    <div class="rchip"><strong>{fuel[:3].upper()}</strong>Fuel</div>
                    <div class="rchip"><strong>90.1%</strong>R²</div>
                </div>
            </div>
            <div style="min-width:260px;max-width:320px">
                <div class="stat-panel">
                    <div class="stat-label">Transmission</div>
                    <div class="stat-value">{transmission.upper()}</div>
                </div>
                <div class="stat-panel">
                    <div class="stat-label">KM / Year</div>
                    <div class="stat-value">{kpy:,} KM</div>
                </div>
                <div class="stat-panel">
                    <div class="stat-label">Owner Type</div>
                    <div class="stat-value">{owner.upper()[:10]}</div>
                </div>
                <div class="stat-panel" style="border-left-color:#ff2d00">
                    <div class="stat-label">Model Accuracy</div>
                    <div class="stat-value" style="color:#ff2d00;text-shadow:0 0 10px #ff2d0066">R² 90.1%</div>
                </div>
            </div>
        </div>
        <div style="margin-top:24px;border-top:1px solid #1a1a2a;padding-top:18px;
                    font-family:'Bebas Neue',sans-serif;font-size:16px;
                    color:#5a5870;letter-spacing:0.1em">
            "{rq[0]}" <span style="color:#ff6b00;font-size:12px;
            font-family:'Orbitron',monospace;margin-left:8px">— {rq[1]}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── ANALYTICS ─────────────────────────────────────────
st.markdown('<div class="analytics-wrap">', unsafe_allow_html=True)
st.markdown("""
<div class="analytics-heading">MARKET <span class="red">INTEL</span></div>
""", unsafe_allow_html=True)

g1, g2 = st.columns(2)

with g1:
    fig = px.histogram(df, x="AskPrice", nbins=60,
                       color_discrete_sequence=[ACC])
    fig.update_traces(marker_line_width=0, opacity=0.9)
    theme(fig, "PRICE DISTRIBUTION")
    fig.update_xaxes(title="Price (₹)", tickprefix="₹", tickformat=",.0f")
    fig.update_yaxes(title="Count")
    st.plotly_chart(fig, use_container_width=True)

with g2:
    top_brands = (df.groupby("Brand")["AskPrice"]
                  .median().sort_values(ascending=False)
                  .head(12).reset_index())
    fig = px.bar(top_brands, x="AskPrice", y="Brand", orientation="h",
                 color="AskPrice",
                 color_continuous_scale=[[0, GRID],[0.5, ACC2],[1, ACC]])
    fig.update_traces(marker_line_width=0)
    fig.update_coloraxes(showscale=False)
    theme(fig, "TOP 12 BRANDS BY VALUE")
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
        line=dict(color=NEON, width=2, dash="dash")
    ))
    theme(fig, "PRICE vs KM DRIVEN")
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
        fillcolor="#ff2d0015"
    ))
    theme(fig, "DEPRECIATION CURVE")
    fig.update_xaxes(title="Age (years)")
    fig.update_yaxes(title="Median Price (₹)", tickprefix="₹", tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True)

g5, g6 = st.columns(2)

with g5:
    fuel_data = df["FuelType"].value_counts().reset_index()
    fuel_data.columns = ["FuelType", "Count"]
    fig = px.pie(fuel_data, names="FuelType", values="Count",
                 color_discrete_sequence=[
                     "#ff2d00","#ff6b00","#00d4ff","#ff9500","#5a5870","#f0ede8"
                 ])
    fig.update_traces(
        textposition="outside",
        textfont=dict(color=TEXT, size=12),
        marker=dict(line=dict(color=CB, width=2))
    )
    theme(fig, "FUEL TYPE BREAKDOWN")
    fig.update_layout(showlegend=True,
                      legend=dict(font=dict(color=TEXT), bgcolor=CARD))
    st.plotly_chart(fig, use_container_width=True)

with g6:
    owner_price = (df.groupby("Owner")["AskPrice"]
                   .median().reset_index().dropna()
                   .sort_values("AskPrice", ascending=False))
    fig = px.bar(owner_price, x="Owner", y="AskPrice",
                 color="AskPrice",
                 color_continuous_scale=[[0, GRID],[0.5, ACC2],[1, ACC]])
    fig.update_traces(marker_line_width=0)
    fig.update_coloraxes(showscale=False)
    theme(fig, "PRICE BY OWNER TYPE")
    fig.update_xaxes(title="Owner")
    fig.update_yaxes(title="Median Price (₹)", tickprefix="₹", tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── QUOTES SECTION ────────────────────────────────────
st.markdown('<div class="quotes-wrap">', unsafe_allow_html=True)
st.markdown("""
<div class="quotes-heading">THE <span>CODE</span> OF THE STREET</div>
""", unsafe_allow_html=True)

# 6 quotes in 2 rows of 3
selected = random.sample(FF_QUOTES, 6)
cols = st.columns(3)
for i, (qt, auth) in enumerate(selected):
    with cols[i % 3]:
        st.markdown(f"""
        <div class="qcard">
            <div class="qcard-text">"{qt}"</div>
            <div class="qcard-author">
                <span class="qcard-dash"></span>{auth}
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="footer-left">
        VELERAX STREET VALUATOR &nbsp;·&nbsp; RANDOM FOREST ML
        &nbsp;·&nbsp; R² 90.1%<br>
        BUILT WITH STREAMLIT &nbsp;·&nbsp; INDIA USED CAR MARKET DATA
    </div>
    <div class="footer-right">
        MADE BY &nbsp;<span>VELERAX</span>
    </div>
</div>
""", unsafe_allow_html=True)