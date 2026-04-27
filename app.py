import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import random

st.set_page_config(
    page_title="VELERAX — Car Price Predictor",
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

FF_QUOTES = [
    ("I live my life a quarter mile at a time.", "Dominic Toretto"),
    ("It doesn't matter if you win by an inch or a mile. Winning's winning.", "Dom Toretto"),
    ("I don't have friends. I got family.", "Dominic Toretto"),
    ("Money will come and go. The most important thing will always be family.", "Dom Toretto"),
    ("Danger is the cornerstone of adventure.", "Luke Hobbs"),
    ("I'm a genuine driver. I'm an honest driver.", "Brian O'Conner"),
]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;600;700&family=Poppins:wght@400;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Poppins', 'Segoe UI', sans-serif !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    min-height: 100vh;
    color: #333 !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 24px 24px 40px !important;
    max-width: 980px !important;
    margin: 0 auto !important;
}
.stSidebar { display: none !important; }
div[data-testid="column"] { padding: 0 0.5rem !important; }

/* ── MAIN CONTAINER ── */
.main-card {
    background: #fff;
    border-radius: 20px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.15);
    overflow: hidden;
    margin-bottom: 0;
}

/* ── HEADER ── */
.app-header {
    background: linear-gradient(45deg, #ff6b6b, #feca57);
    padding: 32px 40px;
    text-align: center;
    color: white;
}
.app-header h1 {
    font-size: 2.4rem;
    font-weight: 800;
    margin-bottom: 8px;
    text-shadow: 0 2px 8px rgba(0,0,0,0.15);
}
.app-header p {
    font-size: 1.1rem;
    opacity: 0.92;
    font-weight: 500;
}

/* ── SECTION TITLES ── */
.section-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #333;
    margin-bottom: 20px;
}

/* ── FORM BODY ── */
.form-body { padding: 36px 40px 28px; background: #fff; }

/* ── RESULT BOX ── */
.result-box {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    padding: 28px;
    border-radius: 16px;
    text-align: center;
    margin-bottom: 20px;
}
.result-price {
    font-size: 3rem;
    font-weight: 800;
    margin: 8px 0 4px;
    letter-spacing: -0.02em;
}
.result-sub { font-size: 1rem; opacity: 0.88; margin-bottom: 16px; }
.stat-chips {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-top: 16px;
}
.stat-chip {
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(10px);
    border-radius: 10px;
    padding: 12px;
    text-align: center;
}
.stat-chip-val {
    font-size: 1.4rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 3px;
}
.stat-chip-lbl { font-size: 11px; opacity: 0.8; letter-spacing: 0.05em; }

/* ── QUOTE BOX ── */
.quote-box {
    background: linear-gradient(45deg, #ff6b6b18, #feca5718);
    border-left: 4px solid #ff6b6b;
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin-bottom: 20px;
}
.quote-text { font-size: 14px; color: #555; font-style: italic; line-height: 1.6; }
.quote-author { font-size: 11px; color: #ff6b6b; font-weight: 700;
                letter-spacing: 0.1em; margin-top: 6px; text-transform: uppercase; }

/* ── PREDICT BUTTON ── */
.stButton > button {
    width: 100% !important;
    padding: 18px !important;
    background: linear-gradient(45deg, #ff6b6b, #feca57) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    font-family: 'Poppins', sans-serif !important;
    letter-spacing: 0.04em !important;
    cursor: pointer !important;
    box-shadow: 0 6px 20px rgba(255,107,107,0.35) !important;
    transition: all 0.3s !important;
    margin-top: 8px !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 28px rgba(255,107,107,0.5) !important;
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
    padding: 12px 14px !important;
    transition: all 0.3s !important;
}
.stSelectbox > div > div:focus-within,
.stNumberInput > div > div > input:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102,126,234,0.12) !important;
}
label, .stSelectbox label, .stNumberInput label {
    font-family: 'Poppins', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #333 !important;
    letter-spacing: 0.01em !important;
    text-transform: none !important;
    margin-bottom: 4px !important;
}

/* ── ANALYTICS SECTION ── */
.analytics-wrap {
    background: #f8f9ff;
    border-top: 1px solid #eef0f8;
    padding: 32px 40px 36px;
    border-radius: 0 0 20px 20px;
}
.analytics-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #333;
    margin-bottom: 4px;
}
.analytics-sub { font-size: 13px; color: #888; margin-bottom: 24px; }

/* ── FOOTER ── */
.app-footer {
    text-align: center;
    padding: 20px;
    color: rgba(255,255,255,0.7);
    font-size: 13px;
    font-family: 'Poppins', sans-serif;
    letter-spacing: 0.05em;
}
.app-footer span { color: #feca57; font-weight: 700; font-size: 15px; }
</style>
""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────
st.markdown("""
<div class="main-card">
<div class="app-header">
    <h1>🚗 Ultimate Car Price Predictor</h1>
    <p>R² Score: <strong>90.1%</strong> &nbsp;|&nbsp; Powered by Random Forest ML &nbsp;|&nbsp; India Used Car Market</p>
</div>
""", unsafe_allow_html=True)

# ── FORM BODY ─────────────────────────────────────────
st.markdown('<div class="form-body">', unsafe_allow_html=True)

col_form, col_result = st.columns([1.1, 0.9])

with col_form:
    st.markdown('<div class="section-title">📝 Enter Car Details</div>', unsafe_allow_html=True)

    brand      = st.selectbox("Brand",       sorted(df["Brand"].dropna().unique()))
    model_name = st.selectbox("Model",       sorted(df["model"].dropna().unique()))
    year       = st.number_input("Year",      min_value=1990, max_value=2024, value=2020)
    km_driven  = st.number_input("KM Driven", min_value=0, max_value=500000, value=50000, step=1000)
    transmission = st.selectbox("Transmission", sorted(df["Transmission"].dropna().unique()))
    owner        = st.selectbox("Owner",        sorted(df["Owner"].dropna().unique()))
    fuel         = st.selectbox("Fuel Type",    sorted(df["FuelType"].dropna().unique()))
    predict_btn  = st.button("🔮 PREDICT PRICE")

with col_result:
    st.markdown('<div class="section-title">🎯 Prediction Results</div>', unsafe_allow_html=True)

    q = random.choice(FF_QUOTES)
    st.markdown(f"""
    <div class="quote-box">
        <div class="quote-text">"{q[0]}"</div>
        <div class="quote-author">— {q[1]}</div>
    </div>
    """, unsafe_allow_html=True)

    if predict_btn:
        age      = 2026 - year
        input_df = pd.DataFrame([[
            year, age, km_driven, brand, model_name, transmission, owner, fuel
        ]], columns=["Year","Age","kmDriven","Brand","model","Transmission","Owner","FuelType"])

        price = float(model.predict(input_df)[0])
        low   = int(price * 0.92)
        high  = int(price * 1.08)
        kpy   = int(km_driven / (age + 1))

        # Estimate original price (when new)
        # Use age-based depreciation: ~10% per year compounded
        depreciation_rate = 0.10
        original_estimate = price * ((1 + depreciation_rate) ** age)

        st.markdown(f"""
        <div class="result-box">
            <div style="font-size:13px;opacity:0.85;letter-spacing:0.1em;text-transform:uppercase">
                Estimated Market Price
            </div>
            <div class="result-price">₹{int(price):,}</div>
            <div class="result-sub">Range: ₹{low:,} — ₹{high:,}</div>
            <div class="stat-chips">
                <div class="stat-chip">
                    <div class="stat-chip-val">90.1%</div>
                    <div class="stat-chip-lbl">R² Score</div>
                </div>
                <div class="stat-chip">
                    <div class="stat-chip-val">{age}yr</div>
                    <div class="stat-chip-lbl">Car Age</div>
                </div>
                <div class="stat-chip">
                    <div class="stat-chip-val">{kpy:,}</div>
                    <div class="stat-chip-lbl">KM / Year</div>
                </div>
                <div class="stat-chip">
                    <div class="stat-chip-val">{fuel[:3].upper()}</div>
                    <div class="stat-chip-lbl">Fuel</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── ORIGINAL vs CURRENT PRICE CHART ──────────────
        fig = go.Figure()

        categories = ['Original Price\n(When New)', 'Current Market\nValue', 'Price Range\n(Low)', 'Price Range\n(High)']
        values     = [int(original_estimate), int(price), low, high]
        colors     = ['#667eea', '#ff6b6b', '#feca57', '#48cae4']

        fig.add_trace(go.Bar(
            x=categories,
            y=values,
            marker=dict(
                color=colors,
                line=dict(width=0),
                opacity=0.92
            ),
            text=[f"₹{v:,}" for v in values],
            textposition='outside',
            textfont=dict(size=11, color='#333', family='Poppins, sans-serif'),
        ))

        fig.update_layout(
            title=dict(
                text="Original Price vs Current Market Value",
                font=dict(family="Poppins, sans-serif", size=14, color="#333"),
                x=0
            ),
            paper_bgcolor='#ffffff',
            plot_bgcolor='#ffffff',
            font=dict(family="Poppins, sans-serif", color="#888", size=11),
            margin=dict(l=10, r=10, t=50, b=10),
            xaxis=dict(
                gridcolor='#f0f0f0',
                linecolor='#e1e5e9',
                tickfont=dict(size=10, color='#555')
            ),
            yaxis=dict(
                gridcolor='#f0f0f0',
                linecolor='#e1e5e9',
                tickprefix='₹',
                tickformat=',.0f',
                tickfont=dict(size=10)
            ),
            showlegend=False,
            height=300
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.markdown("""
        <div style="background:#f8f9ff;border:2px dashed #e1e5e9;border-radius:16px;
                    padding:48px 24px;text-align:center;color:#aaa">
            <div style="font-size:3rem;margin-bottom:12px">🚗</div>
            <div style="font-size:15px;font-weight:600;color:#bbb">
                Fill in the details and click<br>Predict Price to see results
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── ANALYTICS ─────────────────────────────────────────
st.markdown('<div class="analytics-wrap">', unsafe_allow_html=True)
st.markdown("""
<div class="analytics-title">📊 Market Analytics</div>
<div class="analytics-sub">Live insights from the Indian used car market dataset</div>
""", unsafe_allow_html=True)

CHART_BG = "#ffffff"
GRID_C   = "#f0f0f8"

def theme(fig, title=""):
    fig.update_layout(
        title=dict(text=title, font=dict(family="Poppins, sans-serif", size=13, color="#333"), x=0),
        paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        font=dict(family="Poppins, sans-serif", color="#888", size=11),
        margin=dict(l=10, r=10, t=44, b=10),
        xaxis=dict(gridcolor=GRID_C, linecolor=GRID_C, zerolinecolor=GRID_C),
        yaxis=dict(gridcolor=GRID_C, linecolor=GRID_C, zerolinecolor=GRID_C),
        showlegend=False
    )
    return fig

g1, g2 = st.columns(2)

with g1:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df["AskPrice"], nbinsx=60,
        marker=dict(color='#667eea', opacity=0.85, line=dict(width=0))
    ))
    theme(fig, "Price Distribution")
    fig.update_xaxes(title="Price (₹)", tickprefix="₹", tickformat=",.0f")
    fig.update_yaxes(title="Count")
    st.plotly_chart(fig, use_container_width=True)

with g2:
    top_brands = (df.groupby("Brand")["AskPrice"]
                  .median().sort_values(ascending=False)
                  .head(10).reset_index())
    fig = go.Figure(go.Bar(
        x=top_brands["AskPrice"].tolist(),
        y=top_brands["Brand"].tolist(),
        orientation='h',
        marker=dict(
            color=top_brands["AskPrice"].tolist(),
            colorscale=[[0,'#feca57'],[1,'#ff6b6b']],
            line=dict(width=0)
        ),
        text=[f"₹{v:,.0f}" for v in top_brands["AskPrice"]],
        textposition='outside',
        textfont=dict(size=10)
    ))
    theme(fig, "Median Price by Brand (Top 10)")
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
        marker=dict(color='#667eea', size=4, opacity=0.35)
    ))
    fig.add_trace(go.Scatter(
        x=x_line.tolist(), y=p(x_line).tolist(), mode="lines",
        line=dict(color="#ff6b6b", width=2, dash="dash")
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
        line=dict(color="#764ba2", width=3),
        marker=dict(color="#ff6b6b", size=8,
                    line=dict(color="#fff", width=2)),
        fill="tozeroy",
        fillcolor="#667eea18"
    ))
    theme(fig, "Price Depreciation by Age")
    fig.update_xaxes(title="Car Age (years)")
    fig.update_yaxes(title="Median Price (₹)", tickprefix="₹", tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True)

g5, g6 = st.columns(2)

with g5:
    fuel_data = df["FuelType"].value_counts().reset_index()
    fuel_data.columns = ["FuelType","Count"]
    fig = go.Figure(go.Pie(
        labels=fuel_data["FuelType"].tolist(),
        values=fuel_data["Count"].tolist(),
        hole=0.45,
        marker=dict(
            colors=['#667eea','#ff6b6b','#feca57','#48cae4','#06d6a0','#f72585'],
            line=dict(color='#ffffff', width=2)
        ),
        textfont=dict(size=12)
    ))
    theme(fig, "Listings by Fuel Type")
    fig.update_layout(showlegend=True,
                      legend=dict(font=dict(color="#555"), bgcolor="#fff"))
    st.plotly_chart(fig, use_container_width=True)

with g6:
    owner_price = (df.groupby("Owner")["AskPrice"]
                   .median().reset_index().dropna()
                   .sort_values("AskPrice", ascending=False))
    fig = go.Figure(go.Bar(
        x=owner_price["Owner"].tolist(),
        y=owner_price["AskPrice"].tolist(),
        marker=dict(
            color=owner_price["AskPrice"].tolist(),
            colorscale=[[0,'#feca57'],[1,'#667eea']],
            line=dict(width=0)
        ),
        text=[f"₹{v:,.0f}" for v in owner_price["AskPrice"]],
        textposition='outside',
        textfont=dict(size=10)
    ))
    theme(fig, "Median Price by Owner Type")
    fig.update_xaxes(title="Owner")
    fig.update_yaxes(title="Median Price (₹)", tickprefix="₹", tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)  # close main-card

# ── FOOTER ────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    Ultimate Car Price Predictor &nbsp;·&nbsp; Random Forest ML &nbsp;·&nbsp; R² 90.1%
    &nbsp;&nbsp;|&nbsp;&nbsp; MADE BY <span>VELERAX</span>
</div>
""", unsafe_allow_html=True)