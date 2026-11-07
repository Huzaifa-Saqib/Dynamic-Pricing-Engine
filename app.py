import io
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
import plotly.express as px
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score, r2_score

# --- PAGE CONFIG ---
st.set_page_config(page_title="PricePilot", layout="wide", page_icon="favicon.png")

# --- SAFE CSS FOR LAYOUT & GRADIENT ---
st.markdown("""
<style>
    /* Top-to-Bottom Gradient: Beige to White */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #F3EFE6 0%, #F3EFE6 15%, #FFFFFF 60%, #FFFFFF 100%) !important;
    }
    
    /* Make header transparent so gradient shows at the very top */
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0) !important;
    }
    
    /* Custom upload message box */
    .upload-msg {
        background-color: #0B5A44;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
        font-size: 1.2rem;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

FEAT = ["log_price", "log_cp", "is_promo", "is_holiday", "is_peak_season"]

# --- FULL WIDTH LOGO ---
if os.path.exists("logo.png"):
    st.image("logo.png", use_container_width=True)
else:
    st.markdown("<h1 style='text-align: center; color: #0B5A44; font-size: 4rem;'>✈️ PricePilot</h1>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- 1. DATA CLEANING ---
@st.cache_data(show_spinner=False)
def load(raw):
    df = pd.read_excel(io.BytesIO(raw))
    df["product_id"] = df["product_id"].astype(str).str.strip().str.upper()
    df = df[(df["cost_price"] > 0) & (df["price_shown"] > 0) & (df["competitor_price"] > 0)].copy()
    
    def to_binary(x):
        return 1 if str(x).strip().upper() in ["YES", "Y", "TRUE", "1", "PROMO"] else 0
        
    df["is_promo"] = df["is_promo"].apply(to_binary)
    df["is_holiday"] = df["is_holiday"].apply(to_binary)
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0).clip(lower=0).astype(int)
    df["bought"] = (df["quantity"] > 0).astype(int)
    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
    return df.dropna(subset=["date"]).copy()

def prep(d):
    d = d.copy()
    d["log_price"] = np.log(d["price_shown"])
    d["log_cp"] = np.log(d["competitor_price"])
    d["is_peak_season"] = d["date"].dt.month.isin([11, 12]).astype(int)
    return d

# --- BULLETPROOF UPLOADER ---
st.sidebar.markdown("### 📁 Data Source")
up = st.sidebar.file_uploader("Upload Excel", type=["xlsx"])
if up is None:
    st.markdown("<div class='upload-msg'>👈 Upload your file in the sidebar to start the engine.</div>", unsafe_allow_html=True)
    st.stop()

df = load(up.getvalue())

# --- SIDEBAR INPUTS ---
st.sidebar.markdown("### 📦 Product Details")
product = st.sidebar.selectbox("Select Product", sorted(df["product_id"].unique()))
p = df[df["product_id"] == product]

st.sidebar.markdown("### 💰 Business Inputs")
price = st.sidebar.number_input("Current Price", 0.01, value=float(p["price_shown"].median()), step=0.5)
cost  = st.sidebar.number_input("Unit Cost", 0.01, value=float(p["cost_price"].median()), step=0.5)
comp  = st.sidebar.number_input("Competitor Price", 0.01, value=float(p["competitor_price"].median()), step=0.5)

st.sidebar.markdown("### 🎯 Scenario Settings")
promo = int(st.sidebar.checkbox("Promotion Active?"))
hol   = int(st.sidebar.checkbox("Holiday Scenario?"))
date  = st.sidebar.date_input("Pricing Date", pd.Timestamp("2025-06-01"))
peak  = int(pd.Timestamp(date).month in [11, 12])

obj_label = st.sidebar.radio("Optimize For", ["Revenue", "Demand"], horizontal=True)
obj = "expected_revenue" if obj_label == "Revenue" else "expected_quantity"

# --- GUARDRAILS ---
with st.sidebar.expander("🛡️ Guardrails", expanded=True):
    g_ceiling = st.number_input("Absolute Price Ceiling", value=float(comp * 1.2), step=1.0)
    g_margin  = st.number_input("Min Gross Margin %", 0.0, 99.0, 20.0, 1.0)
    g_comp    = st.number_input("Max Competitor Index %", 0.0, 500.0, 110.0, 5.0)
    g_dd      = st.number_input("Max Rev Drawdown %", 0.0, 100.0, 15.0, 1.0)
    g_conv    = st.number_input("Min Conversion Rate %", 0.0, 100.0, 2.0, 0.5)
    g_elas    = st.number_input("Max Elasticity SE", 0.0, 100.0, 5.0, 0.1)

# --- PIPELINE ---
with st.status("⚙️ Initializing Pricing Engine...", expanded=True) as status:
    st.write("📊 Step 1: Preparing features and splitting historical data...")
    base = prep(p.sort_values("date"))
    tr, te = base["date"] < "2025-01-01", base["date"] >= "2025-01-01"
    
    st.write("🧠 Step 2: Training Logistic Regression (Purchase Probability)...")
    lm = LogisticRegression(max_iter=1000, C=100).fit(base.loc[tr, FEAT], base.loc[tr, "bought"])
    auc = 0.0
    if base.loc[te, "bought"].nunique() > 1:
        auc = roc_auc_score(base.loc[te, "bought"], lm.predict_proba(base.loc[te, FEAT])[:, 1])
    
    st.write("📈 Step 3: Training Linear Regression (Quantity if Bought)...")
    reg = base[base["bought"] == 1]
    tr2, te2 = reg["date"] < "2025-01-01", reg["date"] >= "2025-01-01"
    Xt, yt = reg.loc[tr2, FEAT], np.log(reg.loc[tr2, "quantity"].clip(lower=1))
    Xv, yv = reg.loc[te2, FEAT], np.log(reg.loc[te2, "quantity"].clip(lower=1))
    
    rm = LinearRegression().fit(Xt, yt)
    r2 = 0.0
    if len(yv) > 0:
        r2 = r2_score(yv, rm.predict(Xv))
    smear = float(np.exp(yt - rm.predict(Xt)).mean())

    st.write("📉 Step 4: Calculating Price Elasticity & Standard Error...")
    ols_df = prep(p[(p["bought"] == 1) & (p["date"] < "2025-01-01")])
    ols_model = sm.OLS(np.log(ols_df["quantity"].clip(lower=1)), sm.add_constant(ols_df[FEAT])).fit()
    elasticity = ols_model.params.get("log_price", 0.0)
    elasticity_se = float(ols_model.bse.get("log_price", 999.0)) 

    st.write("🔮 Step 5: Running price simulations...")
    rng = 7.5 
    prices = np.linspace(price * (1 - rng / 100), price * (1 + rng / 100), 40)
    
    X = pd.DataFrame([dict(log_price=np.log(x), log_cp=np.log(comp), is_promo=promo,
                           is_holiday=hol, is_peak_season=peak) for x in prices], columns=FEAT)
    
    prob = lm.predict_proba(X)[:, 1]
    qty = prob * np.exp(rm.predict(X)) * smear
    rev = prices * qty
    profit = rev - cost * qty
    
    sim = pd.DataFrame({
        "candidate_price": prices,
        "expected_revenue": rev,
        "expected_quantity": qty,
        "expected_profit": profit,
        "conversion_pct": prob * 100,
        "comp_index_pct": (prices / comp * 100)
    })
    sim["gross_margin_pct"] = (sim["expected_profit"] / sim["expected_revenue"] * 100).fillna(0)
    
    st.write("🛡️ Step 6: Applying all Guardrails and generating recommendations...")
    cur_rev = sim.loc[(sim["candidate_price"] - price).abs().idxmin(), "expected_revenue"]
    sim["rev_drawdown_pct"] = ((cur_rev - sim["expected_revenue"]) / cur_rev * 100).clip(lower=0) if cur_rev > 0 else 0.0
    
    sim["passed"] = (
        (sim["candidate_price"] <= g_ceiling) &
        (sim["gross_margin_pct"] >= g_margin) &
        (sim["comp_index_pct"] <= g_comp) &
        (sim["rev_drawdown_pct"] <= g_dd) &
        (sim["conversion_pct"] >= g_conv) &
        (elasticity_se <= g_elas)
    )
    
    if sim[sim["passed"]].empty:
        sim["recommendation"] = "REJECT"
    else:
        best_idx = sim[sim["passed"]][obj].idxmax()
        best_val = sim.loc[best_idx, obj]
        recs = []
        for idx, row in sim.iterrows():
            if not row["passed"]:
                recs.append("REJECT")
            elif idx == best_idx:
                recs.append("ADOPT")
            elif row[obj] >= (best_val * 0.985):
                recs.append("REVIEW")
            else:
                recs.append("REJECT")
        sim["recommendation"] = recs

    # --- SORTING LOGIC ---
    sort_mapping = {"ADOPT": 1, "REVIEW": 2, "REJECT": 3}
    sim["sort_order"] = sim["recommendation"].map(sort_mapping)
    sim = sim.sort_values(by=["sort_order", obj], ascending=[True, False]).drop(columns=["sort_order"])
    status.update(label="✅ Pipeline Complete!", state="complete", expanded=False)

# --- DISPLAY METRICS ---
st.markdown("### 📊 Model Performance")
c1, c2, c3 = st.columns(3)
c1.metric("Logistic AUC", f"{auc:.3f}")
c2.metric("Regression R²", f"{r2:.3f}")
c3.metric("Price Elasticity", f"{elasticity:.3f}")

# --- DISPLAY CHART ---
st.markdown("### 📈 Simulation Results")
fig = px.line(sim, x="candidate_price", y=["expected_revenue", "expected_profit"], markers=True,
              color_discrete_map={"expected_revenue": "#0B5A44", "expected_profit": "#26A65B"})

fig.update_traces(line=dict(width=3), marker=dict(size=6)) 
fig.add_vline(x=price, line_dash="dot", line_color="#E74C3C", annotation_text="Current Price", annotation_font_color="#111111")
fig.update_layout(
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
    font=dict(family="sans serif", color="#111111", size=13),
    xaxis=dict(
        title="Candidate Price",
        title_font=dict(size=14, color="#111111"),
        tickfont=dict(size=12, color="#111111"),
        showgrid=True,
        gridcolor="#E0DCD3",
        zeroline=False
    ),
    yaxis=dict(
        title="Value per Session",
        title_font=dict(size=14, color="#111111"),
        tickfont=dict(size=12, color="#111111"),
        showgrid=True,
        gridcolor="#E0DCD3",
        zeroline=False
    ),
    legend=dict(
        title="",
        font=dict(color="#111111"),
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    margin=dict(l=20, r=20, t=40, b=20)
)
st.plotly_chart(fig, use_container_width=True)

# --- FINAL TABLE (FIXED FORMATTING) ---
st.markdown("### 🎯 Final Recommendations")

# 1. Create a clean copy for display
display_df = sim[["candidate_price", obj, "gross_margin_pct", "recommendation"]].copy()

# 2. Rename columns to remove underscores and Capitalize words
rename_dict = {
    "candidate_price": "Candidate Price",
    "expected_revenue": "Expected Revenue",
    "expected_quantity": "Expected Quantity",
    "gross_margin_pct": "Gross Margin %",
    "recommendation": "Recommendation"
}
display_df = display_df.rename(columns=rename_dict).round(2)

# 3. Apply styling to force black text on the table
styled_df = display_df.style.set_properties(**{
    'color': '#111111',
    'font-size': '14px'
}).set_table_styles([
    {'selector': 'th', 'props': [('color', '#111111'), ('font-weight', 'bold')]}
])

st.dataframe(styled_df, use_container_width=True, hide_index=True)