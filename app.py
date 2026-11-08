import io
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score, r2_score

# --- PAGE CONFIG ---
st.set_page_config(page_title="PricePilot", layout="wide", page_icon="logo2.png")

# --- SAFE CSS FOR LAYOUT & GRADIENT ---
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #F3EFE6 0%, #F3EFE6 15%, #FFFFFF 60%, #FFFFFF 100%) !important;
    }
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0) !important;
    }
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

# --- LOGO — 60% width, centered ---
if os.path.exists("logo.png"):
    logo_left, logo_center, logo_right = st.columns([1, 3, 1])
    with logo_center:
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

    df["is_promo"]   = df["is_promo"].apply(to_binary)
    df["is_holiday"] = df["is_holiday"].apply(to_binary)
    df["quantity"]   = pd.to_numeric(df["quantity"], errors="coerce").fillna(0).clip(lower=0).astype(int)
    df["bought"]     = (df["quantity"] > 0).astype(int)
    df["date"]       = pd.to_datetime(df["date"], format="mixed", errors="coerce")
    return df.dropna(subset=["date"]).copy()


def prep(d):
    d = d.copy()
    d["log_price"]      = np.log(d["price_shown"])
    d["log_cp"]         = np.log(d["competitor_price"])
    d["is_peak_season"] = d["date"].dt.month.isin([11, 12]).astype(int)
    return d


# --- BOOTSTRAP CIs (With Anti-Explosion Fix) ---
@st.cache_data(show_spinner=False)
def bootstrap_cis(raw_bytes, product, prices_list, comp, cost, promo, hol, peak, n_iter=500, seed=42):
    df_b = load(raw_bytes)
    base = prep(df_b[df_b["product_id"] == product].sort_values("date"))
    train = base[base["date"] < "2025-01-01"]
    logi_train = train
    reg_train  = train[(train["bought"] == 1) & (train["quantity"] > 0)]

    # Cap quantity to prevent mathematical explosions during bootstrap
    max_qty_seen = reg_train["quantity"].max() if len(reg_train) > 0 else 10

    prices_arr    = np.array(prices_list)
    rev_samples   = np.full((n_iter, len(prices_arr)), np.nan)
    qty_samples   = np.full((n_iter, len(prices_arr)), np.nan)
    profit_samples = np.full((n_iter, len(prices_arr)), np.nan)
    rng = np.random.default_rng(seed)

    empty_df = pd.DataFrame({
        "candidate_price": prices_arr,
        "rev_ci_lower": np.nan, "rev_ci_upper": np.nan,
        "profit_ci_lower": np.nan, "profit_ci_upper": np.nan,
        "demand_ci_lower": np.nan, "demand_ci_upper": np.nan,
    })
    if len(reg_train) < 5 or logi_train["bought"].nunique() < 2:
        return empty_df

    for i in range(n_iter):
        logi_s = logi_train.sample(frac=1.0, replace=True, random_state=int(rng.integers(1e9)))
        reg_s  = reg_train.sample(frac=1.0, replace=True, random_state=int(rng.integers(1e9)))

        if logi_s["bought"].nunique() < 2 or len(reg_s) < 5:
            continue

        try:
            b_lm = LogisticRegression(max_iter=1000, C=100).fit(logi_s[FEAT], logi_s["bought"])
            b_rm = LinearRegression().fit(reg_s[FEAT], np.log(reg_s["quantity"].clip(lower=1)))
            resid   = np.log(reg_s["quantity"].clip(lower=1)) - b_rm.predict(reg_s[FEAT])
            b_smear = float(np.exp(resid).mean())
        except Exception:
            continue

        X = pd.DataFrame([{
            "log_price": np.log(x),
            "log_cp": np.log(comp),
            "is_promo": promo,
            "is_holiday": hol,
            "is_peak_season": peak,
        } for x in prices_arr], columns=FEAT)

        prob = b_lm.predict_proba(X)[:, 1]
        
        # Clip the raw quantity so it cannot explode to infinity
        raw_qty = np.exp(b_rm.predict(X)) * b_smear
        qty = prob * np.clip(raw_qty, 0, max_qty_seen * 3)
        
        rev  = prices_arr * qty
        prof = rev - (cost * qty)
        
        rev_samples[i]    = rev
        qty_samples[i]    = qty
        profit_samples[i] = prof

    rows = []
    for j, price in enumerate(prices_arr):
        rev_col = rev_samples[:, j]; rev_col = rev_col[~np.isnan(rev_col)]
        prof_col = profit_samples[:, j]; prof_col = prof_col[~np.isnan(prof_col)]
        qty_col = qty_samples[:, j]; qty_col = qty_col[~np.isnan(qty_col)]
        
        if len(rev_col) == 0:
            rows.append({"candidate_price": price,
                         "rev_ci_lower": np.nan, "rev_ci_upper": np.nan,
                         "profit_ci_lower": np.nan, "profit_ci_upper": np.nan,
                         "demand_ci_lower": np.nan, "demand_ci_upper": np.nan})
        else:
            rows.append({"candidate_price": price,
                         "rev_ci_lower":    np.nanpercentile(rev_col, 2.5),
                         "rev_ci_upper":    np.nanpercentile(rev_col, 97.5),
                         "profit_ci_lower": np.nanpercentile(prof_col, 2.5),
                         "profit_ci_upper": np.nanpercentile(prof_col, 97.5),
                         "demand_ci_lower": np.nanpercentile(qty_col, 2.5),
                         "demand_ci_upper": np.nanpercentile(qty_col, 97.5)})
    return pd.DataFrame(rows)


# --- BULLETPROOF UPLOADER ---
st.sidebar.markdown("### 📁 Data Source")
up = st.sidebar.file_uploader("Upload Excel", type=["xlsx"])
if up is None:
    st.markdown("<div class='upload-msg'>👈 Upload your file in the sidebar to start the engine.</div>", unsafe_allow_html=True)
    st.stop()

df = load(up.getvalue())

# --- DETECT PRODUCT NAME COLUMN (for banner) ---
name_col = None
for candidate in ["product_name", "product", "name", "Product Name", "Product", "Name"]:
    if candidate in df.columns:
        name_col = candidate
        break

def product_display(pid):
    if name_col is None:
        return f"Product ID: {pid}"
    try:
        nm = df.loc[df["product_id"] == pid, name_col].iloc[0]
        return f"{nm} ({pid})"
    except Exception:
        return f"Product ID: {pid}"

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
    g_ceiling = st.number_input(
        "Absolute Price Ceiling",
        value=float(comp * 1.2),
        step=1.0,
        key=f"g_ceiling_{product}",
    )
    g_margin = st.number_input("Min Gross Margin %", 0.0, 99.0, 20.0, 1.0)
    g_comp   = st.number_input("Max Competitor Index %", 0.0, 500.0, 110.0, 5.0)
    g_dd     = st.number_input("Max Rev Drawdown %", 0.0, 100.0, 15.0, 1.0)
    g_conv   = st.number_input("Min Conversion Rate %", 0.0, 100.0, 2.0, 0.5)
    g_elas   = st.number_input("Max Elasticity SE", 0.0, 100.0, 5.0, 0.1)

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
    ols_df    = prep(p[(p["bought"] == 1) & (p["date"] < "2025-01-01")])
    ols_model = sm.OLS(np.log(ols_df["quantity"].clip(lower=1)), sm.add_constant(ols_df[FEAT])).fit()
    elasticity    = ols_model.params.get("log_price", 0.0)
    elasticity_se = float(ols_model.bse.get("log_price", 999.0))

    st.write("🔮 Step 5: Running price simulations...")
    
    # --- MATCHES YOUR NEW CODE.PY LOGIC EXACTLY ---
    price_change_pct = 15
    price_step = price * 0.025
    min_price = price * (1 - price_change_pct / 100)
    max_price = price * (1 + price_change_pct / 100)
    
    prices = np.arange(min_price, max_price + price_step, price_step)
    prices = np.append(prices, price)
    prices = np.unique(np.round(prices, 2))

    X = pd.DataFrame([dict(log_price=np.log(x), log_cp=np.log(comp), is_promo=promo,
                           is_holiday=hol, is_peak_season=peak) for x in prices], columns=FEAT)

    prob   = lm.predict_proba(X)[:, 1]
    qty    = prob * np.exp(rm.predict(X)) * smear
    rev    = prices * qty
    profit = rev - cost * qty

    sim = pd.DataFrame({
        "candidate_price": prices,
        "expected_revenue": rev,
        "expected_quantity": qty,
        "expected_profit": profit,
        "conversion_pct": prob * 100,
        "comp_index_pct": (prices / comp * 100),
    })
    sim["gross_margin_pct"] = (sim["expected_profit"] / sim["expected_revenue"] * 100).fillna(0)

    st.write("🎲 Step 5b: Running bootstrap confidence intervals (500 iterations)...")
    ci_df = bootstrap_cis(up.getvalue(), product, prices.tolist(), comp, cost, promo, hol, peak)
    sim   = sim.merge(ci_df, on="candidate_price", how="left")

    st.write("🛡️ Step 6: Applying all Guardrails and generating recommendations...")
    cur_rev = sim.loc[(sim["candidate_price"] - price).abs().idxmin(), "expected_revenue"]
    sim["rev_drawdown_pct"] = ((cur_rev - sim["expected_revenue"]) / cur_rev * 100).clip(lower=0) if cur_rev > 0 else 0.0

    sim["passed"] = (
        (sim["candidate_price"]  <= g_ceiling) &
        (sim["gross_margin_pct"] >= g_margin)  &
        (sim["comp_index_pct"]   <= g_comp)    &
        (sim["rev_drawdown_pct"] <= g_dd)      &
        (sim["conversion_pct"]   >= g_conv)    &
        (elasticity_se           <= g_elas)
    )

    if sim[sim["passed"]].empty:
        sim["recommendation"] = "REJECT"
        reasons = []
        if elasticity_se > g_elas:
            reasons.append(f"Elasticity SE ({elasticity_se:.2f}) exceeds guardrail ({g_elas:.2f}) — model not confident enough to recommend.")
        if (sim["gross_margin_pct"] >= g_margin).sum() == 0:
            reasons.append("No candidate price achieves the minimum gross margin.")
        if (sim["comp_index_pct"] <= g_comp).sum() == 0:
            reasons.append("Competitor index guardrail rejects entire price range.")
        if (sim["conversion_pct"] >= g_conv).sum() == 0:
            reasons.append("Conversion floor rejects entire price range.")
        st.error("🚫 No price passed all guardrails. Blocking constraints:\n\n- " + "\n- ".join(reasons))
    else:
        best_idx = sim[sim["passed"]][obj].idxmax()
        if obj == "expected_revenue":
            adopt_floor = sim.loc[best_idx, "rev_ci_lower"]
            cmp_col     = "rev_ci_upper"
        else:
            adopt_floor = sim.loc[best_idx, "demand_ci_lower"]
            cmp_col     = "demand_ci_upper"

        recs = []
        for idx, row in sim.iterrows():
            if not row["passed"]:
                recs.append("REJECT")
            elif idx == best_idx:
                recs.append("ADOPT")
            elif pd.notna(row[cmp_col]) and row[cmp_col] >= adopt_floor:
                recs.append("REVIEW")
            else:
                recs.append("REJECT")
        sim["recommendation"] = recs

    sort_mapping = {"ADOPT": 1, "REVIEW": 2, "REJECT": 3}
    sim["sort_order"] = sim["recommendation"].map(sort_mapping)
    sim = sim.sort_values(by=["sort_order", obj], ascending=[True, False]).drop(columns=["sort_order"])
    status.update(label="✅ Pipeline Complete!", state="complete", expanded=False)

# --- DISPLAY METRICS ---
st.markdown("### 📊 Model Performance")
c1, c2, c3 = st.columns(3)
c1.metric("Logistic AUC", f"{auc:.3f}" if auc > 0 else "—")
c2.metric("Regression R²", f"{r2:.3f}")
c3.metric("Price Elasticity", f"{elasticity:.3f}")

# --- DISPLAY CHART (Dynamic 1-Line Chart) ---
st.markdown("### 📈 Simulation Results")

fig = go.Figure()

# Determine which data to plot based on user selection
if obj == "expected_revenue":
    ci_lo = "rev_ci_lower"
    ci_hi = "rev_ci_upper"
    trace_name = "Expected Revenue"
else:
    ci_lo = "demand_ci_lower"
    ci_hi = "demand_ci_upper"
    trace_name = "Expected Demand"

# --- THE FIX: Sort the dataframe by price so Plotly draws left-to-right ---
plot_df = sim.dropna(subset=[ci_lo, ci_hi]).sort_values("candidate_price")

# 1. Bootstrap Shadow (Light Blue)
fig.add_trace(go.Scatter(
    x=plot_df["candidate_price"], y=plot_df[ci_hi],
    mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False
))
fig.add_trace(go.Scatter(
    x=plot_df["candidate_price"], y=plot_df[ci_lo],
    mode="lines", line=dict(width=0), fill="tonexty",
    fillcolor="rgba(52, 152, 219, 0.2)", hoverinfo="skip",
    showlegend=False, name="95% CI"
))

# 2. Main Line (Dark Green)
fig.add_trace(go.Scatter(
    x=plot_df["candidate_price"], y=plot_df[obj],
    name=trace_name, mode="lines+markers",
    line=dict(color="#0B5A44", width=3), marker=dict(size=6)
))

# Current Price vertical line
fig.add_vline(x=price, line_dash="dot", line_color="#E74C3C",
              annotation_text="Current Price", annotation_font_color="#111111")

# Layout & Theme
fig.update_layout(
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
    font=dict(family="sans serif", color="#111111", size=13),
    xaxis=dict(
        title="Candidate Price",
        title_font=dict(size=14, color="#111111"),
        tickfont=dict(size=12, color="#111111"),
        showgrid=True, gridcolor="#E0DCD3", zeroline=False
    ),
    yaxis=dict(
        title="Value",
        title_font=dict(size=14, color="#111111"),
        tickfont=dict(size=12, color="#111111"),
        showgrid=True, gridcolor="#E0DCD3", zeroline=False
    ),
    legend=dict(
        title="", font=dict(color="#111111"),
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
    ),
    margin=dict(l=20, r=20, t=40, b=20),
)

# Lock zoom
fig.update_xaxes(fixedrange=True)
fig.update_yaxes(fixedrange=True)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# --- ADOPT BANNER ---
st.markdown("### 🎯 Final Recommendations")

adopt_rows = sim[sim["recommendation"] == "ADOPT"]

if not adopt_rows.empty:
    adopt_price = round(float(adopt_rows.iloc[0]["candidate_price"]), 2)
    adopt_str = f"{int(adopt_price):,}" if adopt_price == int(adopt_price) else f"{adopt_price:,.2f}"
    label_str = product_display(product)

    banner_html = (
        '<div style="background: linear-gradient(90deg, #0B5A44 0%, #0E7A5C 100%); '
        'color: #FFFFFF; padding: 22px 28px; border-radius: 12px; '
        'text-align: center; font-weight: 700; font-size: 1.5rem; '
        'letter-spacing: 0.5px; box-shadow: 0 4px 12px rgba(11, 90, 68, 0.25); '
        'margin: 6px 0 22px 0;">'
        '⭐ ADOPT: ' 
        f'<span style="font-size: 2.1rem; margin-left: 8px;">{adopt_str}</span>'
        '<div style="font-size: 1rem; font-weight: 500; margin-top: 8px; opacity: 0.92;">'
        f'{label_str}'
        '</div>'
        '</div>'
    )
    st.markdown(banner_html, unsafe_allow_html=True)
else:
    label_str = product_display(product)
    no_adopt_html = (
        '<div style="background: #E74C3C; color: #FFFFFF; padding: 18px 24px; '
        'border-radius: 12px; text-align: center; font-weight: 600; '
        'font-size: 1.2rem; margin: 6px 0 22px 0;">'
        f'⚠️ No ADOPT price — no candidate passed all guardrails for <b>{label_str}</b>.'
        '</div>'
    )
    st.markdown(no_adopt_html, unsafe_allow_html=True)

# --- FINAL TABLE ---
display_df = sim[["candidate_price", obj, "gross_margin_pct", "recommendation"]].copy()

rename_dict = {
    "candidate_price": "Candidate Price",
    "expected_revenue": "Expected Revenue",
    "expected_quantity": "Expected Quantity",
    "gross_margin_pct": "Gross Margin %",
    "recommendation": "Recommendation",
}
display_df = display_df.rename(columns=rename_dict).round(2)

styled_df = display_df.style.set_properties(**{
    'color': '#111111',
    'font-size': '14px'
}).set_table_styles([
    {'selector': 'th', 'props': [('color', '#111111'), ('font-weight', 'bold')]}
])

st.dataframe(styled_df, use_container_width=True, hide_index=True)