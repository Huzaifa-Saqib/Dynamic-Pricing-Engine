import pandas as pd
import numpy as np
import openpyxl
import scipy
import random
import statsmodels.api as sm
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

# EDA and Data Cleaning

df = pd.read_excel("pricepilot_sales_data.xlsx")
 
print(df.head())
print(df.columns)
print(df.shape)
print(df.describe())
print(df.isnull().sum())
print(df["session_id"].nunique())
print(df["session_id"].nunique() == 1)
print(df["product_id"].nunique())
print(df["product_id"].nunique() == 1)
print(df["product_id"].value_counts())
 
df["product_id"] = df["product_id"].astype("string").str.strip().str.upper()
 
print(df["product_id"].value_counts())
 
print(df["cost_price"].isnull().sum())
print((df["cost_price"] <= 0).sum())
print(df["cost_price"].describe())
print(df["cost_price"].median())
 
df = df.dropna(subset=["cost_price"])
df = df[df["cost_price"] > 0]
 
print(df["cost_price"].isnull().sum())
print(df["cost_price"].describe())
 
print(df["price_shown"].isnull().sum())
print((df["price_shown"] <= 0).sum())
print(df["price_shown"].describe())
print(df["price_shown"].median())
 
df["price_shown"] = df["price_shown"].abs()
df = df[df["price_shown"].notna() & (df["price_shown"] > 0)]
 
print(df["price_shown"].isnull().sum())
print(df["price_shown"].describe())
print((df["price_shown"] == 0).sum())
 
print(df["competitor_price"].isnull().sum())
print((df["competitor_price"] < 0).sum())
 
df["competitor_price"] = df["competitor_price"].abs()
df = df.dropna(subset=["competitor_price"])
df = df[df["competitor_price"] > 0]
 
print(df["competitor_price"].isnull().sum())
print((df["competitor_price"] < 0).sum())
print(df["competitor_price"].describe())
 
promo_mapping = {
    "YES": 1, "Yes": 1, "yes": 1, "Y": 1, "y": 1,
    "Promo": 1, "TRUE": 1, "True": 1, "1": 1, True: 1,
    "NO": 0, "No": 0, "no": 0, "N": 0, "n": 0,
    "FALSE": 0, "False": 0, "0": 0, False: 0
}
 
holiday_mapping = {
    "YES": 1, "Yes": 1, "yes": 1, "Y": 1, "y": 1,
    "TRUE": 1, "True": 1, "1": 1, True: 1,
    "NO": 0, "No": 0, "no": 0, "N": 0, "n": 0,
    "FALSE": 0, "False": 0, "0": 0, False: 0
}
 
df["is_promo"] = (
    df["is_promo"].astype("string").str.strip().map(promo_mapping).fillna(0).astype(int)
)
 
df["is_holiday"] = (
    df["is_holiday"].astype("string").str.strip().map(holiday_mapping).fillna(0).astype(int)
)
 
print(df["is_promo"].value_counts(dropna=False))
print(df["is_holiday"].value_counts(dropna=False))
 
print(df["quantity"].isnull().sum())
print((df["quantity"] <= 0).sum())
 
df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
 
calc_qty = (df["revenue"] / df["price_shown"]).round()
invalid_qty_mask = (df["quantity"] < 0) | (df["quantity"].isna())
df.loc[invalid_qty_mask, "quantity"] = calc_qty[invalid_qty_mask]
df["quantity"] = df["quantity"].fillna(0).clip(lower=0).astype(int)
 
print(df["quantity"].isnull().sum())
print((df["quantity"] < 0).sum())
print(df["quantity"].describe())
 
print(df["bought"].value_counts())
print(df["bought"].isnull().sum())
 
df["bought"] = np.where(df["quantity"] > 0, 1, 0).astype(int)
 
print(df["bought"].value_counts(dropna=False))
 
print(df["date"].isnull().sum())
print(df["date"].dtypes)
print(df["date"].head(30))
 
df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
df = df.dropna(subset=["date"])
 
print(df["date"].dtype)
print(df["date"].isnull().sum())
print(df["date"].head(30))
 
print(df["revenue"].isnull().sum())
print((df["revenue"] < 0).sum())
print(df["revenue"].describe())
 
df["revenue"] = df["revenue"].fillna(df["price_shown"] * df["quantity"])
 
print(df["revenue"].isnull().sum())
print((df["revenue"] < 0).sum())
print(df["revenue"].describe())
 
print(df.info())
print(df.isnull().sum())
print(df.describe())
print(df.shape)
print(df.head(20))

# SELECT PRODUCT
 
available_products = sorted(df["product_id"].dropna().unique())
 
print("Available products:", available_products)
 
selected_product = input(
    "Enter product ID you want to predict price for: "
).strip().upper()
 
if selected_product not in available_products:
    raise ValueError("Invalid product ID.")
 
feature_cols = [
    "log_price", "log_cp", "is_promo", "is_holiday", "is_peak_season"
]

counts = df["bought"].value_counts()
plt.figure(figsize=(6, 4))
plt.bar(counts.index.map({0: "No", 1: "Yes"}), counts.values, edgecolor="black", width=0.5)
plt.title("Bought vs Not Bought")
plt.xlabel("Bought Status")
plt.ylabel("Count of Sessions")
plt.tight_layout()
#plt.show()

qty = df.loc[(df["bought"] == 1) & (df["quantity"] > 0), "quantity"]
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(qty, bins=10, edgecolor="black")
plt.title("Quantity: Original Distribution")
plt.xlabel("Quantity Purchased")
plt.ylabel("Frequency")
plt.subplot(1, 2, 2)
plt.hist(np.log(qty), bins=10, edgecolor="black")
plt.title("Quantity: Log-Transformed Distribution")
plt.xlabel("Log(Quantity Purchased)")
plt.ylabel("Frequency")
plt.tight_layout()
#plt.show()

# MODELS
 
def train_models(df, selected_product):

    # Logistic Regression 

    logi_df = df[df["product_id"] == selected_product].copy()
    logi_df = logi_df.sort_values("date")
 
    logi_df["log_price"] = np.log(logi_df["price_shown"])
    logi_df["log_cp"] = np.log(logi_df["competitor_price"])
    logi_df["is_peak_season"] = logi_df["date"].dt.month.isin([11, 12]).astype(int)
 
    train_all = logi_df["date"] < "2025-01-01"
    test_all = logi_df["date"] >= "2025-01-01"
 
    X_train_logi = logi_df.loc[train_all, feature_cols]
    y_train_logi = logi_df.loc[train_all, "bought"]
    X_test_logi = logi_df.loc[test_all, feature_cols]
    y_test_logi = logi_df.loc[test_all, "bought"]
 
    logi_model = LogisticRegression(max_iter=1000, C=100)
    logi_model.fit(X_train_logi, y_train_logi)
 
    logi_prob_test = logi_model.predict_proba(X_test_logi)[:, 1]
 
    print("\n--- LOGISTIC REGRESSION PERFORMANCE ---")
    print("AUC:", roc_auc_score(y_test_logi, logi_prob_test))
 
    # Log-Log Regression

    reg_df = df[
        (df["product_id"] == selected_product) & (df["bought"] == 1)
    ].copy()
 
    reg_df = reg_df.sort_values("date")
 
    reg_df["log_price"] = np.log(reg_df["price_shown"])
    reg_df["log_cp"] = np.log(reg_df["competitor_price"])
    reg_df["is_peak_season"] = reg_df["date"].dt.month.isin([11, 12]).astype(int)
 
    train_date = reg_df["date"] < "2025-01-01"
    test_date = reg_df["date"] >= "2025-01-01"
 
    X = reg_df[feature_cols]
    y = reg_df["quantity"]
 
    X_train, X_test = X.loc[train_date], X.loc[test_date]
    y_train = y.loc[train_date]
    y_test = y.loc[test_date]
 
    valid_train = y_train.notna() & (y_train > 0)
    valid_test = y_test.notna() & (y_test > 0)
 
    X_train = X_train.loc[valid_train]
    y_train = y_train.loc[valid_train]
    X_test = X_test.loc[valid_test]
    y_test = y_test.loc[valid_test]
 
    y_train_log = np.log(y_train)
    y_test_log = np.log(y_test)
 
    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train_log)
 
    y_pred_log = reg_model.predict(X_test)
    train_pred_log = reg_model.predict(X_train)
    residuals = y_train_log - train_pred_log
 
    smearing_factor = np.mean(np.exp(residuals))
    y_pred = np.exp(y_pred_log) * smearing_factor
 
    print("Smearing factor:", smearing_factor)
    print("\n--- LOG-LOG REGRESSION PERFORMANCE ---")
    print("MAE:", mean_absolute_error(y_test_log, y_pred_log))
    print("RMSE:", np.sqrt(mean_squared_error(y_test_log, y_pred_log)))
    print("R²:", r2_score(y_test_log, y_pred_log))
 
    print("\n--- QUANTITY METRICS (ORIGINAL SCALE) ---")
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R²:", r2_score(y_test, y_pred))

    print("\n--- DEMAND ELASTICITY COEFFICIENTS ---")
 
    for col, coef in zip(feature_cols, reg_model.coef_):
        print(f"{col} coefficient: {coef:.2f}")
 
    return logi_model, reg_model, smearing_factor

# INPUTS
 
def get_inputs(selected_product):
    current_price = float(input("Enter current selling price: "))
    cost_price = float(input("Enter unit cost of the product: "))
    competitor_price = float(input("Enter competitor price of the product: "))
    is_promo = int(input("Is a promotion active? (1 = Yes, 0 = No): "))
    is_holiday = int(input("Is it a holiday scenario? (1 = Yes, 0 = No): "))
 
    pricing_date = pd.to_datetime(
        input("Enter date (YYYY-MM-DD): "),
        errors="raise"
    )
 
    is_peak_season = int(pricing_date.month in [11, 12])
 
    if cost_price <= 0 or competitor_price <= 0:
        raise ValueError("Cost and competitor price must be positive.")
 
    if is_promo not in [0, 1] or is_holiday not in [0, 1]:
        raise ValueError("Promotion and holiday must be 1 or 0.")
 
    print("\n--- BUSINESS INPUTS ---")
    print("Product:", selected_product)
    print("Current Price:", current_price)
    print("Unit cost:", cost_price)
    print("Competitor price:", competitor_price)
    print("Promotion active:", is_promo)
    print("Holiday scenario:", is_holiday)
    print("Pricing date:", pricing_date.date())
    print("Peak season:", is_peak_season)
 
    return (current_price, cost_price, competitor_price,
            is_promo, is_holiday, pricing_date, is_peak_season)
 
 
logi_model, reg_model, smearing_factor = train_models(df, selected_product)
 
(current_price, cost_price, competitor_price,
 is_promo, is_holiday, pricing_date, is_peak_season) = get_inputs(selected_product)
 
price_change_pct = 5
price_step = current_price * 0.01
 
min_price = current_price * (1 - price_change_pct/100 )
max_price = current_price * (1 + price_change_pct/100 )

# SIMULATOR

def simulate_price(
    candidate_price,
    cost_price,
    competitor_price,
    is_promo,
    is_holiday,
    pricing_date,
    logi_model,
    reg_model,
    is_peak_season,
    smearing_factor
):

    X_candidate = pd.DataFrame([{
        "log_price": np.log(candidate_price),
        "log_cp": np.log(competitor_price),
        "is_promo": is_promo,
        "is_holiday": is_holiday,
        "is_peak_season": is_peak_season
    }], columns=feature_cols)

    purchase_probability = logi_model.predict_proba(
        X_candidate
    )[0, 1]

    predicted_log_quantity = reg_model.predict(
        X_candidate
    )[0]

    predicted_quantity_if_bought = (
        np.exp(predicted_log_quantity) * smearing_factor
    )

    expected_quantity = (
        purchase_probability * predicted_quantity_if_bought
    )

    expected_conversions = purchase_probability
    expected_revenue = candidate_price * expected_quantity
    expected_cost = cost_price * expected_quantity
    expected_gross_profit = expected_revenue - expected_cost

    if expected_revenue > 0:
        gross_margin = (
            expected_gross_profit / expected_revenue
        ) * 100
    else:
        gross_margin = 0

    return {
        "candidate_price": candidate_price,
        "purchase_probability": purchase_probability,
        "expected_quantity_if_bought": predicted_quantity_if_bought,
        "expected_conversions_per_session": expected_conversions,
        "expected_quantity_per_session": expected_quantity,
        "expected_revenue_per_session": expected_revenue,
        "expected_gross_profit_per_session": expected_gross_profit,
        "gross_margin_pct": gross_margin
    }

# SIMULATION

candidate_prices = np.arange(
    min_price, max_price + price_step, price_step
)

candidate_prices = np.append(candidate_prices, current_price)
candidate_prices = np.unique(np.round(candidate_prices, 2))

simulation_results = []

for price in candidate_prices:
    result = simulate_price(
        price,
        cost_price,
        competitor_price,
        is_promo,
        is_holiday,
        pricing_date,
        logi_model,
        reg_model,
        is_peak_season,
        smearing_factor
    )

    simulation_results.append(result)

results_df = pd.DataFrame(simulation_results)

print("\n--- CANDIDATE PRICE SIMULATION RESULTS ---")
print(results_df.round(1).to_string(index=False))

# GUARDRAILS

def get_guardrails():
    guardrails = {}
    while True:
        try:
            absolute_ceiling = float(input(
                "Enter absolute price ceiling (maximum permitted selling price): "
            ))
            if absolute_ceiling <= 0:
                print("Validation Error: Absolute ceiling must be greater than 0.")
                continue
            break
        except ValueError:
            print("Validation Error: Please enter valid number.")

    while True:
        try:
            min_margin_floor_pct = float(input(
                "Enter minimum gross margin % (of selling price): "
            ))
            if min_margin_floor_pct < 0 or min_margin_floor_pct >= 100:
                print("Validation Error: Minimum margin must be between 0 and 99.")
                continue
            break
        except ValueError:
            print("Validation Error: Please enter valid number.")
    
    while True:
        try:
            comp_index_pct = float(input(
                "Enter Competitor Index %: "
            ))
            if comp_index_pct < 0 or comp_index_pct >= 100:
                print ("Validation Error: Competitor Index % must be between 0 and 99.")
                continue
            break
        except ValueError:
            print("Validation Error: Please enter valid number.")
    
    while True:
        try:
            max_revenue_drawdown_pct = float(input(
                "Enter maximum revenue drawdown %: "
            ))
            if max_revenue_drawdown_pct < 0 or max_revenue_drawdown_pct > 100:
                print("Validation Error: Enter a value between 0 and 100.")
                continue 
            break
        except ValueError:
            print("Validation Error: Please enter a valid number.")

    while True:
        try:
            min_conversion_floor_pct = float(input(
                "Enter minimum conversion rate %: "
            ))
            if min_conversion_floor_pct < 0 or min_conversion_floor_pct > 100:
                print("Validation Error: Enter a value between 0 and 100.")
                continue
            break
        except ValueError:
            print("Validation Error: Please enter a valid number.")

    while True:
        try:
            max_elasticity_variance_limit = float(input(
                "Enter maximum elasticity variance limit: "
            ))
            if max_elasticity_variance_limit < 0:
                print("Validation Error: Variance limit cannot be negative.")
                continue
            break
        except ValueError:
            print("Validation Error: Please enter a valid number.")


    guardrails = {
        "absolute_ceiling": absolute_ceiling,
        "min_margin_floor_pct": min_margin_floor_pct,
        "competitor_index_pct": comp_index_pct,
        "max_revenue_drawdown_pct": max_revenue_drawdown_pct,
        "min_conversion_floor_pct": min_conversion_floor_pct,
        "max_elasticity_variance_limit": max_elasticity_variance_limit
    }

    return guardrails


# APPLY GUARDRAILS
guardrails = get_guardrails()


def elasticity_std_error(df, selected_product):

    d = df[(df["product_id"] == selected_product) & (df["bought"] == 1)].copy()
    d["log_price"] = np.log(d["price_shown"])
    d["log_cp"] = np.log(d["competitor_price"])
    d["is_peak_season"] = d["date"].dt.month.isin([11, 12]).astype(int)

    d = d[(d["date"] < "2025-01-01") & (d["quantity"] > 0)]

    X = sm.add_constant(d[feature_cols])    
    y = np.log(d["quantity"])       
    model = sm.OLS(y, X).fit()
    elasticity_se = model.bse["log_price"]
    return float(elasticity_se)
    print(elasticity_se)


def apply_guardrails(results_df, guardrails, current_price, cost_price,
                     competitor_price, elasticity_se):
    g = results_df.copy()

    baseline_idx = (g["candidate_price"] - current_price).abs().idxmin()
    baseline_revenue = g.loc[baseline_idx, "expected_revenue_per_session"]

    g["unit_margin_pct"] = (g["candidate_price"] - cost_price) / g["candidate_price"] * 100
    g["competitor_dev_pct"] = (g["candidate_price"] - competitor_price).abs() / competitor_price * 100
    g["conversion_pct"] = g["purchase_probability"] * 100

    if baseline_revenue > 0:
        g["revenue_drawdown_pct"] = (
            (baseline_revenue - g["expected_revenue_per_session"]) / baseline_revenue * 100
        )
    else:
        g["revenue_drawdown_pct"] = 0.0

    g["elasticity_se"] = elasticity_se

    g["ok_ceiling"] = g["candidate_price"] <= guardrails["absolute_ceiling"]
    g["ok_margin"] = g["unit_margin_pct"] >= guardrails["min_margin_floor_pct"]
    g["ok_competitor"] = g["competitor_dev_pct"] <= guardrails["competitor_index_pct"]
    g["ok_drawdown"] = g["revenue_drawdown_pct"] <= guardrails["max_revenue_drawdown_pct"]
    g["ok_conversion"] = g["conversion_pct"] >= guardrails["min_conversion_floor_pct"]
    g["ok_elasticity"] = g["elasticity_se"] <= guardrails["max_elasticity_variance_limit"]

    check_cols = ["ok_ceiling", "ok_margin", "ok_competitor",
                  "ok_drawdown", "ok_conversion", "ok_elasticity"]
    g["passes_all"] = g[check_cols].all(axis=1)

    labels = {
        "ok_ceiling": "ceiling",
        "ok_margin": "margin",
        "ok_competitor": "competitor index",
        "ok_drawdown": "revenue drawdown",
        "ok_conversion": "conversion",
        "ok_elasticity": "elasticity variance",
    }
    g["violations"] = g[check_cols].apply(
        lambda row: ", ".join(labels[c] for c in check_cols if not row[c]), axis=1
    )

    valid_df = g[g["passes_all"]].copy()
    rejected_df = g[~g["passes_all"]].copy()
    return valid_df, rejected_df


elasticity_se = elasticity_std_error(df, selected_product)
print(f"\nElasticity (log_price) standard error: {elasticity_se:.4f}")

valid_df, rejected_df = apply_guardrails(
    results_df, guardrails, current_price, cost_price,
    competitor_price, elasticity_se
)

display_cols = [
    "candidate_price", "purchase_probability", "expected_quantity_per_session",
    "expected_revenue_per_session", "expected_gross_profit_per_session",
    "unit_margin_pct"
]

print("\n--- PRICES REMOVED BY GUARDRAILS ---")
if rejected_df.empty:
    print("None. All candidate prices passed.")
else:
    print(rejected_df[["candidate_price", "violations"]].to_string(index=False))

print("\n--- PRICES REMAINING AFTER GUARDRAILS ---")
if valid_df.empty:
    print("No candidate price satisfies all guardrails.")
else:
    print(valid_df[display_cols].round(2).to_string(index=False))

