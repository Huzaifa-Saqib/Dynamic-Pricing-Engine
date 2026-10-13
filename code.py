import pandas as pd
import numpy as np
import openpyxl
import scipy
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

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

available_products = sorted(df["product_id"].dropna().unique())

print("Available products:", available_products)

selected_product = input(
    "Enter product ID you want to predict price for: "
).strip().upper()

if selected_product not in available_products:
    raise ValueError("Invalid product ID.")

logi_df = df[df["product_id"] == selected_product].copy()
logi_df = logi_df.sort_values("date")

logi_df["log_price"] = np.log(logi_df["price_shown"])
logi_df["log_cp"] = np.log(logi_df["competitor_price"])
logi_df["is_peak_season"] = logi_df["date"].dt.month.isin([11, 12]).astype(int)

train_all = logi_df["date"] < "2025-01-01"
test_all = logi_df["date"] >= "2025-01-01"

feature_cols = [
    "log_price", "log_cp", "is_promo", "is_holiday", "is_peak_season"
]

X_train_logi = logi_df.loc[train_all, feature_cols]
y_train_logi = logi_df.loc[train_all, "bought"]
X_test_logi = logi_df.loc[test_all, feature_cols]
y_test_logi = logi_df.loc[test_all, "bought"]

logi_model = LogisticRegression(max_iter=1000, C=100)
logi_model.fit(X_train_logi, y_train_logi)

logi_prob_test = logi_model.predict_proba(X_test_logi)[:, 1]

print("\n--- LOGISTIC REGRESSION PERFORMANCE ---")
print("AUC:", roc_auc_score(y_test_logi, logi_prob_test))

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
y_pred = np.exp(y_pred_log)

print("\n--- LOG-LOG REGRESSION PERFORMANCE ---")
print("MAE:", mean_absolute_error(y_test_log, y_pred_log))
print("RMSE:", np.sqrt(mean_squared_error(y_test_log, y_pred_log)))
print("R²:", r2_score(y_test_log, y_pred_log))

print("\n--- DEMAND ELASTICITY COEFFICIENTS ---")

for col, coef in zip(feature_cols, reg_model.coef_):
    print(f"{col} coefficient: {coef:.2f}")

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

price_change_pct = 25
price_step = 25

min_price = current_price * (1 - price_change_pct/100 )
max_price = current_price * (1 + price_change_pct/100 )