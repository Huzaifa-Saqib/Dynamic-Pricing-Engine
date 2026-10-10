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
print(df["session_id"].nunique()==1)
print(df["product_id"].nunique())
print(df["product_id"].nunique()==1)
print(df["product_id"].value_counts())

df["product_id"] = df["product_id"].str.strip()
df["product_id"] = df["product_id"].str.upper()
print(df["product_id"].value_counts())

print(df["cost_price"].isnull().sum())
print((df["cost_price"]<=0).sum())
print(df["cost_price"].describe())
print(df["cost_price"].median())
df = df.dropna(subset=["cost_price"])
print(df["cost_price"].isnull().sum())
print(df["cost_price"].describe())

print(df["price_shown"].isnull().sum())
print((df["price_shown"]<=0).sum())
print(df["price_shown"].describe())
print(df["price_shown"].median())
df["price_shown"] = df["price_shown"].abs()
df = df[df["price_shown"].notna() & (df["price_shown"] != 0)]
print(df["price_shown"].isnull().sum())
print(df["price_shown"].describe())
print((df["price_shown"]==0).sum())

print(df["competitor_price"].isnull().sum())
print((df["competitor_price"] < 0).sum())
df["competitor_price"] = df["competitor_price"].abs()
df = df.dropna(subset=["competitor_price"])
print(df["competitor_price"].isnull().sum())
print((df["competitor_price"] < 0).sum())
print(df["competitor_price"].describe())

promo_mapping = {
    'YES': 1, 'Yes': 1, 'yes': 1, 'Y': 1, 'y': 1, 
    'Promo': 1, 'TRUE': 1, 'True': 1, '1': 1, True: 1,
    'NO': 0, 'No': 0, 'no': 0, 'N': 0, 'n': 0, 
    'FALSE': 0, 'False': 0, '0': 0, False: 0
}
df["is_promo"] = df["is_promo"].map(promo_mapping).fillna(0).astype(int)
print(df["is_promo"].value_counts(dropna=False))

holiday_mapping = {
    'YES': 1, 'Yes': 1, 'yes': 1, 'Y': 1, 'y': 1, 
    'TRUE': 1, 'True': 1, '1': 1, True: 1,
    'NO': 0, 'No': 0, 'no': 0, 'N': 0, 'n': 0, 
    'FALSE': 0, 'False': 0, '0': 0, False: 0
}
df["is_holiday"] = df["is_holiday"].map(holiday_mapping).fillna(0).astype(int)
print(df["is_holiday"].value_counts(dropna=False))

print(df["quantity"].isnull().sum())
print((df["quantity"] <= 0).sum())
calc_qty = (df["revenue"] / df["price_shown"]).round()
invalid_qty_mask = (df["quantity"] < 0) | (df["quantity"].isna())
df.loc[invalid_qty_mask, "quantity"] = calc_qty[invalid_qty_mask]
df["quantity"] = df["quantity"].fillna(0).astype(int)
print(df["quantity"].isnull().sum())
print((df["quantity"] < 0).sum())
print(df["quantity"].describe())

print(df["bought"].value_counts())
print(df["bought"].isnull().sum())
df["bought"] = np.where(df["quantity"] > 0, 1, 0)
df["bought"] = df["bought"].astype(int)
print(df["bought"].value_counts(dropna=False))

print(df["date"].isnull().sum())
print(df["date"].dtypes)
print(df["date"].head(30))
df["date"] = pd.to_datetime(df["date"], format = "mixed", errors = "coerce")
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

logi_df = df[df["product_id"] == "P001"].copy().sort_values("date")
logi_df["log_price"] = np.log(logi_df["price_shown"])
logi_df["log_cp"] = np.log(logi_df["competitor_price"])
logi_df["is_peak_season"] = logi_df["date"].dt.month.isin([11, 12]).astype(int)

train_all = logi_df["date"] < "2025-01-01"
test_all = logi_df["date"] >= "2025-01-01"

feature_cols = ["log_price", "log_cp", "is_promo", "is_holiday", "is_peak_season"]

logi_model = LogisticRegression(max_iter=1000, C=100)
logi_model.fit(logi_df.loc[train_all, feature_cols], logi_df.loc[train_all, "bought"])
logi_prob_test = logi_model.predict_proba(logi_df.loc[test_all, feature_cols])[:, 1]

print("\n--- LOGISTIC REGRESSION PERFORMANCE ---")
print("AUC:", roc_auc_score(logi_df.loc[test_all, "bought"], logi_prob_test))


reg_df = df[(df["product_id"] == "P001") & (df["bought"] == 1)].copy()
reg_df = reg_df.sort_values("date")

train_date = reg_df["date"] < "2025-01-01"
test_date = reg_df["date"] >= "2025-01-01"

reg_df["log_price"] = np.log(reg_df["price_shown"])
reg_df["log_cp"] = np.log(reg_df["competitor_price"])
reg_df["is_peak_season"] = reg_df["date"].dt.month.isin([11, 12]).astype(int)

X = reg_df[feature_cols]
y = reg_df["quantity"]

X_train, X_test = X.loc[train_date], X.loc[test_date]
y_train, y_test = y.loc[train_date], y.loc[test_date]

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