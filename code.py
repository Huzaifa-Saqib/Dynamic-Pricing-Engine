import pandas as pd
import numpy as np
import openpyxl
import scipy
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_excel("pricepilot_sales_data_raw.xlsx")
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

product_df = df[df["product_id"] == "P001"].copy()
product_df = product_df.sort_values("date")

train_date = product_df['date'] < "2025-01-01"
test_date = product_df['date'] >= "2025-01-01"

product_df["log_price"] = np.log(product_df["price_shown"])
product_df["log_cp"] = np.log(product_df["competitor_price"])

feature_cols = [
    "log_price",
    "log_cp",
    "is_promo",
    "is_holiday"
]

X = product_df[feature_cols]
y = product_df["quantity"]

X_train = X.loc[train_date]
X_test = X.loc[test_date]
y_train = y.loc[train_date]
y_test = y.loc[test_date]

y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

model = LinearRegression()
model.fit(X_train, y_train_log)
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)

print("\n--- ORIGINAL SESSION-LEVEL REGRESSION PERFORMANCE ---")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R²:", r2_score(y_test, y_pred))

print("\n--- DEMAND ELASTICITY COEFFICIENTS ---")
for col, coef in zip(feature_cols, model.coef_):
    print(f"{col} coefficient: {coef:.2f}")

from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

df_p001 = product_df.copy()
df_p001["price_ratio"] = df_p001["price_shown"] / df_p001["competitor_price"]
df_p001["price_diff"] = df_p001["price_shown"] - df_p001["competitor_price"]
df_p001["day_of_week"] = df_p001["date"].dt.dayofweek
df_p001["is_weekend"] = df_p001["day_of_week"].isin([5, 6]).astype(int)

features_ds = ["price_shown", "competitor_price", "price_ratio", "price_diff", "is_promo", "is_holiday", "day_of_week", "is_weekend"]

df_train_raw = df_p001[df_p001['date'] < "2025-01-01"]
df_test_raw = df_p001[df_p001['date'] >= "2025-01-01"]

train_majority = df_train_raw[df_train_raw.bought == 1]
train_minority = df_train_raw[df_train_raw.bought == 0]

train_majority_downsampled = resample(
    train_majority, 
    replace=False,    
    n_samples=len(train_minority),   
    random_state=42
)

df_train_balanced = pd.concat([train_majority_downsampled, train_minority]).sample(frac=1, random_state=42)

X_train_ds = df_train_balanced[features_ds]
y_train_ds = df_train_balanced["bought"]

X_test_ds = df_test_raw[features_ds]
y_test_ds = df_test_raw["bought"]

ds_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
ds_model.fit(X_train_ds, y_train_ds)

y_pred_ds = ds_model.predict(X_test_ds)
y_prob_ds = ds_model.predict_proba(X_test_ds)[:, 1]

print("\n" + "="*40)
print("   BALANCED DOWNSAMPLED MODEL PERFORMANCE")
print("="*40)
print("Accuracy Score:", accuracy_score(y_test_ds, y_pred_ds))
print("ROC-AUC Score:", roc_auc_score(y_test_ds, y_prob_ds))
print("\nDetailed Matrix Breakdown:\n", classification_report(y_test_ds, y_pred_ds, zero_division=0))
print("="*40)