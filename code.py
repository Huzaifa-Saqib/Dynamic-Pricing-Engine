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
df["product_id"] = df["product_id"].str.capitalize()
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


#My code starts
print("\n--- Cleaning competitor_price ---")
print("Initial nulls:", df["competitor_price"].isnull().sum())
print("Initial negative values:", (df["competitor_price"] < 0).sum())


df["competitor_price"] = df["competitor_price"].abs()


df["competitor_price"] = df.groupby("product_id")["competitor_price"].transform(
    lambda x: x.fillna(x.median())
)

print("Cleaned nulls:", df["competitor_price"].isnull().sum())
print("Cleaned negatives:", (df["competitor_price"] < 0).sum())
print(df["competitor_price"].describe())


promo_mapping = {
    'YES': True, 'Yes': True, 'yes': True, 'Y': True, 'y': True, 
    'Promo': True, 'TRUE': True, 'True': True, '1': True, 1: True, True: True,
    'NO': False, 'No': False, 'no': False, 'N': False, 'n': False, 
    'FALSE': False, 'False': False, '0': False, 0: False, False: False
}

df["is_promo"] = df["is_promo"].map(promo_mapping).fillna(False).astype(bool)

print("\n--- is_promo Value Counts ---")
print(df["is_promo"].value_counts(dropna=False))


holiday_mapping = {
    'YES': True, 'Yes': True, 'yes': True, 'Y': True, 'y': True, 
    'TRUE': True, 'True': True, '1': True, 1: True, True: True,
    'NO': False, 'No': False, 'no': False, 'N': False, 'n': False, 
    'FALSE': False, 'False': False, '0': False, 0: False, False: False
}

df["is_holiday"] = df["is_holiday"].map(holiday_mapping).fillna(False).astype(bool)

print("\n--- is_holiday Value Counts ---")
print(df["is_holiday"].value_counts(dropna=False))


print("\n--- Cleaning quantity ---")
print("Initial nulls:", df["quantity"].isnull().sum())
print("Initial invalid (<= 0):", (df["quantity"] <= 0).sum())


calc_qty = (df["revenue"] / df["price_shown"]).round()
invalid_qty_mask = (df["quantity"] <= 0) | (df["quantity"].isna())

df.loc[invalid_qty_mask, "quantity"] = calc_qty[invalid_qty_mask]
df["quantity"] = df["quantity"].fillna(0).astype(int)

print("Cleaned nulls:", df["quantity"].isnull().sum())
print("Cleaned negative count:", (df["quantity"] < 0).sum())
print(df["quantity"].describe())