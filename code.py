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

