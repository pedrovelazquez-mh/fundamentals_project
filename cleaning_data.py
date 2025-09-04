import os 
import pandas as pd
import numpy as np
import sys
from collections import defaultdict

config_dir = r"C:\Users\Pedro\Research\Fundamentals\Bloomberg"
if config_dir not in sys.path:
    sys.path.insert(0, config_dir)
import config_fundamentals as cfg

fundamentals_data={}
for archivo in os.listdir(cfg.carpeta_fundamentals):
    if archivo.endswith(".csv"):
        nombre_df=os.path.splitext(archivo)[0]
        ruta_completa=os.path.join(cfg.carpeta_fundamentals,archivo)
        fundamentals_data[nombre_df]=pd.read_csv(ruta_completa,header=1)


fundamentals_data = {nombre: df.iloc[1:].reset_index(drop=True) for nombre, df in fundamentals_data.items()}
for nombre in fundamentals_data:
    fundamentals_data[nombre].columns.values[0] = "Date"

for nombre in list(fundamentals_data):
    fundamentals_data[nombre]["Date"] = pd.to_datetime(
        fundamentals_data[nombre]["Date"])
    fundamentals_data[nombre] = (
        fundamentals_data[nombre]
          .dropna(subset=["Date"])
          .sort_values("Date")
          .groupby(fundamentals_data[nombre]["Date"].dt.to_period("Q"), group_keys=False)
          .apply(lambda g: g.ffill().tail(1))
          .reset_index(drop=True))
    fundamentals_data[nombre]["Date"] = (
        fundamentals_data[nombre]["Date"]
          .dt.to_period("Q")
          .apply(lambda p: p.end_time.normalize()))

for ticker, df in fundamentals_data.items():
    if "HISTORICAL_MARKET_CAP" in df.columns:
        fundamentals_data[ticker] = df.drop(columns=["HISTORICAL_MARKET_CAP"])

price_data={}
for archivo in os.listdir(cfg.carpeta_prices):
    if archivo.endswith(".csv"):
        nombre_df = os.path.splitext(archivo)[0]
        ruta_completa = os.path.join(cfg.carpeta_prices, archivo)
        price_data[nombre_df] = pd.read_csv(ruta_completa,header=0)
        
price_data = {
    nombre: df
    for nombre, df in price_data.items()
    if nombre == "MERVAL.BA" or nombre.replace(".BA", "") in cfg.ticker_sector}
          
price_data = {nombre: df.iloc[1:].reset_index(drop=True) for nombre, df in price_data.items()}
for nombre in price_data:
    price_data[nombre].columns.values[0] = "Date"

for ticker, df_fund in fundamentals_data.items():
    ticker_price = price_data.get(ticker + ".BA")  
    if ticker_price is not None and len(ticker_price) > 2:
        fecha_minima = pd.to_datetime(ticker_price.iloc[2]["Date"])
        df_fund["Date"] = pd.to_datetime(df_fund["Date"])
        fundamentals_data[ticker] = df_fund[df_fund["Date"] >= fecha_minima].reset_index(drop=True)

for df in fundamentals_data.values():
    c = df.drop(columns="Date").notna().mean()
    w =cfg.W_min + (cfg.W_max - cfg.W_min) * (1 - c)
    for col in df.columns.drop("Date"):
        lower, upper = df[col].quantile([w[col], 1 - w[col]])     
        df[col] = df[col].clip(lower=lower, upper=upper)

data_base = []
for ticker, df in fundamentals_data.items():
    df = df.copy()
    df['ticker'] = ticker
    data_base.append(df)
data_base = pd.concat(data_base, ignore_index=True)
data_base['Date'] = pd.to_datetime(data_base['Date'])
data_base = data_base.drop(columns=[col for col in data_base.columns if "GROWTH" in col])
data_base = data_base.drop(columns=[col for col in data_base.columns if "GROW" in col])

data_base = (data_base[[c for c in data_base.columns if c in ["Date", "ticker", "Sector"]
            or (c in cfg.factor_meta and cfg.factor_meta[c]["factor"] == cfg.one_factor)]] 
              if hasattr(cfg, "one_factor") and cfg.one_factor else data_base)
