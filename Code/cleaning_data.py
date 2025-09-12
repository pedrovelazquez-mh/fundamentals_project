import os 
import pandas as pd
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

market_caps_list = []
for ticker, df in fundamentals_data.items():
    if "HISTORICAL_MARKET_CAP" in df.columns and "Date" in df.columns:
        temp = df[["Date", "HISTORICAL_MARKET_CAP"]].copy()
        temp["Ticker"] = ticker
        market_caps_list.append(temp)
market_caps = pd.concat(market_caps_list, ignore_index=True)

for ticker, df in fundamentals_data.items():
    if "HISTORICAL_MARKET_CAP" in df.columns:
        fundamentals_data[ticker] = df.drop(columns=["HISTORICAL_MARKET_CAP"])


market_caps = market_caps[~((market_caps['Date'] < pd.to_datetime(cfg.strategy_initial_time)))]
market_caps = market_caps.sort_values(["Ticker", "Date"])
market_caps["HISTORICAL_MARKET_CAP"] = (market_caps.groupby("Ticker")["HISTORICAL_MARKET_CAP"].ffill().bfill())
market_caps = market_caps.loc[market_caps['Ticker'].isin(cfg.ticker_sector)].reset_index(drop=True)
market_caps["Sector"]=market_caps["Ticker"].map(cfg.ticker_sector)


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


merval = price_data["MERVAL.BA"]
merval['Date'] = pd.to_datetime(merval['Date'])
merval = merval[~((merval['Date'] < pd.to_datetime(cfg.strategy_initial_time)))]
merval = merval[~((merval['Date'] > pd.to_datetime(cfg.ending_time)))]

data_base_precios = []
for ticker, df in price_data.items():
    df = df.copy()
    df['ticker'] = ticker
    data_base_precios.append(df)
  
data_base_precios = pd.concat(data_base_precios, ignore_index=True)
data_base_precios['Date'] = pd.to_datetime(data_base_precios['Date'])
data_base_precios['ticker'] = data_base_precios['ticker'].str.replace(".BA","",regex=False)
data_base_precios = data_base_precios.loc[data_base_precios['ticker'].isin(cfg.ticker_sector)].reset_index(drop=True)
data_base_precios = data_base_precios[~((data_base_precios['Date'] < pd.to_datetime(cfg.strategy_initial_time)))]
data_base_precios = data_base_precios[~((data_base_precios['Date'] > pd.to_datetime(cfg.ending_time)))]
data_base_precios["Sector"]=data_base_precios["ticker"].map(cfg.ticker_sector)
data_base_precios["close"]=data_base_precios["close"].astype(float)
data_base_precios = data_base_precios.sort_values(['ticker', 'Date'])
data_base_precios['close'] = (data_base_precios.groupby('ticker')['close'].ffill())

def forward_fill_missing_closes(df, merval):
    ref_dates = pd.to_datetime(merval['Date'].unique())
    all_tickers = df['ticker'].unique()
    sector_map = df.dropna(subset=['Sector']).drop_duplicates('ticker').set_index('ticker')['Sector'].to_dict()
    rows_to_add = []
    for date in ref_dates:
        presentes = set(df.loc[df['Date'] == date, 'ticker'])
        for ticker in all_tickers:
            if ticker not in presentes:
                prev = df[(df['ticker'] == ticker) & (df['Date'] < date)]['close']
                close_val = prev.iloc[-1] if not prev.empty else pd.NA
                rows_to_add.append({
                    'Date'   : date,
                    'open'   : pd.NA,
                    'high'   : pd.NA,
                    'low'    : pd.NA,
                    'close'  : close_val,
                    'volume' : pd.NA,
                    'ticker' : ticker,
                    'Sector' : sector_map.get(ticker, pd.NA)})
    if rows_to_add:
        df = pd.concat([df, pd.DataFrame(rows_to_add)], ignore_index=True)
    df = df.sort_values(['ticker', 'Date'])
    df['close'] = df.groupby('ticker')['close'].ffill().bfill()
    return df.sort_values(['Date', 'ticker']).reset_index(drop=True)
data_base_precios = forward_fill_missing_closes(data_base_precios,merval)
