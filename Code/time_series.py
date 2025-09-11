#Librerias
import os
import pandas as pd
import numpy as np
from collections import defaultdict
import config_fundamentals as cfg   #modulo
import cleaning_data as cd    #modulo
import backtest as bt    #modulo
import HRP as hrp #modulo
import plotly.graph_objects as go
import plotly.io as pio


data_base=cd.data_base
data_base = data_base[~((data_base['Date'] < pd.to_datetime(cfg.corr_initial_time)))]
data_base = data_base[~((data_base['Date'] >= pd.to_datetime(cfg.ending_time)))]
cols = [c for c in data_base.columns if c not in ('Date', 'ticker')]
# data_base['pct_nan'] =  data_base[cols].isna().mean(axis=1)
data_base = data_base[data_base["ticker"].isin(cfg.ticker_sector.keys())]

def rank_to_symm(series):
    r = series.rank(method='average')
    N = series.notna().sum()
    if N <= 1:
        return pd.Series(np.nan, index=series.index)
    return 2 * ((r - 1) / (N - 1)) - 1

window = 6
def rolling_rank_to_symm(x):
    return x.rolling(window, min_periods=3).apply(
        lambda s: rank_to_symm(s).iloc[-1], raw=False)
data_base[ [c+"_rank" for c in cols] ] = (
    data_base.groupby("ticker")[cols].transform(rolling_rank_to_symm))
for col in data_base.columns:
    if col.endswith("_rank"):
        base_col = col.replace("_rank", "")
        if base_col in cfg.factor_meta:
            if cfg.factor_meta[base_col]["invert_sign"]:
                data_base[col] = -data_base[col]
  
weights_ts = hrp.compute_hrp_factor_weights(
    data_base=data_base,
    factor_meta=cfg.factor_meta,
    sectoral=False)  #el false es para determinar que en el HRP no tome pesos diferenciales por sector(es decir,  promedie la matriz de correlacion entre métricas de todos los sectores)

weights_ts = weights_ts[((weights_ts['Date'] > pd.to_datetime(cfg.strategy_initial_time)))]
data_base = data_base[((data_base['Date'] > pd.to_datetime(cfg.strategy_initial_time)))]


def build_data_base_factors(data_base: pd.DataFrame,
                            weights_ts: pd.DataFrame,
                            factor_meta: dict) -> pd.DataFrame:
    df = data_base.copy()
    w = weights_ts.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    w["Date"] = pd.to_datetime(w["Date"])
    factor_to_cols = {}
    for base, meta in factor_meta.items():
        col = f"{base}_rank"
        if col in df.columns:
            factor_to_cols.setdefault(meta["factor"], []).append(col)
    factors = ["value_score", "quality_score", "credit_score"]
    factors = [f for f in factors if f in factor_to_cols]  # por si falta alguno
    printed_missing_weight = set()  # (date, col)
    rows = []
    for date, df_day in df.groupby("Date"):
        wrow = w[w["Date"] == date]
        if wrow.empty:
            print(f"[AVISO] No hay fila de pesos en weights_ts para la fecha {date.date()}. Omito la fecha.")
            continue
        wser = wrow.iloc[0]  # Series con pesos de ese día
        for _, r in df_day.iterrows():
            out = {"Date": date, "ticker": r["ticker"]}
            for f in factors:
                cols = factor_to_cols.get(f, [])
                if not cols:
                    out[f] = np.nan
                    continue
                vals = []
                wts  = []
                for col in cols:
                    x = r.get(col, np.nan)
                    w_m = wser.get(col, np.nan)
                    if (pd.isna(w_m) or col not in wser.index) and (date, col) not in printed_missing_weight:
                        print(f"[INFO] {date.date()} | '{col}' sin peso asignado en weights_ts → uso 0.")
                        printed_missing_weight.add((date, col))
                    if pd.isna(w_m):
                        w_m = 0.0
                    vals.append(x)
                    wts.append(float(w_m))
                vals = np.asarray(vals, dtype=float)
                wts  = np.asarray(wts, dtype=float)
                msk = ~np.isnan(vals)
                w_eff = wts[msk]
                v_eff = vals[msk]
                denom = w_eff.sum()
                if denom <= 0:
                    out[f] = np.nan  
                else:
                    out[f] = float(np.dot(v_eff, w_eff / denom))
            rows.append(out)
    data_base_factors = pd.DataFrame(rows, columns=["Date", "ticker", "value_score", "quality_score", "credit_score"])
    data_base_factors = data_base_factors.sort_values(["Date", "ticker"]).reset_index(drop=True)
    return data_base_factors

data_base_factors = build_data_base_factors(
    data_base=data_base,
    weights_ts=weights_ts,          
    factor_meta=cfg.factor_meta)

factors=["value_score","quality_score","credit_score"]

data_base_factors = (
    data_base_factors
    .sort_values(by=["ticker", "Date"]))

data_base_factors[factors] = (
    data_base_factors
    .groupby("ticker")[factors]
    .ffill())

data_base_factors[factors]=data_base_factors[factors].fillna(0)   #es por si falta el primer valor(se evita bfill)
data_base_factors["Sector"]=data_base_factors["ticker"].map(cfg.ticker_sector)

def calcular_w_score(row):
    columnas_validas = [col for col in cfg.factor_weights if pd.notnull(row[col])]
    if not columnas_validas:
        return np.nan
    total_peso = sum(cfg.factor_weights[col] for col in columnas_validas)
    pesos_norm = {col: cfg.factor_weights[col]/total_peso for col in columnas_validas}
    return sum(row[col] * pesos_norm[col] for col in columnas_validas)
data_base_factors["W_score"] = data_base_factors.apply(calcular_w_score, axis=1)

def build_portfolio_binario(df: pd.DataFrame, min_per_sector: int = cfg.min_empresas) -> pd.DataFrame:    #para seguir las ponderaciones del merval, tiene que haber siempre un minimo de 2 por sector
    df = df.sort_values(['Date', 'ticker']).copy()
    df['in_base'] = df['W_score'] > 0
    out = []
    for date, d in df.groupby('Date', sort=False):
        base = d.loc[d['in_base'], ['ticker','Sector','W_score']].copy()
        sel = set(base['ticker'])
        cnt = base['Sector'].value_counts().to_dict()
        for s in d['Sector'].dropna().unique():
            n = cnt.get(s, 0)
            if n < min_per_sector:
                k = min_per_sector - n
                cand = (
                    d[(d['Sector'] == s) & (~d['ticker'].isin(sel)) & (d['W_score'].le(0)) & (d['W_score'].notna())]
                      .sort_values(['W_score','ticker'], ascending=[False, True])
                )
                take = cand.head(k)['ticker'].tolist()
                sel.update(take)
                cnt[s] = n + len(take)
        out.append({'Date': date, 'tickers_in_portfolio': sorted(sel)})
    return pd.DataFrame(out)

# def build_portfolio_binario(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.sort_values(['Date', 'ticker'])
#     df['in_portfolio'] = df['W_score'] > 0
#     return (
#         df[df['in_portfolio']]
#           .groupby('Date')['ticker']
#           .apply(list)
#           .reset_index()
#           .rename(columns={'ticker': 'tickers_in_portfolio'}))

def build_portfolio_cuartil(df: pd.DataFrame) -> pd.DataFrame:
    pct = cfg.top_percentil
    df = df.sort_values(['Date', 'ticker'])
    df['W_threshold'] = (
        df.groupby(['Date', 'Sector'])['W_score']
          .transform(lambda x: x.quantile(pct)))
    return (
        df.query('W_score >= W_threshold')
          .groupby('Date')['ticker']
          .apply(list)
          .reset_index()
          .rename(columns={'ticker': 'tickers_in_portfolio'}))

if cfg.metodo_portafolio == "binario":
    portafolio = build_portfolio_binario(data_base_factors)
elif cfg.metodo_portafolio == "cuartil":
    portafolio = build_portfolio_cuartil(data_base_factors)
else:
    raise ValueError(f"Método no reconocido en config: {cfg.metodo_portafolio!r}")

# market_caps=cd.market_caps
# market_caps = market_caps[market_caps["Ticker"].isin(cfg.ticker_sector.keys())]



"""BackTesting"""
resultado_naive, portafolio_fundamental, logs = bt.backtest_sector_lagged_strategy(
    portafolio=portafolio,
    data_base_precios=cd.data_base_precios,
    merval=cd.merval)
resultado_naive['Date'] = resultado_naive.index
