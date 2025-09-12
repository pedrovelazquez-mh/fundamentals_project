import os
import pandas as pd
import numpy as np
import cleaning_data as cd   #modulo
import backtest as bt  #modulo
import config_fundamentals as cfg #modulo
import HRP as hrp #modulo 

data_base=cd.data_base
data_base = data_base[~((data_base['Date'] < pd.to_datetime(cfg.corr_initial_time)))]
data_base = data_base[~((data_base['Date'] >= pd.to_datetime(cfg.ending_time)))]
data_base = data_base[data_base["ticker"].isin(cfg.ticker_sector.keys())]
data_base["Sector"]=data_base["ticker"].map(cfg.ticker_sector)
cols_metricas = [c for c in data_base.columns if c not in ['Date', 'ticker', 'Sector']]


"""Cross-Sectional"""
def normalize_cross_sectional(df, col):
    def z(s):
        iqr = s.quantile(0.75) - s.quantile(0.25)
        if iqr > 0:
            return ((s - s.median()) / iqr).where(s.notna())
        else:
            return pd.Series([float('nan')] * len(s), index=s.index)
    return df.groupby(['Sector', 'Date'], group_keys=False)[col].apply(z)

data_base_cross_sectional = data_base.copy()
for col in cols_metricas:
    if col in data_base_cross_sectional.columns:
        data_base_cross_sectional[f"{col}_zscore"] = normalize_cross_sectional(
            data_base_cross_sectional, col)

cols_z = [f"{col}_zscore" for col in cols_metricas]
data_base_cross_sectional = data_base_cross_sectional[
    ['Date', 'ticker', 'Sector']
    + [z for z in cols_z if z in data_base_cross_sectional.columns]]

for metric, meta in cfg.factor_meta.items():
    if meta["invert_sign"]:
        col_name = f"{metric}_zscore"
        if col_name in data_base_cross_sectional.columns:
            data_base_cross_sectional[col_name] = data_base_cross_sectional[col_name].apply(
                lambda x: -x if x != 0 else x)

def rank_to_symm(series):
    r = series.rank(method='average', ascending=True)  # peor->1, mejor->N
    N = series.notna().sum()
    if N <= 1:
        return pd.Series(np.nan, index=series.index)
    return 2 * ((r - 1) / (N - 1)) - 1

for col in [c for c in data_base_cross_sectional.columns if c.endswith('_zscore')]:
    data_base_cross_sectional[col] = (
        data_base_cross_sectional
        .groupby(['Sector','Date'], group_keys=False)[col]
        .apply(rank_to_symm))

data_base_cross_sectional.rename(columns={col: col.replace('_zscore', '_rank') for col in data_base_cross_sectional.columns if col.endswith('_zscore')}, inplace=True)

weights_cs = hrp.compute_hrp_factor_weights(
    data_base=data_base_cross_sectional,
    factor_meta=cfg.factor_meta,
    sectoral=True) #el true es para determinar que en el HRP  tome pesos diferenciales por sector(es decir,  promedie la matriz de correlacion entre métricas del mismo sector)


weights_cs = weights_cs[((weights_cs['Date'] > pd.to_datetime(cfg.strategy_initial_time)))]
data_base_cross_sectional = data_base_cross_sectional[((data_base_cross_sectional['Date'] > pd.to_datetime(cfg.strategy_initial_time)))]


def build_portfolio_factors_sectoral(
    data_base: pd.DataFrame,
    weights_cs: pd.DataFrame,
    factor_meta: dict
) -> pd.DataFrame:
    """
    Combina ranks por ticker con pesos sectoriales por fecha (HRP sectoral=True)
    para obtener factor scores por ticker y fecha.
    Salida: DataFrame con columnas: Date, ticker, <factor>_score
    """
    import numpy as np
    import pandas as pd

    df = data_base.copy()
    w  = weights_cs.copy()

    df["Date"] = pd.to_datetime(df["Date"])
    w["Date"]  = pd.to_datetime(w["Date"])

    if "Sector" not in df.columns:
        raise ValueError("data_base debe contener la columna 'Sector'.")

    if not {"Date", "Sector"}.issubset(set(w.columns)):
        raise ValueError("weights_cs (sectoral=True) debe contener columnas 'Date' y 'Sector'.")

    # Mapeo: factor -> columnas de métricas *_rank
    factor_to_cols = {}
    for base, meta in factor_meta.items():
        col = f"{base}_rank"
        if col in df.columns:
            factor_to_cols.setdefault(meta["factor"], []).append(col)

    factor_names = sorted(factor_to_cols.keys())          # ej: ['credit','quality','value']
    out_factor_cols = [f"{f}_score" for f in factor_names]

    # Índice rápido de pesos: (Date, Sector) -> Series de pesos
    w_idx = {(d, s): g.iloc[0] for (d, s), g in w.groupby(["Date", "Sector"], sort=False)}

    rows = []
    for date, day in df.groupby("Date"):
        for _, r in day.iterrows():
            sector = r["Sector"]
            wser = w_idx.get((date, sector), None)  # puede faltar la fila de ese sector/fecha

            out = {"Date": date, "ticker": r["ticker"]}
            for f, cols in factor_to_cols.items():
                vals = np.array([r.get(c, np.nan) for c in cols], dtype=float)

                if wser is None:
                    # Fallback simple: equal-weight con las métricas disponibles
                    m = ~np.isnan(vals)
                    out[f"{f}_score"] = float(vals[m].mean()) if m.any() else np.nan
                    continue

                # Peso 0 si no está la columna en weights (simple y robusto)
                wts = np.array([float(wser.get(c, 0.0)) for c in cols], dtype=float)

                m = ~np.isnan(vals)
                denom = wts[m].sum()
                out[f"{f}_score"] = float(np.dot(vals[m], wts[m] / denom)) if denom > 0 else np.nan

            rows.append(out)

    out = pd.DataFrame(rows, columns=["Date", "ticker"] + out_factor_cols)
    return out.sort_values(["Date", "ticker"]).reset_index(drop=True)

data_base_cross_sectional_factors = build_portfolio_factors_sectoral(
    data_base=data_base_cross_sectional,  # requiere Date, ticker, Sector, *_rank
    weights_cs=weights_cs,                # salida de sectoral=True
    factor_meta=cfg.factor_meta
)

data_base_cross_sectional_factors.columns = [c.replace("_score_score", "_score") for c in data_base_cross_sectional_factors.columns]
factors=["value_score","quality_score","credit_score"]

data_base_cross_sectional_factors = (
    data_base_cross_sectional_factors
    .sort_values(by=["ticker", "Date"]))

data_base_cross_sectional_factors[factors] = (
    data_base_cross_sectional_factors
    .groupby("ticker")[factors]
    .ffill())

data_base_cross_sectional_factors[factors]=data_base_cross_sectional_factors[factors].fillna(0)   #es por si falta el primer valor(se evita bfill)

data_base_cross_sectional_factors["Sector"]=data_base_cross_sectional_factors["ticker"].map(cfg.ticker_sector)


def calcular_w_score(row):
    columnas_validas = [col for col in cfg.factor_weights if pd.notnull(row[col])]
    if not columnas_validas:
        return np.nan
    total_peso = sum(cfg.factor_weights[col] for col in columnas_validas)
    pesos_norm = {col: cfg.factor_weights[col]/total_peso for col in columnas_validas}
    return sum(row[col] * pesos_norm[col] for col in columnas_validas)
data_base_cross_sectional_factors["W_score"] = data_base_cross_sectional_factors.apply(calcular_w_score, axis=1)

data_base_cross_sectional_factors["W_score"] = (
    data_base_cross_sectional_factors
    .sort_values(by=["ticker", "Date"])
    .groupby("ticker")["W_score"]
    .fillna(method="ffill"))
data_base_cross_sectional_factors["W_score"]=data_base_cross_sectional_factors["W_score"].fillna(0)
def build_portfolio_binario(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['Date', 'ticker'])
    df['in_portfolio'] = df['W_score'] > 0
    return (
        df[df['in_portfolio']]
          .groupby('Date')['ticker']
          .apply(list)
          .reset_index()
          .rename(columns={'ticker': 'tickers_in_portfolio'}))

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
    portafolio = build_portfolio_binario(data_base_cross_sectional_factors)
elif cfg.metodo_portafolio == "cuartil":
    portafolio = build_portfolio_cuartil(data_base_cross_sectional_factors)
else:
    raise ValueError(f"Método no reconocido en config: {cfg.metodo_portafolio!r}")
    

"""BackTesting
resultado_naive, portafolio_fundamental, logs = bt.backtest_sector_lagged_strategy(
    portafolio=portafolio,
    data_base_precios=cd.data_base_precios,
    merval=cd.merval)
 #   market_cap=cd.market_caps)
resultado_naive['Date'] = resultado_naive.index
"""


"""Earnings release"""
# releasing_data = {}
# for archivo in os.listdir(cfg.carpeta_releasing):
#     if archivo.endswith(".xlsx"):
#         nombre_df = os.path.splitext(archivo)[0]
#         ruta_completa = os.path.join(cfg.carpeta_releasing, archivo)
#         releasing_data[nombre_df] = (
#             pd.read_excel(ruta_completa, header=0)   
#               .drop(index=0)                         
#               .iloc[:, [0, 1, 2]])
#         releasing_data[nombre_df].iloc[:, 0] = pd.to_datetime(releasing_data[nombre_df].iloc[:, 0])
#         releasing_data[nombre_df].iloc[:, 2] = pd.to_datetime(releasing_data[nombre_df].iloc[:, 2],format="%m/%y") + pd.offsets.MonthEnd(0)
        
# releasing_data = {
#     ticker: data
#     for ticker, data in releasing_data.items()
#     if ticker in cfg.ticker_sector}

# for tkr in releasing_data:
#     releasing_data[tkr]["Fecha an"] = pd.to_datetime(releasing_data[tkr]["Fecha an"])
#     releasing_data[tkr]["Per final"] = pd.to_datetime(releasing_data[tkr]["Per final"])
#     releasing_data[tkr]["lag"] = (
#         releasing_data[tkr]["Fecha an"].dt.normalize()
#         - releasing_data[tkr]["Per final"].dt.normalize()
#     ).dt.days
#     releasing_data[tkr]["mes"] = releasing_data[tkr]["Per final"].dt.month

# releasing_df = []
# for ticker, df in releasing_data.items():
#     df = df.copy()
#     df['ticker'] = ticker
#     releasing_df.append(df)
# releasing_df = pd.concat(releasing_df, ignore_index=True)
