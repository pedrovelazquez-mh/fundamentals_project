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
    

"""BackTesting"""
resultado_naive, portafolio_fundamental, logs = bt.backtest_sector_lagged_strategy(
    portafolio=portafolio,
    data_base_precios=cd.data_base_precios,
    merval=cd.merval)
resultado_naive['Date'] = resultado_naive.index


# from plotly.subplots import make_subplots
# import plotly.graph_objects as go

# # =========================
# # 1) PANEL 1–3 (tu código)
# # =========================
# merval['Date'] = pd.to_datetime(merval['Date'])
# merval = merval.set_index('Date').sort_index()

# resultado_merval = (
#     merval[['close']]
#     .rename(columns={'close': 'valor'})
#     .reindex(resultado_naive.index)
#     .ffill()
# )

# V0_naive  = resultado_naive['valor'].iloc[0]
# V0_merval = resultado_merval['valor'].iloc[0]
# capital_naive  = resultado_naive['valor']  / V0_naive
# capital_merval = resultado_merval['valor'] / V0_merval
# outperf_ratio = (capital_naive / capital_merval).rename("Ratio Naive/Merval")


# q_naive  = resultado_naive['valor'].resample('Q').last().pct_change() * 100
# q_merval = resultado_merval['valor'].resample('Q').last().pct_change() * 100
# idx = q_naive.index.intersection(q_merval.index)

# wins = (q_naive.loc[idx] > q_merval.loc[idx]).sum()
# total = len(idx)
# win_rate = wins / total * 100 if total > 0 else np.nan

# rf_daily = 0.0 / 252
# retornos_naive = resultado_naive['valor'].pct_change().dropna()
# exceso_retornos = retornos_naive - rf_daily
# sharpe_ratio = (exceso_retornos.mean() / exceso_retornos.std()) * np.sqrt(252) if exceso_retornos.std() > 0 else np.nan

# # =========================
# # 2) PANEL 4 (área apilada)
# # =========================
# df = portafolio_fundamental.copy()
# df.index = pd.to_datetime(df.index)
# df = df.sort_index().fillna(0)

# # Detectar columna de efectivo si existe
# cash_col = 'efectivo' if 'efectivo' in df.columns else None
# comp_cols = [c for c in df.columns if c != cash_col]

# # Orden global por peso promedio (grandes al fondo de la pila)
# order = (df[comp_cols].mean().sort_values(ascending=False)).index.tolist()
# ordered_cols = ([cash_col] + order) if cash_col else order

# # Normalizar por fila (proporciones 0–1)
# row_sum = df[ordered_cols].sum(axis=1).replace(0, np.nan)
# df_norm = df[ordered_cols].div(row_sum, axis=0).fillna(0)

# # =========================
# # 3) FIGURA CON 4 SUBPLOTS
# # =========================
# fig = make_subplots(
#     rows=4, cols=1,
#     shared_xaxes=True,
#     vertical_spacing=0.07,
#     subplot_titles=(
#         "Capital Acumulado Diario: Estrategia Naive vs. Merval",
#         "Ratio de Capital: Naive / Merval",
#         "Rendimiento Trimestral: Estrategia Naive vs. Merval",
#         "Pesos del Portafolio a lo largo del tiempo"
#     ),
#     row_heights=[0.34, 0.18, 0.18, 0.30]
# )

# # --- Row 1: capital acumulado
# fig.add_trace(
#     go.Scatter(x=capital_naive.index, y=capital_naive, mode='lines',
#                 name='Estrategia Naive', line=dict(color='blue')),
#     row=1, col=1
# )
# fig.add_trace(
#     go.Scatter(x=capital_merval.index, y=capital_merval, mode='lines',
#                 name='Merval', line=dict(color='red', dash='dash')),
#     row=1, col=1
# )

# # --- Row 2: ratio
# fig.add_trace(
#     go.Scatter(
#         x=outperf_ratio.index, y=outperf_ratio, mode='lines',
#         name='Ratio Naive / Merval', line=dict(color='green'),
#         hovertemplate='Fecha=%{x|%Y-%m-%d}<br>Ratio=%{y:.3f}<extra></extra>'
#     ),
#     row=2, col=1
# )
# fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=2, col=1)

# # --- Row 3: barras trimestrales
# fig.add_trace(
#     go.Bar(x=idx, y=q_naive.loc[idx], name="Estrategia Naive", marker_color='blue'),
#     row=3, col=1
# )
# fig.add_trace(
#     go.Bar(x=idx, y=q_merval.loc[idx], name="Merval", marker_color='red'),
#     row=3, col=1
# )

# # --- Row 4: área apilada de pesos
# for col in ordered_cols:
#     fig.add_trace(
#         go.Scatter(
#             x=df_norm.index,
#             y=df_norm[col],
#             mode='lines',
#             stackgroup='one',
#             name=col,
#             hovertemplate="%{x|%Y-%m-%d}<br>%{fullData.name}: %{y:.2%}<extra></extra>"
#         ),
#         row=4, col=1
#     )

# # --- Anotaciones y ejes
# anotacion_texto = (
#     f"<span style='margin-right:30px'>Win Rate: {win_rate:.2f}% ({wins}/{total})</span>"
#     f"<span>     Sharpe Ratio (anualizado): {sharpe_ratio:.2f}</span>"
# )
# fig.add_annotation(
#     text=anotacion_texto,
#     x=0.5, y=1, xref='paper', yref='paper',
#     showarrow=False, font=dict(size=14), align="center"
# )

# fig.update_yaxes(title_text="Capital Acumulado", type="log", row=1, col=1)
# fig.update_yaxes(title_text="Ratio", tickformat=".2f", row=2, col=1)
# fig.update_yaxes(title_text="Trimestral [%]", tickformat=".2f", ticksuffix="%", row=3, col=1)
# fig.update_yaxes(title_text="Peso", tickformat=".0%", range=[0, 1], row=4, col=1)

# fig.update_layout(
#     height=1300, width=1000,
#     title_text="Estrategia Naive vs. Merval (Capital, Ratio, Retorno Trimestral y Pesos del Portafolio)",
#     barmode='group',
#     template="plotly_white",
#     legend=dict(
#         orientation="h",              # horizontal
#         yanchor="top", y=-0.25,       # debajo de todo
#         xanchor="center", x=0.5,      # centrada
#         bordercolor="gray", borderwidth=1,
#         bgcolor="rgba(255,255,255,0.8)"
#     ),
#     hovermode="x unified"
# )

# # Exportá la versión combinada
# fig.write_html("naive_vs_merval_Ranks.html")







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
