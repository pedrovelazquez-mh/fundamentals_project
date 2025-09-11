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





# resultado_naive = resultado_naive[((resultado_naive['Date'] > pd.to_datetime("2008-05-20")))]
# cross_sectional=pd.read_csv(r"C:\Users\Pedro\Research\Fundamentals\Bloomberg\Versión 1\cross_sectional.csv")
# cross_sectional["Date"]=pd.to_datetime(cross_sectional["Date"])
# cross_sectional = cross_sectional[((cross_sectional['Date'] > pd.to_datetime("2008-05-20")))]
# V0_cross_sectional = cross_sectional['valor'].iloc[0]
# cross_sectional["valor"]=cross_sectional["valor"]/V0_cross_sectional
# cross_sectional=cross_sectional.rename(columns={"valor":"cross_sectional","activos":"portafolio_cross_sectional"})
# V0_naive  = resultado_naive['valor'].iloc[0]
# resultado_naive["valor"]=resultado_naive["valor"]/V0_naive
# resultado_naive=resultado_naive.rename(columns={"valor":"time_series","activos":"portafolio_time_series"})

# blend=pd.merge(cross_sectional,resultado_naive,on="Date")
# blend["ret_cross_sectional"] = blend["cross_sectional"].pct_change()
# blend["ret_time_series"] = blend["time_series"].pct_change()
# blend = blend.drop("Unnamed: 0", axis=1)
# correlacion = blend["ret_cross_sectional"].corr(blend["ret_time_series"])
# print(correlacion)



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
#                name='Estrategia Naive', line=dict(color='blue')),
#     row=1, col=1
# )
# fig.add_trace(
#     go.Scatter(x=capital_merval.index, y=capital_merval, mode='lines',
#                name='Merval', line=dict(color='red', dash='dash')),
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
# fig.write_html("time_series_vs_merval.html")

