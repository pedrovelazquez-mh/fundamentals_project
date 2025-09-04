import os
import pandas as pd
import numpy as np
import sys
from collections import defaultdict
config_dir = r"C:\Users\Pedro\Research\Fundamentals\Bloomberg"
if config_dir not in sys.path:
    sys.path.insert(0, config_dir)
import config_fundamentals as cfg
import cleaning_data as cd
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list
from scipy.spatial.distance import squareform
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
  

def _pairwise_valid_counts(X: pd.DataFrame) -> pd.DataFrame:
    B = X.notna().astype(int)
    return pd.DataFrame(B.T.dot(B), index=X.columns, columns=X.columns)

def _fisher_avg_corr(mats, counts, all_cols):
    k = len(all_cols)
    Z_sum = np.zeros((k, k), dtype=float)
    W_sum = np.zeros((k, k), dtype=float)

    for R_i, N_i in zip(mats, counts):
        Ri = R_i.reindex(index=all_cols, columns=all_cols)
        Ni = N_i.reindex(index=all_cols, columns=all_cols).fillna(0.0)
        arr = Ri.values
        arr = np.clip(arr, -0.999999, 0.999999)
        Zi = np.arctanh(arr)  # Fisher z
        Wi = np.maximum(Ni.values - 3.0, 0.0) 
        mask = np.isfinite(Zi) & (Wi > 0)
        Z_sum[mask] += (Zi * Wi)[mask]
        W_sum[mask] += Wi[mask]

    with np.errstate(divide='ignore', invalid='ignore'):
        Z_bar = np.divide(Z_sum, W_sum, out=np.zeros_like(Z_sum), where=W_sum > 0)
    R_bar = np.tanh(Z_bar)
    np.fill_diagonal(R_bar, 1.0)
    R_bar = pd.DataFrame(R_bar, index=all_cols, columns=all_cols)
    return R_bar

def _nearest_psd_correlation(R: pd.DataFrame, eps=1e-8) -> pd.DataFrame:
    A = 0.5 * (R.values + R.values.T)
    vals, vecs = np.linalg.eigh(A)
    vals_clipped = np.clip(vals, eps, None)
    A_psd = (vecs * vals_clipped) @ vecs.T
    d = np.sqrt(np.clip(np.diag(A_psd), eps, None))
    A_corr = (A_psd / d).T / d
    np.fill_diagonal(A_corr, 1.0)
    return pd.DataFrame(A_corr, index=R.index, columns=R.columns)

def _hrp_weights_from_corr(R: pd.DataFrame, linkage_method="average") -> pd.Series:
    D = np.sqrt(np.clip(0.5 * (1.0 - R.values), 0.0, None))
    Z = linkage(squareform(D, checks=False), method=linkage_method)
    order = leaves_list(optimal_leaf_ordering(Z, squareform(D, checks=False)))
    R_ord = R.iloc[order, order]
    cols_ord = R_ord.columns

    def get_ivp(cov):
        iv = 1.0 / np.clip(np.diag(cov), 1e-12, None)
        return iv / iv.sum()

    def get_cluster_var(cov):
        w = get_ivp(cov)
        return float(w.T @ cov @ w)

    # Bisección recursiva (Lopez de Prado)
    w = pd.Series(1.0, index=cols_ord, dtype=float)
    clusters = [np.array(range(len(cols_ord)))]
    while len(clusters) > 0:
        new_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            split = int(len(cluster) / 2)
            c1 = cluster[:split]
            c2 = cluster[split:]
            cov = R_ord.values
            cov_1 = cov[np.ix_(c1, c1)]
            cov_2 = cov[np.ix_(c2, c2)]
            var1 = get_cluster_var(cov_1)
            var2 = get_cluster_var(cov_2)
            # asignación inversa al riesgo
            alpha = 1.0 - var1 / (var1 + var2)
            w.iloc[c1] *= alpha
            w.iloc[c2] *= (1.0 - alpha)
            if len(c1) > 1: new_clusters.append(c1)
            if len(c2) > 1: new_clusters.append(c2)
        clusters = new_clusters

    w = w.reindex(R.columns).fillna(0.0)
    s = w.sum()
    if s <= 0:
        w = pd.Series(1.0 / len(R.columns), index=R.columns, dtype=float)
    else:
        w = w / s
    return w
def _to_series(x, idx):
    s = pd.Series(0.0, index=idx, dtype=float)
    if x is None:
        return s
    if isinstance(x, pd.Series):
        y = x.reindex(idx).fillna(0.0).astype(float)
    else:
        y = pd.Series(x, index=idx).fillna(0.0).astype(float)
    y[y < 0] = 0.0
    if y.sum() > 0:
        y = y / y.sum()
    return y
def _smooth(w_hrp: pd.Series, w_prev: pd.Series | None, alpha: float) -> pd.Series:
    idx = w_hrp.index
    w_prev = _to_series(w_prev, idx)
    w_eff = alpha * w_hrp + (1 - alpha) * w_prev
    w_eff = w_eff.clip(lower=0.0)
    s = w_eff.sum()
    if s <= 0:
        return pd.Series(1.0 / len(idx), index=idx, dtype=float)
    return w_eff / s

def compute_hrp_factor_weights_timeseries(
    data_base: pd.DataFrame,
    factor_meta: dict,
    alpha: float = 0.1,
    linkage_method: str = "average",
    min_obs_per_pair: int = 8,    # mínimo para computar correlación a partir de un ticker
    min_metrics_per_factor: int = 2,  # si <2 → igual-ponderado dentro del factor
    use_psd_projection: bool = True,
    dates: list | None = None,    # si None usa todas las fechas ordenadas
):
   
    print("Preparando columnas y mapeos de factores_HRP INTRA FACTORS...")
    base_to_factor = {k: v["factor"] for k, v in factor_meta.items()}
    rank_cols_all = [c for c in data_base.columns if c.endswith("_rank")]
    rank_cols_mapped = [c for c in rank_cols_all if c.replace("_rank", "") in base_to_factor]

    dropped = sorted(set(rank_cols_all) - set(rank_cols_mapped))
    if dropped:
        print(f"Aviso: {len(dropped)} columnas *_rank* no mapeadas a factor y se ignoran (peso=0). Ejemplo: {dropped[:3]}")
    factor_to_rankcols = {}
    for base, fact in base_to_factor.items():
        rc = base + "_rank"
        if rc in rank_cols_mapped:
            factor_to_rankcols.setdefault(fact, []).append(rc)
    if dates is None:
        dates = sorted(pd.to_datetime(data_base["Date"]).dropna().unique())
    else:
        dates = sorted(pd.to_datetime(d) for d in dates)
    prev_eff_by_factor: dict[str, pd.Series] = {}
    out_cols = ["Date"] + sorted(rank_cols_mapped) + sorted(dropped)  # dropped van al final como 0
    out_rows = []
    print(f"Iterando {len(dates)} fechas (expanding, sin mirar futuro)...")
    for t in dates:
        df_t = data_base[pd.to_datetime(data_base["Date"]) <= t]
        mats, counts = [], []
        for tkr, sub in df_t.groupby("ticker"):
            X = sub[rank_cols_mapped].copy()
            X = X.loc[:, X.std(numeric_only=True) > 0]
            if X.shape[1] < 2 or len(X) < min_obs_per_pair:
                continue
            R_i = X.corr(method="pearson")
            N_i = _pairwise_valid_counts(X)
            valid = (N_i.values >= min_obs_per_pair)
            if valid.sum() == 0:
                continue
            mats.append(R_i)
            counts.append(N_i)

        if not mats:
            # si no hay info todavía, igual-ponderado por factor
            print(f"{t.date()} | sin matrices válidas; aplico igual-ponderado por factor.")
            row = {"Date": pd.to_datetime(t)}
            # pesos iguales en cada factor
            for fact, cols in factor_to_rankcols.items():
                if len(cols) == 0:
                    continue
                w_eff = pd.Series(1.0 / len(cols), index=cols, dtype=float)
                prev_eff_by_factor[fact] = w_eff
                row.update({c: w_eff.get(c, 0.0) for c in cols})
            for c in dropped:
                row[c] = 0.0
            out_rows.append(row)
            continue

        # correlación promedio (Fisher-z)
        all_cols = sorted({c for R_i in mats for c in R_i.columns})
        R_bar = _fisher_avg_corr(mats, counts, all_cols)
        if use_psd_projection:
            R_bar = _nearest_psd_correlation(R_bar)

        row = {"Date": pd.to_datetime(t)}
        for fact, cols in factor_to_rankcols.items():
            cols_f = [c for c in cols if c in R_bar.columns]
            if len(cols_f) >= min_metrics_per_factor:
                Rf = R_bar.loc[cols_f, cols_f]
                w_hrp = _hrp_weights_from_corr(Rf, linkage_method=linkage_method)
            else:
                w_hrp = pd.Series(1.0 / max(len(cols_f), 1), index=cols_f, dtype=float)

            # suavizado temporal con pesos efectivos previos del mismo factor
            w_prev = prev_eff_by_factor.get(fact, None)
            w_prev = None if w_prev is None else w_prev.reindex(cols_f)
            w_eff = _smooth(w_hrp, w_prev, alpha=alpha)

            # renormalizar dentro del factor (invariante)
            if w_eff.sum() > 0:
                w_eff = w_eff / w_eff.sum()
            prev_eff_by_factor[fact] = w_eff.copy()
            for c in cols:
                row[c] = float(w_eff.get(c, 0.0))
        for c in dropped:
            row[c] = 0.0

        out_rows.append(row)
        if len(out_rows) % 10 == 0:
            print(f"  ... procesadas {len(out_rows)} fechas, última: {t.date()}")
    out = pd.DataFrame(out_rows, columns=out_cols)
    out = out.sort_values("Date").reset_index(drop=True)
    return out
              
weights_ts = compute_hrp_factor_weights_timeseries(
    data_base=data_base,
    factor_meta=cfg.factor_meta)
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
    factor_meta=cfg.factor_meta     
)

















# import pandas as pd, numpy as np
# import plotly.graph_objects as go
# import plotly.io as pio

# df = data_base_factors.copy()
# cols = ["value_score", "quality_score", "credit_score"]

# # Correlaciones por ticker (todo el período), filtrando columnas con poca info
# by_ticker = {}
# for tkr, g in df.groupby("ticker"):
#     valid = [c for c in cols if g[c].notna().sum() >= 3 and g[c].nunique(dropna=True) > 1]
#     if len(valid) < 2: 
#         continue
#     R = g[valid].corr()
#     np.fill_diagonal(R.values, 1.0)
#     by_ticker[tkr] = R

# if not by_ticker:
#     raise ValueError("Sin matrices válidas.")

# tickers = sorted(by_ticker)
# t0 = tickers[0]
# R0 = by_ticker[t0]

# fig = go.Figure(go.Heatmap(
#     z=R0.values, x=R0.columns, y=R0.index,
#     zmin=-1, zmax=1, colorscale="RdBu",
#     colorbar=dict(title="ρ"),
#     hovertemplate="X:%{x}<br>Y:%{y}<br>ρ=%{z:.3f}<extra></extra>"
# ))

# fig.update_layout(
#     title=f"Correlación de factores — {t0} (todo el período)",
#     xaxis=dict(tickangle=45), margin=dict(l=60, r=60, t=80, b=60),
#     updatemenus=[dict(
#         type="dropdown", x=1.0, xanchor="right", y=1.15, yanchor="top", showactive=True,
#         buttons=[dict(
#             label=t,
#             method="update",
#             args=[{"z":[by_ticker[t].values], "x":[by_ticker[t].columns], "y":[by_ticker[t].index]},
#                   {"title": f"Correlación de factores — {t} (todo el período)"}]
#         ) for t in tickers]
#     )]
# )

# fig.write_html("corr_factors.html")
# print("HTML escrito: corr_factors_por_ticker.html")















# df = weights_ts.copy()
# df['Date'] = pd.to_datetime(df['Date'])
# df = df.sort_values('Date')

# # Mapear factor -> columnas *_rank presentes
# factor_to_cols = {}
# for base, meta in cfg.factor_meta.items():
#     col = f"{base}_rank"
#     if col in df.columns:
#         factor_to_cols.setdefault(meta["factor"], []).append(col)

# title_map = {"value_score": "Value", "quality_score": "Quality", "credit_score": "Credit"}
# present_factors = [f for f in ["value_score","quality_score","credit_score"]
#                    if f in factor_to_cols and factor_to_cols[f]]

# if not present_factors:
#     raise ValueError("No hay columnas *_rank* mapeadas a factores en weights_ts.")

# default_factor = present_factors[0]

# fig = go.Figure()
# trace_factors = []  # para saber a qué factor pertenece cada traza

# for f in present_factors:
#     for col in sorted(factor_to_cols[f]):
#         fig.add_trace(go.Scatter(
#             x=df["Date"], y=df[col], mode="lines", name=col,
#             visible=(f == default_factor),
#             hovertemplate="Fecha: %{x|%Y-%m-%d}<br>Métrica: "+col+"<br>Peso: %{y:.4f}<extra></extra>"
#         ))
#         trace_factors.append(f)

# # Dropdown sin "Todos"
# buttons = []
# for f in present_factors:
#     vis = [tf == f for tf in trace_factors]
#     buttons.append(dict(
#         label=title_map.get(f, f),
#         method="update",
#         args=[{"visible": vis},
#               {"title": f"Pesos por métrica — {title_map.get(f, f)}"}]
#     ))

# fig.update_layout(
#     title=f"Pesos por métrica — {title_map.get(default_factor, default_factor)}",
#     xaxis_title="Fecha",
#     yaxis_title="Peso",
#     yaxis=dict(range=[0.05, 0.3]),
#     legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.2, yanchor="top"),
#     margin=dict(l=60, r=30, t=80, b=90),
#     hovermode="x unified",
#     updatemenus=[dict(
#         type="dropdown",
#         x=1.0, xanchor="right",
#         y=1.15, yanchor="top",
#         showactive=True,
#         direction="down",
#         buttons=buttons
#     )]
# )

# fig.write_html("pesos_factores.html")




















# import plotly.subplots as sp
# import plotly.graph_objects as go

# metrics = ["AVERAGE_PRICE_TO_BOOK_RATIO","AVERAGE_PRICE_EARNINGS_RATIO","RETURN_ON_ASSET"]
# fig = sp.make_subplots(
#     rows=len(metrics), cols=1, shared_xaxes=True,
#     subplot_titles=metrics,
#     specs=[[{"secondary_y": True}] for _ in metrics])
# for i, m in enumerate(metrics, start=1):
#     fig.add_trace(
#         go.Scatter(x=data_base["Date"], y=data_base[m],
#                    name=f"{m} raw", line=dict(color="blue")),
#         row=i, col=1, secondary_y=False)
#     rank_col = f"{m}_rank"
#     if rank_col in data_base:
#         fig.add_trace(
#             go.Scatter(x=data_base["Date"], y=data_base[rank_col],
#                        name=f"{m} rank", line=dict(color="red", dash="dot")),
#             row=i, col=1, secondary_y=True)
# fig.update_layout(
#     height=300*len(metrics),
#     hovermode="x unified")
# for i, m in enumerate(metrics, start=1):
#     fig.update_yaxes(title_text="Raw", row=i, col=1, secondary_y=False)
#     fig.update_yaxes(title_text="Rank", row=i, col=1, secondary_y=True)
# fig.write_html("rank_ALUAR.html")

                
                

# import numpy as np, pandas as pd, plotly.graph_objects as go, logging, traceback
# from scipy.spatial.distance import squareform
# from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list

# logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
# log = logging.getLogger("corr-heatmap")

# corr_method = "pearson"
# linkage_method = "average"
# write_file = "Matrices de correlacion.html"

# def cluster_order(corr: pd.DataFrame, method="average"):
#     corr = corr.copy()
#     np.fill_diagonal(corr.values, 1.0)
#     dist = np.clip(1.0 - corr.values, 0.0, 2.0)
#     Z = linkage(squareform(dist, checks=False), method=method)
#     idx = leaves_list(optimal_leaf_ordering(Z, squareform(dist, checks=False)))
#     return corr.index[idx].tolist()
# try:
#     if "data_base" not in globals():
#         raise NameError("El objeto 'data_base' no está en el entorno.")
#     rank_cols = [c for c in data_base.columns if c.endswith("_rank")]
#     if not rank_cols: raise ValueError("No hay columnas *_rank.")
#     by_ticker = {}
#     log.info("Iniciando agrupación por ticker...")
#     for tkr, df in data_base.groupby("ticker"):
#         try:
#             sub = df[rank_cols]
#             cols = sub.columns[sub.notna().any()]
#             if len(cols) < 2: 
#                 log.warning(f"[{tkr}] <2 columnas con datos. Omito.")
#                 continue
#             sub = sub[cols]
#             nonconst = sub.std(numeric_only=True)
#             sub = sub[nonconst.index[nonconst > 0]]
#             if sub.shape[1] < 2:
#                 log.warning(f"[{tkr}] <2 no constantes. Omito.")
#                 continue
#             corr = sub.corr(method=corr_method)
#             finite = np.isfinite(corr.values)
#             keep = corr.columns[finite.all(0) & finite.all(1)]
#             if len(keep) >= 2:
#                 corr = corr.loc[keep, keep]
#             else:
#                 log.warning(f"[{tkr}] corr con NaN/Inf. Uso orden original.")
#                 corr_ord = corr.fillna(0.0)
#                 by_ticker[tkr] = {
#                     "z": corr_ord.values, "x": corr_ord.columns.tolist(),
#                     "y": corr_ord.index.tolist(), "text": corr_ord.round(2).values,
#                     "n_obs": len(sub)}
#                 continue
#             try:
#                 order = cluster_order(corr, linkage_method)
#                 corr_ord = corr.loc[order, order]
#                 log.info(f"[{tkr}] Clustering OK. Orden: {order}")
#             except Exception as e:
#                 log.error(f"[{tkr}] Error clustering. Fallback. {e}")
#                 log.debug(traceback.format_exc())
#                 corr_ord = corr

#             if not np.isfinite(corr_ord.values).all():
#                 log.error(f"[{tkr}] No finitos tras limpieza. Reemplazo y sigo.")
#                 arr = np.nan_to_num(corr_ord.values, nan=0.0, posinf=1.0, neginf=-1.0)
#                 corr_ord = pd.DataFrame(arr, index=corr_ord.index, columns=corr_ord.columns)
#             by_ticker[tkr] = {
#                 "z": corr_ord.values, "x": corr_ord.columns.tolist(),
#                 "y": corr_ord.index.tolist(), "text": corr_ord.round(2).values,
#                 "n_obs": len(sub)}
#         except Exception as e_t:
#             log.error(f"Error en {tkr}: {e_t}")
#             log.debug(traceback.format_exc())
#     if not by_ticker:
#         raise ValueError("Sin matrices válidas para graficar.")
#     tickers = sorted(by_ticker.keys())
#     t0 = tickers[0]
#     log.info(f"Se graficará {len(tickers)} tickers. Ticker inicial: {t0}")
#     fig = go.Figure(data=go.Heatmap(
#         z=by_ticker[t0]["z"], x=by_ticker[t0]["x"], y=by_ticker[t0]["y"],
#         colorscale="RdBu", zmin=-1, zmax=1, colorbar=dict(title="Correlación"),
#         text=by_ticker[t0]["text"], texttemplate="%{text}",
#         hovertemplate="X:%{x}<br>Y:%{y}<br>ρ=%{z:.3f}<extra></extra>"))
#     fig.update_layout(
#         title=f"Matriz de Correlación por Ticker — {t0} (n={by_ticker[t0]['n_obs']})",
#         xaxis=dict(tickangle=45), yaxis=dict(autorange="reversed"),
#         updatemenus=[dict(type="dropdown", x=1.0, xanchor="right", y=1.15, yanchor="top",
#                           direction="down", showactive=True,
#                           buttons=[dict(label=t, method="update",
#                             args=[{"z":[by_ticker[t]["z"]], "x":[by_ticker[t]["x"]],
#                                     "y":[by_ticker[t]["y"]], "text":[by_ticker[t]["text"]]},
#                                   {"title": f"Matriz de Correlación por Ticker — {t} (n={by_ticker[t]['n_obs']})"}]
#                           ) for t in tickers])], margin=dict(l=60, r=60, t=90, b=60))
#     fig.write_html(write_file)
#     log.info(f"Archivo HTML escrito: {write_file}")
# except Exception as e:
#     log.critical(f"Fallo crítico: {e}")
#     log.debug(traceback.format_exc())
#     raise    
   

   
    
   
    
   