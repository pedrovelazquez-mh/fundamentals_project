import os
import pandas as pd
import numpy as np
from collections import Counter
from collections import defaultdict
import sys
import glob

config_dir = r"C:\Users\Pedro\Research\Fundamentals\Bloomberg"
if config_dir not in sys.path:
    sys.path.insert(0, config_dir)
import config_fundamentals as cfg
import cleaning_data as cd

fundamentals_data=cd.fundamentals_data

data_base = []
for ticker, df in fundamentals_data.items():
    df = df.copy()
    df['ticker'] = ticker
    data_base.append(df)
data_base = pd.concat(data_base, ignore_index=True)

data_base['Date'] = pd.to_datetime(data_base['Date'])
data_base["Sector"]=data_base["ticker"].map(cfg.ticker_sector)
data_base = data_base.drop(columns=[col for col in data_base.columns if "GROWTH" in col])
data_base = data_base.drop(columns=[col for col in data_base.columns if "GROW" in col])

data_base = data_base[~((data_base['Date'] < pd.to_datetime(cfg.corr_initial_time)))]
data_base = data_base[~((data_base['Date'] >= pd.to_datetime(cfg.ending_time)))]

data_base = (data_base[[c for c in data_base.columns if c in ["Date", "ticker", "Sector"]
            or (c in cfg.factor_meta and cfg.factor_meta[c]["factor"] == cfg.one_factor)]] 
              if hasattr(cfg, "one_factor") and cfg.one_factor else data_base)

cols_metricas = [c for c in data_base.columns if c not in ['Date', 'ticker', 'Sector']]


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
        
          
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

def get_quasi_diag(link):
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0])
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()

def get_cluster_var(cov, c_items):
    cov_ = cov.iloc[c_items, c_items]
    ivp = 1. / np.diag(cov_)
    ivp /= ivp.sum()
    w_ = ivp.reshape(-1, 1)
    c_var = np.dot(np.dot(w_.T, cov_), w_)[0, 0]
    return c_var

def get_rec_bipart(cov, sort_ix):
    w = pd.Series(1.0, index=sort_ix)
    c_items = [sort_ix]
    while len(c_items) > 0:
        c_items = [i[int(j):int(k)] for i in c_items for j, k in ((0, len(i)/2), (len(i)/2, len(i))) if len(i) > 1]
        for i in range(0, len(c_items), 2):
            c_items0 = c_items[i]
            c_items1 = c_items[i+1]
            c_var0 = get_cluster_var(cov, c_items0)
            c_var1 = get_cluster_var(cov, c_items1)
            alpha = 1 - c_var0 / (c_var0 + c_var1)
            w[c_items0] *= alpha
            w[c_items1] *= 1 - alpha
    return w

def calcular_hrp_por_sector(historical_data, cols_z, min_non_na=20):
    corrs = {}
    for sector, df_sector in historical_data.groupby("Sector"):
        df_signals = df_sector[cols_z]
        valid_cols = [col for col in df_signals.columns if df_signals[col].notna().sum() >= min_non_na]
        df_signals = df_signals[valid_cols]

        if df_signals.shape[1] < 2:
            continue 
        corr = df_signals.corr(min_periods=2)
        d_corr = np.sqrt(0.5 * (1 - corr))
        d_corr = d_corr.fillna(0)
        if not np.allclose(d_corr, d_corr.T, atol=1e-10):
            print(f"Advertencia: matriz no simétrica en sector {sector}")
            continue
        try:
            link = linkage(squareform(d_corr.values), method='average')
        except ValueError as e:
            print(f"Error en linkage para sector {sector}: {e}")
            continue

        sort_ix = get_quasi_diag(link)
        weights = get_rec_bipart(cov=corr, sort_ix=sort_ix)
        weights.index = [corr.columns[i] for i in weights.index]
        corrs[sector] = weights
    return corrs

def calcular_pesos_hrp_dinamico(data_base_cross_sectional, cols_z, cfg, metodo_ventana="expanding", alpha=0.1):
    cfg.strategy_initial_time = pd.to_datetime(cfg.strategy_initial_time)
    cfg.corr_initial_time = pd.to_datetime(cfg.corr_initial_time)
    cfg.ending_time = pd.to_datetime(cfg.ending_time)
    data_base_cross_sectional = data_base_cross_sectional.copy()
    data_base_cross_sectional['Date'] = pd.to_datetime(data_base_cross_sectional['Date'])
    fechas = pd.to_datetime(data_base_cross_sectional['Date'].unique())
    fechas = sorted([f for f in fechas if cfg.strategy_initial_time <= f <= cfg.ending_time])
    pesos_hrp = {}
    pesos_previos = {}  
    for fecha_actual in fechas:
        if metodo_ventana == "expanding":
            data_hasta_t = data_base_cross_sectional[
                (data_base_cross_sectional['Date'] >= cfg.corr_initial_time) &
                (data_base_cross_sectional['Date'] <= fecha_actual)
            ]
        elif metodo_ventana == "rolling":
            fechas_validas = sorted(data_base_cross_sectional['Date'].unique())
            idx_actual = fechas_validas.index(fecha_actual)
            fechas_ventana = fechas_validas[max(0, idx_actual - 11):idx_actual + 1]
            data_hasta_t = data_base_cross_sectional[data_base_cross_sectional['Date'].isin(fechas_ventana)]
        else:
            raise ValueError("metodo_ventana debe ser 'expanding' o 'rolling'")
        pesos_crudos_por_sector = calcular_hrp_por_sector(data_hasta_t, cols_z)
        for sector, pesos_crudos in pesos_crudos_por_sector.items():
            pesos_crudos = pesos_crudos.copy()
            if sector in pesos_previos:
                pesos_previos_sector = pesos_previos[sector]
                all_signals = sorted(set(pesos_crudos.index).union(set(pesos_previos_sector.index)))
                pesos_crudos = pesos_crudos.reindex(all_signals, fill_value=0)
                pesos_previos_sector = pesos_previos_sector.reindex(all_signals, fill_value=0)
                pesos_efectivos = alpha * pesos_crudos + (1 - alpha) * pesos_previos_sector
            else:
                pesos_efectivos = pesos_crudos
            pesos_previos[sector] = pesos_efectivos
            pesos_hrp[(sector, fecha_actual)] = pesos_efectivos
    return pesos_hrp

pesos_expanding = calcular_pesos_hrp_dinamico(
    data_base_cross_sectional=data_base_cross_sectional,
    cols_z=cols_z,
    cfg=cfg,
    metodo_ventana="expanding")

hrps_por_sector = {
    sector: pd.DataFrame({
        fecha: pesos_expanding[(sector, fecha)]
        for (s, fecha) in pesos_expanding if s == sector
    }).T.sort_index().astype(float)
    for sector in set(s for s, _ in pesos_expanding)}

for sector, df in hrps_por_sector.items():
    df["Sector"] = sector








HRP_materials=hrps_por_sector.get("Basic Materials")
HRP_utilities=hrps_por_sector.get("Utilities")
HRP_energy=hrps_por_sector.get("Energy")
HRP_financials=hrps_por_sector.get("Financials")

HRP_materials.to_csv(r"C:\Users\Pedro\Research\Fundamentals\Bloomberg\HRP_metrics\HRP_value_Basic Materials.csv")
HRP_utilities.to_csv(r"C:\Users\Pedro\Research\Fundamentals\Bloomberg\HRP_metrics\HRP_value_Utilities.csv")
HRP_energy.to_csv(r"C:\Users\Pedro\Research\Fundamentals\Bloomberg\HRP_metrics\HRP_value_Energy.csv")
HRP_financials.to_csv(r"C:\Users\Pedro\Research\Fundamentals\Bloomberg\HRP_metrics\HRP_value_Financials.csv")







# import plotly.graph_objects as go

# def graficar_pesos_por_sector_plotly(pesos_hrp_dinamicos, sector, cols_z, nombre_archivo_html=None):
#     claves_sector = [(s, fecha) for (s, fecha) in pesos_hrp_dinamicos if s == sector]
#     claves_sector = sorted(claves_sector, key=lambda x: x[1])
#     df_pesos = pd.DataFrame(index=[fecha for (_, fecha) in claves_sector], columns=cols_z)
#     for (s, fecha) in claves_sector:
#         pesos = pesos_hrp_dinamicos[(s, fecha)]
#         for m in pesos.index:
#             df_pesos.loc[fecha, m] = pesos[m]
#     df_pesos = df_pesos.astype(float)
#     fig = go.Figure()
#     for col in df_pesos.columns:
#         fig.add_trace(go.Scatter(
#             x=df_pesos.index,
#             y=df_pesos[col],
#             mode='lines',
#             name=col
#         ))

#     fig.update_layout(
#         title=f"Evolución de pesos HRP - Sector: {sector}",
#         xaxis_title="Fecha",
#         yaxis_title="Peso HRP",
#         template="plotly_white",
#         legend=dict(orientation="v", x=1.02, y=1),
#         margin=dict(r=120))
#     if nombre_archivo_html:
#         fig.write_html(nombre_archivo_html)

#     fig.show()
#     return df_pesos


# df_evolucion_materials = graficar_pesos_por_sector_plotly(
#     pesos_expanding,
#     sector="Basic Materials",
#     cols_z=cols_z,
#     nombre_archivo_html="pesos_hrp_Materials_expanding.html"
# )

