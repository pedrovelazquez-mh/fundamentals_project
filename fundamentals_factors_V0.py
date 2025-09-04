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

fundamentals_data = {}
for archivo in os.listdir(cfg.carpeta_fundamentals):
    if archivo.endswith(".csv"):
        nombre_df = os.path.splitext(archivo)[0]
        ruta_completa = os.path.join(cfg.carpeta_fundamentals, archivo)
        fundamentals_data[nombre_df] = pd.read_csv(ruta_completa,header=1)
        
fundamentals_data = {
    ticker: data
    for ticker, data in fundamentals_data.items()
    if ticker in cfg.ticker_sector}
       
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
    ticker_price = price_data.get(ticker + ".BA")  # Asumiendo claves con .BA
    if ticker_price is not None and len(ticker_price) > 2:
        fecha_minima = pd.to_datetime(ticker_price.iloc[2]["Date"])  # 3ra fila de precios
        df_fund["Date"] = pd.to_datetime(df_fund["Date"])
        fundamentals_data[ticker] = df_fund[df_fund["Date"] >= fecha_minima].reset_index(drop=True)



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
market_caps["HISTORICAL_MARKET_CAP"] = (market_caps.groupby("Ticker")["HISTORICAL_MARKET_CAP"].ffill())
market_caps["HISTORICAL_MARKET_CAP"] = (market_caps.groupby("Ticker")["HISTORICAL_MARKET_CAP"].bfill())

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
data_base["Sector"]=data_base["ticker"].map(cfg.ticker_sector)
data_base = data_base.drop(columns=[col for col in data_base.columns if "GROWTH" in col])
data_base = data_base.drop(columns=[col for col in data_base.columns if "GROW" in col])

data_base = data_base[~((data_base['Date'] < pd.to_datetime(cfg.strategy_initial_time)))]
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
    
for col in cols_z:
    if col in data_base_cross_sectional.columns:
        data_base_cross_sectional[col] = data_base_cross_sectional[col].apply(
            lambda x: np.nan if pd.isna(x) else (-1 if x < 0 else (1 if x > 0 else 0)))            

ruta = cfg.carpeta_hrp_metrics
prefijos = ["HRP_credit", "HRP_value", "HRP_quality"]
HRP_metrics_weights = {}
for prefijo in prefijos:
    dfs = []
    for f in os.listdir(ruta):
        if f.startswith(prefijo) and f.endswith(".csv"):
            df = pd.read_csv(os.path.join(ruta, f))
            df.rename(columns={df.columns[0]: "Date"}, inplace=True)
            df["Date"] = pd.to_datetime(df["Date"])
            cols_redondear = [col for col in df.columns if col not in ["Date", "Sector"]]
            df[cols_redondear] = df[cols_redondear].round(2)
            dfs.append(df)
    HRP_metrics_weights[prefijo] = pd.concat(dfs, ignore_index=True)

def calcular_factor_score(factor_name, data_zscores, hrp_weights_dict):
    hrp_key   = "HRP_" + factor_name.split("_")[0]
    pesos_df  = hrp_weights_dict[hrp_key].copy()
    metricas_hrp = [
        c for c in pesos_df.columns
        if c.endswith("_zscore") and c in data_zscores.columns]
    if not metricas_hrp:
        return pd.Series(float("nan"), index=data_zscores.index)
    cols_base = ["Date", "Sector", "ticker"] + metricas_hrp
    zscore_df = data_zscores[cols_base].copy()
    merged = zscore_df.merge(
        pesos_df,
        on=["Date", "Sector"],
        how="left",
        suffixes=("", "_weight"))
    def _score(row):
        valid = [
            m for m in metricas_hrp
            if pd.notna(row[m]) and pd.notna(row[f"{m}_weight"])]
        if not valid:
            return float("nan")
        zs = np.array([row[m] for m in valid], dtype=float)
        ws = np.array([row[f"{m}_weight"] for m in valid], dtype=float)
        ws /= ws.sum()
        return float((zs * ws).sum())
    scores = merged.apply(_score, axis=1)
    scores.index = zscore_df.index
    return scores
data_base_cross_sectional_factors = data_base_cross_sectional[["Date", "ticker", "Sector"]].copy()

if hasattr(cfg, "one_factor") and cfg.one_factor:
    factors = [cfg.one_factor]  
else:
    factors = [
        key.replace("HRP_", "") + "_score"
        for key in HRP_metrics_weights.keys()]
for factor in factors:
    data_base_cross_sectional_factors[factor] = calcular_factor_score(
        factor_name=factor,
        data_zscores=data_base_cross_sectional,
        hrp_weights_dict=HRP_metrics_weights)
    
for col in ["value_score", "credit_score", "quality_score"]:
    if col not in data_base_cross_sectional_factors.columns:
        data_base_cross_sectional_factors[col] = np.nan

data_base_cross_sectional_factors = (
    data_base_cross_sectional_factors
    .sort_values(by=["ticker", "Date"]))

data_base_cross_sectional_factors[factors] = (
    data_base_cross_sectional_factors
    .groupby("ticker")[factors]
    .ffill())

data_base_cross_sectional_factors[factors]=data_base_cross_sectional_factors[factors].fillna(0)   #es por si falta el primer valor(se evita bfill)

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

data_base_cross_sectional_factors["W_score"]=data_base_cross_sectional_factors["W_score"].fillna(0)   #es por si falta el primer valor(se evita bfill)

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

data_base_cross_sectional_factors = (
    data_base_cross_sectional_factors
    .sort_values(by=["ticker", "Date"])
    .groupby("ticker")
    .apply(lambda df_t: df_t.assign(W_score=df_t["W_score"].ffill()))
    .reset_index(drop=True))

if cfg.metodo_portafolio == "binario":
    portafolio = build_portfolio_binario(data_base_cross_sectional_factors)
elif cfg.metodo_portafolio == "cuartil":
    portafolio = build_portfolio_cuartil(data_base_cross_sectional_factors)
else:
    raise ValueError(f"Método no reconocido en config: {cfg.metodo_portafolio!r}")

"""BackTesting"""
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

def naive_backtest_strategy(portafolio, data_base_precios, merval, 
                            capital_inicial=cfg.CAPITAL_INICIAL, 
                            quarter_lag=cfg.QUARTER_LAG, 
                            annual_lag=cfg.ANNUAL_LAG):
    portafolio['Date'] = pd.to_datetime(portafolio['Date'])
    data_base_precios['Date'] = pd.to_datetime(data_base_precios['Date'])
    merval['Date'] = pd.to_datetime(merval['Date'])
    trading_days = pd.DatetimeIndex(merval['Date'].unique()).sort_values()
    
    precio_por_ticker = {
        t: df.set_index('Date')['close'].sort_index()
        for t, df in data_base_precios.groupby('ticker')}
    rebalanceos = []
    for _, row in portafolio.iterrows():
        fecha_balance = row['Date']
        lag = annual_lag if fecha_balance.month == 12 and fecha_balance.day == 31 else quarter_lag
        fecha_rebalanceo = fecha_balance + pd.Timedelta(days=lag)
        if fecha_rebalanceo > trading_days.max():
            break
        if fecha_rebalanceo not in trading_days:
            next_idx = trading_days.searchsorted(fecha_rebalanceo)
            if next_idx >= len(trading_days):
                break
            fecha_rebalanceo = trading_days[next_idx]
        rebalanceos.append((fecha_rebalanceo, row['tickers_in_portfolio']))

    resultado = pd.DataFrame(columns=["valor", "activos"])
    portafolio_diario = {}
    efectivo_actual = capital_inicial

    for i, (fecha_ini, tickers) in enumerate(rebalanceos):
        if i < len(rebalanceos) - 1:
            proximo_reb = rebalanceos[i + 1][0]
            idx = trading_days.searchsorted(proximo_reb)
            fin_periodo = trading_days[idx - 1]
        else:
            fin_periodo = trading_days.max()

        # AGRUPAR TICKERS POR SECTOR
        sector_to_tickers = {}
        for t in tickers:
            sector = cfg.ticker_sector.get(t)
            if sector:
                sector_to_tickers.setdefault(sector, []).append(t)

        # CAPITAL POR TICKER SEGÚN PESO SECTORIAL
        capital_asignado = {}
        for sector, tickers_sector in sector_to_tickers.items():
            peso_sector = cfg.sector_weights.get(sector, 0)
            capital_sector = efectivo_actual * peso_sector
            capital_por_ticker = capital_sector / len(tickers_sector)
            for t in tickers_sector:
                capital_asignado[t] = capital_por_ticker

        # BUSCAR FECHA DE COMPRA PARA CADA TICKER Y CALCULAR CANTIDAD COMPRADA
        cantidades_compradas = {}
        fechas_compra = {}
        for t in tickers:
            serie_precios = precio_por_ticker.get(t, pd.Series(dtype=float))
            # fechas_validas = serie_precios.loc[serie_precios.index >= fecha_ini]
            fecha_ini = pd.Timestamp(fecha_ini)  # asegura que sea tz-naive
            mask = pd.to_datetime(serie_precios.index) >= fecha_ini
            fechas_validas = serie_precios.loc[mask]
            if not fechas_validas.empty:
                fecha_compra = fechas_validas.index[0]
                precio_compra = fechas_validas.iloc[0]
                cantidades_compradas[t] = capital_asignado[t] / precio_compra
                fechas_compra[t] = fecha_compra
            else:
                cantidades_compradas[t] = 0.0
                fechas_compra[t] = None
                print(f"[{fecha_ini.date()}] No se pudo comprar {t}: sin precios posteriores disponibles. Capital asignado no invertido.")

        fechas = trading_days[(trading_days >= fecha_ini) & (trading_days <= fin_periodo)]
        for fecha in fechas:
            valor_total = 0
            activos = []
            diario = {}
            for t in tickers:
                if fechas_compra[t] is not None and fecha >= fechas_compra[t]:
                    precio_actual = precio_por_ticker[t].get(fecha, None)
                    if precio_actual is not None:
                        valor_ticker = cantidades_compradas[t] * precio_actual
                        valor_total += valor_ticker
                        diario[t] = valor_ticker
                        activos.append(t)
                    else:
                        diario[t] = 0.0
                else:
                    diario[t] = 0.0
            resultado.loc[fecha, "valor"] = valor_total
            resultado.loc[fecha, "activos"] = ', '.join(sorted(activos))
            diario["efectivo"] = 0.0  # No se mantiene efectivo
            portafolio_diario[fecha] = diario
        efectivo_actual = resultado.loc[fechas[-1], "valor"]
    portafolio_df = pd.DataFrame(portafolio_diario).T
    portafolio_df = portafolio_df.sort_index()
    return resultado, portafolio_df

resultado_naive, portafolio_fundamental = naive_backtest_strategy(
    portafolio=portafolio,
    data_base_precios=data_base_precios,
    merval=merval)
resultado_naive['Date'] = resultado_naive.index




# import plotly.graph_objects as go

# # % por fecha
# pf_pct = portafolio_fundamental.div(portafolio_fundamental.sum(axis=1), axis=0) * 100
# pf_pct = pf_pct.fillna(0)

# fechas = pf_pct.index.strftime("%Y-%m-%d").tolist()
# tickers = pf_pct.columns.tolist()

# # --- Función para armar el pie filtrado ---
# def pie_for_date(i):
#     valores = pf_pct.iloc[i].values
#     labels = [tickers[j] for j, v in enumerate(valores) if v > 0]   # solo positivos
#     values = [v for v in valores if v > 0]
#     return go.Pie(labels=labels, values=values, hole=0.3,
#                   textinfo="label+percent", insidetextorientation="auto")

# # --- Primer fecha ---
# fig = go.Figure(data=[pie_for_date(0)],
#                 layout=go.Layout(title=f"Composición del Portafolio ({fechas[0]})"))

# # --- Slider sin animación ---
# steps = []
# for i, fecha in enumerate(fechas):
#     valores = pf_pct.iloc[i].values
#     labels = [tickers[j] for j, v in enumerate(valores) if v > 0]
#     values = [v for v in valores if v > 0]

#     step = {
#         "method": "update",
#         "args": [{"labels": [labels], "values": [values]},
#                   {"title": f"Composición del Portafolio ({fecha})"}],
#         "label": fecha
#     }
#     steps.append(step)

# sliders = [{
#     "steps": steps,
#     "x": 0.1,
#     "y": -0.2
# }]

# fig.update_layout(sliders=sliders)

# fig.write_html("pie_portafolio_slider.html")













# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np

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

# outperf_ratio = capital_naive / capital_merval

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

# texto_stats = f"Win Rate: {win_rate:.2f}% ({wins}/{total})   |   Sharpe Ratio (anualizado): {sharpe_ratio:.2f}"

# # ==== 1) Capital acumulado ====

# plt.figure(figsize=(10, 6))
# plt.plot(capital_naive.index, capital_naive, label="Estrategia Naive", color='blue')
# plt.plot(capital_merval.index, capital_merval, label="Merval", color='red', linestyle='--')
# plt.yscale('log')
# plt.ylabel("Capital Acumulado")
# plt.title("Capital Acumulado Diario: Estrategia Naive vs. Merval")
# plt.legend()

# def add_nice_labels(x, y, color, n_labels=4, y_offset_factor=1.1):
#     np.random.seed(42)
#     idx_all = np.arange(len(x))

#     # posiciones equidistantes + último valor
#     step = len(x) // (n_labels - 1)
#     idx_labels = list(range(0, len(x), step))[:n_labels-1]
#     if (len(x)-1) not in idx_labels:
#         idx_labels.append(len(x)-1)

#     for i in idx_labels:
#         plt.scatter(x[i], y[i], color=color, edgecolor='black', zorder=5, s=40)
#         plt.text(
#             x[i], y[i] * y_offset_factor,  # << desplazamiento vertical
#             f"{y[i]:,.0f}",
#             fontsize=8, color=color, weight='bold',
#             ha='center', va='bottom',    # centrado horizontal
#             bbox=dict(facecolor='white', edgecolor=color, boxstyle='round,pad=0.2', alpha=0.7)
#         )

# # Uso
# add_nice_labels(capital_naive.index, capital_naive.values, 'blue', n_labels=4, y_offset_factor=1.70)
# add_nice_labels(capital_merval.index, capital_merval.values, 'red', n_labels=4, y_offset_factor=1.70)

# plt.tight_layout()
# plt.show()

# import matplotlib.pyplot as plt
# import matplotlib.ticker as mticker
# import pandas as pd
# import numpy as np
# import matplotlib.dates as mdates

# # datos limpios
# r = outperf_ratio.dropna()
# x = r.index.to_pydatetime()
# y = pd.to_numeric(r.values, errors='coerce')
# mask = np.isfinite(y)
# x = np.array(x)[mask]
# y = y[mask]

# last_date, last_val = x[-1], y[-1]
# peak_idx = np.argmax(y)
# peak_date, peak_val = x[peak_idx], y[peak_idx]

# fig, ax = plt.subplots(figsize=(11, 6))

# # sombreado donde ratio > 1
# ax.fill_between(x, 1.0, y, where=(y >= 1.0), alpha=0.10, color='green')

# # línea principal
# ax.plot(x, y, color='green', linewidth=2.2, label="Ratio Naive / Merval")

# # baseline
# ax.axhline(1.0, color='gray', linestyle='--', linewidth=1)
# ax.text(x[-1], 1.0, "  Naive = Merval", va='bottom', ha='left', color='gray')

# # anotaciones de máximo y último valor
# ax.scatter([peak_date, last_date], [peak_val, last_val],
#             s=40, color='green', edgecolor='black', zorder=5)
# ax.annotate(f"Máximo: {peak_val:.2f}x", xy=(peak_date, peak_val),
#             xytext=(10, 12), textcoords="offset points",
#             bbox=dict(facecolor='white', edgecolor='green',
#                       boxstyle='round,pad=0.2', alpha=0.8))
# ax.annotate(f"Último: {last_val:.2f}x", xy=(last_date, last_val),
#             xytext=(10, 12), textcoords="offset points",
#             bbox=dict(facecolor='white', edgecolor='green',
#                       boxstyle='round,pad=0.2', alpha=0.8))

# # estética de ejes
# ax.set_title("Ratio de Capital: Naive / Merval")
# ax.set_ylabel("Ratio")
# ax.grid(True, axis='y', linestyle=':', alpha=0.4)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# # ticks y formato
# ax.yaxis.set_major_locator(mticker.MaxNLocator(8))
# ax.yaxis.set_major_formatter(mticker.FuncFormatter(
#     lambda v, _: f"{v:.0f}x" if v >= 1 else f"{v:.2f}x"))

# ax.xaxis.set_major_locator(mdates.YearLocator(2))
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# ax.margins(x=0.01, y=0.05)
# ax.legend(frameon=True, framealpha=0.9, edgecolor='lightgray', loc='upper left')

# plt.tight_layout()
# plt.show()

# import matplotlib.pyplot as plt
# import matplotlib.ticker as mticker
# import matplotlib.dates as mdates
# import numpy as np
# import pandas as pd

# # --- datos trimestrales (ya los tenés calculados) ---
# # q_naive, q_merval en %, idx = intersección de fechas

# qN = q_naive.loc[idx].dropna()
# qM = q_merval.loc[idx].dropna()
# # alinear otra vez por las dudas
# common = qN.index.intersection(qM.index)
# qN = qN.loc[common]
# qM = qM.loc[common]

# x = common.to_pydatetime()

# # desplazamiento horizontal (en días) para separar barras
# off_days = 20
# x_left  = [d - pd.Timedelta(days=off_days) for d in x]
# x_right = [d + pd.Timedelta(days=off_days) for d in x]

# # ancho de cada barra (en días)
# barw = 36

# fig, ax = plt.subplots(figsize=(12, 6))

# # barras
# bars_naive = ax.bar(x_left,  qN.values, width=barw, align='center',
#                     label='Estrategia Naive', color='#1565c0', alpha=0.85,
#                     edgecolor='black', linewidth=0.6)
# bars_mer   = ax.bar(x_right, qM.values, width=barw, align='center',
#                     label='Merval', color='#e53935', alpha=0.85,
#                     edgecolor='black', linewidth=0.6)

# # baseline en 0
# ax.axhline(0, color='gray', linestyle='--', linewidth=1)

# # grilla sutil
# ax.grid(True, axis='y', linestyle=':', alpha=0.35)

# # formato eje Y como porcentaje
# ax.set_ylabel("Trimestral [%]")
# ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
# ax.yaxis.set_major_locator(mticker.MaxNLocator(10))

# # ticks de fechas: 1 cada año (o cada 2, si hay muchos)
# ax.xaxis.set_major_locator(mdates.YearLocator(2))
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# # título y estética de ejes
# ax.set_title("Rendimiento Trimestral: Estrategia Naive vs. Merval")
# for s in ("top", "right"):
#     ax.spines[s].set_visible(False)

# # límites Y (ajustá si querés)
# ylim = max(abs(qN.min()), abs(qN.max()), abs(qM.min()), abs(qM.max()))
# ylim = min(max(ylim, 30), 90)   # entre 30% y 90% aprox.
# ax.set_ylim(-ylim, ylim)

# # anotaciones de extremos (máximo y mínimo) para cada serie
# def annotate_extremes(ax, xdates, yvals, color):
#     if len(yvals) == 0:
#         return
#     iy_max = int(np.nanargmax(yvals))
#     iy_min = int(np.nanargmin(yvals))
#     for iy, tag in [(iy_max, "Máx"), (iy_min, "Mín")]:
#         ax.annotate(f"{tag}: {yvals[iy]:.0f}%",
#                     xy=(xdates[iy], yvals[iy]),
#                     xytext=(0, 12 if yvals[iy] >= 0 else -16),
#                     textcoords="offset points",
#                     ha='center', va='bottom' if yvals[iy] >= 0 else 'top',
#                     fontsize=8, color=color,
#                     bbox=dict(facecolor='white', edgecolor=color,
#                               boxstyle='round,pad=0.2', alpha=0.85))
# annotate_extremes(ax, x_left,  qN.values, '#1565c0')
# annotate_extremes(ax, x_right, qM.values, '#e53935')

# ax.legend(frameon=True, framealpha=0.95, edgecolor='lightgray', loc='upper left')
# plt.tight_layout()
# plt.show()


from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import pandas as pd

merval['Date'] = pd.to_datetime(merval['Date'])
merval = merval.set_index('Date').sort_index()

resultado_merval = (
    merval[['close']]
    .rename(columns={'close': 'valor'})
    .reindex(resultado_naive.index)
    .ffill())

V0_naive  = resultado_naive['valor'].iloc[0]
V0_merval = resultado_merval['valor'].iloc[0]
capital_naive  = resultado_naive['valor']  / V0_naive
capital_merval = resultado_merval['valor'] / V0_merval
outperf_ratio = (capital_naive / capital_merval).rename("Ratio Naive/Merval")
q_naive  = resultado_naive['valor'].resample('Q').last().pct_change() * 100
q_merval = resultado_merval['valor'].resample('Q').last().pct_change() * 100
idx = q_naive.index.intersection(q_merval.index)

wins = (q_naive.loc[idx] > q_merval.loc[idx]).sum()
total = len(idx)
win_rate = wins / total * 100 if total > 0 else np.nan

rf_daily = 0.0 / 252
retornos_naive = resultado_naive['valor'].pct_change().dropna()
exceso_retornos = retornos_naive - rf_daily
sharpe_ratio = (exceso_retornos.mean() / exceso_retornos.std()) * np.sqrt(252) if exceso_retornos.std() > 0 else np.nan

fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=(
        "Capital Acumulado Diario: Estrategia Naive vs. Merval",
        "Ratio de Capital: Naive / Merval",
        "Rendimiento Trimestral: Estrategia Naive vs. Merval"
    )
)

fig.add_trace(
    go.Scatter(
        x=capital_naive.index, y=capital_naive, mode='lines',
        name='Estrategia Naive', line=dict(color='blue')
    ),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(
        x=capital_merval.index, y=capital_merval, mode='lines',
        name='Merval', line=dict(color='red', dash='dash')
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=outperf_ratio.index, y=outperf_ratio, mode='lines',
        name='Ratio Naive / Merval', line=dict(color='green'),
        hovertemplate='Fecha=%{x|%Y-%m-%d}<br>Ratio=%{y:.3f}<extra></extra>'
    ),
    row=2, col=1
)
fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=2, col=1)

fig.add_trace(
    go.Bar(x=idx, y=q_naive.loc[idx], name="Estrategia Naive", marker_color='blue'),
    row=3, col=1
)
fig.add_trace(
    go.Bar(x=idx, y=q_merval.loc[idx], name="Merval", marker_color='red'),
    row=3, col=1
)

anotacion_texto = (
    f"<span style='margin-right:30px'>Win Rate: {win_rate:.2f}% ({wins}/{total})</span>"
    f"<span>     Sharpe Ratio (anualizado): {sharpe_ratio:.2f}</span>"
)
fig.add_annotation(
    text=anotacion_texto,
    x=0.5, y=1, xref='paper', yref='paper',
    showarrow=False, font=dict(size=14), align="center"
)

fig.update_yaxes(title_text="Capital Acumulado", type="log", row=1, col=1)
fig.update_yaxes(title_text="Ratio", tickformat=".2f", row=2, col=1)
fig.update_yaxes(title_text="Trimestral [%]", tickformat=".2f", ticksuffix="%", row=3, col=1)

fig.update_layout(
    height=1000, width=1000,
    title_text=(
        f"Estrategia Naive vs. Merval "
        f"(Capital Acumulado, Ratio y Retorno Trimestral - lag Q={cfg.QUARTER_LAG}, A={cfg.ANNUAL_LAG})"
    ),
    barmode='group',
    template="plotly_white",
    legend=dict(
        x=1.02, y=1, xanchor="left", yanchor="top",
        bordercolor="gray", borderwidth=1,
        bgcolor="rgba(255,255,255,0.8)"
    )
)
# fig.write_html(f"naive_vs_merval_lag_{cfg.QUARTER_LAG}_{cfg.ANNUAL_LAG}_{cfg.metodo_portafolio}-{cfg.strategy_initial_time}_{cfg.ending_time}.html")
fig.write_html(f"naive_vs_merval_lag_{cfg.QUARTER_LAG}_{cfg.ANNUAL_LAG}.html")
