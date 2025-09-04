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

data_base=cd.data_base
data_base = data_base[~((data_base['Date'] < pd.to_datetime(cfg.strategy_initial_time)))]
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
        .apply(rank_to_symm)
    )


fundamentals_data={
    ticker:data
    for ticker, data in fundamentals_data.items()
    if ticker in cfg.ticker_sector
}


# for col in cols_z:
#     if col in data_base_cross_sectional.columns:
#         data_base_cross_sectional[col] = data_base_cross_sectional[col].apply(
#             lambda x: np.nan if pd.isna(x) else (-1 if x < 0 else (1 if x > 0 else 0)))            


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
price_data=cd.price_data
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




def backtest_sector_lagged_strategy(portafolio, data_base_precios, merval, CFG=cfg):

    portafolio = portafolio.copy()
    data_base_precios = data_base_precios.copy()
    portafolio['Date'] = pd.to_datetime(portafolio['Date'])
    data_base_precios['Date'] = pd.to_datetime(data_base_precios['Date'])
    merval = merval.copy()
    merval['Date'] = pd.to_datetime(merval['Date'])

    # --------- Series de precios por ticker (pivot sin ffill) ----------
    # Asumimos que forward_fill_missing_closes() ya dejó cerrado el panel en el calendario de MERVAL.
    # Por eso, NO hacemos .ffill() acá para no "ejecutar" con ffill.
    trading_days = pd.DatetimeIndex(merval['Date'].unique()).sort_values()   # <-- (1) MISMO CALENDARIO QUE NAIVE

    # Garantizar que los tickers están limpios (column 'ticker' sin .BA si así lo usás aguas arriba)
    # data_base_precios ya viene en ese formato en tu pipeline.
    panel = (data_base_precios
             .pivot_table(index='Date', columns='ticker', values='close', aggfunc='last')
             .reindex(trading_days)
             .sort_index())

    # Validación suave: si aún quedaran NaNs, preferimos “next available” (forward) SOLO para *valuación* diaria.
    # Para la *ejecución* usamos el valor de ese día en panel (que debería existir gracias al forward_fill previo).
    panel_val = panel.ffill()  # solo para valuación del NAV diario

    # --------- Config ----------
    sector_of = dict(CFG.ticker_sector)                    # ticker -> sector
    lag_by_sector = dict(getattr(CFG, "lag_by_sector", {}))# sector -> [qlag, alag]
    sector_weights_target = dict(CFG.sector_weights)       # sector -> peso objetivo
    capital_inicial = float(CFG.CAPITAL_INICIAL)
    quarter_fallback = int(getattr(CFG, "QUARTER_LAG", 0))
    annual_fallback  = int(getattr(CFG, "ANNUAL_LAG", 0))

    # --------- Construcción de eventos por ciclo/sector ----------
    eventos = []  # (fecha_evento, ciclo_id, sector, tickers_sector_en_ese_ciclo)
    sectores_por_ciclo = {}
    cycle_sector_selection = {}  # (ciclo_id, sector) -> tuple(tickers de ese sector en ese ciclo)

    for ciclo_id, row in portafolio.reset_index(drop=True).iterrows():
        fecha_base = pd.Timestamp(row['Date'])
        es_anual = (fecha_base.month == 12 and fecha_base.day == 31)

        # agrupación sectorial
        secmap = defaultdict(list)
        for t in row['tickers_in_portfolio']:
            s = sector_of.get(t)
            if s is not None:
                secmap[s].append(t)
        sectores_por_ciclo[ciclo_id] = sorted(secmap.keys())

        for s, ts in secmap.items():
            if s in lag_by_sector:
                qlag, alag = lag_by_sector[s]
                lag_days = int(alag if es_anual else qlag)
            else:
                lag_days = int(annual_fallback if es_anual else quarter_fallback)

            fecha_evt = fecha_base + pd.Timedelta(days=lag_days)
            # snap al próximo día de MERVAL (igual que en naive con trading_days de merval)
            idx = trading_days.searchsorted(fecha_evt, side='left')
            if idx >= len(trading_days):
                continue
            fecha_evt = trading_days[idx]

            ts_sorted = tuple(sorted(ts))
            eventos.append((fecha_evt, ciclo_id, s, ts_sorted))
            cycle_sector_selection[(ciclo_id, s)] = ts_sorted

    eventos.sort(key=lambda x: x[0])
    eventos_restantes_por_ciclo = {cid: set(secs) for cid, secs in sectores_por_ciclo.items()}

    # --------- Estado del portafolio ----------
    universe = sorted(set(c for c in panel.columns if c in sector_of))
    qty  = {t: 0.0 for t in universe}
    cash = capital_inicial
    last_selection_by_sector = {}  # sector -> última selección intra
    logs = []

    resultado = pd.DataFrame(index=trading_days, columns=["valor", "activos"], dtype=object)
    resultado["valor"] = 0.0
    diarios = {}

    # --------- Helpers ----------
    def sector_value(fecha):
        px = panel_val.loc[fecha]  # valuación con ffill permitido para NAV
        vals = defaultdict(float)
        for t in universe:
            s = sector_of.get(t)
            if s is None:
                continue
            p = float(px[t]) if pd.notna(px[t]) else 0.0
            if p > 0 and qty[t] != 0.0:
                vals[s] += qty[t] * p
        return vals

    def rebalance_intrasector(fecha, sector, tickers_target):
        """Mantiene el VALOR del sector y redistribuye equiponderado.
           Ejecución usa panel.loc[fecha, t] (cierre del día, sin ffill local)."""
        nonlocal cash, qty
        px_exec = panel.loc[fecha]  # <-- (2) PRECIO DE EJECUCIÓN = CIERRE REAL DEL DÍA (sin ffill aquí)

        curr_sector = [t for t in universe if sector_of.get(t) == sector]
        # Valor previo del sector con precio de ejecución del día
        current_val = 0.0
        for t in curr_sector:
            p = px_exec.get(t)
            if pd.isna(p) or p <= 0:
                continue
            current_val += qty[t] * float(p)

        valid_targets = [t for t in tickers_target if t in panel.columns]
        last_selection_by_sector[sector] = tuple(valid_targets)

        if current_val <= 0.0:
            return  # no creamos desde 0 en el intra

        # vender los que salen
        to_sell = [t for t in curr_sector if t not in valid_targets and qty[t] != 0.0]
        for t in to_sell:
            p = px_exec.get(t)
            if pd.isna(p) or p <= 0:
                continue
            cash += qty[t] * float(p)
            qty[t] = 0.0

        n = len(valid_targets)
        if n == 0:
            return
        per_value = current_val / n

        # ajustar cantidades para equiponderar en valor
        for t in valid_targets:
            p = px_exec.get(t)
            if pd.isna(p) or p <= 0:
                continue
            target_qty = per_value / float(p)
            delta_qty  = target_qty - qty[t]
            cash -= delta_qty * float(p)
            qty[t] = target_qty

    def rebalance_macro_to_targets(fecha, ciclo_id):
        """Alinea a sector_weights sobre el NAV del día (ejecución con precios del día, sin ffill local).
           SIN reparto de cash residual (cash = 0.0 al final)."""
        nonlocal cash, qty
        px_exec = panel.loc[fecha]     # precios de ejecución exactos del día
        px_val  = panel_val.loc[fecha] # valuación con ffill para NAV

        sectores_presentes = sectores_por_ciclo.get(ciclo_id, [])
        if not sectores_presentes:
            return

        # Valor por sector + NAV (valuación con px_val)
        vals = defaultdict(float)
        for t in universe:
            s = sector_of.get(t)
            if s is None:
                continue
            p = float(px_val.get(t)) if pd.notna(px_val.get(t)) else 0.0
            if p > 0 and qty[t] != 0.0:
                vals[s] += qty[t] * p
        nav = cash + sum(vals.values())
        if nav <= 0.0:
            return

        # Targets normalizados SOLO entre presentes (igual que mencionabas)
        targets = {s: sector_weights_target.get(s, 0.0) for s in sectores_presentes}
        ssum = sum(targets.values())
        if ssum > 0:
            targets = {s: w / ssum for s, w in targets.items()}

        # Map sector -> tickers (universo)
        by_sector_all = defaultdict(list)
        for t in universe:
            s = sector_of.get(t)
            if s is not None:
                by_sector_all[s].append(t)

        # Escalado por sector hacia target_v (ejecución con px_exec)
        for s in sectores_presentes:
            v = float(vals.get(s, 0.0))
            target_v = float(targets.get(s, 0.0) * nav)

            if v > 0.0 and target_v > 0.0:
                r = target_v / v
                for t in by_sector_all.get(s, []):
                    p = px_exec.get(t)
                    if pd.isna(p) or p <= 0.0:
                        continue
                    new_qty = qty[t] * r
                    delta_qty = new_qty - qty[t]
                    cash -= delta_qty * float(p)
                    qty[t] = new_qty

            elif v == 0.0 and target_v > 0.0:
                # crear desde 0 usando selección del ciclo (o última intra si existe)
                sel = list(last_selection_by_sector.get(s, []))
                if not sel:
                    sel = list(cycle_sector_selection.get((ciclo_id, s), []))
                valid = [t for t in sel if t in panel.columns]
                if not valid:
                    continue
                per = target_v / len(valid)
                # asegurar que no quede nada "viejo"
                for t in by_sector_all.get(s, []):
                    p = px_exec.get(t)
                    if pd.isna(p) or p <= 0.0 or qty[t] == 0.0:
                        continue
                    cash += qty[t] * float(p)
                    qty[t] = 0.0
                for t in valid:
                    p = px_exec.get(t)
                    if pd.isna(p) or p <= 0.0:
                        continue
                    target_qty = per / float(p)
                    delta_qty = target_qty - qty[t]
                    cash -= delta_qty * float(p)
                    qty[t] = target_qty

            else:
                # target_v == 0.0: cerrar sector
                for t in by_sector_all.get(s, []):
                    p = px_exec.get(t)
                    if pd.isna(p) or p <= 0.0 or qty[t] == 0.0:
                        continue
                    cash += qty[t] * float(p)
                    qty[t] = 0.0

        # ---- SIN reparto de cash residual para emular naive ----
        cash = 0.0  # <-- (3) IGUALAMOS AL NAIVE: no dejamos cash “suelto”

    # --------- Loop principal ----------
    eventos_idx = 0
    eventos_len = len(eventos)

    for fecha in trading_days:
        # ejecutar eventos del día
        while eventos_idx < eventos_len and eventos[eventos_idx][0] == fecha:
            _, ciclo_id, sector, ts = eventos[eventos_idx]

            # Intra (cash-neutral, usa precios del día sin ffill)
            rebalance_intrasector(fecha, sector, list(ts))

            # Si cerró el ciclo, Macro (sin cash residual)
            eventos_restantes_por_ciclo[ciclo_id].discard(sector)
            if not eventos_restantes_por_ciclo[ciclo_id]:
                rebalance_macro_to_targets(fecha, ciclo_id)

            eventos_idx += 1

        # Valuación diaria (usamos panel_val con ffill SOLO para valuación)
        px = panel_val.loc[fecha]
        valor_pos = 0.0
        activos = []
        for t in universe:
            p = float(px.get(t)) if pd.notna(px.get(t)) else 0.0
            if p > 0.0 and qty[t] != 0.0:
                v = qty[t] * p
                valor_pos += v
                activos.append(t)
        nav = cash + valor_pos
        resultado.loc[fecha, "valor"] = float(nav)
        resultado.loc[fecha, "activos"] = ", ".join(sorted(set(activos)))
        diarios_row = {t: float(qty[t] * (px.get(t) if pd.notna(px.get(t)) else 0.0)) for t in universe}
        diarios_row["efectivo"] = float(cash)
        diarios[fecha] = diarios_row
    portafolio_df = pd.DataFrame(diarios).T.sort_index()
    return resultado, portafolio_df, logs

resultado_naive, portafolio_fundamental, logs = backtest_sector_lagged_strategy(
    portafolio=portafolio,
    data_base_precios=data_base_precios,
    merval=merval)
resultado_naive['Date'] = resultado_naive.index

from plotly.subplots import make_subplots
import plotly.graph_objects as go

# =========================
# 1) PANEL 1–3 (tu código)
# =========================
merval['Date'] = pd.to_datetime(merval['Date'])
merval = merval.set_index('Date').sort_index()

resultado_merval = (
    merval[['close']]
    .rename(columns={'close': 'valor'})
    .reindex(resultado_naive.index)
    .ffill()
)

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

# =========================
# 2) PANEL 4 (área apilada)
# =========================
df = portafolio_fundamental.copy()
df.index = pd.to_datetime(df.index)
df = df.sort_index().fillna(0)

# Detectar columna de efectivo si existe
cash_col = 'efectivo' if 'efectivo' in df.columns else None
comp_cols = [c for c in df.columns if c != cash_col]

# Orden global por peso promedio (grandes al fondo de la pila)
order = (df[comp_cols].mean().sort_values(ascending=False)).index.tolist()
ordered_cols = ([cash_col] + order) if cash_col else order

# Normalizar por fila (proporciones 0–1)
row_sum = df[ordered_cols].sum(axis=1).replace(0, np.nan)
df_norm = df[ordered_cols].div(row_sum, axis=0).fillna(0)

# =========================
# 3) FIGURA CON 4 SUBPLOTS
# =========================
fig = make_subplots(
    rows=4, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.07,
    subplot_titles=(
        "Capital Acumulado Diario: Estrategia Naive vs. Merval",
        "Ratio de Capital: Naive / Merval",
        "Rendimiento Trimestral: Estrategia Naive vs. Merval",
        "Pesos del Portafolio a lo largo del tiempo"
    ),
    row_heights=[0.34, 0.18, 0.18, 0.30]
)

# --- Row 1: capital acumulado
fig.add_trace(
    go.Scatter(x=capital_naive.index, y=capital_naive, mode='lines',
               name='Estrategia Naive', line=dict(color='blue')),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=capital_merval.index, y=capital_merval, mode='lines',
               name='Merval', line=dict(color='red', dash='dash')),
    row=1, col=1
)

# --- Row 2: ratio
fig.add_trace(
    go.Scatter(
        x=outperf_ratio.index, y=outperf_ratio, mode='lines',
        name='Ratio Naive / Merval', line=dict(color='green'),
        hovertemplate='Fecha=%{x|%Y-%m-%d}<br>Ratio=%{y:.3f}<extra></extra>'
    ),
    row=2, col=1
)
fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=2, col=1)

# --- Row 3: barras trimestrales
fig.add_trace(
    go.Bar(x=idx, y=q_naive.loc[idx], name="Estrategia Naive", marker_color='blue'),
    row=3, col=1
)
fig.add_trace(
    go.Bar(x=idx, y=q_merval.loc[idx], name="Merval", marker_color='red'),
    row=3, col=1
)

# --- Row 4: área apilada de pesos
for col in ordered_cols:
    fig.add_trace(
        go.Scatter(
            x=df_norm.index,
            y=df_norm[col],
            mode='lines',
            stackgroup='one',
            name=col,
            hovertemplate="%{x|%Y-%m-%d}<br>%{fullData.name}: %{y:.2%}<extra></extra>"
        ),
        row=4, col=1
    )

# --- Anotaciones y ejes
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
fig.update_yaxes(title_text="Peso", tickformat=".0%", range=[0, 1], row=4, col=1)

fig.update_layout(
    height=1300, width=1000,
    title_text="Estrategia Naive vs. Merval (Capital, Ratio, Retorno Trimestral y Pesos del Portafolio)",
    barmode='group',
    template="plotly_white",
    legend=dict(
        orientation="h",              # horizontal
        yanchor="top", y=-0.25,       # debajo de todo
        xanchor="center", x=0.5,      # centrada
        bordercolor="gray", borderwidth=1,
        bgcolor="rgba(255,255,255,0.8)"
    ),
    hovermode="x unified"
)

# Exportá la versión combinada
fig.write_html("naive_vs_merval_Ranks.html")





"""Time-Series"""
# def calcular_zscore_temporal(data_base, cols_metricas, window, min_periods, tipo_ventana):
#     data_base_time_series = data_base.sort_values(['ticker', 'Date']).copy()
#     for col in cols_metricas:
#         def z_robusto(x):
#             if tipo_ventana == 'expanding':
#                 med = x.expanding(min_periods=min_periods).median()
#                 iqr = x.expanding(min_periods=min_periods).quantile(0.75) - \
#                       x.expanding(min_periods=min_periods).quantile(0.25)
#             elif tipo_ventana == 'rolling':
#                 med = x.rolling(window=window, min_periods=min_periods).median()
#                 iqr = x.rolling(window=window, min_periods=min_periods).quantile(0.75) - \
#                       x.rolling(window=window, min_periods=min_periods).quantile(0.25)
#             else:
#                 raise ValueError("tipo_ventana debe ser 'rolling' o 'expanding'")
#             return (x - med) / iqr.replace(0, np.nan)
#         data_base_time_series[f"{col}_z_ts"] = data_base_time_series.groupby("ticker")[col].transform(z_robusto)
#     return data_base_time_series

# data_base_time_series = calcular_zscore_temporal(
#     data_base=data_base,
#     cols_metricas=cols_metricas,
#     window=6,
#     min_periods=4,
#     tipo_ventana='rolling')

# cols_z_ts = ['Date', 'ticker', 'Sector'] + [col for col in data_base_time_series.columns if col.endswith('_z_ts')]
# data_base_time_series = data_base_time_series[cols_z_ts]

# for metric, meta in cfg.factor_meta.items():
#     if meta["invert_sign"]:
#         col_name = f"{metric}_z_ts"
#         if col_name in data_base_time_series.columns:
#             data_base_time_series[col_name] = data_base_time_series[col_name].apply(
#                 lambda x: -x if x != 0 else x)


# for col in cols_z:
#     if col in data_base_cross_sectional.columns:
#         data_base_cross_sectional[col] = data_base_cross_sectional[col].apply(
#             lambda x: np.nan if pd.isna(x) else (-1 if x < 0 else (1 if x > 0 else 0)))            





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
