import backtest as bt
import cleaning_data as cd
import config_fundamentals as cfg
import pandas as pd
import time_series as ts
import Cross_Sectional as cs
import numpy as np

time_series_factors=ts.data_base_factors
cross_sectional_factors=cs.data_base_cross_sectional_factors

blend = (time_series_factors.rename(columns={'W_score':'W_ts'})
        .merge(cross_sectional_factors.rename(columns={'W_score':'W_cs'}),
               on=['Date','ticker','Sector']))
blend = blend.loc[:, ~blend.columns.str.startswith(('value','quality','credit'))]

blend["blend_score"]=(blend["W_ts"]+blend["W_cs"])/2

def build_portfolio_binario(df: pd.DataFrame, 
                            min_per_sector: int = 2,
                            umbral_entrada: float = 0.3,
                            umbral_salida: float = 0.3,
                            Smin: float = 0, 
                            delta: float = 0.6) -> pd.DataFrame:
    """
    Reglas:
    - Entrada si blend_score > umbral_entrada.
    - Salida si blend_score < -umbral_salida.
    - Al menos una señal (W_ts, W_cs) debe ser fuerte (>|Smin|).
    - Si discrepan en signo, solo entrar si la contraria es débil (<delta).
    - Siempre garantizar >= min_per_sector por sector, completando con 'menos peores'.
    """
    df = df.sort_values(['Date', 'ticker']).copy()
    out = []

    for date, d in df.groupby('Date', sort=False):
        sel = set()
        cnt = {}

        # --- 1. Evaluar reglas de elegibilidad ---
        for _, row in d.iterrows():
            ts, cs, blend = row['W_ts'], row['W_cs'], row['blend_score']

            # histéresis
            if blend < -umbral_salida:
                continue

            # fuerza mínima
            if max(abs(ts), abs(cs)) <= Smin:
                continue

            # misma dirección
            if np.sign(ts) == np.sign(cs):
                if blend > umbral_entrada:
                    sel.add(row['ticker'])
                continue

            # direcciones opuestas
            if np.sign(ts) != np.sign(cs):
                if blend > umbral_entrada and min(abs(ts), abs(cs)) < delta:
                    sel.add(row['ticker'])

        # --- 2. Constraint sectorial ---
        base = d.loc[d['ticker'].isin(sel), ['ticker','Sector','blend_score']]
        cnt = base['Sector'].value_counts().to_dict()

        for s in d['Sector'].dropna().unique():
            n = cnt.get(s, 0)
            if n < min_per_sector:
                k = min_per_sector - n
                # tomar "menos peores" según blend_score
                cand = (
                    d[(d['Sector'] == s) & (~d['ticker'].isin(sel)) & (d['blend_score'].notna())]
                    .sort_values(['blend_score','ticker'], ascending=[False, True])
                )
                take = cand.head(k)['ticker'].tolist()
                sel.update(take)
                cnt[s] = n + len(take)
        out.append({'Date': date, 'tickers_in_portfolio': sorted(sel)})
    return pd.DataFrame(out)
portafolio = build_portfolio_binario(blend)




"""BackTesting"""
resultado_naive, portafolio_fundamental, logs = bt.backtest_sector_lagged_strategy(
    portafolio=portafolio,
    data_base_precios=cd.data_base_precios,
    merval=cd.merval)
resultado_naive['Date'] = resultado_naive.index
# portafolio_fundamental["suma_total"] = portafolio_fundamental.sum(axis=1)
# portafolio_fundamental["retorno_diario"] = (portafolio_fundamental["suma_total"].pct_change())

