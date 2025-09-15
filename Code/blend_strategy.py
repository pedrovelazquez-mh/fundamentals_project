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

time_series_factors=ts.data_base_factors
cross_sectional_factors=cs.data_base_cross_sectional_factors

blend = (time_series_factors.rename(columns={'W_score':'W_ts'})
        .merge(cross_sectional_factors.rename(columns={'W_score':'W_cs'}),
               on=['Date','ticker','Sector']))
blend = blend.loc[:, ~blend.columns.str.startswith(('value','quality','credit'))]

blend["blend_score"]=(blend["W_ts"]+blend["W_cs"])/2


def build_portfolio_binario(df: pd.DataFrame, min_per_sector: int = cfg.min_empresas) -> pd.DataFrame:    #para seguir las ponderaciones del merval, tiene que haber siempre un minimo de 2 por sector
    df = df.sort_values(['Date', 'ticker']).copy()
    df['in_base'] = df['blend_score'] > 0
    out = []
    for date, d in df.groupby('Date', sort=False):
        base = d.loc[d['in_base'], ['ticker','Sector','blend_score']].copy()
        sel = set(base['ticker'])
        cnt = base['Sector'].value_counts().to_dict()
        for s in d['Sector'].dropna().unique():
            n = cnt.get(s, 0)
            if n < min_per_sector:
                k = min_per_sector - n
                cand = (
                    d[(d['Sector'] == s) & (~d['ticker'].isin(sel)) & (d['blend_score'].le(0)) & (d['blend_score'].notna())]
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
 #   market_caps=cd.market_caps)
resultado_naive['Date'] = resultado_naive.index
# portafolio_fundamental["suma_total"] = portafolio_fundamental.sum(axis=1)
# portafolio_fundamental["retorno_diario"] = (portafolio_fundamental["suma_total"].pct_change())

