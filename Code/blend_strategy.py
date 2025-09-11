import backtest as bt
import cleaning_data as cd
import config_fundamentals as cfg
import pandas as pd
import time_series as ts
import Cross_Sectional as cs


time_series_factors=ts.data_base_factors
cross_sectional_factors=cs.data_base_cross_sectional_factors

blend = (time_series_factors.rename(columns={'W_score':'W_ts'})
        .merge(cross_sectional_factors.rename(columns={'W_score':'W_cs'}),
               on=['Date','ticker','Sector']))
blend = blend.loc[:, ~blend.columns.str.startswith(('value','quality','credit'))]

blend["blend_score"]=(blend["W_ts"]+blend["W_cs"])/2


def build_portfolio_binario(df: pd.DataFrame, min_per_sector: int = 2) -> pd.DataFrame:    #para seguir las ponderaciones del merval, tiene que haber siempre un minimo de 2 por sector
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

resultado_naive['Date'] = resultado_naive.index

portafolio_fundamental["suma_total"] = portafolio_fundamental.sum(axis=1)

portafolio_fundamental["retorno_diario"] = (
    portafolio_fundamental["suma_total"].pct_change()
)




from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
# =========================
# 1) PANEL 1–3 (tu código)
# =========================
merval=cd.merval
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
fig.write_html("blend vs_merval.html")

