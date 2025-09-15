import pandas as pd
import config_fundamentals as cfg
from collections import defaultdict


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

        if not valid_targets:
            return

        # ---- NUEVO: pesos por market cap con último dato conocido (ffill) ----
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
        # cash = 0.0  # <-- (3) IGUALAMOS AL NAIVE: no dejamos cash “suelto”

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
