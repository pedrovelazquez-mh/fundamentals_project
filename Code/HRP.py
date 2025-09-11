import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list
from scipy.spatial.distance import squareform
import config_fundamentals as cfg

"""HRP"""   
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


def compute_hrp_factor_weights(
    data_base: pd.DataFrame,
    factor_meta: dict,
    alpha: float = cfg.alpha,
    linkage_method: str = "average",
    min_obs_per_pair: int = 8,    # mínimo para computar correlación a partir de un ticker
    min_metrics_per_factor: int = 2,  # si <2 → igual-ponderado dentro del factor
    use_psd_projection: bool = True,
    dates: list | None = None,    # si None usa todas las fechas ordenadas
    sectoral: bool | None = None       #False=global, True=por sector
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

    # Estado para suavizado temporal
    prev_eff_by_key: dict[object, pd.Series] = {}  # key=fact (global) o (sector,fact) (sectoral)

    out_cols = (["Date", "Sector"] if sectoral else ["Date"]) + sorted(rank_cols_mapped) + sorted(dropped)
    out_rows = []

    modo = "sectorial" if sectoral else "global"
    print(f"Iterando {len(dates)} fechas (expanding, sin mirar futuro)... Modo: {modo}")

    for t in dates:
        df_t = data_base[pd.to_datetime(data_base["Date"]) <= t]

        if not sectoral:
            # ====== MODO GLOBAL (igual al original) ======
            mats, counts = [], []
            for tkr, sub in df_t.groupby("ticker"):
                X = sub[rank_cols_mapped].copy()
                X = X.loc[:, X.std(numeric_only=True) > 0]
                if X.shape[1] < 2 or len(X) < min_obs_per_pair:
                    continue
                R_i = X.corr(method="pearson")
                N_i = _pairwise_valid_counts(X)
                if (N_i.values >= min_obs_per_pair).sum() == 0:
                    continue
                mats.append(R_i)
                counts.append(N_i)

            if not mats:
                # igual-ponderado por factor
                row = {"Date": pd.to_datetime(t)}
                for fact, cols in factor_to_rankcols.items():
                    if len(cols) == 0:
                        continue
                    w_eff = pd.Series(1.0 / len(cols), index=cols, dtype=float)
                    prev_eff_by_key[fact] = w_eff
                    row.update({c: w_eff.get(c, 0.0) for c in cols})
                for c in dropped:
                    row[c] = 0.0
                out_rows.append(row)
                continue

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

                key = fact
                w_prev = prev_eff_by_key.get(key, None)
                w_prev = None if w_prev is None else w_prev.reindex(cols_f)
                w_eff = _smooth(w_hrp, w_prev, alpha=alpha)
                if w_eff.sum() > 0:
                    w_eff = w_eff / w_eff.sum()
                prev_eff_by_key[key] = w_eff.copy()

                for c in cols:
                    row[c] = float(w_eff.get(c, 0.0))
            for c in dropped:
                row[c] = 0.0
            out_rows.append(row)

        else:
            # ====== MODO SECTORIAL ======
            # Recolectar matrices por sector
            mats_by_sector: dict[str, list] = {}
            counts_by_sector: dict[str, list] = {}
            for tkr, sub in df_t.groupby("ticker"):
                X = sub[rank_cols_mapped].copy()
                X = X.loc[:, X.std(numeric_only=True) > 0]
                if X.shape[1] < 2 or len(X) < min_obs_per_pair:
                    continue
                R_i = X.corr(method="pearson")
                N_i = _pairwise_valid_counts(X)
                if (N_i.values >= min_obs_per_pair).sum() == 0:
                    continue
                # Sector del ticker (último disponible)
                s_vals = sub["Sector"].dropna()
                sector = s_vals.iloc[-1] if len(s_vals) else "Unknown"
                mats_by_sector.setdefault(sector, []).append(R_i)
                counts_by_sector.setdefault(sector, []).append(N_i)

            # Determinar sectores presentes en la fecha (existentes en datos hasta t)
            sectores_t = sorted(df_t["Sector"].dropna().unique())

            if len(sectores_t) == 0:
                # No hay sectores identificados: caer a global igual-ponderado
                row = {"Date": pd.to_datetime(t), "Sector": "Unknown"}
                for fact, cols in factor_to_rankcols.items():
                    if len(cols) == 0:
                        continue
                    w_eff = pd.Series(1.0 / len(cols), index=cols, dtype=float)
                    prev_eff_by_key[("Unknown", fact)] = w_eff
                    row.update({c: w_eff.get(c, 0.0) for c in cols})
                for c in dropped:
                    row[c] = 0.0
                out_rows.append(row)
                continue

            for s in sectores_t:
                mats_s = mats_by_sector.get(s, [])
                counts_s = counts_by_sector.get(s, [])

                if not mats_s:
                    # Sin matrices válidas en este sector: igual-ponderado por factor
                    row = {"Date": pd.to_datetime(t), "Sector": s}
                    for fact, cols in factor_to_rankcols.items():
                        if len(cols) == 0:
                            continue
                        w_eff = pd.Series(1.0 / len(cols), index=cols, dtype=float)
                        prev_eff_by_key[(s, fact)] = w_eff
                        row.update({c: w_eff.get(c, 0.0) for c in cols})
                    for c in dropped:
                        row[c] = 0.0
                    out_rows.append(row)
                    continue

                all_cols_s = sorted({c for R_i in mats_s for c in R_i.columns})
                R_bar_s = _fisher_avg_corr(mats_s, counts_s, all_cols_s)
                if use_psd_projection:
                    R_bar_s = _nearest_psd_correlation(R_bar_s)

                row = {"Date": pd.to_datetime(t), "Sector": s}
                for fact, cols in factor_to_rankcols.items():
                    cols_f = [c for c in cols if c in R_bar_s.columns]
                    if len(cols_f) >= min_metrics_per_factor:
                        Rf = R_bar_s.loc[cols_f, cols_f]
                        w_hrp = _hrp_weights_from_corr(Rf, linkage_method=linkage_method)
                    else:
                        w_hrp = pd.Series(1.0 / max(len(cols_f), 1), index=cols_f, dtype=float)

                    key = (s, fact)
                    w_prev = prev_eff_by_key.get(key, None)
                    w_prev = None if w_prev is None else w_prev.reindex(cols_f)
                    w_eff = _smooth(w_hrp, w_prev, alpha=alpha)
                    if w_eff.sum() > 0:
                        w_eff = w_eff / w_eff.sum()
                    prev_eff_by_key[key] = w_eff.copy()

                    for c in cols:
                        row[c] = float(w_eff.get(c, 0.0))
                for c in dropped:
                    row[c] = 0.0
                out_rows.append(row)

        if len(out_rows) % 10 == 0:
            print(f"  ... procesadas {len(out_rows)} filas, última fecha: {t.date()}")

    out = pd.DataFrame(out_rows, columns=out_cols)
    out = out.sort_values(["Date", "Sector"] if sectoral else ["Date"]).reset_index(drop=True)
    return out
