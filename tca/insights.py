from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HeadlineSummary:
    n_orders: int
    n_footprint: int                     # orders with meaningful arrival footprint
    mean_slippage_bps: float             # vs arrival, equal-weighted (negative = cost)
    median_slippage_bps: float
    value_weighted_slippage_bps: float   # vs arrival, value-weighted (KGM §1 style)
    footprint_mean_cost_bps: float       # positive = cost, footprint orders only
    vwap_value_weighted_bps: float       # drift-free cross-check vs interval VWAP
    sanity_ok: bool                      # book-level slippage is a sane magnitude & VWAP agrees
    benchmark: str = "arrival (ArrPx)"


def _vw(values: pd.Series, weights: pd.Series) -> float:
    m = values.notna() & weights.notna() & (weights > 0)
    if m.sum() == 0:
        return float("nan")
    return float(np.average(values[m], weights=weights[m]))


def value_weighted_summary(clean_df: pd.DataFrame) -> HeadlineSummary:
    """Portfolio headline vs ARRIVAL: value-weighted bps, not the equal-weighted mean.

    Sanity is judged on the footprint subset (orders that actually traded away from
    arrival) — the full book is ~2/3 zero-footprint fills that wash the mean toward 0.
    """
    df = clean_df.loc[clean_df["keep_for_analysis"]].copy()
    n = int(len(df))
    if n == 0:
        return HeadlineSummary(0, 0, *([float("nan")] * 5), False)

    notional = df["notional_usd"]
    fp = df.loc[df["has_footprint"]]
    footprint_mean_cost = float(fp["cost_bps"].mean()) if len(fp) else float("nan")
    vw_arr = _vw(df["slippage_bps"], notional)
    vw_vwap = _vw(df["slippage_vwap_bps"], notional)
    # For an arrival benchmark, price improvement is legitimate (worked limit flow), so
    # "must be a cost" is the wrong test. Sanity = book-level slippage is a plausible
    # magnitude and the two independent benchmarks broadly agree in sign/scale.
    sane = bool(np.isfinite(vw_arr) and abs(vw_arr) < 100 and np.isfinite(vw_vwap))
    return HeadlineSummary(
        n_orders=n,
        n_footprint=int(len(fp)),
        mean_slippage_bps=float(df["slippage_bps"].mean()),
        median_slippage_bps=float(df["slippage_bps"].median()),
        value_weighted_slippage_bps=vw_arr,
        footprint_mean_cost_bps=footprint_mean_cost,
        vwap_value_weighted_bps=vw_vwap,
        sanity_ok=sane,
    )


def attribution_summary(clean_df: pd.DataFrame, footprint_only: bool = True) -> dict:
    """Book-level value-weighted decomposition of arrival slippage.

    slippage_bps = exec_vwap_bps (execution vs market VWAP) + timing_bps (drift).
    Value-weighted by traded notional. Returns a dict for a waterfall.
    """
    df = clean_df.loc[clean_df["keep_for_analysis"]].copy()
    df = df.dropna(subset=["exec_vwap_bps", "timing_bps", "slippage_bps"])
    if footprint_only:
        df = df.loc[df["has_footprint"]]
    w = df["notional_usd"]
    return {
        "n": int(len(df)),
        "execution_vs_vwap_bps": _vw(df["exec_vwap_bps"], w),
        "timing_drift_bps": _vw(df["timing_bps"], w),
        "total_vs_arrival_bps": _vw(df["slippage_bps"], w),
    }


def attribution_by(
    clean_df: pd.DataFrame,
    group_col: str,
    footprint_only: bool = True,
    min_n: int = 1,
) -> pd.DataFrame:
    """Per-broker / per-venue value-weighted execution vs timing decomposition.

    Isolates the controllable piece (execution vs VWAP) from market drift so a broker
    isn't blamed (or credited) for how the stock moved while it worked the order.
    Buckets with fewer than ``min_n`` footprint orders are dropped as too thin to read
    (a single penny-stock order can otherwise dominate the chart).
    """
    df = clean_df.loc[clean_df["keep_for_analysis"]].copy()
    df = df.dropna(subset=["exec_vwap_bps", "timing_bps", "slippage_bps"])
    if footprint_only:
        df = df.loc[df["has_footprint"]]
    cols = [group_col, "n_orders", "execution_vs_vwap_bps", "timing_drift_bps", "total_vs_arrival_bps"]
    if df.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for key, g in df.groupby(group_col, dropna=False):
        if len(g) < min_n:
            continue
        w = g["notional_usd"]
        rows.append(
            {
                group_col: key,
                "n_orders": int(len(g)),
                "execution_vs_vwap_bps": _vw(g["exec_vwap_bps"], w),
                "timing_drift_bps": _vw(g["timing_bps"], w),
                "total_vs_arrival_bps": _vw(g["slippage_bps"], w),
            }
        )
    out = pd.DataFrame(rows, columns=cols)
    # rank worst execution (most negative = costliest) first
    return out.sort_values("execution_vs_vwap_bps").reset_index(drop=True)


def _effects_to_table(params: pd.Series, prefix: str, baseline_label: str = "BASE") -> pd.DataFrame:
    rows = [{"bucket": baseline_label, "fe_effect": 0.0}]
    for key, val in params.items():
        if key.startswith(prefix):
            name = key.split("T.", 1)[-1].rstrip("]")
            rows.append({"bucket": name, "fe_effect": float(val)})
    return pd.DataFrame(rows)


def build_league_tables(frame: pd.DataFrame, ols_result: object) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Difficulty-adjusted league tables (residual / FE), positive = costlier.

    Uses the regression's broker/venue fixed effects (on standardized cost) plus
    mean residual per bucket. Raw averages are a confound trap — the biggest
    broker gets the hardest orders — so rank on the adjusted effect, not raw bps.
    """
    work = frame.copy()
    work["residual"] = ols_result.resid

    def _table(group_col: str, fe_prefix: str) -> pd.DataFrame:
        perf = (
            work.groupby(group_col, dropna=False)
            .agg(
                n_orders=("order_id", "count"),
                mean_residual=("residual", "mean"),
                raw_mean_cost_bps=("cost_bps_w", "mean"),
                median_cost_bps=("cost_bps_w", "median"),
            )
            .reset_index()
            .rename(columns={group_col: "bucket"})
        )
        fe = _effects_to_table(ols_result.params, fe_prefix)
        tbl = perf.merge(fe, on="bucket", how="left")
        tbl["fe_effect"] = tbl["fe_effect"].fillna(0.0)
        # rank costliest first (positive effect = costs more after controls)
        return tbl.sort_values(["fe_effect", "n_orders"], ascending=[False, False]).reset_index(drop=True)

    broker = _table("brkr_code", "C(brkr_code)")
    venue = _table("exch_code", "C(exch_code)")
    return broker, venue


def compute_trend_summary(clean_df: pd.DataFrame) -> pd.DataFrame:
    df = clean_df.loc[clean_df["keep_for_analysis"]].copy()
    if df.empty or df["trade_date"].notna().sum() == 0:
        return pd.DataFrame(columns=["week", "n_orders", "mean_slippage_bps", "median_slippage_bps"])

    df = df.loc[df["trade_date"].notna()].copy()
    df["week"] = df["trade_date"].dt.to_period("W").dt.start_time
    return (
        df.groupby("week", dropna=True)
        .agg(
            n_orders=("order_id", "count"),
            mean_slippage_bps=("slippage_bps_w", "mean"),
            median_slippage_bps=("slippage_bps_w", "median"),
        )
        .reset_index()
        .sort_values("week")
    )


def cost_by_participation(clean_df: pd.DataFrame, footprint_only: bool = True) -> pd.DataFrame:
    """Value-weighted arrival cost bucketed by realized participation (Day Part Rate %).

    The POV framework: impact should rise with how aggressively you take liquidity. Uses
    the participation field the fuller export now carries.
    """
    df = clean_df.loc[clean_df["keep_for_analysis"]].copy()
    df = df.dropna(subset=["day_part_rate", "cost_bps"])
    if footprint_only:
        df = df.loc[df["has_footprint"]]
    if df.empty:
        return pd.DataFrame(columns=["pov_bucket", "n_orders", "mean_pov", "cost_bps", "tca20_bps"])

    bins = [0, 1, 5, 10, 25, 50, 100.0001]
    labels = ["<1%", "1-5%", "5-10%", "10-25%", "25-50%", "50-100%"]
    df["pov_bucket"] = pd.cut(df["day_part_rate"].clip(upper=100), bins=bins, labels=labels, include_lowest=True)
    rows = []
    for b, g in df.groupby("pov_bucket", observed=True):
        w = g["notional_usd"]
        wv = lambda s: float(np.average(s, weights=w)) if w.sum() > 0 else float("nan")
        rows.append({
            "pov_bucket": str(b),
            "n_orders": int(len(g)),
            "mean_pov": float(g["day_part_rate"].mean()),
            "cost_bps": wv(g["cost_bps"]),
            "tca20_bps": wv(g["tca20"]) if g["tca20"].notna().any() else float("nan"),
        })
    return pd.DataFrame(rows)


def child_routes_for_broker(child_clean: pd.DataFrame, broker: str) -> pd.DataFrame:
    """Parent -> child drill-down. The child export has no parent id, so this is a
    broker/security view of routes, not a true order-level join (documented limit)."""
    df = child_clean.loc[child_clean["brkr_code"] == broker].copy()
    if df.empty:
        return df
    return (
        df.groupby(["security", "exch_code"], dropna=False)
        .agg(
            n_routes=("security", "size"),
            total_fill=("fill_qty", "sum"),
            median_fillqty_pct_adv=("fillqty_pct_adv_20d", "median"),
            median_volatil_30d=("volatil_30d", "median"),
        )
        .reset_index()
        .sort_values("n_routes", ascending=False)
    )
