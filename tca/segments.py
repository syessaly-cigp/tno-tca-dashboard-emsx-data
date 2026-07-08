"""Row segmentation for the Market-Order TCA framework.

Differentiating factors matter in TCA: the same broker looks good on large-cap US names
and bad on small-cap Asian ones. This module tags each order with the grouping factors in
the framework spec — region, industry, market-cap group, ADV% group, spread bucket,
direction — and computes per-group cost statistics (mean/median/std/t-stat).

Cost convention here follows the framework spec: **positive = cost** (= `cost_bps`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Approximate FX to USD (mid-2026, indicative — the book is ~75% USD so cap buckets are
# robust to this). MC Lst Trd is in millions of the pricing currency; GBp market caps are
# reported in GBP millions, so use the GBP rate for GBp names.
FX_TO_USD = {
    "USD": 1.00, "EUR": 1.08, "GBp": 1.27, "GBP": 1.27, "CHF": 1.12,
    "HKD": 0.128, "JPY": 0.0067, "CNY": 0.14, "AUD": 0.66, "CAD": 0.73,
}
# FX for ORDER NOTIONAL: prices for GBp names are quoted in PENCE, so pence -> USD = 1.27/100.
# (Distinct from market-cap FX above, where MC is reported in GBP millions.) Used to convert
# each order's local-ccy notional to USD so value-weighting is comparable across regions.
FX_ORDER_TO_USD = {**FX_TO_USD, "GBp": 0.0127, "GBP": 1.27}

# Primary-exchange MIC → region
_MIC_REGION = {
    # Americas
    "ARCX": "Americas", "XNGS": "Americas", "XNMS": "Americas", "XNYS": "Americas",
    "XNCM": "Americas", "BATS": "Americas", "OOTC": "Americas", "XASE": "Americas",
    "IEXG": "Americas", "XTSE": "Americas", "XTSX": "Americas",
    # Europe (incl. UK / Switzerland)
    "XLON": "Europe", "XSWX": "Europe", "XVTX": "Europe", "XPAR": "Europe",
    "XETR": "Europe", "XFRA": "Europe", "MTAA": "Europe", "XMIL": "Europe",
    "XAMS": "Europe", "XBRU": "Europe", "XMAD": "Europe", "XSTO": "Europe",
    "XHEL": "Europe", "XCSE": "Europe", "XOSL": "Europe", "XLIS": "Europe",
    # Asia-Pacific
    "XHKG": "Asia-Pacific", "XTKS": "Asia-Pacific", "XJPX": "Asia-Pacific",
    "XSHG": "Asia-Pacific", "XSHE": "Asia-Pacific", "XSEC": "Asia-Pacific",
    "XASX": "Asia-Pacific", "XKRX": "Asia-Pacific", "XTAI": "Asia-Pacific",
    "XSES": "Asia-Pacific",
}
_CCY_REGION = {
    "USD": "Americas", "CAD": "Americas", "EUR": "Europe", "CHF": "Europe",
    "GBp": "Europe", "GBP": "Europe", "HKD": "Asia-Pacific", "JPY": "Asia-Pacific",
    "CNY": "Asia-Pacific", "AUD": "Asia-Pacific",
}

MKTCAP_ORDER = ["Small Cap", "Mid Cap", "Large Cap"]
ADV_ORDER = ["Small", "Medium", "Large", "Very Large"]
SPREAD_ORDER = ["Tight", "Medium", "Wide", "Very Wide"]

MKTCAP_DEFS = {"Large Cap": "> USD 10bn", "Mid Cap": "USD 2–10bn", "Small Cap": "< USD 2bn"}
ADV_DEFS = {"Small": "< 1% ADV", "Medium": "1–5% ADV", "Large": "5–10% ADV", "Very Large": "> 10% ADV"}
SPREAD_DEFS = {"Tight": "0–10 bps", "Medium": "10–20 bps", "Wide": "20–50 bps", "Very Wide": "> 50 bps"}


def _mktcap_group(usd_bn: float) -> object:
    if not np.isfinite(usd_bn):
        return np.nan
    if usd_bn > 10:
        return "Large Cap"
    if usd_bn >= 2:
        return "Mid Cap"
    return "Small Cap"


def _adv_group(pct: float) -> object:
    if not np.isfinite(pct):
        return np.nan
    if pct < 1:
        return "Small"
    if pct < 5:
        return "Medium"
    if pct < 10:
        return "Large"
    return "Very Large"


def _spread_bucket(bps: float) -> object:
    if not np.isfinite(bps) or bps < 0:
        return np.nan
    if bps <= 10:
        return "Tight"
    if bps <= 20:
        return "Medium"
    if bps <= 50:
        return "Wide"
    return "Very Wide"


def add_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Tag each order with region / industry / market-cap / ADV% / spread / direction."""
    fx = df["currency"].map(FX_TO_USD).fillna(1.0)
    df["market_cap_usd_bn"] = df.get("mc_last_trade") * fx / 1000.0 if "mc_last_trade" in df else np.nan

    mic = df.get("primary_mic", pd.Series(index=df.index, dtype=object)).astype(str).str.upper()
    region = mic.map(_MIC_REGION)
    df["region"] = region.fillna(df["currency"].map(_CCY_REGION)).fillna("Other")

    df["industry"] = df.get("gics_sector", pd.Series(index=df.index, dtype=object))
    df["industry"] = df["industry"].replace({"": np.nan, "nan": np.nan}).fillna("Unknown")

    df["direction"] = np.where(df["side"].astype(str).str.lower().str.startswith("b"), "Buy", "Sell")

    # FX-converted USD notional — the correct weight for value-weighting across regions
    df["notional_usd"] = df["notional_local"].abs() * df["currency"].map(FX_ORDER_TO_USD).fillna(1.0)

    df["mktcap_group"] = df["market_cap_usd_bn"].map(_mktcap_group)
    df["adv_group"] = df["qty_pct_adv_20d"].map(_adv_group)
    df["spread_bucket"] = df["spread_bps"].map(_spread_bucket)

    for col, order in [("mktcap_group", MKTCAP_ORDER), ("adv_group", ADV_ORDER),
                       ("spread_bucket", SPREAD_ORDER)]:
        df[col] = pd.Categorical(df[col], categories=order, ordered=True)
    return df


# Benchmark → cost column (positive = cost). Mirrors Bloomberg BTCA's multi-benchmark view.
BENCHMARK_COLS = {
    "Arrival": "cost_bps",
    "Interval VWAP": "cost_vwap_bps",
    "Open (diagnostic)": "cost_open_bps",
}
# fields offered as pivot categories (label → column)
PIVOT_CATEGORIES = {
    "Region": "region", "Broker": "brkr_code", "Venue": "exch_code",
    "Industry": "industry", "Market-cap group": "mktcap_group", "ADV% group": "adv_group",
    "Spread bucket": "spread_bucket", "Direction": "direction", "Order type": "order_type",
    "Currency": "currency",
}


def btca_pivot(
    clean_df: pd.DataFrame,
    categories: list[str],
    benchmarks: list[str],
    stat: str = "Value-weighted",
    min_n: int = 1,
    add_total: bool = True,
) -> pd.DataFrame:
    """Bloomberg-BTCA-style pivot: group by any categories, cost vs each benchmark.

    Categories are column names (region, brkr_code, …); benchmarks are keys of
    BENCHMARK_COLS. `stat` ∈ {Value-weighted, Mean, Median}. Cost is positive = cost,
    value-weighted by traded notional (BTCA convention). Dynamic — add/remove categories
    or benchmarks and the table recomputes.
    """
    d = clean_df.loc[clean_df["keep_for_analysis"]].copy()
    if not categories:
        categories = []
    if categories:
        d = d.dropna(subset=categories)
    cost_cols = {b: BENCHMARK_COLS[b] for b in benchmarks if b in BENCHMARK_COLS}

    def _agg(g: pd.DataFrame) -> dict:
        rec = {"n_orders": int(len(g))}
        w = g["notional_usd"]
        for bname, col in cost_cols.items():
            s = pd.to_numeric(g[col], errors="coerce")
            m = s.notna()
            if m.sum() == 0:
                val = np.nan
            elif stat == "Mean":
                val = float(s[m].mean())
            elif stat == "Median":
                val = float(s[m].median())
            else:  # Value-weighted
                ww = w[m]
                val = float(np.average(s[m], weights=ww)) if ww.sum() > 0 else float(s[m].mean())
            rec[f"{bname} cost (bps)"] = val
        return rec

    rows = []
    if categories:
        for key, g in d.groupby(categories, observed=True, dropna=True):
            if len(g) < min_n:
                continue
            rec = dict(zip(categories, key if isinstance(key, tuple) else (key,)))
            rec.update(_agg(g))
            rows.append(rec)
    out = pd.DataFrame(rows)
    if not out.empty:
        first_bench = f"{benchmarks[0]} cost (bps)" if benchmarks else "n_orders"
        out = out.sort_values(first_bench, ascending=False).reset_index(drop=True)
        if add_total:
            total = dict.fromkeys(categories, "ALL")
            total.update(_agg(d))
            out = pd.concat([out, pd.DataFrame([total])], ignore_index=True)
    return out


def cost_stats(clean_df: pd.DataFrame, by: list[str], min_n: int = 5) -> pd.DataFrame:
    """Per-group arrival-cost stats (positive = cost): n, mean, median, std, t-stat, VW mean.

    t-stat tests whether the group's mean cost differs from zero (|t| > ~2 ⇒ significant).
    """
    df = clean_df.loc[clean_df["keep_for_analysis"]].copy()
    df = df.dropna(subset=["cost_bps"] + by)
    rows = []
    for key, g in df.groupby(by, observed=True, dropna=True):
        if len(g) < min_n:
            continue
        c = g["cost_bps"]
        n = len(c)
        std = float(c.std(ddof=1)) if n > 1 else np.nan
        tstat = float(c.mean() / (std / np.sqrt(n))) if (std and std > 0) else np.nan
        w = g["notional_usd"]
        vw = float(np.average(c, weights=w)) if w.sum() > 0 else np.nan
        rec = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        rec.update(n_orders=n, mean_cost_bps=float(c.mean()), median_cost_bps=float(c.median()),
                   std_cost_bps=std, t_stat=tstat, vw_cost_bps=vw)
        rows.append(rec)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("mean_cost_bps", ascending=False).reset_index(drop=True)
    return out
