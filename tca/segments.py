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

    df["mktcap_group"] = df["market_cap_usd_bn"].map(_mktcap_group)
    df["adv_group"] = df["qty_pct_adv_20d"].map(_adv_group)
    df["spread_bucket"] = df["spread_bps"].map(_spread_bucket)

    for col, order in [("mktcap_group", MKTCAP_ORDER), ("adv_group", ADV_ORDER),
                       ("spread_bucket", SPREAD_ORDER)]:
        df[col] = pd.Categorical(df[col], categories=order, ordered=True)
    return df


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
        w = g["notional_local"].abs()
        vw = float(np.average(c, weights=w)) if w.sum() > 0 else np.nan
        rec = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        rec.update(n_orders=n, mean_cost_bps=float(c.mean()), median_cost_bps=float(c.median()),
                   std_cost_bps=std, t_stat=tstat, vw_cost_bps=vw)
        rows.append(rec)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("mean_cost_bps", ascending=False).reset_index(drop=True)
    return out
