"""Arrival vs Interval-VWAP gap analysis (market orders, GTC excluded).

For each category cell: n, value-weighted & mean cost vs Arrival (A) and vs VWAP (V),
t-stats, and the gap A-V (= timing drift). Positive = cost. Weighting = FX-adjusted USD.
"""
import numpy as np, pandas as pd
from tca.pipeline import run_parent_pipeline

pd.set_option("display.width", 200, "display.max_columns", 30)
art = run_parent_pipeline("data_trades_new2.csv", "180days_child_order_data.csv", exclude_gtc=True)
clean = art.clean
mkt = clean.loc[clean["keep_for_analysis"] & (clean["order_type"] == "Market")].copy()
print(f"Population: market orders, GTC excluded -> n = {len(mkt)}")
print(f"Book value-weighted:  Arrival = {np.average(mkt['cost_bps'], weights=mkt['notional_usd']):.2f} bps   "
      f"VWAP = {np.average(mkt['cost_vwap_bps'].fillna(0), weights=mkt['notional_usd']):.2f} bps")


def _t(s):
    s = s.dropna(); n = len(s)
    sd = s.std(ddof=1)
    return (s.mean() / (sd / np.sqrt(n))) if (n > 1 and sd > 0) else np.nan


def _vw(s, w):
    m = s.notna()
    return np.average(s[m], weights=w[m]) if m.sum() and w[m].sum() > 0 else np.nan


def analyze(df, by, min_n):
    rows = []
    for key, g in df.groupby(by, observed=True, dropna=True):
        if len(g) < min_n:
            continue
        w = g["notional_usd"]
        A, V = g["cost_bps"], g["cost_vwap_bps"]
        rec = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        rec.update(n=len(g),
                   A_vw=_vw(A, w), A_mean=A.mean(), A_t=_t(A),
                   V_vw=_vw(V, w), V_mean=V.mean(), V_t=_t(V))
        rec["gap_vw"] = rec["A_vw"] - rec["V_vw"]
        # read: execution (from V) and drift (from gap)
        exe = "exec worse" if rec["V_vw"] > 5 else ("exec beat mkt" if rec["V_vw"] < -5 else "exec ~ mkt")
        dft = "adverse drift" if rec["gap_vw"] > 5 else ("favorable drift" if rec["gap_vw"] < -5 else "no drift")
        rec["read"] = f"{exe} | {dft}"
        rows.append(rec)
    out = pd.DataFrame(rows)
    return out.sort_values("A_vw", ascending=False).reset_index(drop=True) if not out.empty else out


CUTS = [
    ("Direction", ["direction"], 30),
    ("Spread bucket x Direction", ["spread_bucket", "direction"], 25),
    ("ADV% group x Direction", ["adv_group", "direction"], 25),
    ("Market-cap x Direction", ["mktcap_group", "direction"], 30),
    ("Region x Direction", ["region", "direction"], 30),
    ("Broker x Direction", ["brkr_code", "direction"], 30),
    ("Region x Broker", ["region", "brkr_code"], 25),
]
cols = ["n", "A_vw", "V_vw", "gap_vw", "A_t", "V_t", "read"]
for title, by, mn in CUTS:
    print(f"\n{'='*100}\n{title}  (min-n={mn})\n{'='*100}")
    res = analyze(mkt, by, mn)
    if res.empty:
        print("  (no cell meets min-n)")
        continue
    show = res[by + cols].copy()
    for c in ["A_vw", "V_vw", "gap_vw", "A_t", "V_t"]:
        show[c] = show[c].round(1)
    print(show.to_string(index=False))
