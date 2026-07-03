"""Kissell–Glantz–Malamut (2004) ex-ante layer.

Companion to KGM_TC_MODEL.md. Three pieces, kept separate from the ex-post
measurement (modeling.py):

  §2  faithful impact calibration  I = a1 * X^a2 * (alpha/v + (1-alpha)/X),  alpha = 0.95
  §3  single-security cost & risk of a *schedule* (Eq. 7)
  §4  Efficient Trading Frontier over a family of schedules + decision criteria

CRITICAL: the ETF is built by VARYING the schedule, NOT extracted from realized
orders. The 951-order cross-section's only role here is to calibrate the impact
model. Never let a modeled cost overwrite a measured cost.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

ALPHA = 0.95  # temporary/permanent split — FIXED (KGM find it stable; ~900 orders can't estimate it)
TRADING_DAYS = 252.0


# ---------------------------------------------------------------------------
# §2  faithful impact calibration  (a1, a2),  alpha fixed
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FaithfulImpact:
    a1: float
    a2: float
    a1_se: float
    a2_se: float
    n: int
    note: str


def calibrate_faithful_impact(clean_df: pd.DataFrame) -> FaithfulImpact:
    """Fit MI($/share) = a1 * X^a2 * (alpha/v + (1-alpha)/X), alpha = 0.95.

    Dependent = realized $/share over open (cost_bps/1e4 * open_px). X = FillQty.
    v = same-side market volume over the interval, proxied by 0.5 * ADV20 (no
    intraday volume on hand). FLAG: mixes 9 currencies in the $/share dependent —
    indicative only; the bps curve in modeling.py is the multi-currency-robust
    headline. Use this only for the temp/perm decomposition the ETF needs.
    """
    df = clean_df.loc[clean_df["keep_for_analysis"]].copy()
    X = pd.to_numeric(df["fill_qty"], errors="coerce")
    adv = pd.to_numeric(df["adv_shares_20d"], errors="coerce")
    px = pd.to_numeric(df["arr_px"], errors="coerce")            # arrival is the impact reference
    mi = pd.to_numeric(df["cost_bps"], errors="coerce") / 1e4 * px  # $/share, +=cost
    v_side = 0.5 * adv

    mask = X.notna() & v_side.notna() & mi.notna() & (X > 0) & (v_side > 0)
    Xv, vv, yv = X[mask].to_numpy(), v_side[mask].to_numpy(), mi[mask].to_numpy()
    if len(Xv) < 10:
        raise ValueError("Too few points to calibrate faithful impact.")

    def model(data, a1, a2):
        x, v = data
        return a1 * np.power(x, a2) * (ALPHA / v + (1.0 - ALPHA) / x)

    try:
        popt, pcov = curve_fit(
            model, (Xv, vv), yv, p0=[1e-3, 0.5],
            bounds=([0, 0], [np.inf, 2.0]), maxfev=40000,
        )
        a1, a2 = float(popt[0]), float(popt[1])
        perr = np.sqrt(np.diag(pcov))
        a1_se, a2_se = float(perr[0]), float(perr[1])
    except Exception as exc:  # pragma: no cover - defensive
        a1, a2, a1_se, a2_se = float("nan"), float("nan"), float("nan"), float("nan")
        return FaithfulImpact(a1, a2, a1_se, a2_se, int(len(Xv)), f"fit failed: {exc}")

    note = (
        "alpha=0.95 fixed; v proxied by 0.5*ADV20; $/share dependent mixes currencies "
        "(indicative). Headline impact = pragmatic bps curve."
    )
    return FaithfulImpact(a1, a2, a1_se, a2_se, int(len(Xv)), note)


# ---------------------------------------------------------------------------
# §3  single-security cost & risk of a schedule  (Eq. 7)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OrderSpec:
    """Everything one single-security ETF needs."""

    label: str
    x_shares: float
    adv_shares: float
    sigma_annual: float   # decimal return vol (e.g. 0.25), from Volatil 30D / 100
    open_px: float
    b1: float             # pragmatic curve: full-order cost_bps = b1 * (X/ADV)^b2
    b2: float
    currency: str = ""


def _full_order_impact_dollar(spec: OrderSpec) -> float:
    """Total $ impact I of trading the whole order, from the calibrated bps curve."""
    x_over_adv = spec.x_shares / spec.adv_shares if spec.adv_shares > 0 else np.nan
    cost_bps_full = spec.b1 * (x_over_adv ** spec.b2)
    return cost_bps_full / 1e4 * spec.x_shares * spec.open_px


def schedule_cost_risk_bps(
    spec: OrderSpec, horizon_days: float, periods_per_day: float = 13.0
) -> tuple[float, float]:
    """Constant-rate (TWAP) liquidation of X over `horizon_days`; Eq. 7 cost & risk.

    Returns (cost_bps, risk_bps), both as bps of traded notional. E[trend] = 0
    (intraday drift negligible vs vol), so the price-trend term drops.
    """
    n = max(1, int(round(horizon_days * periods_per_day)))
    X = spec.x_shares
    x_j = X / n                                   # equal slices
    v_j = spec.adv_shares * (horizon_days / n)    # market volume per period
    I = _full_order_impact_dollar(spec)           # total $ impact scale

    # temporary impact: sum_j 0.95 * I * x_j^2 / (X * (x_j + 0.5 v_j))
    temp = np.sum(ALPHA * I * x_j ** 2 / (X * (x_j + 0.5 * v_j)))
    # permanent impact: sum_j 0.05 * I * x_j / X = 0.05 * I
    perm = (1.0 - ALPHA) * I
    cost_dollar = temp + perm

    # risk: sigma_$ per period * sqrt(sum r_j^2), r_j = residual shares from period j on
    j = np.arange(1, n + 1)
    r_j = X * (n - j + 1) / n
    daily_vol = spec.sigma_annual / np.sqrt(TRADING_DAYS)
    sigma_dollar_period = daily_vol * np.sqrt(horizon_days / n) * spec.open_px
    risk_dollar = sigma_dollar_period * np.sqrt(np.sum(r_j ** 2))

    notional = X * spec.open_px
    return (cost_dollar / notional * 1e4, risk_dollar / notional * 1e4)


def build_etf(
    spec: OrderSpec,
    horizons_days: np.ndarray | None = None,
    periods_per_day: float = 13.0,
) -> pd.DataFrame:
    """Trace the Efficient Trading Frontier by varying the trading horizon.

    Short horizon = aggressive (high cost / low risk); long horizon = passive
    (low cost / high risk). One frontier per ORDER — never one for the dataset.
    """
    if horizons_days is None:
        horizons_days = np.geomspace(0.05, 20.0, 40)
    rows = []
    for h in horizons_days:
        cost_bps, risk_bps = schedule_cost_risk_bps(spec, float(h), periods_per_day)
        rows.append({"horizon_days": float(h), "cost_bps": cost_bps, "risk_bps": risk_bps})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# §4  decision criteria overlay
# ---------------------------------------------------------------------------

def decision_points(etf: pd.DataFrame, r_star: float | None = None, lam: float = 0.05,
                    c_star: float | None = None) -> dict:
    """Mark the optimal frontier point for each fund goal (KGM §3.2)."""
    out: dict = {}
    e = etf.dropna(subset=["cost_bps", "risk_bps"]).copy()
    if e.empty:
        return out

    # Goal 1: min cost s.t. risk <= R*
    if r_star is None:
        r_star = float(e["risk_bps"].median())
    feasible = e.loc[e["risk_bps"] <= r_star]
    if not feasible.empty:
        out["goal1_min_cost_risk_cap"] = feasible.loc[feasible["cost_bps"].idxmin()].to_dict()
    out["r_star"] = r_star

    # Goal 2: min Cost + lambda * Risk  (tangency to the ETF)
    obj = e["cost_bps"] + lam * e["risk_bps"]
    out["goal2_min_cost_plus_lambda_risk"] = e.loc[obj.idxmin()].to_dict()
    out["lambda"] = lam

    # Goal 3: price improvement — max (C* - cost) / risk  (Sharpe-like)
    if c_star is None:
        c_star = float(e["cost_bps"].max())
    ratio = (c_star - e["cost_bps"]) / e["risk_bps"].replace(0, np.nan)
    if ratio.notna().any():
        out["goal3_price_improvement"] = e.loc[ratio.idxmax()].to_dict()
    out["c_star"] = c_star
    return out


# ---------------------------------------------------------------------------
# helpers to build a representative single-security spec from cleaned data
# ---------------------------------------------------------------------------

def order_spec_from_row(row: pd.Series, b1: float, b2: float) -> OrderSpec:
    return OrderSpec(
        label=str(row.get("security", row.get("order_id", "order"))),
        x_shares=float(row["fill_qty"]),
        adv_shares=float(row["adv_shares_20d"]),
        sigma_annual=float(row["volatil_30d"]) / 100.0,
        open_px=float(row["arr_px"]),   # price level = arrival (open is snapshot-contaminated)
        b1=b1,
        b2=b2,
        currency=str(row.get("currency", "")),
    )


def pick_representative_order(clean_df: pd.DataFrame, difficulty_q: float = 0.90) -> pd.Series | None:
    """A higher-difficulty, volatility-available order to seed an illustrative ETF.

    A median-%ADV order is so small that impact is ~0 and the frontier is flat;
    pick a harder order (default 90th-pct %ADV) so cost actually trades off vs risk.
    """
    df = clean_df.loc[
        clean_df["keep_for_analysis"]
        & clean_df["has_volatility"]
        & clean_df["adv_shares_20d"].gt(0)
        & clean_df["volatil_30d"].gt(0)
        & clean_df["arr_px"].gt(0)
    ].copy()
    if df.empty:
        return None
    target = df["qty_pct_adv_20d"].quantile(difficulty_q)
    idx = (df["qty_pct_adv_20d"] - target).abs().idxmin()
    return df.loc[idx]
