from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import curve_fit


@dataclass(frozen=True)
class RegressionSuite:
    frame: pd.DataFrame
    ols_result: object          # standardized slippage, broker+venue FE, clustered SE
    q95_result: object          # 95th-pct quantile regression (tail drivers)
    formula: str


@dataclass(frozen=True)
class ImpactCurve:
    frame: pd.DataFrame
    b1: float                   # slip_bps ≈ b1 * (X/ADV)^b2
    b2: float
    b1_se: float
    b2_se: float
    n: int


def fit_regression_suite(clean_df: pd.DataFrame) -> RegressionSuite:
    """Regress standardized cost on difficulty controls + entry timing.

    Broker and venue enter as fixed effects; SEs are clustered by security
    (orders in the same name are not independent). The dependent variable is the
    z-scored *cost* (positive = cost) so a positive broker effect = costlier.
    """
    df = clean_df.loc[clean_df["keep_for_analysis"]].copy()
    if df.empty:
        raise ValueError("No rows available for modeling after cleaning.")

    required = [
        "cost_bps_w",
        "qty_pct_adv_20d_w",
        "bid_ask_sprd_w",
        "fill_qty_w",
        "volatil_30d",
        "day_part_rate_w",
        "entry_minute",
        "side",
        "brkr_code",
        "exch_code",
        "security",
    ]
    df = df.dropna(subset=required).copy()
    if df.empty:
        raise ValueError("No rows available after dropping missing model fields.")

    sd = df["cost_bps_w"].std(ddof=0)
    df["cost_z"] = 0.0 if (pd.isna(sd) or sd == 0) else (df["cost_bps_w"] - df["cost_bps_w"].mean()) / sd

    # difficulty controls now include Volatil 30D and participation (Day Part Rate),
    # both newly available on the parent export.
    rhs = (
        "qty_pct_adv_20d_w + bid_ask_sprd_w + np.log1p(fill_qty_w) + "
        "volatil_30d + day_part_rate_w + entry_minute + "
        "C(side) + C(brkr_code) + C(exch_code)"
    )
    formula = f"cost_z ~ {rhs}"

    groups = df["security"].replace("", np.nan).fillna(df["order_id"])
    ols = smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": groups})

    q_formula = f"cost_bps_w ~ {rhs}"
    q95 = smf.quantreg(q_formula, data=df).fit(q=0.95, max_iter=20000)

    return RegressionSuite(frame=df, ols_result=ols, q95_result=q95, formula=formula)


def _sqrt_law(x: np.ndarray, b1: float, b2: float) -> np.ndarray:
    return b1 * np.power(x, b2)


def fit_impact_curve(clean_df: pd.DataFrame, footprint_only: bool = True) -> ImpactCurve:
    """Pragmatic, currency-robust impact curve: cost_bps ≈ b1 * (X/ADV)^b2.

    Fit in bps vs (X/ADV) so it is unit-free across the 9 currencies (no $/share
    issue), per KGM_TC_MODEL.md §2. Now measured vs ARRIVAL (the correct reference
    for impact). ~66% of orders fill at arrival with ~0 footprint and would flatten
    b1, so by default we fit on the footprint subset (|arrival cost| > 0.5 bps),
    i.e. the orders that actually moved the price. Indicative shape, wide error bars.
    """
    df = clean_df.loc[clean_df["keep_for_analysis"]].copy()
    if footprint_only:
        df = df.loc[df["has_footprint"]].copy()
    x = pd.to_numeric(df["x_over_adv"], errors="coerce")
    y = pd.to_numeric(df["cost_bps"], errors="coerce")
    mask = x.notna() & y.notna() & (x > 0)
    df = df.loc[mask].copy()
    xv, yv = x[mask].to_numpy(), y[mask].to_numpy()
    if len(xv) < 10:
        raise ValueError("Too few points to fit the impact curve.")

    try:
        popt, pcov = curve_fit(
            _sqrt_law, xv, yv, p0=[50.0, 0.5], bounds=([0, 0], [np.inf, 2.0]), maxfev=20000
        )
        b1, b2 = float(popt[0]), float(popt[1])
        perr = np.sqrt(np.diag(pcov))
        b1_se, b2_se = float(perr[0]), float(perr[1])
    except Exception:
        # fall back to a fixed square-root exponent if NLLS misbehaves
        b2, b2_se = 0.5, float("nan")
        b1 = float(np.nanmedian(yv / np.sqrt(xv)))
        b1_se = float("nan")

    df = df.assign(impact_fit_bps=_sqrt_law(xv, b1, b2), sqrt_x_over_adv=np.sqrt(xv))
    return ImpactCurve(frame=df, b1=b1, b2=b2, b1_se=b1_se, b2_se=b2_se, n=int(len(xv)))
