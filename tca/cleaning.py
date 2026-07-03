from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QualityReport:
    """Coverage / data-quality counts emitted before any aggregation."""

    total_rows: int
    filled_or_partfilled_rows: int
    kept_rows: int
    kept_pct: float
    part_filled_rows: int
    dropped_missing_arrival: int
    dropped_missing_core_fields: int
    flagged_negative_spread: int
    flagged_avgpx_outside_hilo: int   # open/low/high snapshot mismatch (informational)
    flagged_extreme_adv: int
    flagged_multi_day_gtc: int
    footprint_orders: int             # orders with meaningful arrival slippage (|bps|>0.5)
    currency_counts: dict = field(default_factory=dict)
    securities_with_vol: int = 0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _coalesce(df: pd.DataFrame, candidates: Iterable[str], default=np.nan) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series([default] * len(df), index=df.index)


def _num(series: pd.Series) -> pd.Series:
    """Coerce to numeric, tolerating thousands-separators / stray text.

    Several benchmark fields (ArrPx, IntervalVWAP, the provided *Bps* columns) come
    through the grid as TEXT, so a plain to_numeric drops them.
    """
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def _winsorize(series: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")
    if clean.notna().sum() < 5:
        return clean
    lo, hi = clean.quantile(lower_q), clean.quantile(upper_q)
    return clean.clip(lower=lo, upper=hi)


def _side_factor(side: pd.Series) -> np.ndarray:
    """+1 for sell, -1 for buy → 'negative = cost' for both (CLAUDE.md convention)."""
    s = side.astype(str).str.strip().str.lower()
    return np.where(s.str.startswith("b"), -1.0, 1.0)


def _slip_bps(avg: pd.Series, bench: pd.Series, sf: np.ndarray) -> pd.Series:
    """Side-adjusted slippage in bps of the benchmark (negative = cost).

    Verified against Bloomberg's `AvgPx Vs ArrPx (Bps)`: positive = price
    improvement, negative = cost, for both buys and sells. A non-positive or missing
    benchmark yields NaN (a few IntervalVWAP rows are 0 and would divide to ±inf).
    """
    bench = bench.where(bench > 0)
    return sf * (avg - bench) / bench * 1e4


# ---------------------------------------------------------------------------
# child routes (still used for parent→child drill-down; volatility now on parent)
# ---------------------------------------------------------------------------

def clean_child_orders(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df["security"] = _coalesce(df, ["Security"]).astype(str).str.strip()
    df["brkr_code"] = _coalesce(df, ["Brkr Code"]).astype(str).str.strip()
    df["exch_code"] = _coalesce(df, ["Exch Code"]).astype(str).str.strip()
    df["side"] = _coalesce(df, ["Side"]).astype(str).str.strip().str.lower()
    df["fill_qty"] = _num(_coalesce(df, ["FillQty"]))
    df["avg_px"] = _num(_coalesce(df, ["AvgPx"]))
    df["volatil_30d"] = _num(_coalesce(df, ["Volatil 30D"]))
    df["avg_vol_20d"] = _num(_coalesce(df, ["Avg Vol 20D"]))
    df["fillqty_pct_adv_20d"] = _num(_coalesce(df, ["FillQty % Avg Vol 20D"]))
    df["create_time"] = pd.to_datetime(
        _coalesce(df, ["Create Time (As of)"]), errors="coerce", format="mixed"
    )
    return df


def security_volatility_map(child_clean: pd.DataFrame) -> pd.DataFrame:
    return (
        child_clean.groupby("security", dropna=True)
        .agg(volatil_30d_child=("volatil_30d", "median"),
             avg_vol_20d_child=("avg_vol_20d", "median"))
        .reset_index()
    )


# ---------------------------------------------------------------------------
# parent orders — ARRIVAL-benchmark Implementation Shortfall
# ---------------------------------------------------------------------------

def clean_parent_orders(
    raw_df: pd.DataFrame, child_clean: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Phase-1 cleaning + slippage on the parent table.

    PRIMARY benchmark = arrival price (`ArrPx`) → trading-related Implementation
    Shortfall (Perold). The open benchmark is retained only as a DIAGNOSTIC column
    because Open/Low/High are an export-time snapshot (AvgPx lands inside [Low,High]
    ~7% of the time). Interval VWAP is a drift-free cross-check.

    Sign conventions (both kept):
      * ``slippage_bps`` / ``slippage_*_bps``  — negative = cost (CLAUDE.md).
      * ``cost_bps``                           — positive = cost (= -slippage_bps),
                                                 used for the impact fit & regression.
    """
    df = raw_df.copy()

    # identifiers / categoricals
    df["status"] = _coalesce(df, ["Status"]).astype(str).str.strip()
    df["side"] = _coalesce(df, ["Side"]).astype(str).str.strip().str.lower()
    df["security"] = _coalesce(df, ["Security"]).astype(str).str.strip()
    df["isin"] = _coalesce(df, ["ISIN"]).astype(str).str.strip()
    df["name"] = _coalesce(df, ["Name"]).astype(str).str.strip()
    df["order_id"] = _coalesce(df, ["ID"]).astype(str).str.strip()
    df["brkr_code"] = _coalesce(df, ["Brkr Code", "Def Brkr Code"]).astype(str).str.strip()
    df["exch_code"] = _coalesce(df, ["Exch Code"]).astype(str).str.strip()
    df["currency"] = _coalesce(df, ["Curncy"]).astype(str).str.strip()
    df["strategy"] = _coalesce(df, ["Strategy Name"]).astype(str).str.strip()
    df["gics_sector"] = _coalesce(df, ["GICS Sector"]).astype(str).str.strip()
    df["tif"] = _coalesce(df, ["TIF"]).astype(str).str.strip()
    df["handling_inst"] = _coalesce(df, ["Handling Inst"]).astype(str).str.strip()

    # numerics — prices & benchmarks
    df["qty"] = _num(_coalesce(df, ["Qty"]))
    df["fill_qty"] = _num(_coalesce(df, ["FillQty"]))
    df["avg_px"] = _num(_coalesce(df, ["AvgPx"]))
    df["arr_px"] = _num(_coalesce(df, ["ArrPx"]))
    df["interval_vwap"] = _num(_coalesce(df, ["IntervalVWAP"]))
    df["open_px"] = _num(_coalesce(df, ["Open Px", "OpenPx"]))
    df["yest_cls_px"] = _num(_coalesce(df, ["Yest Cls Px"]))
    df["low"] = _num(_coalesce(df, ["Low"]))
    df["high"] = _num(_coalesce(df, ["High"]))
    df["bid_ask_sprd"] = _num(_coalesce(df, ["Bid Ask Sprd"]))
    df["pct_filled"] = _num(_coalesce(df, ["% Filled"]))
    df["value_local"] = _num(_coalesce(df, ["Value (Local)"]))

    # numerics — difficulty controls (Volatil 30D & %ADV now ship on the parent)
    df["volatil_30d"] = _num(_coalesce(df, ["Volatil 30D"]))
    df["avg_vol_30d"] = _num(_coalesce(df, ["Avg Vol 30D"]))
    df["day_part_rate"] = _num(_coalesce(df, ["Day Part Rate %"]))          # realized participation (POV)
    df["fillqty_pct_vwap_vol"] = _num(_coalesce(df, ["FillQty % VWAP Vol"]))
    df["tca20"] = _num(_coalesce(df, ["TCA (20%)"]))                         # Bloomberg pre-trade cost estimate (bps, 20% POV)
    df["rsi_14d"] = _num(_coalesce(df, ["RSI 14D"]))                         # short-term momentum signal
    df["qty_pct_adv_20d"] = _num(
        _coalesce(df, ["FillQty % Avg Vol 20D", "Qty % Avg Vol 20D"])
    )
    # provided Bloomberg benchmark bps (positive = cost, their convention) — reference only
    df["bbg_avgpx_vs_arr_bps"] = _num(_coalesce(df, ["AvgPx Vs ArrPx (Bps)"]))
    df["bbg_avgpx_vs_vwap_bps"] = _num(_coalesce(df, ["AvgPx Vs IntervalVWAP (Bps)"]))
    df["openpx_vs_yestcls_bps"] = _num(_coalesce(df, ["OpenPx Vs Yest Cls Px (Bps)"]))

    # timing
    df["create_time"] = pd.to_datetime(
        _coalesce(df, ["Create Time (As of)"]), errors="coerce", format="mixed"
    )
    df["trade_date"] = pd.to_datetime(
        _coalesce(df, ["Trade Date"]), errors="coerce", format="mixed"
    )

    # keep only orders that actually traded
    status_l = df["status"].str.lower()
    df = df.loc[status_l.isin(["filled", "part-filled", "part filled", "partially filled"])].copy()
    df["flag_part_fill"] = status_l.reindex(df.index).isin(
        ["part-filled", "part filled", "partially filled"]
    )

    # ADV in shares, back-derived from the exact ratio, and unit-free X/ADV
    ratio = df["qty_pct_adv_20d"] / 100.0
    df["adv_shares_20d"] = np.where(ratio > 0, df["fill_qty"] / ratio, np.nan)
    df["x_over_adv"] = ratio

    # volatility fallback from child if a parent row is missing it
    if child_clean is not None and len(child_clean):
        df = df.merge(security_volatility_map(child_clean), on="security", how="left")
        df["volatil_30d"] = df["volatil_30d"].fillna(df["volatil_30d_child"])
    df["has_volatility"] = df["volatil_30d"].notna()

    # --- slippage: PRIMARY = arrival, plus VWAP & open diagnostics (negative = cost) ---
    sf = _side_factor(df["side"])
    df["slippage_bps"] = _slip_bps(df["avg_px"], df["arr_px"], sf)       # PRIMARY (IS)
    df["slippage_vwap_bps"] = _slip_bps(df["avg_px"], df["interval_vwap"], sf)
    df["slippage_open_bps"] = _slip_bps(df["avg_px"], df["open_px"], sf)  # diagnostic only
    df["cost_bps"] = -df["slippage_bps"]                                  # positive = cost
    df["slippage_cash_local"] = (df["avg_px"] - df["arr_px"]) * df["fill_qty"] * sf
    df["notional_local"] = df["fill_qty"] * df["arr_px"]

    # Drift-free attribution on a COMMON arrival denominator so the pieces add exactly:
    #   slippage_bps = exec_vwap_bps  +  timing_bps
    #   exec_vwap_bps : execution vs the market's own interval VWAP (controllable skill)
    #   timing_bps    : market drift between arrival and the trading window (mostly not)
    # Both keep negative = cost, positive = improvement.
    _arr = df["arr_px"].where(df["arr_px"] > 0)
    _vwap = df["interval_vwap"].where(df["interval_vwap"] > 0)
    df["exec_vwap_bps"] = sf * (df["avg_px"] - _vwap) / _arr * 1e4
    df["timing_bps"] = sf * (_vwap - df["arr_px"]) / _arr * 1e4
    # overnight/delay context: arrival vs open, and open vs prior close
    df["delay_arr_vs_open_bps"] = (df["arr_px"] - df["open_px"]) / df["open_px"] * 1e4

    # spread in bps (driver of the Bloomberg TCA model) and the vendor cost-surprise
    df["spread_bps"] = df["bid_ask_sprd"].where(df["bid_ask_sprd"] >= 0) / df["arr_px"].where(df["arr_px"] > 0) * 1e4
    # cost surprise = realized cost − Bloomberg's ex-ante estimate (positive = worse than predicted)
    df["cost_surprise_bps"] = df["cost_bps"] - df["tca20"]

    # entry-timing control
    df["entry_hour"] = df["create_time"].dt.hour
    df["entry_minute"] = df["entry_hour"].fillna(0) * 60.0 + df["create_time"].dt.minute.fillna(0)

    # --- flags ---
    df["flag_negative_spread"] = df["bid_ask_sprd"] < 0
    df["flag_avgpx_outside_hilo"] = (
        df["avg_px"].notna() & df["low"].notna() & df["high"].notna()
        & ((df["avg_px"] < df["low"]) | (df["avg_px"] > df["high"]))
    )
    df["flag_extreme_adv"] = df["qty_pct_adv_20d"] > 100.0
    span = df["create_time"].dt.normalize() != df["trade_date"].dt.normalize()
    df["flag_multi_day_gtc"] = df["tif"].str.upper().str.contains("GTC", na=False) & span.fillna(False)
    df["flag_missing_core"] = (
        df["arr_px"].isna() | df["avg_px"].isna() | df["fill_qty"].isna()
        | df["side"].eq("") | df["brkr_code"].eq("") | df["exch_code"].eq("")
    )
    # orders that actually left a footprint vs arrival (informative for impact/attribution)
    df["has_footprint"] = df["cost_bps"].abs() > 0.5

    # winsorize continuous vars at 1/99 pct
    for col in ["slippage_bps", "slippage_vwap_bps", "cost_bps", "exec_vwap_bps",
                "timing_bps", "qty_pct_adv_20d", "bid_ask_sprd", "fill_qty", "day_part_rate",
                "tca20", "spread_bps", "cost_surprise_bps"]:
        df[f"{col}_w"] = _winsorize(df[col])

    df["keep_for_analysis"] = (
        (~df["flag_negative_spread"])
        & (~df["flag_missing_core"])
        & df["arr_px"].gt(0)
        & df["slippage_bps"].notna()
    )
    return df.reset_index(drop=True)


def build_data_quality_report(clean_df: pd.DataFrame) -> tuple[QualityReport, pd.DataFrame]:
    total = int(len(clean_df))
    kept = int(clean_df["keep_for_analysis"].sum())

    report = QualityReport(
        total_rows=total,
        filled_or_partfilled_rows=total,
        kept_rows=kept,
        kept_pct=(100.0 * kept / total) if total else 0.0,
        part_filled_rows=int(clean_df["flag_part_fill"].sum()),
        dropped_missing_arrival=int(clean_df["arr_px"].isna().sum()),
        dropped_missing_core_fields=int(clean_df["flag_missing_core"].sum()),
        flagged_negative_spread=int(clean_df["flag_negative_spread"].sum()),
        flagged_avgpx_outside_hilo=int(clean_df["flag_avgpx_outside_hilo"].sum()),
        flagged_extreme_adv=int(clean_df["flag_extreme_adv"].sum()),
        flagged_multi_day_gtc=int(clean_df["flag_multi_day_gtc"].sum()),
        footprint_orders=int((clean_df["keep_for_analysis"] & clean_df["has_footprint"]).sum()),
        currency_counts=dict(clean_df["currency"].value_counts()),
        securities_with_vol=int(clean_df["has_volatility"].sum()),
    )

    reasons = pd.DataFrame(
        {
            "reason": [
                "Filled / part-filled in scope",
                "Missing ArrPx (dropped from y)",
                "Missing core fields (dropped)",
                "Negative bid/ask spread (dropped)",
                "AvgPx outside [Low,High] (open snapshot — informational)",
                ">100% ADV (flagged, winsorized)",
                "Multi-day GTC (flagged, kept)",
                "Part-filled (flagged, kept)",
                "Kept for analysis",
                "  of which have arrival footprint (|bps|>0.5)",
            ],
            "count": [
                total,
                report.dropped_missing_arrival,
                report.dropped_missing_core_fields,
                report.flagged_negative_spread,
                report.flagged_avgpx_outside_hilo,
                report.flagged_extreme_adv,
                report.flagged_multi_day_gtc,
                report.part_filled_rows,
                kept,
                report.footprint_orders,
            ],
        }
    )
    return report, reasons
