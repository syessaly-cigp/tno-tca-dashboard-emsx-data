"""Unit tests for the pure analysis functions (run before trusting the app).

Primary benchmark is ARRIVAL (ArrPx). Sign convention verified against Bloomberg's
`AvgPx Vs ArrPx (Bps)`: slippage negative = cost, positive = price improvement.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tca.cleaning import clean_child_orders, clean_parent_orders, build_data_quality_report
from tca.kissell import ALPHA, OrderSpec, build_etf, decision_points, schedule_cost_risk_bps
from tca.modeling import fit_impact_curve, fit_regression_suite
from tca.insights import attribution_by, attribution_summary, value_weighted_summary

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "td_data_full.csv"
CHILD = ROOT / "180days_child_order_data.csv"


def _toy_parent() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Status": ["Filled", "Filled", "Part-filled"],
            "Side": ["Buy", "Sell", "Buy"],
            "Security": ["AAA", "AAA", "BBB"],
            "ID": ["1", "2", "3"],
            "Brkr Code": ["X", "X", "Y"],
            "Exch Code": ["E1", "E1", "E2"],
            "Curncy": ["USD", "USD", "USD"],
            "Qty": [1000, 2000, 500],
            "FillQty": [1000, 2000, 250],
            "AvgPx": [100.5, 99.5, 50.25],   # buy paid up, sell sold down, buy paid up
            "ArrPx": [100.0, 100.0, 50.0],   # PRIMARY benchmark
            "IntervalVWAP": [100.2, 99.8, 50.1],
            "Open Px": [100.0, 100.0, 50.0],
            "Low": [99.0, 98.0, 49.0],
            "High": [102.0, 101.0, 51.0],
            "Yest Cls Px": [99.5, 99.5, 49.5],
            "Bid Ask Sprd": [0.02, 0.02, 0.05],
            "Qty % Avg Vol 20D": [10.0, 20.0, 5.0],
            "Volatil 30D": [25.0, 25.0, 40.0],
            "Day Part Rate %": [3.0, 5.0, 2.0],
            "Create Time (As of)": ["01/02/24 10:00", "01/02/24 11:00", "01/03/24 09:30"],
            "Trade Date": ["01/02/24", "01/02/24", "01/03/24"],
            "% Filled": [100, 100, 50],
            "TIF": ["DAY", "DAY", "DAY"],
            "Value (Local)": [100500, 199000, 12562],
        }
    )


def test_sign_convention_negative_is_cost_vs_arrival():
    clean = clean_parent_orders(_toy_parent())
    buy = clean.loc[clean["order_id"] == "1"].iloc[0]
    # buy paid 100.5 vs arrival 100 -> 50 bps cost -> slippage negative, cost_bps positive
    assert buy["slippage_bps"] == pytest.approx(-50.0, rel=1e-6)
    assert buy["cost_bps"] == pytest.approx(50.0, rel=1e-6)
    sell = clean.loc[clean["order_id"] == "2"].iloc[0]
    # sell sold 99.5 vs arrival 100 -> 50 bps cost too
    assert sell["slippage_bps"] == pytest.approx(-50.0, rel=1e-6)


def test_vwap_and_open_diagnostics_present():
    clean = clean_parent_orders(_toy_parent())
    row = clean.loc[clean["order_id"] == "1"].iloc[0]
    # buy 100.5 vs vwap 100.2 -> ~ -29.94 bps
    assert row["slippage_vwap_bps"] == pytest.approx(-10000 * (100.5 - 100.2) / 100.2, rel=1e-6)
    assert "slippage_open_bps" in clean.columns


def test_adv_shares_back_derived():
    clean = clean_parent_orders(_toy_parent())
    row = clean.loc[clean["order_id"] == "1"].iloc[0]
    # FillQty 1000 at 10% ADV -> ADV = 10000
    assert row["adv_shares_20d"] == pytest.approx(10000.0, rel=1e-6)
    assert row["x_over_adv"] == pytest.approx(0.10, rel=1e-6)


def test_quality_report_counts():
    clean = clean_parent_orders(_toy_parent())
    report, reasons = build_data_quality_report(clean)
    assert report.total_rows == 3
    assert report.part_filled_rows == 1
    assert report.kept_rows == 3          # all have ArrPx and core fields
    assert report.footprint_orders == 3   # all 3 are >0.5 bps from arrival
    assert {"reason", "count"}.issubset(reasons.columns)


def test_value_weighted_summary_sane():
    clean = clean_parent_orders(_toy_parent())
    h = value_weighted_summary(clean)
    assert h.benchmark.startswith("arrival")
    assert h.n_footprint == 3
    # every toy order is a ~50 bps cost -> value-weighted slippage negative but modest
    assert -100 < h.value_weighted_slippage_bps < 0
    assert h.sanity_ok


def test_attribution_decomposition_adds_up():
    clean = clean_parent_orders(_toy_parent())
    row = clean.loc[clean["order_id"] == "1"].iloc[0]
    # identity on common arrival denominator: slippage = execution-vs-VWAP + timing
    assert row["slippage_bps"] == pytest.approx(row["exec_vwap_bps"] + row["timing_bps"], rel=1e-9)
    # buy: arrival 100, vwap 100.2 -> stock ticked up before we traded = adverse timing (cost)
    assert row["timing_bps"] < 0
    tbl = attribution_by(clean, "brkr_code")
    assert {"execution_vs_vwap_bps", "timing_drift_bps", "total_vs_arrival_bps"}.issubset(tbl.columns)
    book = attribution_summary(clean)
    assert book["total_vs_arrival_bps"] == pytest.approx(
        book["execution_vs_vwap_bps"] + book["timing_drift_bps"], abs=1.0
    )
    # min_n drops thin buckets: with 3 toy orders (X:2, Y:1), min_n=2 keeps only X
    keep2 = attribution_by(clean, "brkr_code", footprint_only=False, min_n=2)
    assert list(keep2["brkr_code"]) == ["X"]
    assert (keep2["n_orders"] >= 2).all()


def test_etf_is_downward_sloping_and_decisions_present():
    spec = OrderSpec(label="T", x_shares=50000, adv_shares=100000, sigma_annual=0.30,
                     open_px=100.0, b1=200.0, b2=0.5, currency="USD")
    etf = build_etf(spec)
    assert etf["cost_bps"].iloc[0] > etf["cost_bps"].iloc[-1]
    assert etf["risk_bps"].iloc[0] < etf["risk_bps"].iloc[-1]
    pts = decision_points(etf, r_star=float(etf["risk_bps"].median()), lam=0.05)
    assert "goal2_min_cost_plus_lambda_risk" in pts


def test_permanent_impact_fraction():
    spec = OrderSpec(label="T", x_shares=10000, adv_shares=100000, sigma_annual=0.2,
                     open_px=50.0, b1=100.0, b2=0.5, currency="USD")
    cost_long, _ = schedule_cost_risk_bps(spec, horizon_days=200.0)
    full_bps = spec.b1 * (spec.x_shares / spec.adv_shares) ** spec.b2
    assert cost_long == pytest.approx((1 - ALPHA) * full_bps, rel=0.05)


def test_segments_and_cost_stats():
    from tca.segments import add_segments, cost_stats
    base = _toy_parent()
    base["MC Lst Trd"] = [15000.0, 5000.0, 1000.0]      # USD m -> 15bn, 5bn, 1bn
    base["Primary Exchange MIC (FIGI or Tkr+YKey)"] = ["XNYS", "XLON", "XHKG"]
    base["Bid Ask Sprd"] = [0.02, 0.20, 0.60]           # -> tight / medium+ / wider in bps
    clean = clean_parent_orders(base)
    # market-cap buckets from MC Lst Trd (USD m / 1000 = bn)
    caps = dict(zip(clean["order_id"], clean["mktcap_group"].astype(str)))
    assert caps["1"] == "Large Cap" and caps["2"] == "Mid Cap" and caps["3"] == "Small Cap"
    # region mapped from MIC
    assert clean.loc[clean.order_id == "1", "region"].iloc[0] == "Americas"
    assert clean.loc[clean.order_id == "2", "region"].iloc[0] == "Europe"
    # direction present, cost_stats returns the stat columns incl t-stat
    cs = cost_stats(clean, ["direction"], min_n=1)
    assert {"mean_cost_bps", "median_cost_bps", "std_cost_bps", "t_stat", "n_orders"}.issubset(cs.columns)


def test_order_type_segregation():
    base = _toy_parent()
    base["LmtPx"] = ["MKT", "101.5", "MKT"]     # market, limit, market
    clean = clean_parent_orders(base)
    types = dict(zip(clean["order_id"], clean["order_type"]))
    assert types["1"] == "Market" and types["3"] == "Market"
    assert types["2"] == "Limit"
    assert clean.loc[clean.order_id == "2", "lmt_px"].iloc[0] == pytest.approx(101.5)


def test_tca_model_reproduces_bloomberg():
    # synthetic: TCA20 built exactly as spread + sqrt(adv) + vol, model must recover R²≈1
    import numpy as np
    from tca.tca_model import fit_tca_model, tca_component_decomposition
    rng = np.random.default_rng(0)
    n = 200
    spread = rng.uniform(1, 40, n)
    adv = rng.uniform(0.1, 25, n)
    vol = rng.uniform(5, 60, n)
    base = pd.DataFrame({
        "Status": ["Filled"] * n, "Side": ["Buy"] * n,
        "Security": [f"S{i%20}" for i in range(n)], "ID": [str(i) for i in range(n)],
        "Brkr Code": ["X"] * n, "Exch Code": ["E"] * n, "Curncy": ["USD"] * n,
        "Qty": 1000, "FillQty": 1000, "AvgPx": 100.0, "ArrPx": 100.0,
        "IntervalVWAP": 100.0, "Open Px": 100.0, "Low": 99.0, "High": 101.0,
        "Bid Ask Sprd": spread / 1e4 * 100.0,            # -> spread_bps ≈ spread
        "Qty % Avg Vol 20D": adv, "Volatil 30D": vol,
        "TCA (20%)": 2 + 0.4 * spread + 3.0 * np.sqrt(adv) + 0.2 * vol,
        "Day Part Rate %": 20.0, "Create Time (As of)": "01/02/24 10:00", "Trade Date": "01/02/24",
        "% Filled": 100, "TIF": "DAY", "Value (Local)": 100000,
    })
    clean = clean_parent_orders(base)
    m = fit_tca_model(clean)
    assert m.r2 > 0.98
    dec = tca_component_decomposition(m)
    assert set(dec["component"]) and abs(dec["bps"].sum()) > 0


requires_data = pytest.mark.skipif(not FULL.exists(), reason="full CSV not present")


@requires_data
def test_real_pipeline_runs():
    from tca.io import load_parent_orders

    child = clean_child_orders(pd.read_csv(CHILD)) if CHILD.exists() else None
    clean = clean_parent_orders(load_parent_orders(FULL), child)
    assert len(clean) > 950
    assert clean["keep_for_analysis"].sum() > 950     # arrival is populated ~99.6%
    # sign check vs Bloomberg's provided field: cost_bps == -(their improvement field)
    fp = clean.loc[clean["keep_for_analysis"] & clean["has_footprint"]].dropna(
        subset=["cost_bps", "bbg_avgpx_vs_arr_bps"]
    )
    corr = fp["cost_bps"].corr(fp["bbg_avgpx_vs_arr_bps"])
    assert corr < -0.99
    reg = fit_regression_suite(clean)
    assert reg.ols_result.nobs > 800
    imp = fit_impact_curve(clean)
    assert 0.0 < imp.b2 < 2.0
