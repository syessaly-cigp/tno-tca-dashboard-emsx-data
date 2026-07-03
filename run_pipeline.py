from __future__ import annotations

from pathlib import Path

from tca.pipeline import run_parent_pipeline


def main() -> None:
    # full export with participation + Bloomberg TCA(20%) + arrival/VWAP benchmarks
    parent = Path("data_trades_new2.csv")
    for alt in ("td_data_full.csv", "180days_parent_order_data.csv"):
        if not parent.exists():
            parent = Path(alt)
    child = Path("180days_child_order_data.csv")
    art = run_parent_pipeline(parent, child if child.exists() else None)

    q, h = art.quality, art.headline
    print("=== Coverage / data quality ===")
    print(f"rows_in_scope      = {q.total_rows}")
    print(f"rows_kept          = {q.kept_rows} ({q.kept_pct:.1f}%)")
    print(f"missing_arrival    = {q.dropped_missing_arrival}")
    print(f"neg_spread_flagged = {q.flagged_negative_spread}")
    print(f"open_snapshot_flag = {q.flagged_avgpx_outside_hilo} (AvgPx outside [Low,High]; informational)")
    print(f"footprint_orders   = {q.footprint_orders}  (|arrival cost|>0.5 bps)")
    print(f"multi_day_gtc      = {q.flagged_multi_day_gtc}")

    print(f"\n=== Headline slippage vs {h.benchmark} (negative = cost) ===")
    print(f"equal-weighted mean       = {h.mean_slippage_bps:8.2f} bps")
    print(f"median                    = {h.median_slippage_bps:8.2f} bps")
    print(f"value-weighted            = {h.value_weighted_slippage_bps:8.2f} bps")
    print(f"footprint mean COST (+ve) = {h.footprint_mean_cost_bps:8.2f} bps   "
          f"(n={h.n_footprint}, sanity_ok={h.sanity_ok})")
    print(f"VWAP cross-check (val-wt) = {h.vwap_value_weighted_bps:8.2f} bps")

    print("\n=== Impact curve  cost_bps ~ b1*(X/ADV)^b2  (footprint orders, vs arrival) ===")
    print(f"b1={art.impact.b1:.3f} (se {art.impact.b1_se:.3f})  "
          f"b2={art.impact.b2:.3f} (se {art.impact.b2_se:.3f})  n={art.impact.n}")

    print("\n=== Difficulty-adjusted broker league (costliest first) ===")
    print(art.broker_table.head(6).to_string(index=False))

    print(f"\n=== Illustrative ETF for {art.etf_spec_label} ===")
    if not art.etf.empty:
        print(art.etf.iloc[[0, len(art.etf) // 2, -1]].to_string(index=False))


if __name__ == "__main__":
    main()
