from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .cleaning import (
    QualityReport,
    build_data_quality_report,
    clean_child_orders,
    clean_parent_orders,
)
from .insights import (
    HeadlineSummary,
    build_league_tables,
    child_routes_for_broker,
    compute_trend_summary,
    value_weighted_summary,
)
from .io import load_child_orders, load_parent_orders
from .kissell import (
    FaithfulImpact,
    build_etf,
    calibrate_faithful_impact,
    order_spec_from_row,
    pick_representative_order,
)
from .modeling import ImpactCurve, RegressionSuite, fit_impact_curve, fit_regression_suite
from .tca_model import TCAModel, fit_tca_model


@dataclass(frozen=True)
class PipelineArtifacts:
    raw_parent: pd.DataFrame
    raw_child: pd.DataFrame
    clean: pd.DataFrame
    child: pd.DataFrame
    quality: QualityReport
    quality_reasons: pd.DataFrame
    headline: HeadlineSummary
    regression: RegressionSuite
    impact: ImpactCurve
    faithful_impact: FaithfulImpact
    tca_model: TCAModel | None
    broker_table: pd.DataFrame
    venue_table: pd.DataFrame
    trend_table: pd.DataFrame
    etf: pd.DataFrame                 # illustrative single-security frontier
    etf_spec_label: str


def run_parent_pipeline(
    parent_csv: str | Path,
    child_csv: str | Path | None = None,
    exclude_gtc: bool = False,
) -> PipelineArtifacts:
    raw_parent = load_parent_orders(parent_csv)

    if child_csv is None:
        guess = Path(parent_csv).with_name("180days_child_order_data.csv")
        child_csv = guess if guess.exists() else None
    raw_child = load_child_orders(child_csv) if child_csv else pd.DataFrame()
    child = clean_child_orders(raw_child) if len(raw_child) else pd.DataFrame()

    clean = clean_parent_orders(raw_parent, child if len(child) else None)
    if exclude_gtc:
        # drop Good-Till-Cancelled orders and re-run everything on the remainder
        clean = clean.loc[~clean["is_gtc"]].reset_index(drop=True)
    quality, reasons = build_data_quality_report(clean)
    headline = value_weighted_summary(clean)

    regression = fit_regression_suite(clean)
    impact = fit_impact_curve(clean)
    try:
        faithful = calibrate_faithful_impact(clean)
    except Exception as exc:  # keep pipeline alive if the $/share fit fails
        faithful = FaithfulImpact(float("nan"), float("nan"), float("nan"), float("nan"), 0, str(exc))
    try:
        tca_model = fit_tca_model(clean)
    except Exception:
        tca_model = None

    broker_table, venue_table = build_league_tables(regression.frame, regression.ols_result)
    trend_table = compute_trend_summary(clean)

    # illustrative ETF for a representative order (KGM §4)
    rep = pick_representative_order(clean)
    if rep is not None:
        spec = order_spec_from_row(rep, impact.b1, impact.b2)
        etf = build_etf(spec)
        etf_label = spec.label
    else:
        etf, etf_label = pd.DataFrame(), "(no volatility-covered order)"

    return PipelineArtifacts(
        raw_parent=raw_parent,
        raw_child=raw_child,
        clean=clean,
        child=child,
        quality=quality,
        quality_reasons=reasons,
        headline=headline,
        regression=regression,
        impact=impact,
        faithful_impact=faithful,
        tca_model=tca_model,
        broker_table=broker_table,
        venue_table=venue_table,
        trend_table=trend_table,
        etf=etf,
        etf_spec_label=etf_label,
    )
