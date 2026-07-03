"""TCA / best-execution analysis package.

Three layers kept separate so the same logic serves a notebook and the app:
  * data layer     -> io.py, cleaning.py
  * analysis layer -> modeling.py (ex-post), kissell.py (ex-ante), insights.py
  * orchestration  -> pipeline.py  (view layer lives in app.py)
"""

from .io import load_parent_orders, load_child_orders
from .cleaning import (
    clean_parent_orders,
    clean_child_orders,
    build_data_quality_report,
    security_volatility_map,
)
from .modeling import fit_regression_suite, fit_impact_curve
from .insights import (
    value_weighted_summary,
    attribution_summary,
    attribution_by,
    cost_by_participation,
    build_league_tables,
    compute_trend_summary,
    child_routes_for_broker,
)
from .tca_model import (
    fit_tca_model,
    tca_component_decomposition,
    cost_surprise_by,
)
from .kissell import (
    calibrate_faithful_impact,
    build_etf,
    schedule_cost_risk_bps,
    decision_points,
    order_spec_from_row,
    pick_representative_order,
)
from .pipeline import run_parent_pipeline, PipelineArtifacts

__all__ = [
    "load_parent_orders",
    "load_child_orders",
    "clean_parent_orders",
    "clean_child_orders",
    "build_data_quality_report",
    "security_volatility_map",
    "fit_regression_suite",
    "fit_impact_curve",
    "value_weighted_summary",
    "attribution_summary",
    "attribution_by",
    "cost_by_participation",
    "fit_tca_model",
    "tca_component_decomposition",
    "cost_surprise_by",
    "build_league_tables",
    "compute_trend_summary",
    "child_routes_for_broker",
    "calibrate_faithful_impact",
    "build_etf",
    "schedule_cost_risk_bps",
    "decision_points",
    "order_spec_from_row",
    "pick_representative_order",
    "run_parent_pipeline",
    "PipelineArtifacts",
]
