"""Equity TCA / Best-Execution dashboard.

Framework-segregated: each tab is one cost lens (Arrival IS, VWAP, Participation,
Bloomberg-TCA model, Market Impact, Difficulty-adjusted league, Efficient Frontier),
with its own methodology, standardized visuals, and a broker breakdown so the
frameworks can be compared side by side. Data is hardcoded for demo deployment.
Analysis logic lives in the tested `tca` package; this file is presentation only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from tca.pipeline import run_parent_pipeline
from tca.insights import attribution_summary
from tca.segments import (
    cost_stats, btca_pivot, arrival_vwap_trend_scan, BENCHMARK_COLS, PIVOT_CATEGORIES,
    MKTCAP_DEFS, ADV_DEFS, SPREAD_DEFS,
    MKTCAP_ORDER, ADV_ORDER, SPREAD_ORDER,
)

# ---------------------------------------------------------------------------
# config + shared styling (standardized visuals)
# ---------------------------------------------------------------------------
st.set_page_config(page_title="CIGP — Equity TCA", layout="wide")

# --- Poppins font + CIGP gold theme (injected CSS) -------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    /* Import the Material icon font ourselves with display=block: the icon text stays
       INVISIBLE until the font loads, so the ligature word (e.g. keyboard_arrow_right)
       can never flash as text or overlap. Guarantees the arrows render in any browser. */
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,400,0,0&display=block');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,400,0,0&display=block');

    /* Apply Poppins only to text elements. Do NOT target bare span/div or [class*=css]:
       those directly match Streamlit's Material icon <span>s and would clobber the icon
       font, printing the ligature name (e.g. keyboard_arrow_right). */
    html, body, .stApp, .stMarkdown, p, label,
    h1, h2, h3, h4, h5, h6, button, input, textarea, select,
    .stMarkdown p, .stMarkdown li, .stMarkdown span,
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
    [data-baseweb="tab"] div, [data-testid="stWidgetLabel"] {
        font-family: 'Poppins', sans-serif !important;
    }

    /* Canonical Material icon rendering — force ligature shaping so arrows/chevrons
       render as glyphs (expander toggles, sidebar collapse, select dropdowns). */
    [data-testid="stIconMaterial"], span.material-symbols-rounded,
    span.material-symbols-outlined, span.material-icons {
        font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
        font-weight: normal !important;
        font-style: normal !important;
        line-height: 1 !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        white-space: nowrap !important;
        word-wrap: normal !important;
        direction: ltr !important;
        font-feature-settings: 'liga' !important;
        -webkit-font-feature-settings: 'liga' !important;
        font-variant-ligatures: normal !important;
        overflow: hidden !important;      /* if the font ever fails, clip instead of overlap */
        max-width: 1.5em !important;
    }
    h1, h2, h3, h4 { color: #23262B; font-weight: 600; letter-spacing: -0.01em; }
    h1 { font-weight: 700; }
    [data-testid="stMetricValue"] { color: #8A6C28; font-weight: 600; }
    [data-testid="stMetricLabel"] { color: #6B7280; }
    .stTabs [data-baseweb="tab"] { font-weight: 500; }
    .stTabs [aria-selected="true"] { color: #B08D3C !important; }
    .stTabs [data-baseweb="tab-highlight"] { background-color: #B08D3C !important; }
    [data-testid="stSidebar"] { background-color: #FAF6EC; border-right: 1px solid #E7DCC3; }
    div.stButton > button, .stDownloadButton > button { border-color: #B08D3C; color: #8A6C28; }
    a { color: #8A6C28; }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_FILE = "data_trades_new2.csv"        # hardcoded for demo deployment
CHILD_FILE = "180days_child_order_data.csv"

# CIGP gold-centred palette (+ compatible bronze/champagne/teal/terracotta/burgundy)
GOLD = "#B08D3C"       # primary gold
GOLD_DK = "#8A6C28"    # deep bronze gold
GOLD_LT = "#D8C48E"    # champagne
SAND = "#C9A24B"       # warm secondary gold
TEAL = "#2E6F63"       # gold-compatible green (improvement)
CLAY = "#A6462F"       # terracotta (cost)
BURG = "#7B3B47"       # burgundy accent
SLATE = "#6B7280"      # neutral grey
INK = "#23262B"        # near-black text
SEQ = [GOLD, TEAL, GOLD_DK, SLATE, BURG, SAND, INK]
DIVERGING = [CLAY, "#EAD9B0", "#D7E3D5", TEAL]     # neg = cost → pos = improvement
GOLD_SEQ = ["#F4ECD8", GOLD_LT, GOLD, GOLD_DK]


def _style(fig, height: int = 380, ytitle: str = "", xtitle: str = ""):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(family="Poppins, sans-serif", size=13, color=INK),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        colorway=SEQ,
    )
    if ytitle:
        fig.update_yaxes(title_text=ytitle)
    if xtitle:
        fig.update_xaxes(title_text=xtitle)
    return fig


def hbar(df, cat, val, title, height=340, center_zero=True):
    """Standardized diverging horizontal bar (terracotta = cost, teal = improvement/cheaper)."""
    d = df.dropna(subset=[val]).copy()
    fig = px.bar(d, x=val, y=cat, orientation="h", color=val,
                 color_continuous_scale=DIVERGING if center_zero else GOLD_SEQ,
                 color_continuous_midpoint=0 if center_zero else None)
    fig.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
    return _style(fig, height=height, xtitle=title)


def metric_row(items):
    cols = st.columns(len(items))
    for c, (label, value, help_) in zip(cols, items):
        c.metric(label, value, help=help_)


def framework_box(title, formula, assumptions, validity):
    """Standardized per-tab methodology block."""
    st.markdown(f"### {title}")
    if formula:
        st.latex(formula)
    with st.expander("Methodology — assumptions & empirical validity", expanded=False):
        st.markdown("**Assumptions & controls**")
        st.markdown(assumptions)
        st.markdown("**Is it a valid benchmark / formula?**")
        st.markdown(validity)


@st.cache_data(show_spinner="Running TCA pipeline…")
def load(exclude_gtc: bool):
    child = CHILD_FILE if Path(CHILD_FILE).exists() else None
    return run_parent_pipeline(DATA_FILE, child, exclude_gtc=exclude_gtc)


def _fmt(x, d=2):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{d}f}"


# ---------------------------------------------------------------------------
# sidebar (rendered first so the GTC filter re-runs the pipeline before load)
# ---------------------------------------------------------------------------
st.sidebar.markdown("## Equity TCA")
st.sidebar.caption("Best-execution analytics · multi-framework")
st.sidebar.markdown("---")
exclude_gtc = st.sidebar.toggle(
    "Exclude GTC orders", value=True,
    help="Good-Till-Cancelled orders span multiple sessions (stale arrival); excluded by default. "
         "The whole analysis re-runs on the remainder.",
)

try:
    art = load(exclude_gtc)
except Exception as exc:
    st.error(f"Could not load {DATA_FILE}: {exc}")
    st.stop()

q, h = art.quality, art.headline
clean = art.clean
kept = clean.loc[clean["keep_for_analysis"]].copy()

ccy_opts = sorted(kept["currency"].dropna().unique())
sel_ccy = st.sidebar.multiselect("Currency", ccy_opts, default=[])
fp_only = st.sidebar.toggle("Footprint orders only", value=True,
                            help="Orders that actually traded away from arrival (|cost| > 0.5 bps).")
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Dataset:** `{DATA_FILE}`  \n"
    f"**GTC excluded:** {'yes' if exclude_gtc else 'no'}  \n"
    f"**Orders in scope:** {q.total_rows} · **kept** {q.kept_rows}  \n"
    f"**Footprint:** {q.footprint_orders}  \n"
    f"**Currencies:** {len(q.currency_counts)}  \n"
    f"**Benchmark:** arrival (ArrPx)"
)

view = kept if not sel_ccy else kept[kept["currency"].isin(sel_ccy)]
if fp_only:
    view = view[view["has_footprint"]]

st.title("Equity TCA — Best-Execution Dashboard")
st.caption(
    "One book of orders seen through multiple cost frameworks, plus EDA and a segmented "
    "Market-Order TCA view. Unless a tab states otherwise, negative bps = **cost**, "
    "positive = **price improvement**. Primary benchmark: arrival price."
)

TABS = st.tabs([
    "Overview",
    "EDA",
    "Market-Order TCA",
    "BTCA Pivot",
    "Methodology",
])
T_OVERVIEW, T_EDA, T_MOTCA, T_BTCA, T_METHOD = range(5)

# ===========================================================================
# Overview
# ===========================================================================
with TABS[T_OVERVIEW]:
    st.markdown("### Book at a glance")
    metric_row([
        ("Orders kept", f"{q.kept_rows}/{q.total_rows}", "After dropping missing arrival / neg spread"),
        ("Footprint orders", f"{h.n_footprint}", "Traded away from arrival"),
        ("VW cost vs arrival", f"{_fmt(h.value_weighted_slippage_bps)} bps", "Value-weighted Implementation Shortfall"),
        ("VW cost vs VWAP", f"{_fmt(h.vwap_value_weighted_bps)} bps", "Drift-free execution cross-check"),
    ])

    # ---- auto-generated Trends panel (Arrival vs VWAP gap scan) --------------------
    st.markdown("#### Trends — Arrival vs VWAP (auto-generated)")
    st.caption("Market orders, GTC-excluded. Positive = cost; value-weighted (FX-USD). "
               "Gap = Arrival − VWAP ≈ timing drift. Only cells that clear min-n, are significant "
               "(|t| ≥ 2) and aren't single-ticket artefacts are shown. Honours the currency filter.")
    trend_min = st.slider("Minimum orders per cell", 15, 60, 25, key="ov_trend_minn")
    tsrc = clean if not sel_ccy else clean[clean["currency"].isin(sel_ccy)]
    scan = arrival_vwap_trend_scan(tsrc, min_n=trend_min)
    robust = scan[scan["robust"]] if not scan.empty else scan
    if robust.empty:
        st.warning("No robust trend at this minimum sample / filter — loosen min-n or clear the filter.")
    else:
        for _, r in robust.head(8).iterrows():
            st.markdown(
                f"- **{r['segment']}** — Arrival **{r['A_vw']:+.1f}** / VWAP **{r['V_vw']:+.1f}** bps "
                f"(gap {r['gap_vw']:+.1f}), n={int(r['n'])}, t={r['A_t']:+.1f}  →  _{r['read']}_"
            )
        top = robust.head(6).iloc[::-1]
        melt = top.melt(id_vars=["segment"], value_vars=["A_vw", "V_vw"],
                        var_name="benchmark", value_name="bps")
        melt["benchmark"] = melt["benchmark"].map({"A_vw": "vs Arrival", "V_vw": "vs VWAP"})
        fig = px.bar(melt, x="bps", y="segment", color="benchmark", orientation="h",
                     barmode="group", color_discrete_sequence=[GOLD, TEAL])
        st.plotly_chart(_style(fig, height=320, xtitle="cost (bps, +=cost)"), use_container_width=True)
        with st.expander("All screened cells (incl. non-robust)"):
            st.dataframe(scan.round(1)[["segment", "n", "A_vw", "V_vw", "gap_vw", "A_t", "V_t", "read", "robust"]],
                         use_container_width=True, hide_index=True)

    st.markdown("#### Framework comparison — value-weighted cost of the book (bps)")
    comp = pd.DataFrame({
        "framework": ["Arrival IS", "Interval VWAP", "Bloomberg TCA(20%) est.", "Execution vs VWAP", "Timing drift"],
        "bps": [
            h.value_weighted_slippage_bps,
            h.vwap_value_weighted_bps,
            -float(np.average(kept["tca20"].dropna())) if kept["tca20"].notna().any() else np.nan,
            attribution_summary(clean, footprint_only=True)["execution_vs_vwap_bps"],
            attribution_summary(clean, footprint_only=True)["timing_drift_bps"],
        ],
    })
    fig = px.bar(comp, x="framework", y="bps", color="bps",
                 color_continuous_scale=[CLAY, "#fee2e2", "#dcfce7", TEAL], color_continuous_midpoint=0,
                 text=comp["bps"].round(1))
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(_style(fig, ytitle="bps (neg = cost)"), use_container_width=True)
    st.info(
        "Arrival IS and Interval VWAP are the realized benchmarks; **Bloomberg TCA(20%)** is an "
        "ex-ante estimate (negated for comparability). **Execution vs VWAP** and **timing drift** "
        "decompose arrival cost into controllable skill vs market drift."
    )
    st.markdown("#### Currency mix")
    cc = pd.Series(q.currency_counts).sort_values(ascending=False).reset_index()
    cc.columns = ["currency", "orders"]
    st.plotly_chart(_style(px.bar(cc, x="currency", y="orders", color_discrete_sequence=[GOLD]), height=300),
                    use_container_width=True)

# ===========================================================================
# EDA — exploratory data analysis
# ===========================================================================
with TABS[T_EDA]:
    st.markdown("### Exploratory data analysis")
    st.caption("Distributions, category mix, missingness and driver correlations before any modelling. "
               "Honours the sidebar currency filter.")
    eda = kept if not sel_ccy else kept[kept["currency"].isin(sel_ccy)]

    st.markdown("#### Numeric distributions")
    num_specs = [
        ("cost_bps_w", "Arrival cost (bps, +=cost)"),
        ("spread_bps", "Bid-ask spread (bps)"),
        ("qty_pct_adv_20d", "Order size (% ADV)"),
        ("day_part_rate", "Participation / POV (%)"),
        ("volatil_30d", "30-day volatility (%)"),
        ("market_cap_usd_bn", "Market cap (USD bn)"),
    ]
    cols = st.columns(3)
    for i, (col, lab) in enumerate(num_specs):
        d = eda[col].replace([np.inf, -np.inf], np.nan).dropna()
        if col in ("qty_pct_adv_20d", "market_cap_usd_bn", "spread_bps"):
            d = d[d > 0]
            fig = px.histogram(np.log10(d), nbins=40, color_discrete_sequence=[GOLD])
            fig.update_xaxes(title=f"log10 {lab}")
        else:
            fig = px.histogram(d, nbins=40, color_discrete_sequence=[GOLD])
            fig.update_xaxes(title=lab)
        cols[i % 3].plotly_chart(_style(fig, height=260), use_container_width=True)

    st.markdown("#### Category mix")
    cat_specs = [("region", "Region"), ("mktcap_group", "Market-cap group"),
                 ("adv_group", "ADV% group"), ("spread_bucket", "Spread bucket"),
                 ("direction", "Direction"), ("brkr_code", "Broker")]
    ccols = st.columns(3)
    for i, (col, lab) in enumerate(cat_specs):
        vc = eda[col].value_counts(dropna=False).reset_index()
        vc.columns = [lab, "orders"]
        fig = px.bar(vc, x=lab, y="orders", color_discrete_sequence=[GOLD_DK])
        ccols[i % 3].plotly_chart(_style(fig, height=260), use_container_width=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("#### Field coverage")
        cov_fields = ["arr_px", "interval_vwap", "open_px", "tca20", "day_part_rate",
                      "volatil_30d", "spread_bps", "market_cap_usd_bn", "industry", "rsi_14d"]
        cov = pd.DataFrame({
            "field": cov_fields,
            "coverage_%": [round(100 * eda[f].notna().mean(), 1) for f in cov_fields],
        }).sort_values("coverage_%")
        st.dataframe(cov, use_container_width=True, hide_index=True)
    with c2:
        st.markdown("#### Driver correlations")
        corr_cols = ["cost_bps_w", "spread_bps", "qty_pct_adv_20d", "day_part_rate",
                     "volatil_30d", "tca20"]
        cm = eda[corr_cols].replace([np.inf, -np.inf], np.nan).corr()
        fig = px.imshow(cm, text_auto=".2f", color_continuous_scale=[CLAY, "#f4f1e9", TEAL],
                        zmin=-1, zmax=1, aspect="auto")
        st.plotly_chart(_style(fig, height=300), use_container_width=True)

# ===========================================================================
# Market-Order TCA framework (segmented)
# ===========================================================================
with TABS[T_MOTCA]:
    st.markdown("### Market-Order TCA Framework")
    st.markdown(
        "**Core question:** where do we see higher or lower arrival-to-execution slippage for "
        "**market orders** in **2026 H1**? Segmentation matters — the same broker looks good on "
        "large-cap US names and poor on small-cap Asian ones."
    )
    st.latex(r"\text{Cost bps} = \text{SideSign}\times\frac{\text{ExecAvgPx}-\text{ArrPx}}{\text{ArrPx}}\times10^4,"
             r"\quad \text{Buy}=+1,\ \text{Sell}=-1\ \Rightarrow\ \textbf{positive = cost}")

    with st.expander("Data fields, grouping methodology & bucket definitions", expanded=False):
        st.markdown(
            "**Data fields** — *Order:* trade date, arrival time, arrival price, direction, broker · "
            "*Market:* arrival price, average price, market cap, region, industry · "
            "*Cost:* Mean / Median / Std-Dev cost (bps), cost t-stat · "
            "*Further:* AVAT, historical volatility, interval VWAP.\n\n"
            "**Grouping methodology**\n"
            "- **Broker quality** — algo/execution quality by context: Region → Industry → Broker → Direction\n"
            "- **Size effect** — cost of small-cap trading: Market-Cap Group → Direction\n"
            "- **Market impact** — cost of high participation: ADV% Group → Direction\n"
            "- **Liquidity cost** — cost of crossing wide spreads: Spread Bucket → Direction\n\n"
            "*Buckets are initial proposals; feature sets and ML-based feature selection can be explored "
            "once per-cell sample sizes are sufficient.*"
        )
        b1, b2, b3 = st.columns(3)
        b1.markdown("**Market cap**\n\n" + "\n".join(f"- {k}: {v}" for k, v in MKTCAP_DEFS.items()))
        b2.markdown("**ADV%**\n\n" + "\n".join(f"- {k}: {v}" for k, v in ADV_DEFS.items()))
        b3.markdown("**Spread**\n\n" + "\n".join(f"- {k}: {v}" for k, v in SPREAD_DEFS.items()))

    # ---- order segregation: Market vs Limit (LmtPx = "MKT" flags market orders) --------
    st.markdown("#### Order segregation")
    seg = (kept.groupby("order_type")
           .apply(lambda g: pd.Series({
               "n_orders": len(g),
               "mean_cost_bps": np.average(g["cost_bps"], weights=g["notional_usd"])
               if g["notional_usd"].sum() > 0 else np.nan,
               "median_cost_bps": g["cost_bps"].median(),
               "footprint": int(g["has_footprint"].sum())}), include_groups=False)
           .reset_index())
    sc1, sc2 = st.columns([1, 1])
    with sc1:
        fig = px.bar(seg, x="order_type", y="n_orders", color="order_type",
                     color_discrete_sequence=[GOLD, SLATE, BURG], text="n_orders")
        st.plotly_chart(_style(fig, height=280, ytitle="orders"), use_container_width=True)
    with sc2:
        st.dataframe(seg.round(2), use_container_width=True, hide_index=True)
    st.caption("Market orders (`LmtPx = MKT`) are the framework's population. Limit orders are segregated "
               "out — their opportunity cost is a separate study (see Future Analysis).")

    otype = st.radio("Population", ["Market orders", "Limit orders", "All orders"], horizontal=True)
    type_map = {"Market orders": "Market", "Limit orders": "Limit", "All orders": None}
    fp_seg = st.toggle("Footprint orders only", value=False, key="motca_fp",
                       help="~2/3 fill at arrival (median cost 0); footprint isolates orders that moved.")
    src = clean.copy()
    if type_map[otype] is not None:
        src = src[src["order_type"] == type_map[otype]]
    if fp_seg:
        src = src[src["has_footprint"] | ~src["keep_for_analysis"]]

    # ---- book segregation treemap across factors (market orders sized by n, coloured by cost) ---
    tre = src.loc[src["keep_for_analysis"]].dropna(subset=["region", "mktcap_group", "direction"]).copy()
    if not tre.empty:
        tre["mktcap_group"] = tre["mktcap_group"].astype(str)
        agg = (tre.groupby(["region", "mktcap_group", "direction"], observed=True)
               .agg(n=("cost_bps", "size"), cost=("cost_bps", "mean")).reset_index())
        agg = agg[agg["n"] >= 3]
        if not agg.empty:
            fig = px.treemap(agg, path=[px.Constant("Book"), "region", "mktcap_group", "direction"],
                             values="n", color="cost", color_continuous_scale=[TEAL, "#f4f1e9", CLAY],
                             color_continuous_midpoint=0)
            st.plotly_chart(_style(fig, height=420).update_layout(margin=dict(t=30, l=10, r=10, b=10)),
                            use_container_width=True)
            st.caption("Book segregated Region → Market-cap → Direction. Box size = order count, "
                       "colour = mean cost (red = costlier, teal = cheaper).")

    # ---- grouping methodology ---------------------------------------------------------
    st.markdown("#### Segmented cost analysis")
    method = st.radio("Grouping methodology", ["Broker quality", "Size effect", "Market impact", "Liquidity cost"],
                      horizontal=True)
    cfg = {
        "Broker quality": (["region", "brkr_code", "direction"], "brkr_code", "direction", None),
        "Size effect": (["mktcap_group", "direction"], "mktcap_group", "direction", MKTCAP_ORDER),
        "Market impact": (["adv_group", "direction"], "adv_group", "direction", ADV_ORDER),
        "Liquidity cost": (["spread_bucket", "direction"], "spread_bucket", "direction", SPREAD_ORDER),
    }
    by, primary, color, order = cfg[method]
    min_n = st.slider("Minimum orders per cell", 3, 40, 5, key="motca_minn")
    stats = cost_stats(src, by, min_n=min_n)
    if stats.empty:
        st.warning(f"No cell has ≥ {min_n} orders for this grouping/population.")
    else:
        cat_orders = {primary: order} if order else {}
        fig = px.bar(stats, x=primary, y="mean_cost_bps", color=color, barmode="group",
                     color_discrete_sequence=[GOLD, TEAL, SLATE, BURG],
                     category_orders=cat_orders, hover_data=["n_orders", "t_stat", "median_cost_bps"])
        st.plotly_chart(_style(fig, ytitle="mean cost (bps, +=cost)"), use_container_width=True)
        st.caption("|t-stat| ≳ 2 ⇒ the group's mean cost is statistically different from zero.")
        show = stats.copy()
        for c in ["mean_cost_bps", "median_cost_bps", "std_cost_bps", "t_stat", "vw_cost_bps"]:
            show[c] = show[c].round(2)
        st.dataframe(show, use_container_width=True, hide_index=True)
        sig = stats.loc[stats["t_stat"].abs() >= 2].sort_values("mean_cost_bps", ascending=False)
        if not sig.empty:
            top = sig.iloc[0]
            grp = " / ".join(str(top[c]) for c in by)
            st.success(f"**Most significant cost cell:** {grp} — mean {top['mean_cost_bps']:.1f} bps "
                       f"(t = {top['t_stat']:.1f}, n = {int(top['n_orders'])}).")

    with st.expander("Future analysis (roadmap)"):
        st.markdown(
            "1. **Opportunity cost** — study price movement *after* order completion; compare average price "
            "to day VWAP and close price.\n"
            "2. **Limit orders** — opportunity cost of a potentially better average price when the market never "
            "reaches the initial limit (segregated above; 131 limit orders here).\n"
            "3. **Operational cost** — cost of live orders between the actual placed time and the arrival time.\n"
            "4. **More data** — actual market volumes and price within the trading lifespan, to split slippage "
            "into **momentum cost, speed cost, liquidity premium**."
        )

# ===========================================================================
# BTCA Pivot — Bloomberg-style multi-benchmark cost pivot
# ===========================================================================
with TABS[T_BTCA]:
    st.markdown("### BTCA Pivot — cost by category vs benchmarks")
    st.markdown(
        "Pick any **categories** as the grouping columns "
        "(region, broker, market cap, …) and the table shows **cost (bps) vs each benchmark** "
        "as value columns."
        + ("  \n_GTC orders are excluded (sidebar)._" if exclude_gtc else "")
    )

    cat_labels = list(PIVOT_CATEGORIES.keys())
    c1, c2, c3 = st.columns([2, 2, 1])
    sel_cat_labels = c1.multiselect("Categories (grouping columns)", cat_labels,
                                    default=["Region", "Broker"])
    sel_bench = c2.multiselect("Benchmarks (value columns)", list(BENCHMARK_COLS.keys()),
                               default=["Arrival", "Interval VWAP", "Bloomberg TCA(20%) est."])
    stat = c3.selectbox("Statistic", ["Value-weighted", "Mean", "Median"], index=0)
    min_n = st.slider("Minimum orders per row", 1, 40, 5, key="btca_minn")

    categories = [PIVOT_CATEGORIES[l] for l in sel_cat_labels]
    if not categories or not sel_bench:
        st.info("Pick at least one category and one benchmark.")
    else:
        piv = btca_pivot(clean, categories, sel_bench, stat=stat, min_n=min_n)
        if piv.empty:
            st.warning(f"No group has ≥ {min_n} orders.")
        else:
            bench_cols = [f"{b} cost (bps)" for b in sel_bench]
            pretty = piv.rename(columns={c: l for l, c in PIVOT_CATEGORIES.items()})

            def _color_cost(v):  # red = cost (+), teal = improvement (−); no matplotlib needed
                if not isinstance(v, (int, float)) or pd.isna(v):
                    return ""
                a = 0.12 + 0.55 * min(abs(v) / 50.0, 1.0)
                rgb = "166,70,47" if v > 0 else "46,111,99"
                return f"background-color: rgba({rgb},{a:.2f})"

            _style_df = pretty.style.format({c: "{:,.1f}" for c in bench_cols})
            # pandas ≥2.1 renamed Styler.applymap → Styler.map; support both
            _elem = getattr(_style_df, "map", None) or _style_df.applymap
            styled = _elem(_color_cost, subset=bench_cols)
            st.dataframe(styled, use_container_width=True, hide_index=True)
            st.caption(f"{stat} cost, positive = cost. Last row = ALL (book total). "
                       "Colour scale ±50 bps (red = costlier). Cost is FX-neutral (bps ratio); "
                       "value-weighting uses **FX-converted USD notional** so mixed-currency groups "
                       "aren't dominated by JPY/GBp.")

            # heatmap when exactly one category and ≥1 benchmark
            body = piv[piv[categories[0]] != "ALL"] if categories else piv
            if len(categories) == 1 and not body.empty:
                melt = body.melt(id_vars=categories, value_vars=bench_cols,
                                 var_name="benchmark", value_name="cost")
                fig = px.bar(melt, x=categories[0], y="cost", color="benchmark", barmode="group",
                             color_discrete_sequence=[GOLD, TEAL, SLATE])
                st.plotly_chart(_style(fig, ytitle="cost (bps, +=cost)"), use_container_width=True)

            csv = piv.to_csv(index=False).encode("utf-8")
            st.download_button("Download pivot (CSV)", csv, "btca_pivot.csv", "text/csv")

    st.caption("Open benchmark is a **diagnostic** — Open/Low/High are an export snapshot, so its "
               "column is noisy; rely on Arrival and Interval VWAP.")

# ===========================================================================
# Methodology
# ===========================================================================
with TABS[T_METHOD]:
    st.markdown("### How everything is calculated")
    st.markdown(
        "Full reference: **METHODOLOGY.md** in the project root. Sign convention throughout: "
        "**negative bps = cost, positive = price improvement.**"
    )
    st.markdown(f"""
| Framework (tab) | Formula | Uses | Validity |
|---|---|---|---|
| Market-Order TCA | `Cost = SideSign·(AvgPx−ArrPx)/ArrPx·1e4` grouped | region/cap/ADV%/spread | segmented mean/median/std/t-stat |
| Arrival IS | `sf·(AvgPx−ArrPx)/ArrPx·1e4` | arrival price | Perold IS — CFA best-ex standard |
| VWAP + attribution | `slip = exec_vs_vwap + timing` | interval VWAP | standard shortfall decomposition |
| Participation | cost vs `Day Part Rate %` | POV | driver in Almgren/Kissell impact |
| Bloomberg TCA | `TCA20 ≈ c₀+c₁·spread+c₂·√%ADV+c₃·σ` (R²={art.tca_model.r2:.2f}) | spread, size, vol | reproduces vendor √-law model |
| Market impact | `cost ≈ b₁·(X/ADV)^b₂` (b₂={art.impact.b2:.2f}) | size/ADV | Almgren 2005 square-root law |
| Adjusted league | `cost_z ~ βX + δ_broker + φ_venue` | controls + FE | panel FE, clustered SE |
| Efficient frontier | `min Cost s.t. Risk≤R*` | impact + vol | Almgren–Chriss / Kissell 2004 |
""")
    st.markdown("#### Key empirical insights from this book")
    st.markdown(
        f"- **Value-weighted execution is essentially flat vs arrival ({_fmt(h.value_weighted_slippage_bps)} bps)** — "
        "the book trades close to its decision price; cost lives in a minority of footprint orders.\n"
        f"- **Bloomberg's TCA(20%) is a spread + √-size + vol model (R² = {art.tca_model.r2:.2f})**, "
        "independent of participation → confirms it is a fixed-20%-POV pre-trade estimate.\n"
        f"- **Impact exponent b₂ ≈ {art.impact.b2:.2f}** — consistent with the empirical square-root law.\n"
        "- Broker apparent-outperformance is often **timing drift, not skill** — the VWAP attribution separates them.\n"
        "- **Open Px / Low / High remain an export snapshot** (AvgPx in-range only "
        f"{100*(q.total_rows-q.flagged_avgpx_outside_hilo)/q.total_rows:.0f}%) → the open benchmark is retired."
    )
