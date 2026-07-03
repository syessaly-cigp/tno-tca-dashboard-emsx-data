# TCA / Best-Execution Analysis — Project Context & Build Plan

## What this project is

Empirical transaction-cost analysis on EMSX order data exported from Bloomberg,
ending in a Streamlit app that visualizes slippage and attributes it to controllable
levers (broker, venue) after adjusting for order difficulty.

This is an **empirical cross-sectional study**, NOT a Monte Carlo. The records are
realized outcomes, not simulated paths. (MC is a separate later step — see bottom.)

## Data on hand

- `180days_parent_order_data.xlsx` — 951 rows (893 Filled, 58 Part-filled). **Primary unit.**
- `180days_child_order_data.xlsx` — 642 rows (all Filled). Diagnosis/attribution only.
- Treat as two linked tables (`parents`, `routes`); join routes to parent order id.
- NEVER pool parents + routes into one regression (double-counts shares; two arrival prices).

## Slippage definition (the dependent variable)

- **Benchmark = open price** (fallback: arrival price and OMS benchmark both export blocked).
  Slippage = `(AvgPx − OpenPx)`, side-adjusted (negative = cost for both Buy and Sell),
  expressed in **bps of OpenPx**.
- Compute slippage yourself from raw `OpenPx`, `AvgPx`, `Qty` with an explicit sign
  convention; keep `(AvgPx − OpenPx) × FillQty`, FX-converted, only for cash-cost reporting.
- **Interpretation caveat:** AvgPx-vs-Open measures execution cost PLUS intraday drift
  between the open and when trading actually started — it is NOT pure implementation
  shortfall. The later in the day an order was entered, the more it is contaminated by
  drift. Mitigations: (a) **control for time-of-day / entry timing** in the regression to
  partially separate execution from drift; (b) lean on cross-broker *residual* comparison,
  since drift is roughly common across brokers in the same names and largely washes out.
- **Verify Open** = the primary-exchange official opening-auction print for the order's
  own trade date (not a prior session or a composite). A mismatched open date silently
  inflates every number.
- Upsides vs arrival here: robust to the stale-arrival problem on multi-day GTC orders
  (open re-stamps daily), and a clean auction print, so less prone to the garbage values
  seen in Bid-Ask Spread.

## Metric convention

- **Analyze in basis points (bps) only.** Book spans ~9 currencies (USD-dominated) and
  AvgPx from ~0.02 to ~5570, so Net/Cps/% are not comparable across orders. % = bps/100.

## ⚠ Phase 0 blocker — confirm OpenPx exports

The original benchmark fields (`ArrPx`, `AvgPx Vs OMS Benchmark Px`) and every
`Data Export Restricted` column came through **empty**, so the dependent variable was
missing. Plan now uses **OpenPx** as the benchmark — confirm it actually exports populated
before building downstream (it is a standard listed-products field, so it should).

- If OpenPx also exports blocked: re-pull via `blpapi` (often returns fields the Excel
  grid blocks), or capture open prices separately and compute bps. An earlier grid export
  did carry `AvgPx Vs ArrPx (Bps)`, so the restriction is export-path/permission-driven,
  not a hard wall — worth retrying arrival via that route too.

## Usable columns (the X matrix that already works)

- **Difficulty (controls, well-populated):** `FillQty % Avg Vol 20D` (%ADV),
  `Avg Vol 20D`, `Volatil 30D`, `Bid Ask Sprd`, `Side`, plus `Qty`, `AvgPx`,
  `FillQty`, `% Filled`.
- **Levers worth using:** `Brkr Code` (9 values, real spread) and `Exch Code` (24 venues).
- **Levers too thin in this data — do NOT build findings on them:** `Strategy Name`
  (~73% missing), `GICS Sector` (~57% missing), `Handling Inst` (~97% "ANY", no variation).
  Note them and improve capture going forward.

## Cleaning rules (Phase 1 — apply BEFORE any aggregation)

- Keep filled orders; for the 58 part-fills, compute slippage on the filled portion and
  treat opportunity cost on the unfilled portion separately.
- `Bid Ask Sprd` has impossible negatives (down to −1655): drop/flag.
- `Avg Vol 20D` has zeros: guard against division blow-ups in %ADV.
- `FillQty % Avg Vol 20D` runs to 1358%: winsorize continuous vars at 1st/99th pct
  (winsorize, don't delete).
- `Create Time (As of)` is MM/DD/YY strings: parse to datetime; also derive **time-of-day
  / entry timing** — needed as a regression control because the open benchmark folds in
  intraday drift before trading started.
- NEVER use "P/L (T) vs ArrPx" as slippage — it is mark-to-market drift, not execution.
- Open re-stamps daily, so multi-day GTC orders are less of a problem than under arrival;
  still flag them, since AvgPx-vs-Open over a multi-day order spans several opens.
- Emit a coverage / data-quality report of what was dropped and why.

## Statistical framework (Phases 2–3)

1. **Distribution:** mean / median / dispersion / tails of bps slippage. Mean is
   *expected* to be a cost — judge vs order difficulty, not vs zero.
2. **Standardize** slippage (z-score vs expected, or as fraction of spread/vol) so
   orders are comparable.
3. **Regress** standardized slippage on the difficulty controls **plus time-of-day /
   entry timing** (to soak up intraday drift that the open benchmark includes), with
   **broker and venue as fixed effects**, and **cluster standard errors by security**
   (orders in the same name aren't independent). Each broker/venue effect after controls =
   its difficulty-adjusted execution quality.
4. **Residual league table:** brokers/venues with persistently negative residuals are
   genuinely costing money. Raw broker averages are a confound trap (the biggest broker
   gets the hardest orders).
5. **Quantile regression** for the tail — 95th-pct cost has different drivers than the mean.
6. **Impact curve:** slippage vs %ADV fitted to the square-root law — an output AND the
   calibration for any later MC. With ~900 orders this is a shape, not a precise constant.
7. **Mind cell sizes:** broker × venue × size-bucket thins fast; keep cuts coarse, show n.

## Kissell–Glantz–Malamut (2004) integration — ex-ante best-execution layer

> **Full computational spec (formulas mapped to fields, calibration, ETF loop): see
> `KGM_TC_MODEL.md`.** This section is the summary.

Theory backbone for the open-price benchmark: the paper derives the open-benchmark cost
profile explicitly (§3.1.1, `E(Pb) = P0`). Adding it turns the project from "measure +
rank brokers" into "measure + judge whether each execution was even on the efficient
frontier."

**Two costs, do NOT conflate them:**
- **Ex-post (Eq. 1, measurement):** what an execution actually cost vs open =
  `side*(AvgPx - Open)`. Runs on current data. This is the dependent variable.
- **Ex-ante (Eq. 7, model):** predicts expected cost + risk of a *hypothetical schedule*.
  Powers the Efficient Trading Frontier, which is built by VARYING the schedule — it is
  NOT extracted from realized orders. The 951-order cross-section's only ex-ante role is to
  **calibrate the impact model `I = a1*X^a2` (α = 0.95 fixed)** that feeds the frontier.

**ETF — single security vs trade list:** build ONE ETF per order or per trade-list, never
one frontier for the whole dataset (the frontier depends on that order's size, the stock's
vol/volume, and — for a basket — the covariance). Start single-security (risk term collapses
to one variance, no covariance matrix). A multi-stock list needs the full `($/share)^2`
covariance matrix `Z` across names traded together — heavier; defer or approximate.

**Paradigm shift to respect:** the paper is *ex-ante* (estimate cost/risk to CHOOSE a
strategy); our 951 orders are *ex-post* (what happened). Integration = calibrate the
ex-ante model on the ex-post data, then use it to judge those same executions. Keep
calibration vs evaluation honest — fit impact params on one slice, evaluate on another —
or the loop is circular.

**What maps, and what does not:**
- **IS decomposition (Eq. 1):** we measure ONLY the trading-related term (fills vs open).
  Delay cost needs decision price `Pd` (= OMS/arrival, not available); opportunity cost
  needs per-period unfilled qty. Label the app metric **"trading-related cost vs open,"
  NOT full implementation shortfall.**
- **Market-impact model (Eqs. 3–6) — the prize.** Top-down cost allocation: total dollar
  impact `I = a1·X^a2` (Eq. 5), split temporary/permanent by α. This is the square-root
  impact curve (framework step 6) with explicit temp/perm structure. Inputs `X`, `v` =
  the `FillQty % Avg Vol 20D` / `Avg Vol 20D` fields. **Fit a1, a2 by non-linear
  regression on filled orders; FIX α = 0.95** (paper finds it stable across stocks/days/
  seasons; ~900 orders can't estimate α reliably).
- **ETF + cost profiles (Eqs. 7–8):** with calibrated impact, generate (expected cost,
  risk) for alternative schedules → trace the Efficient Trading Frontier. **Evaluation
  rule (§3, p.40): an execution is "best execution" only if it sits ON the frontier —
  true regardless of realized prices, good or bad.** Catches what residual-vs-broker
  cannot: a lucky cheap fill from a dominated/reckless schedule is NOT best execution.
- **Decision criteria (§3.2) → app dropdown:** (1) min cost s.t. risk ≤ R*; (2) min
  Cost + λ·Risk (λ = risk aversion; tangent to ETF); (3) price improvement = max
  Prob(Cost ≤ C*) (Sharpe-like). Marks the optimal point on each ETF for the fund's goal.

**Equation → module map:**
- Eq. 5 (`I = a1·X^a2`, α = 0.95) → **impact-calibration module** (extends framework step 6).
- Eq. 7 (open-benchmark cost + risk) → **cost-profile generator**. Risk term needs the
  ($/share)² covariance `Z`, derived from `Volatil 30D`.
- §3 / Fig. 1 → **ETF + evaluation view** (new headline Streamlit view): plot each order's
  realized (cost, risk) vs its frontier; flag dominated executions.
- §3.2 / Figs. 2–4 → **decision-criteria overlay** (goal dropdown).

**Caveats (do NOT over-claim):**
- **Calibration hunger:** a1, a2 need many orders across a RANGE of sizes per liquidity
  bucket; ~900 across many names → impact curve indicative, not precise. Stress params.
- **Risk term needs intraday structure:** Eqs. 7–8 assume a multi-period schedule
  (n periods, residual share vectors). Parent rows are outcomes, not paths → to locate the
  REALIZED point on the ETF, reconstruct the actual schedule from **child-route
  timestamps** (this is where the child table earns its place), or assume a schedule shape.
- **Calibration is US-equity-derived:** α = 0.95 and the `a1·X^a2` form are US estimates;
  the book is 9 currencies / 24 venues → treat as a starting prior, check residuals by
  region, segment (e.g. Asian small-caps) if they don't fit.

**Build order within this layer:** impact-calibration module FIRST (dependency for cost
profiles, the ETF, and the MC), then cost-profile generator, then ETF view, then decision overlay.

## The app (Phase 4)

Three layers, kept separate so the same logic serves a notebook and the UI:

- **Data layer:** cleaning pipeline → one tidy parent table + one linked child table → parquet.
- **Analysis layer:** standardization + regression + impact fit as pure functions
  returning dataframes (testable, no UI).
- **View layer:** Streamlit + Plotly (pure Python, sits on pandas/statsmodels; do not
  reach for Dash/React at this scale).

Views, in build order:
1. **Data-quality panel** (coverage, fill %, negative-spread / >100%-ADV flags) — build first.
2. Slippage **distribution** with filters (date, currency, broker).
3. **Conditional cuts** — boxplots by broker and venue, n shown.
4. **Impact curve** — slippage vs %ADV scatter + fitted sqrt line.
5. **Difficulty-adjusted league table** — regression residuals by broker/venue (the
   decision-support view).
6. **Time trends** over the 180 days (after Create Time is parsed).
7. **Parent → child drill-down** — click a flagged broker, see its routes.
8. **ETF best-execution view** (Kissell layer) — each order's realized (cost, risk) vs its
   efficient trading frontier; flag dominated executions; goal-dropdown overlay. The
   headline decision-support view once the impact model is calibrated.

## Build sequence

- **Phase 0:** confirm `OpenPx` exports populated (the y). Nothing else until this works.
- **Phase 1:** cleaning pipeline + data-quality report on the two files.
- **Phase 2:** EDA — distributions, conditional cuts.
- **Phase 3:** regression + impact fit.
- **Phase 3.5 (Kissell layer):** impact-calibration module (`I = a1·X^a2`, α = 0.95) →
  cost-profile generator → ETF evaluation → decision-criteria overlay.
- **Phase 4:** wrap analysis functions in Streamlit.
- Do NOT build the app before the analysis functions exist and are tested — the app is a
  thin shell over tested logic, not a place to debug statistics.

## Monte Carlo (separate, later — do NOT build on this dataset)

- Realized orders are outcomes, not paths. MC is for forward-looking *schedule design*.
- Simulate price paths (drift ≈ 0 intraday, vol-dominated); apply temporary + permanent
  (square-root) impact; compute IS per path; compare schedules (TWAP / VWAP / AC-optimal)
  on the (expected cost, cost std-dev) frontier.
- Needs intraday price/volume + an impact model. Dominant model risk is the impact
  coefficient — take from literature and **stress-sweep**; never trust a single value.
- ~10k paths for a stable mean/95th pct; tie N to the cost gap you must resolve. MC
  standard error is sampling noise only — it says nothing about model error.
- The empirical impact curve (stat framework step 6) calibrates this.

## Reading anchors

Perold (1988) IS; Almgren & Chriss (2000) optimal execution; Almgren, Thum, Hauptmann &
Li (2005) direct estimation of equity market impact (impact coefficient); **Kissell,
Glantz & Malamut (2004)** — open-benchmark cost profile, top-down impact model (α = 0.95),
ETF + decision criteria (the ex-ante layer above); Kissell (2006/2013) fuller cost model;
López de Prado (backtest-overfitting discipline).

## Stack notes

- Python: `pandas` (tables), `statsmodels` (regression, clustered SE, quantile reg),
  `plotly` (charts), `streamlit` (app). Intermediate data as parquet.
- Bloomberg: `blpapi` for re-extraction (and IntradayBar/IntradayTick for any MC data);
  `//blp/emsx.history` for fills (subscribe, don't poll).
