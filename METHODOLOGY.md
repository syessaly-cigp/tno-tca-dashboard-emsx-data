# TCA Methodology — Formulas, Assumptions, Controls & Data

Reference for every quantity the pipeline computes. Each section gives the **formula**,
the **assumptions** behind it, the **control variables** it uses, and **what the data
represents**. Sign convention is fixed throughout:

> **Negative bps = cost, positive bps = price improvement.** A parallel `cost_bps = −slippage_bps`
> (positive = cost) is used only where a cost must be positive (impact fit, regression, league).

Notation: `i` indexes a parent order. `side` = +1 buy / −1 sell only where noted; in code the
side factor `sf` is **−1 for buy, +1 for sell** so that both sides express cost as negative.

---

## 0. What the data represents

| Table | Rows | Grain | Role |
|---|---|---|---|
| `parents` (`td_data_full.xlsx` → `td_data_full.csv`) | 974 (916 Filled, 58 Part-filled) | one **parent order** = a realized execution outcome | **Primary unit.** All slippage, regression, impact, ETF. |
| `routes` (`180days_child_order_data.csv`) | 642 (all Filled) | one **child route** (a slice sent to a venue) | Diagnosis / drill-down only. **No parent ID** → joined by `Security`. |

- **Cross-sectional, ex-post.** Each row is what *happened*, not a simulated path. Never pool
  parents + routes into one regression (double-counts shares; two arrival prices).
- **9 currencies** (USD-dominated), prices from ~0.02 to ~5,570 → **all analysis in bps**, never
  cash/price levels across orders. `% = bps / 100`.
- **Prices are in each order's local currency.** Cash figures (`slippage_cash_local`,
  `notional_local`) are local-ccy and used only for weighting/reporting, not cross-order ranking.

### Key raw fields used
`AvgPx` (achieved), `ArrPx` (arrival / decision price), `IntervalVWAP` (market VWAP over the
order's fill window), `Open Px`/`Low`/`High` (**export-time snapshot — diagnostic only**),
`Yest Cls Px`, `FillQty`, `Qty`, `Qty % Avg Vol 20D` (%ADV), `Volatil 30D` (annualized % vol),
`Day Part Rate %` (participation), `Bid Ask Sprd`, `Side`, `Brkr Code`, `Exch Code`, `Curncy`,
`Create Time (As of)`, `Trade Date`, `Value (Local)`.

---

## 1. Slippage (the dependent variable)

**Primary — Implementation Shortfall vs arrival:**
```
slippage_bps_i      =  sf_i · (AvgPx_i − ArrPx_i) / ArrPx_i · 1e4
cost_bps_i          = −slippage_bps_i
slippage_cash_local = (AvgPx_i − ArrPx_i) · FillQty_i · sf_i        # local ccy, reporting only
notional_local_i    =  FillQty_i · ArrPx_i
```

**Secondary benchmarks (same fills, different reference):**
```
slippage_vwap_bps_i =  sf_i · (AvgPx_i − IntervalVWAP_i) / IntervalVWAP_i · 1e4   # drift-free cross-check
slippage_open_bps_i =  sf_i · (AvgPx_i − OpenPx_i) / OpenPx_i · 1e4               # DIAGNOSTIC ONLY
delay_arr_vs_open_bps = (ArrPx_i − OpenPx_i) / OpenPx_i · 1e4                     # context
```

**Assumptions**
- **Arrival price = the decision price** at order entry; slippage vs arrival ≈ trading-related
  Implementation Shortfall (not full IS — no delay/opportunity terms, see §9).
- Sign convention verified against Bloomberg's own `AvgPx Vs ArrPx (Bps)` (buys and sells match
  to the decimal; `corr(cost_bps, bbg) = −1.00`).
- A non-positive or missing benchmark → `NaN` (a handful of `IntervalVWAP = 0` rows).
- **Open Px / Low / High are an export-time snapshot** (AvgPx sits inside `[Low, High]` only ~7%
  of the time), so the open benchmark is **retired to a diagnostic column** and never gated on.

**Data represented:** one bps cost per filled parent order, comparable across currencies.

---

## 2. Attribution decomposition (execution skill vs market drift)

On a **common arrival denominator** so the pieces add up exactly:
```
exec_vwap_bps_i =  sf_i · (AvgPx_i − IntervalVWAP_i) / ArrPx_i · 1e4     # execution vs market
timing_bps_i    =  sf_i · (IntervalVWAP_i − ArrPx_i) / ArrPx_i · 1e4     # drift while working

⇒  slippage_bps_i = exec_vwap_bps_i + timing_bps_i        (exact identity)
```

**Interpretation / assumptions**
- `exec_vwap_bps` = did we beat the market's own VWAP over the fill window → **controllable**
  (broker/venue/algo skill), largely drift-free.
- `timing_bps` = how the stock moved between arrival and the window → **mostly not controllable**
  (context, not broker blame). Entry timing is the only partly-controllable part.
- Uses `ArrPx` as the common denominator (not `IntervalVWAP`) so the two terms sum to
  `slippage_bps` exactly rather than approximately.

**Control variables:** none (pure identity). Aggregated **value-weighted by `notional_local`**.

---

## 3. Cleaning, flags & winsorization

**Derived quantities**
```
ADV_shares_20d_i =  FillQty_i / (Qty%ADV_i / 100)          # back-derived exact 20-day ADV
x_over_adv_i     =  Qty%ADV_i / 100                          # order size as fraction of ADV
entry_minute_i   =  hour·60 + minute  of Create Time         # time-of-day control
```

**Flags** (warn/keep unless noted)
| Flag | Rule | Action |
|---|---|---|
| `flag_negative_spread` | `Bid Ask Sprd < 0` | **drop** from analysis |
| `flag_missing_core` | any of ArrPx/AvgPx/FillQty/side/broker/venue missing | **drop** |
| `flag_avgpx_outside_hilo` | AvgPx ∉ `[Low, High]` | **warn only** (open snapshot) |
| `flag_extreme_adv` | `Qty%ADV > 100` | flag, winsorized |
| `flag_multi_day_gtc` | TIF contains "GTC" and Create date ≠ Trade date | flag, kept |
| `flag_part_fill` | status part-filled | flag, kept (slippage on filled portion) |
| `has_footprint` | `|cost_bps| > 0.5` | subset marker (see §6) |

**Keep rule**
```
keep_for_analysis = ¬neg_spread ∧ ¬missing_core ∧ (ArrPx > 0) ∧ slippage_bps not NaN
```

**Winsorization** (guard fat tails without deleting): every continuous variable clipped to its
**1st / 99th percentile** — `slippage_bps, slippage_vwap_bps, cost_bps, exec_vwap_bps,
timing_bps, Qty%ADV, Bid Ask Sprd, FillQty, Day Part Rate`. Winsorized columns carry a `_w` suffix.

**Assumption:** winsorize, don't delete — a 1358% ADV or a −1655 spread is a data glitch, not a
reason to lose the order's other fields.

---

## 4. Difficulty-adjusted regression (the league table)

**Model (OLS):**
```
cost_z_i = β0
         + β1·Qty%ADV_w      + β2·BidAskSprd_w   + β3·log(1+FillQty_w)
         + β4·Volatil30D     + β5·DayPartRate_w  + β6·entry_minute
         + Σ γ_s·1[side]      + Σ δ_b·1[broker]   + Σ φ_v·1[venue]
         + ε_i
```
where the dependent is the **standardized cost**:
```
cost_z_i = (cost_bps_w_i − mean(cost_bps_w)) / sd(cost_bps_w)
```
Standard errors **clustered by `Security`**. Also fit a **95th-percentile quantile regression**
with the same RHS on `cost_bps_w` (tail drivers differ from the mean).

**Control variables (difficulty controls — hold order difficulty fixed):**
| Control | Field | Why |
|---|---|---|
| Size vs liquidity | `Qty % Avg Vol 20D` (winsorized) | bigger vs ADV ⇒ more impact |
| Spread | `Bid Ask Sprd` (winsorized) | wider ⇒ costlier to cross |
| Absolute size | `log(1 + FillQty)` | scale, diminishing |
| Volatility | `Volatil 30D` | riskier names cost more |
| Participation | `Day Part Rate %` (winsorized) | trading faster ⇒ more impact |
| Time of day | `entry_minute` | soaks up intraday drift folded into the benchmark |
| Direction | `Side` fixed effect | buys/sells differ |

**Fixed effects / levers:** `Broker` (δ_b) and `Venue` (φ_v) — the coefficients of interest.
After controls, δ_b / φ_v = that broker/venue's **difficulty-adjusted execution quality**
(positive = costlier). The residual league ranks on these, **not** raw averages (the biggest
broker gets the hardest orders — a confound trap).

**Assumptions**
- Orders in the same security are correlated → cluster by security.
- Broker FE is only meaningfully estimable for the ~3 large brokers (ICBI, CLLT, BTIA); thin
  brokers carry wide error bars — show `n`.
- Linear/log-linear functional form; controls are proxies, not exhaustive.

---

## 5. Market-impact curve (square-root law)

**Pragmatic headline (currency-robust, in bps):**
```
cost_bps_i ≈ b1 · (X_i / ADV_i)^b2          fit by non-linear least squares (scipy.curve_fit)
```
**Faithful Kissell (Eq. 5, $/share, temp/perm structure):**
```
MI($/share)_i = a1 · X_i^{a2} · ( α / v_i + (1−α) / X_i ),   α = 0.95 (FIXED)
   dependent  = cost_bps_i / 1e4 · ArrPx_i          # realized $/share over arrival
   X_i = FillQty_i,   v_i = 0.5 · ADV_shares_20d_i   # "one side" of ADV
```

**Assumptions**
- **Fit on the footprint subset** (`|cost_bps| > 0.5`): ~2/3 of orders fill at arrival with zero
  footprint and would flatten `b1`. The informative n is ~330 orders.
- Impact measured **vs arrival** (the correct reference for impact).
- `α = 0.95` fixed (Kissell find it stable; ~900 orders can't estimate it).
- `v ≈ 0.5·ADV20` because no intraday same-side volume is available — **approximate**.
- Bps form mixes no currencies (unit-free); the $/share faithful form **does** mix 9 currencies →
  indicative only. Both have wide error bars — stress them, segment by region if residuals misbehave.

**Control variables:** size/liquidity ratio `X/ADV` (and `X`, `v` in the faithful form). No FE.

---

## 6. Ex-ante cost & risk of a schedule (Kissell Eq. 7, single security)

Split an order of `X` shares into `n` equal slices over a horizon of `H` days
(`n = round(H · periods_per_day)`, `periods_per_day = 13` ≈ half-hour buckets), constant-rate (TWAP):
```
x_j = X / n                                    # shares per period
v_j = ADV · (H / n)                            # market volume per period
I   = ( b1·(X/ADV)^b2 / 1e4 ) · X · ArrPx      # total $ impact of the full order (from §5)

Cost($)      = Σ_j 0.95 · I · x_j² / ( X·(x_j + 0.5·v_j) )   # temporary impact
             + 0.05 · I                                       # permanent impact
Risk($)      = σ$_period · sqrt( Σ_j r_j² ),   r_j = Σ_{k≥j} x_k = X·(n−j+1)/n
   σ$_period = (Volatil30D / 100) / sqrt(252) · sqrt(H/n) · ArrPx     # per-period $ vol

Cost_bps = Cost($) / (X·ArrPx) · 1e4      Risk_bps = Risk($) / (X·ArrPx) · 1e4
```

**Assumptions**
- **E[price trend Δp] = 0** on the cost axis (intraday drift negligible vs vol) → the trend term
  drops. Keep it only if explicitly modeling a signal.
- `Volatil 30D` is an **annualized return vol in %** → `/100` to decimal, `/sqrt(252)` to daily.
- **Price level = ArrPx**, not open (open is snapshot-contaminated).
- Temporary/permanent split fixed by `α = 0.95`.
- **Single security** ⇒ risk is a scalar variance; a multi-name basket would need the full
  `($/share)²` covariance matrix `Z` (deferred).
- Constant-rate (TWAP) schedule family; other trajectories (Almgren) would shift the curve.

---

## 7. Efficient Trading Frontier + decision criteria (Kissell §3–4)

Trace `(Cost_bps, Risk_bps)` from §6 while **varying the horizon `H`** (aggressive short-H = high
cost / low risk → passive long-H = low cost / high risk). Lower-left envelope = the frontier.
Optimal point per fund goal:
```
Goal 1  min Cost   s.t. Risk ≤ R*                      (risk-budget)
Goal 2  min ( Cost + λ · Risk )                        (mean–variance, λ = risk aversion)
Goal 3  max ( C* − Cost ) / Risk                       (price-improvement / Sharpe-like)
```
**Rule:** an execution is *best execution* only if it sits **on** the frontier — regardless of the
realized price (catches a lucky cheap fill from a reckless schedule). **One frontier per order**,
never one for the whole book.

**Assumptions:** built by varying the *schedule*, **not** extracted from realized fills. The
cross-section's only role in this layer is to calibrate `b1, b2` (§5).

---

## 8. Portfolio headline (value-weighting)

```
value_weighted_slippage_bps = Σ_i ( slippage_bps_i · w_i ) / Σ_i w_i ,   w_i = |notional_local_i|
```
Reported alongside the equal-weighted mean, the median, and the VWAP cross-check
(`value_weighted(slippage_vwap_bps)`).

**Assumptions**
- Value-weighting (by traded notional) is the book-level number that matters — the equal-weighted
  mean is dominated by tiny orders.
- **Sanity check** (arrival benchmark): value-weighted slippage is a *plausible magnitude*
  (`|vw| < 100 bps`) and the VWAP cross-check is finite. For an arrival benchmark, price
  *improvement* is legitimate (worked limit flow), so "must be a cost" is **not** the test.

---

## 9. What this measures — and what it does not

- **Measures:** trading-related cost vs arrival (fills vs decision price) + a VWAP cross-check +
  a difficulty-adjusted broker/venue attribution + an indicative impact curve + an illustrative ETF.
- **Does NOT measure full Implementation Shortfall:** the **delay** term needs a decision price
  distinct from arrival, and the **opportunity** term needs per-period unfilled quantity — label
  the metric **"trading-related cost vs arrival," not full IS.**
- **Cannot yet** place each *realized* execution as a point on its ETF (needs the intraday
  child-route fill schedule) or build a multi-name basket ETF (needs the covariance matrix `Z`).
- Calibration is US-equity-derived (`α = 0.95`, the `a1·X^a2` form) applied to a 9-currency /
  24-venue book — a starting prior, not gospel.

---

## 10. Symbol glossary

| Symbol | Meaning | Source field |
|---|---|---|
| `AvgPx` | achieved average execution price | `AvgPx` |
| `ArrPx` | arrival / decision price (primary benchmark) | `ArrPx` |
| `IntervalVWAP` | market VWAP over the order's fill window | `IntervalVWAP` |
| `X` | order size in shares | `FillQty` |
| `ADV` | 20-day average daily volume (shares) | back-derived from `Qty % Avg Vol 20D` |
| `v` | market volume in a period | `ADV · H/n` (proxy `0.5·ADV` for "one side") |
| `σ` | annualized return volatility | `Volatil 30D` / 100 |
| `sf` | side factor (−1 buy, +1 sell) | `Side` |
| `w` | value weight | `|FillQty · ArrPx|` |
| `α` | temp/perm impact split | fixed 0.95 |
| `b1, b2` | pragmatic impact coefficients | fitted (§5) |
| `a1, a2` | faithful impact coefficients | fitted (§5) |
| `λ` | risk aversion | user input (§7) |
