# KGM (2004) TC Model & Efficient Trading Frontier — Computational Spec

Maps Kissell–Glantz–Malamut (2004) opening-price benchmark
(§3.1.1, Eqs. 5 & 7) onto the available data fields. Notation uses the project's columns:
`AvgPx`, `Open` (= P0), `FillQty`, `Avg Vol 20D` (ADV proxy), `Volatil 30D`, `Side`.

## 0. The split that governs the whole build

- **Ex-post (measurement):** what an execution actually cost vs open. Runs on current data.
- **Ex-ante (model):** Eq. 7 predicts expected cost + risk of a *schedule*. Powers the ETF.
  Built by VARYING the schedule — NOT extracted from realized orders.
- The 951-order cross-section's only role in the ex-ante world is to **calibrate `a1, a2`**.

Keep these in separate modules. Never let the modeled cost overwrite the measured cost.

---

## 1. Ex-post realized cost (open benchmark)

Per order i (side = +1 buy, −1 sell):
```
slip_bps_i  = side_i * (AvgPx_i - Open_i) / Open_i * 1e4        # the dependent variable
cost_dollar_i (per share) = side_i * (AvgPx_i - Open_i)         # $/share, = MI proxy below
cash_cost_i = side_i * (AvgPx_i - Open_i) * FillQty_i           # then FX -> base ccy
```
Portfolio: total cash = Σ cash_cost_i; headline bps = **value-weighted**
`Σ cash_cost_i / Σ(FillQty_i * Open_i, FX) * 1e4` — NOT the equal-weighted mean of slip_bps.

Sanity gate before trusting any aggregate: side-adjusted mean must be POSITIVE and in the
tens of bps; every AvgPx must sit inside its day's [Low, High]; verify currency/units (GBp).

---

## 2. Impact-model calibration (`I = a1 * X^a2`, α = 0.95 fixed)

**Faithful form (Eq. 5), one-day interval reduction:**
```
MI_i($/share) = a1 * X_i^a2 * ( α / v_side_i  +  (1-α) / X_i ),   α = 0.95
```
- Dependent `MI_i($/share)` = realized `side_i*(AvgPx_i - Open_i)` (the $/share paid over open).
- `X_i` = order size in shares = `FillQty_i`.
- `v_side_i` = same-side market volume over the interval. No intraday volume on hand →
  proxy with `Avg Vol 20D` (or 0.5*ADV for "one side"). FLAG: this is approximate; the clean
  version needs intraday same-side volume. Document the proxy used.
- Fit `a1, a2` by non-linear least squares (`scipy.optimize.curve_fit`), α fixed at 0.95.

**Pragmatic form (preferred headline, multi-currency-robust):** fit in bps vs %ADV,
dropping the temp/perm split:
```
slip_bps_i ≈ b1 * (X_i / ADV_i) ^ b2          # pure square-root-law impact curve
```
Use this for the headline impact curve (unit-free, no $/share currency issue). Use the
faithful Eq. 5 form only when you need the temporary/permanent decomposition for the ETF.

Caveat (already in CLAUDE.md): ~900 orders across many names → `a1,a2`/`b1,b2` are
indicative, wide error bars. Stress them; segment by region if residuals misbehave.

---

## 3. Ex-ante cost & risk for a schedule (single security, Eq. 7)

Order of `X` shares split into `n` periods as schedule `{x_j}`, j = 1..n.
`I = a1 * X^a2` (calibrated, total $ impact). `v_j` = expected market volume in period j
(volume profile). `Δp` = expected per-period price trend.

```
Cost($) =  Σ_j  x_j * j * Δp                          # price-trend term
         + Σ_j  0.95 * I * x_j^2 / ( X * (x_j + 0.5*v_j) )   # temporary impact
         + Σ_j  0.05 * I * x_j / X                     # permanent impact

Risk($) =  sqrt( Σ_j  r_j^2 * σ_$^2 )  =  σ_$ * sqrt( Σ_j r_j^2 )   # single security
           where r_j = shares still to trade from period j onward = Σ_{k>=j} x_k
```
- Set **E[Δp] = 0** for the cost axis (intraday drift negligible vs vol) → first term drops in
  expectation. Keep it only if explicitly modeling a trend.
- `σ_$` = per-period $/share volatility = `Volatil 30D` (as a return vol) × `Open`, scaled to
  the period length. Convert Cost and Risk to bps by dividing by `X*Open` and ×1e4.
- Volume profile `v_j`: assume flat (`ADV/n`) or U-shaped intraday curve for an illustrative
  ETF; use the real interval profile only when placing realized executions.

**Multi-stock trade list:** Cost sums over stocks i; Risk uses the full `($/share)^2`
covariance matrix `Z`: `Risk = sqrt( Σ_j r_j^T Z r_j )`, where `r_j` is the residual-share
vector across names. Needs a returns covariance matrix for the basket, converted to
($/share)^2. Heavier data lift — start single-name (Z scalar), then extend.

---

## 4. Efficient Trading Frontier construction

1. **Parameterize a family of schedules** from aggressive to passive — e.g. constant-rate
   liquidation over horizon `n`, with `n` from 1 (immediate: max cost / min risk) to large
   (slow: min cost / max risk); or Almgren closed-form trajectories indexed by risk aversion λ.
2. For each schedule compute `(Cost_bps, Risk_bps)` from §3.
3. Plot Cost (y) vs Risk (x); the lower-left envelope is the ETF (cost decreasing in risk).
4. **Decision criteria (overlay):**
   - Goal 1 — min cost s.t. Risk ≤ R*: vertical line at R*, read cost off the ETF.
   - Goal 2 — min `Cost + λ*Risk`: line of slope −λ slid to tangency with the ETF.
   - Goal 3 — price improvement, max `(C* − E[Cost]) / Risk`: line from C* tangent to the ETF.

One ETF per ORDER or per TRADE-LIST — never one frontier for the whole dataset.

---

## 5. What the current data can and cannot do

- CAN: realized cost vs open (§1); calibrate `a1,a2` / `b1,b2` (§2); draw an illustrative ETF
  for a representative order/list from the calibrated model (§3–4).
- CANNOT yet: place each REALIZED execution as a point on its ETF — needs the realized
  intraperiod schedule `{x_j}`, i.e. **child-route fill timestamps** (capture going forward).
- CANNOT precisely: multi-stock portfolio ETF without the `($/share)^2` covariance matrix `Z`.

## 6. Build order

calibrate impact (`a1,a2`) → single-security cost/risk functions → ETF + decision overlay →
(later, with child timestamps) realized-point placement on the frontier.
