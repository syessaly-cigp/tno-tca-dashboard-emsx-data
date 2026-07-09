# Arrival vs Interval-VWAP — Cost Gap Analysis (2026 H1)

Reproducible via `analysis_arrival_vwap.py`. All figures in **bps, positive = cost**.

## 1. Logic

For the same fills, two benchmarks isolate different things:

- **Arrival cost (A)** = `side·(AvgPx − ArrPx)/ArrPx·1e4` — vs the decision price. Contains
  execution/impact **plus** market drift while the order worked.
- **Interval-VWAP cost (V)** = `side·(AvgPx − IntervalVWAP)/…` — vs the market's own average
  over the fill window. Drift-free; measures **execution vs the market**.
- **Gap A − V ≈ timing drift** — how the price moved against/for your side while trading.

Read rule:
| Condition | Meaning | Controllable |
|---|---|---|
| A ≈ V | genuine execution/impact | yes |
| A ≫ V | adverse drift (market moved against you) | mostly no (urgency) |
| A ≪ V | favorable drift **masking weak execution** — red flag | yes (algo) |

Judge **broker/algo skill on V**, total outcome on A, and attribute the gap to timing.

## 2. Parameters selected

| Parameter | Value | Why |
|---|---|---|
| Population | **Market orders** (`LmtPx=MKT`), **GTC excluded** | the framework's scope; GTC span multiple sessions (stale arrival) |
| n (market, ex-GTC) | **805** | |
| Weighting | **FX-adjusted USD notional** (value-weighted) + equal-weighted **mean** | VW = money view; mean = typical order; FX so JPY/GBp don't dominate |
| Benchmarks | **Arrival** + **Interval VWAP** | the gap is the signal |
| Statistics | VW cost, mean cost, **t-stat** (H₀: mean = 0), gap A−V | significance + concentration check |
| Min-n | **25–30** per cell (30 for 1-D cuts, 25 for 2-D) | your trend bar; raise to 50 for action |
| Trend bar | n ≥ 25 **and** \|t\| ≥ 2 **and** VW/mean agree in sign | avoids single-ticket artefacts |

Book baseline: **Arrival −0.6 bps, VWAP +0.9 bps** — the book essentially trades at its
decision price; findings are read against this, not against zero.

## 3. Categories analysed

Direction · Spread bucket×Direction · ADV%×Direction · Market-cap×Direction ·
Region×Direction · Broker×Direction · Region×Broker.
(Cells below min-n are dropped — notably the extreme high-participation / very-wide-spread
cells, which are too thin in market-orders-only to conclude on; see §6.)

## 4. Robust trends (n ≥ 25, |t| ≥ 2, VW & mean agree)

Ranked by economic size:

| # | Segment | n | A (VW) | V (VW) | Gap | t (A / V) | Read | Action |
|---|---|---|---|---|---|---|---|---|
| 1 | **Small-cap · Sell** | 61 | **+15.4** | +8.5 | **+6.9** | 3.0 / 1.9 | exec worse **and** adverse drift | slow the schedule / source block liquidity on small-cap sells |
| 2 | **Americas · BTIA** | 115 | +3.9 | **+12.8** | **−8.8** | 2.0 / 2.2 | favorable drift **masks weak execution** | review BTIA US algo — arrival flatters it | 
| 3 | **Europe · Buy** | 151 | +7.4 | **+6.3** | +1.1 | 1.7 / 2.6 | genuine execution cost vs market, no drift | review EU buy execution / venue |
| 4 | **BTIA · Sell** | 85 | +4.8 | +5.9 | −1.1 | 2.0 / 1.6 | executes worse than market | broker-quality watch |
| 5 | **Europe · Sell** | 84 | +4.5 | +2.0 | +2.5 | 2.2 / 1.0 | mild cost, small adverse drift | monitor |
| 6 | **Asia-Pacific · ICBI** | 43 | −6.0 | +3.9 | −9.9 | 2.5 / −0.2 | arrival "gain" is **drift, not skill** (V is a cost) | don't credit ICBI Asia on arrival |
| 7 | **CLLT · Buy** | 140 | +1.9 | +1.7 | +0.2 | 5.5 / 2.1 | tiny but highly consistent execution cost | negligible; benchmark |

## 5. Insights

- **The one genuinely expensive, controllable pocket is small-cap sells (+15 bps, t=3.0).**
  Uniquely, *both* components hurt: ~+8.5 bps execution (worse than VWAP) **and** ~+7 bps
  adverse drift. This is where scheduling/liquidity-sourcing saves the most.
- **Two brokers look better on arrival than they execute.** Americas/BTIA (A +3.9 but
  **V +12.8**) and Asia/ICBI (A −6 but **V +3.9**) both benefit from favorable drift that hides
  weak execution vs the market. Ranking brokers on arrival alone would mis-rate them — the
  drift-stripped **V** is the honest broker-quality metric here.
- **European flow carries a real execution cost** on both sides (buys V +6.3 t=2.6; sells
  A +4.5 t=2.2), with little drift → a venue/algo question, not a timing one.
- **US buys and Asia buys show large arrival "improvements" (−9 to −10 bps) that are drift**,
  not skill (gaps −3 to −6, means insignificant) — do not book these as execution wins.
- **Sells cost, buys don't** at the book level (Sell A +2.2 t=2.9 vs Buy −2.0 n.s.) — the
  cost in this book is concentrated on the sell side.

## 6. Caveats (what the data can't yet support)

- **Extreme cells are too thin.** Very-Large-ADV and Very-Wide-spread cells — where the
  biggest raw costs live (e.g. the +161 bps Very-Large-ADV sell seen without a size gate) —
  have n < 25 in market-orders-only, so they **do not qualify as trends**. Flag for more data.
- **Value-weighted ≠ significant.** Some VW cells are large but statistically insignificant
  (e.g. Mid-cap Buy VW +17.3 but t=−0.9) → driven by one or two big tickets, **not a trend**.
  Excluded above by the VW/mean-agreement rule.
- Market cap uses approximate static FX; industry (GICS) is ~43% populated so Region→Industry→
  Broker chains thin out — kept to 2-D cuts here.
- Cost is trading-related IS vs arrival, not full implementation shortfall (no delay/opportunity).
