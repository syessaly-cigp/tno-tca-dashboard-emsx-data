"""Generate the Arrival-vs-VWAP report as a formatted .docx with gold-themed charts.

Fully offline: charts drawn with PIL (Windows fonts), the .docx assembled as OOXML (no
python-docx). Run:  python build_report_docx.py
"""
from __future__ import annotations

import os
import io
import zipfile
from html import escape

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tca.pipeline import run_parent_pipeline
from tca.segments import arrival_vwap_trend_scan

# ---- CIGP gold palette (RGB) ------------------------------------------------
GOLD = (176, 141, 60)
GOLD_DK = (138, 108, 40)
TEAL = (46, 111, 99)
CLAY = (166, 70, 47)
INK = (35, 38, 43)
GREY = (107, 114, 128)
GRID = (231, 220, 195)
CREAM = (250, 246, 236)
WHITE = (255, 255, 255)


def _font(size, bold=False):
    names = (["segoeuib.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf"])
    for nm in names:
        p = f"C:/Windows/Fonts/{nm}"
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _text_w(d, text, font):
    return d.textbbox((0, 0), text, font=font)[2]


def chart_grouped_hbars(cats, series: dict, colors, title, subtitle=""):
    """series: {name: [values]} aligned to cats. Returns PNG bytes + (w,h)."""
    S = 2  # supersample
    W, rowh = 1500, 70
    top, bot, left, right = 120, 70, 430, 90
    H = top + len(cats) * rowh + bot
    img = Image.new("RGB", (W * S, H * S), WHITE)
    d = ImageDraw.Draw(img)
    f_title = _font(34 * S, bold=True)
    f_sub = _font(22 * S)
    f_cat = _font(22 * S)
    f_val = _font(20 * S, bold=True)
    f_leg = _font(20 * S)

    d.text((left * S, 34 * S), title, font=f_title, fill=INK)
    if subtitle:
        d.text((left * S, 78 * S), subtitle, font=f_sub, fill=GREY)

    vals = [v for vs in series.values() for v in vs]
    vmin, vmax = min(vals + [0]), max(vals + [0])
    span = (vmax - vmin) or 1
    pad = span * 0.15
    vmin, vmax = vmin - pad, vmax + pad
    span = vmax - vmin
    plot_l, plot_r = left * S, (W - right) * S

    def x_of(v):
        return plot_l + (v - vmin) / span * (plot_r - plot_l)

    x0 = x_of(0)
    # zero line
    d.line([(x0, (top - 10) * S), (x0, (top + len(cats) * rowh) * S)], fill=GRID, width=2 * S)

    names = list(series.keys())
    g = len(names)
    barh = int(rowh * 0.72 / g)
    gap = int(rowh * 0.14)
    for i, cat in enumerate(cats):
        ytop = (top + i * rowh) * S + gap * S
        d.text((30 * S, ytop + (barh * g * S) // 2 - 12 * S), cat, font=f_cat, fill=INK)
        for j, nm in enumerate(names):
            v = series[nm][i]
            y = ytop + j * barh * S
            xa, xb = (x0, x_of(v)) if v >= 0 else (x_of(v), x0)
            d.rectangle([xa, y, xb, y + barh * S - 6 * S], fill=colors[j])
            lx = x_of(v) + (6 * S if v >= 0 else -6 * S - _text_w(d, f"{v:+.1f}", f_val))
            d.text((lx, y + 2 * S), f"{v:+.1f}", font=f_val, fill=INK)
    # legend
    lx = plot_l
    for j, nm in enumerate(names):
        d.rectangle([lx, (H - 46) * S, lx + 28 * S, (H - 22) * S], fill=colors[j])
        d.text((lx + 38 * S, (H - 48) * S), nm, font=f_leg, fill=INK)
        lx += (60 + _text_w(d, nm, f_leg) // S + 40) * S
    d.text((plot_l, (H - 46) * S + 0), "", font=f_leg, fill=INK)

    img = img.resize((W, H), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue(), (W, H)


def chart_diverging(cats, values, title, subtitle=""):
    """Single-series diverging bars (clay = +/adverse, teal = −/favorable)."""
    S = 2
    W, rowh = 1500, 60
    top, bot, left, right = 120, 50, 430, 90
    H = top + len(cats) * rowh + bot
    img = Image.new("RGB", (W * S, H * S), WHITE)
    d = ImageDraw.Draw(img)
    d.text((left * S, 34 * S), title, font=_font(34 * S, bold=True), fill=INK)
    if subtitle:
        d.text((left * S, 78 * S), subtitle, font=_font(22 * S), fill=GREY)
    f_cat, f_val = _font(22 * S), _font(20 * S, bold=True)
    vmax = max(abs(min(values + [0])), abs(max(values + [0]))) or 1
    vmax *= 1.2
    plot_l, plot_r = left * S, (W - right) * S
    x0 = (plot_l + plot_r) / 2

    def x_of(v):
        return x0 + (v / vmax) * (plot_r - x0)

    d.line([(x0, (top - 8) * S), (x0, (top + len(cats) * rowh) * S)], fill=GRID, width=2 * S)
    for i, cat in enumerate(cats):
        y = (top + i * rowh) * S + 10 * S
        v = values[i]
        col = CLAY if v > 0 else TEAL
        xa, xb = (x0, x_of(v)) if v >= 0 else (x_of(v), x0)
        d.rectangle([xa, y, xb, y + 30 * S], fill=col)
        d.text((30 * S, y + 2 * S), cat, font=f_cat, fill=INK)
        lx = x_of(v) + (6 * S if v >= 0 else -6 * S - _text_w(d, f"{v:+.1f}", f_val))
        d.text((lx, y + 3 * S), f"{v:+.1f}", font=f_val, fill=INK)
    img = img.resize((W, H), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue(), (W, H)


# ---- minimal OOXML .docx builder -------------------------------------------
NS = ('xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
      'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
      'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
      'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
      'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"')


def _run(text, bold=False, size=22, color="23262B"):
    rpr = f'<w:rPr><w:rFonts w:ascii="Segoe UI" w:hAnsi="Segoe UI"/>{"<w:b/>" if bold else ""}' \
          f'<w:sz w:val="{size}"/><w:color w:val="{color}"/></w:rPr>'
    return f'<w:r>{rpr}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def para(text="", bold=False, size=22, color="23262B", after=120, align=None):
    runs = ""
    # simple **bold** inline
    parts = text.split("**")
    for i, seg in enumerate(parts):
        if seg:
            runs += _run(seg, bold=(bold or i % 2 == 1), size=size, color=color)
    jc = f'<w:jc w:val="{align}"/>' if align else ""
    return f'<w:p><w:pPr><w:spacing w:after="{after}"/>{jc}</w:pPr>{runs}</w:p>'


def heading(text, size=30, color="8A6C28", before=200, after=100):
    return (f'<w:p><w:pPr><w:spacing w:before="{before}" w:after="{after}"/></w:pPr>'
            f'{_run(text, bold=True, size=size, color=color)}</w:p>')


def bullet(text):
    return para("•  " + text, size=21, after=60)


def table(headers, rows, widths):
    def cell(text, w, bold=False, fill=None, cc="23262B"):
        shd = f'<w:shd w:val="clear" w:fill="{fill}"/>' if fill else ""
        return (f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>{shd}'
                f'<w:tcMar><w:top w:w="40"/><w:bottom w:w="40"/><w:left w:w="80"/><w:right w:w="80"/></w:tcMar>'
                f'</w:tcPr><w:p><w:pPr><w:spacing w:after="0"/></w:pPr>{_run(str(text), bold=bold, size=19, color=cc)}</w:p></w:tc>')
    borders = ('<w:tblBorders>' + ''.join(
        f'<w:{s} w:val="single" w:sz="4" w:color="E7DCC3"/>'
        for s in ["top", "left", "bottom", "right", "insideH", "insideV"]) + '</w:tblBorders>')
    tbl = f'<w:tbl><w:tblPr><w:tblW w:w="{sum(widths)}" w:type="dxa"/>{borders}</w:tblPr>'
    tbl += '<w:tr>' + ''.join(cell(h, w, bold=True, fill="B08D3C", cc="FFFFFF")
                              for h, w in zip(headers, widths)) + '</w:tr>'
    for r in rows:
        tbl += '<w:tr>' + ''.join(cell(c, w) for c, w in zip(r, widths)) + '</w:tr>'
    return tbl + '</w:tbl>' + para("", after=80)


def image_para(rid, w_px, h_px, target_in=6.6):
    emu_w = int(target_in * 914400)
    emu_h = int(emu_w * h_px / w_px)
    return (f'<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:after="160"/></w:pPr><w:r><w:drawing>'
            f'<wp:inline distT="0" distB="0" distL="0" distR="0">'
            f'<wp:extent cx="{emu_w}" cy="{emu_h}"/><wp:effectExtent l="0" t="0" r="0" b="0"/>'
            f'<wp:docPr id="{rid}" name="Picture {rid}"/>'
            f'<wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>'
            f'<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            f'<pic:pic><pic:nvPicPr><pic:cNvPr id="{rid}" name="Picture {rid}"/><pic:cNvPicPr/></pic:nvPicPr>'
            f'<pic:blipFill><a:blip r:embed="rId{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{emu_w}" cy="{emu_h}"/></a:xfrm>'
            f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic>'
            f'</a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>')


def build_docx(path, body_xml, images):
    """images: list of (rid, png_bytes)."""
    ct = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
          '<Default Extension="xml" ContentType="application/xml"/>'
          '<Default Extension="png" ContentType="image/png"/>'
          '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
          '</Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rIdDoc" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
            '</Relationships>')
    doc = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
           f'<w:document {NS}><w:body>{body_xml}'
           f'<w:sectPr><w:pgSz w:w="12240" w:h="15840"/>'
           f'<w:pgMar w:top="1200" w:right="1200" w:bottom="1200" w:left="1200"/></w:sectPr></w:body></w:document>')
    drels = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
             '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">']
    for rid, _ in images:
        drels.append(f'<Relationship Id="rId{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/image{rid}.png"/>')
    drels.append('</Relationships>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", ct)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", doc)
        z.writestr("word/_rels/document.xml.rels", "".join(drels))
        for rid, png in images:
            z.writestr(f"word/media/image{rid}.png", png)


# ---- assemble the report ----------------------------------------------------
def main():
    art = run_parent_pipeline("data_trades_new2.csv", "180days_child_order_data.csv", exclude_gtc=True)
    clean = art.clean
    mkt = clean[clean["keep_for_analysis"] & (clean["order_type"] == "Market")]
    book_A = float(np.average(mkt["cost_bps"], weights=mkt["notional_usd"]))
    book_V = float(np.average(mkt["cost_vwap_bps"].fillna(0), weights=mkt["notional_usd"]))
    scan = arrival_vwap_trend_scan(clean, min_n=25)
    robust = scan[scan["robust"]].head(7).copy()

    short = robust["segment"].str.replace("Region×Broker: ", "", regex=False) \
        .str.replace("Region×Dir: ", "", regex=False).str.replace("Broker×Dir: ", "", regex=False) \
        .str.replace("Cap×Dir: ", "", regex=False).str.replace("Direction: ", "", regex=False)
    cats = short.tolist()[::-1]
    A = robust["A_vw"].round(1).tolist()[::-1]
    V = robust["V_vw"].round(1).tolist()[::-1]
    gap = robust["gap_vw"].round(1).tolist()[::-1]

    png1, (w1, h1) = chart_grouped_hbars(
        cats, {"vs Arrival": A, "vs VWAP": V}, [GOLD, TEAL],
        "Robust trends — cost vs Arrival and vs Interval VWAP",
        "Market orders, GTC-excluded · value-weighted (FX-USD) · positive = cost")
    png2, (w2, h2) = chart_diverging(
        cats, gap, "Timing drift by segment  (Arrival − VWAP)",
        "Positive = adverse drift (market moved against you) · negative = favorable")
    png3, (w3, h3) = chart_grouped_hbars(
        ["Book (market orders)"], {"vs Arrival": [round(book_A, 1)], "vs VWAP": [round(book_V, 1)]},
        [GOLD, TEAL], "Book-level cost", "Value-weighted, positive = cost")

    B = []
    B.append(para("CIGP — Equity TCA", bold=True, size=44, color="B08D3C", after=40))
    B.append(para("Arrival vs Interval-VWAP — Cost Gap Analysis (2026 H1)", bold=True, size=30, color="23262B", after=60))
    B.append(para("Market orders · GTC excluded · value-weighted (FX-adjusted USD) · positive = cost",
                  size=20, color="6B7280", after=200))

    B.append(heading("1 · Objective & logic"))
    B.append(para("Where do we see higher or lower arrival-to-execution slippage for market orders? "
                  "Two benchmarks isolate different things:"))
    B.append(bullet("Arrival cost (A) = side·(AvgPx − ArrPx)/ArrPx·1e4 — vs the decision price; contains execution/impact plus market drift."))
    B.append(bullet("Interval-VWAP cost (V) = side·(AvgPx − VWAP)/… — vs the market’s own average; drift-free execution."))
    B.append(bullet("Gap A − V ≈ timing drift — how the price moved against/for your side while trading."))
    B.append(para("Read: A≈V → genuine execution/impact (controllable). A≫V → adverse drift (timing, mostly not). "
                  "A≪V → favorable drift masking weak execution (red flag). Judge broker skill on V.", after=160))

    B.append(heading("2 · Parameters, statistics & categories"))
    B.append(table(["Parameter", "Value"], [
        ["Population", f"Market orders (LmtPx=MKT), GTC excluded — n = {len(mkt)}"],
        ["Weighting", "FX-adjusted USD notional (value-weighted) + equal-weighted mean"],
        ["Benchmarks", "Arrival + Interval VWAP"],
        ["Statistics", "VW cost, mean cost, t-stat (H0: mean=0), gap A−V"],
        ["Min sample", "25–30 per cell (raise to 50 before acting)"],
        ["Trend bar", "n ≥ 25 and |t| ≥ 2 and VW/mean agree in sign"],
        ["Categories", "Direction; Spread×Dir; ADV%×Dir; Cap×Dir; Region×Dir; Broker×Dir; Region×Broker"],
    ], [2600, 6800]))
    B.append(para(f"Book baseline (market orders): Arrival **{book_A:+.1f}** bps, VWAP **{book_V:+.1f}** bps — "
                  "the book essentially trades at its decision price; findings are read against this.", after=120))
    B.append(image_para(3, w3, h3))

    B.append(heading("3 · Robust trends"))
    B.append(image_para(1, w1, h1))
    B.append(table(["Segment", "n", "A (VW)", "V (VW)", "Gap", "t(A)", "Read"],
                   [[s, int(n), f"{a:+.1f}", f"{v:+.1f}", f"{gp:+.1f}", f"{t:+.1f}", rd]
                    for s, n, a, v, gp, t, rd in zip(
                        robust["segment"], robust["n"], robust["A_vw"], robust["V_vw"],
                        robust["gap_vw"], robust["A_t"], robust["read"])],
                   [3200, 700, 1000, 1000, 900, 800, 2000]))
    B.append(image_para(2, w2, h2))

    B.append(heading("4 · Insights"))
    B.append(bullet("The one sizeable, controllable pocket is small-cap sells (~+15 bps, t=3.0): both execution "
                    "(worse than VWAP) and adverse drift hurt. Slow the schedule / source block liquidity."))
    B.append(bullet("Two brokers look better on arrival than they execute: Americas/BTIA (A +3.9 but V +12.8) and "
                    "Asia/ICBI (A −6 but V +3.9) — favorable drift masks weak execution. Rank brokers on V."))
    B.append(bullet("European flow carries a real execution cost on both sides (buys V +6.3, t=2.6; sells A +4.5, "
                    "t=2.2) with little drift — a venue/algo question, not timing."))
    B.append(bullet("Sells cost, buys don’t at book level (Sell A +2.2, t=2.9 vs Buy −2.0 n.s.)."))

    B.append(heading("5 · Caveats"))
    B.append(bullet("Extreme cells (Very-Large ADV, Very-Wide spread) fall below min-n in market-orders-only, so they "
                    "do NOT qualify as trends — flag for more data."))
    B.append(bullet("Some value-weighted cells are large but insignificant (one/two big tickets) — excluded by the "
                    "VW/mean-agreement rule."))
    B.append(bullet("Market cap uses approximate static FX; GICS industry ~43% populated. Cost is trading-related IS "
                    "vs arrival, not full implementation shortfall."))

    out = "TCA_Arrival_vs_VWAP_Report.docx"
    build_docx(out, "".join(B), [(1, png1), (2, png2), (3, png3)])
    print("wrote", os.path.abspath(out), f"({os.path.getsize(out):,} bytes)")


if __name__ == "__main__":
    main()
