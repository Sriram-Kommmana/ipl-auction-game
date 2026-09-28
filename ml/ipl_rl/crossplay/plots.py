"""Phase 2E.0 plots (static SVG + one HTML page), from matchup-results.json.

    python plots.py <report_dir>        writes <report_dir>/plots/*.svg and plots/index.html

Charts
  1–4  ordered learner × opponent heatmaps: transfer Δ and head-to-head Δ, C1 and C4
       (diverging blue ↔ red around a neutral gray 0; every cell labelled; hover = CI).
  5    Stage A → Stage B validation XI per algorithm (A, C1, C4, S4; 95% CI).
  6    seed stability: the 9 seed-pairing transfer Δs of every ordered pair (C1, C4).
  7    S4 mixed room: transfer Δ per learner and head-to-head vs each algorithm.
No chart ranks the algorithms; rows keep the fixed algorithm order.
"""
import json
import sys
from html import escape
from pathlib import Path

ALGOS = ["ppo", "a2c", "d3qn", "qrdqn", "es"]
NAMES = {"ppo": "PPO", "a2c": "A2C", "d3qn": "D3QN", "qrdqn": "QR-DQN", "es": "OpenAI-ES"}
# reference palette (dataviz): diverging blue ↔ red, neutral gray midpoint; chrome/ink
NEG, MID, POS = (0xB2, 0x2E, 0x2E), (0xF0, 0xEF, 0xEC), (0x1C, 0x5C, 0xAB)
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = {"A": "#52514e", "C1": "#2a78d6", "C4": "#eb6834", "S4": "#1baf7a"}  # categorical slots in fixed order (ink for the Stage-A control)
FONT = "font-family:system-ui,-apple-system,'Segoe UI',sans-serif"


def mix(a, b, t):
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


def diverging(v, vmax):
    t = max(-1.0, min(1.0, v / vmax)) if vmax else 0.0
    return mix(MID, NEG, -t) if t < 0 else mix(MID, POS, t)


def text_on(hexcol):
    r, g, b = (int(hexcol[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lum = 0.2126 * r ** 2.2 + 0.7152 * g ** 2.2 + 0.0722 * b ** 2.2
    return "#ffffff" if lum < 0.33 else INK


def heatmap(title, sub, cells, vmax, fmt="{:+.2f}"):
    cw, ch, left, top = 112, 46, 150, 100
    w, h = left + cw * 5 + 24, top + ch * 5 + 70
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" style="{FONT};background:{SURF}">',
         f'<text x="16" y="24" font-size="15" font-weight="600" fill="{INK}">{escape(title)}</text>',
         f'<text x="16" y="44" font-size="12" fill="{INK2}">{escape(sub)}</text>',
         f'<text x="{left + cw * 2.5}" y="{top - 10}" font-size="11" fill="{MUTED}" text-anchor="middle">opponent →</text>',
         f'<text x="16" y="{top + ch * 2.5}" font-size="11" fill="{MUTED}">learner ↓</text>']
    for j, b in enumerate(ALGOS):
        o.append(f'<text x="{left + cw * j + cw / 2}" y="{top - 28}" font-size="12" fill="{INK2}" text-anchor="middle">{NAMES[b]}</text>')
    for i, a in enumerate(ALGOS):
        o.append(f'<text x="{left - 10}" y="{top + ch * i + ch / 2 + 4}" font-size="12" fill="{INK2}" text-anchor="end">{NAMES[a]}</text>')
        for j, b in enumerate(ALGOS):
            x, y = left + cw * j, top + ch * i
            if a == b:
                o.append(f'<rect x="{x + 1}" y="{y + 1}" width="{cw - 2}" height="{ch - 2}" rx="4" fill="{SURF}" stroke="{GRID}"/><text x="{x + cw / 2}" y="{y + ch / 2 + 4}" font-size="12" fill="{MUTED}" text-anchor="middle">—</text>')
                continue
            c = cells[f"{a}>{b}"]
            col = diverging(c["mean"], vmax)
            lo, hi = c["ci95"]
            o.append(f'<g><title>{NAMES[a]} vs {NAMES[b]}: {c["mean"]:+.3f} (95% CI {lo:+.3f} to {hi:+.3f})</title>'
                     f'<rect x="{x + 1}" y="{y + 1}" width="{cw - 2}" height="{ch - 2}" rx="4" fill="{col}"/>'
                     f'<text x="{x + cw / 2}" y="{y + ch / 2}" font-size="13" font-weight="600" fill="{text_on(col)}" text-anchor="middle">{fmt.format(c["mean"])}</text>'
                     f'<text x="{x + cw / 2}" y="{y + ch / 2 + 14}" font-size="10" fill="{text_on(col)}" text-anchor="middle">[{lo:+.2f}, {hi:+.2f}]</text></g>')
    # legend (diverging ramp)
    ly = top + ch * 5 + 26
    for k in range(41):
        v = -vmax + 2 * vmax * k / 40
        o.append(f'<rect x="{left + k * 7}" y="{ly}" width="7" height="10" fill="{diverging(v, vmax)}"/>')
    o.append(f'<text x="{left}" y="{ly + 24}" font-size="11" fill="{MUTED}">{-vmax:+.1f}</text><text x="{left + 140}" y="{ly + 24}" font-size="11" fill="{MUTED}" text-anchor="middle">0</text><text x="{left + 287}" y="{ly + 24}" font-size="11" fill="{MUTED}" text-anchor="end">{vmax:+.1f} XI</text>')
    o.append("</svg>")
    return "".join(o)


def dotplot(title, sub, rows, series, xmin, xmax, zero=None, xlabel="validation XI"):
    """rows: [(label, {series: (mean, lo, hi)})]."""
    left, right, top, rh = 150, 30, 70, 30
    w = 760
    h = top + rh * len(rows) + 64
    pw = w - left - right
    X = lambda v: left + (v - xmin) / (xmax - xmin) * pw
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" style="{FONT};background:{SURF}">',
         f'<text x="16" y="24" font-size="15" font-weight="600" fill="{INK}">{escape(title)}</text>',
         f'<text x="16" y="44" font-size="12" fill="{INK2}">{escape(sub)}</text>']
    ticks = 6
    for t in range(ticks + 1):
        v = xmin + (xmax - xmin) * t / ticks
        o.append(f'<line x1="{X(v)}" x2="{X(v)}" y1="{top - 6}" y2="{top + rh * len(rows)}" stroke="{GRID}" stroke-width="1"/>'
                 f'<text x="{X(v)}" y="{top + rh * len(rows) + 16}" font-size="11" fill="{MUTED}" text-anchor="middle">{v:.1f}</text>')
    if zero is not None:
        o.append(f'<line x1="{X(zero)}" x2="{X(zero)}" y1="{top - 6}" y2="{top + rh * len(rows)}" stroke="{AXIS}" stroke-width="1.5"/>')
    o.append(f'<text x="{left + pw / 2}" y="{top + rh * len(rows) + 34}" font-size="11" fill="{MUTED}" text-anchor="middle">{escape(xlabel)}</text>')
    offs = {s: (k - (len(series) - 1) / 2) * 6 for k, s in enumerate(series)}
    for i, (label, vals) in enumerate(rows):
        y = top + rh * i + rh / 2
        o.append(f'<text x="{left - 10}" y="{y + 4}" font-size="12" fill="{INK2}" text-anchor="end">{escape(label)}</text>')
        for s in series:
            if s not in vals or vals[s] is None:
                continue
            m, lo, hi = vals[s]
            yy = y + offs[s]
            o.append(f'<g><title>{escape(label)} · {s}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]</title>'
                     f'<line x1="{X(lo)}" x2="{X(hi)}" y1="{yy}" y2="{yy}" stroke="{SERIES.get(s, INK2)}" stroke-width="2"/>'
                     f'<circle cx="{X(m)}" cy="{yy}" r="4.5" fill="{SERIES.get(s, INK2)}" stroke="{SURF}" stroke-width="2"/></g>')
    lx = left
    for s in series:
        o.append(f'<circle cx="{lx}" cy="56" r="4.5" fill="{SERIES.get(s, INK2)}"/><text x="{lx + 9}" y="60" font-size="11" fill="{INK2}">{escape(s)}</text>')
        lx += 12 + 7.2 * len(s) + 26
    o.append("</svg>")
    return "".join(o)


def strips(title, sub, rows, xmin, xmax):
    """rows: [(label, [9 values], colour)] — one dot per seed pairing."""
    left, right, top, rh = 170, 30, 60, 17
    w = 760
    h = top + rh * len(rows) + 50
    pw = w - left - right
    X = lambda v: left + (v - xmin) / (xmax - xmin) * pw
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" style="{FONT};background:{SURF}">',
         f'<text x="16" y="24" font-size="15" font-weight="600" fill="{INK}">{escape(title)}</text>',
         f'<text x="16" y="44" font-size="12" fill="{INK2}">{escape(sub)}</text>']
    for t in range(7):
        v = xmin + (xmax - xmin) * t / 6
        o.append(f'<line x1="{X(v)}" x2="{X(v)}" y1="{top - 4}" y2="{top + rh * len(rows)}" stroke="{GRID}"/><text x="{X(v)}" y="{top + rh * len(rows) + 16}" font-size="11" fill="{MUTED}" text-anchor="middle">{v:+.1f}</text>')
    if xmin < 0 < xmax:
        o.append(f'<line x1="{X(0)}" x2="{X(0)}" y1="{top - 4}" y2="{top + rh * len(rows)}" stroke="{AXIS}" stroke-width="1.5"/>')
    for i, (label, vals, col) in enumerate(rows):
        y = top + rh * i + rh / 2
        o.append(f'<text x="{left - 10}" y="{y + 4}" font-size="11" fill="{INK2}" text-anchor="end">{escape(label)}</text>')
        o.append(f'<line x1="{X(min(vals))}" x2="{X(max(vals))}" y1="{y}" y2="{y}" stroke="{GRID}" stroke-width="2"/>')
        for v in vals:
            o.append(f'<circle cx="{X(v)}" cy="{y}" r="3.2" fill="{col}" fill-opacity="0.8"><title>{escape(label)}: {v:+.3f}</title></circle>')
    o.append(f'<text x="{left + pw / 2}" y="{top + rh * len(rows) + 34}" font-size="11" fill="{MUTED}" text-anchor="middle">transfer Δ XI of each seed pairing (learner seed × opponent seed)</text></svg>')
    return "".join(o)


def main(report_dir):
    report_dir = Path(report_dir)
    R = json.loads((report_dir / "matchup-results.json").read_text())
    out = report_dir / "plots"
    out.mkdir(exist_ok=True)
    svgs = []
    comps = [c for c in ("C1", "C4") if c in R]
    vt = max(abs(R[c][k]["transfer"]["mean"]) for c in comps for k in R[c]) or 1
    vh = max(abs(R[c][k]["h2h"]["mean"]) for c in comps for k in R[c]) or 1
    for c in comps:
        cells_t = {k: {"mean": v["transfer"]["mean"], "ci95": v["transfer"]["ci95"]} for k, v in R[c].items()}
        cells_h = {k: {"mean": v["h2h"]["mean"], "ci95": v["h2h"]["ci95"]} for k, v in R[c].items()}
        label = "C1 head-to-head (1 opponent seat)" if c == "C1" else "C4 saturated (opponent in all 4 other RL seats)"
        svgs.append((f"{c.lower()}_transfer.svg", heatmap(f"{label}: transfer Δ", "learner Stage-B XI − the same export's Stage-A XI on the same auctions (9 seed pairings × 500)", cells_t, vt)))
        svgs.append((f"{c.lower()}_h2h.svg", heatmap(f"{label}: head-to-head Δ", "learner XI − opponent XI in the same auction" + (" (mean of the 4 opponent seats)" if c == "C4" else ""), cells_h, vh)))
    # Stage A → B XI per algorithm
    rows = []
    lo_all, hi_all = [], []
    for a in ALGOS:
        vals = {}
        sa = R["stageA"][a]
        vals["A"] = (sa["xiMean"], sa["xiMean"] - sa["xiSeedStd"], sa["xiMean"] + sa["xiSeedStd"])
        for c in comps:
            xs = [R[c][f"{a}>{b}"]["learnerXI"] for b in ALGOS if b != a]
            m = sum(x["mean"] for x in xs) / len(xs)
            vals[c] = (m, min(x["mean"] for x in xs), max(x["mean"] for x in xs))
        if "S4" in R and a in R["S4"]:
            x = R["S4"][a]["learnerXI"]
            vals["S4"] = (x["mean"], x["ci95"][0], x["ci95"][1])
        rows.append((NAMES[a], vals))
        for v in vals.values():
            lo_all.append(v[1]); hi_all.append(v[2])
    series = ["A"] + comps + (["S4"] if "S4" in R else [])
    svgs.append(("stage_a_to_b_xi.svg", dotplot("Validation XI: Stage A control vs Stage-B rooms", "A: three-seed mean ± seed std · C1/C4: mean over the 4 opponents (bar = worst to best opponent) · S4: mean, 95% CI", rows, series, min(lo_all) - 0.5, max(hi_all) + 0.5)))
    # seed stability
    for c in comps:
        srows = []
        tmin, tmax = 0, 0
        for a in ALGOS:
            for b in ALGOS:
                if a == b:
                    continue
                vals = [v["transfer"] for v in R[c][f"{a}>{b}"]["seedPairs"].values()]
                tmin, tmax = min(tmin, *vals), max(tmax, *vals)
                srows.append((f"{NAMES[a]} vs {NAMES[b]}", vals, SERIES[c]))
        svgs.append((f"{c.lower()}_seed_stability.svg", strips(f"{c}: seed stability of transfer Δ", "each dot = one learner-seed × opponent-seed pairing (500 auctions); gray bar = range", srows, tmin - 0.3, max(tmax, 0) + 0.3)))
    # S4
    if "S4" in R:
        rows = []
        for a in ALGOS:
            s = R["S4"][a]
            vals = {"transfer": (s["transfer"]["mean"], *s["transfer"]["ci95"])}
            rows.append((NAMES[a], vals))
        mins = [r[1]["transfer"][1] for r in rows]
        SERIES["transfer"] = SERIES["S4"]
        svgs.append(("s4_transfer.svg", dotplot("S4 mixed room (learner + the other four algorithms): transfer Δ", "learner XI in S4 − the same export's Stage-A XI on the same auctions (3 seeds × 500), 95% CI", rows, ["transfer"], min(mins) - 0.5, 0.5, zero=0, xlabel="transfer Δ XI")))
        cells = {}
        for a in ALGOS:
            for b in ALGOS:
                if a != b:
                    w = R["S4"][a]["vsEachAlgorithm"][b]["h2h"]
                    cells[f"{a}>{b}"] = {"mean": w["mean"], "ci95": w["ci95"]}
        vs = max(abs(v["mean"]) for v in cells.values()) or 1
        svgs.append(("s4_h2h.svg", heatmap("S4 mixed room: head-to-head Δ", "learner XI − that algorithm's seat XI in the same auction (3 seeds × 500)", cells, vs)))
    for name, svg in svgs:
        (out / name).write_text(svg, encoding="utf-8")
    page = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
            '<title>Phase 2E.0 cross-play plots</title><style>',
            ':root{--bg:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e}',
            '@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0d0d0d;--ink:#ffffff;--ink2:#c3c2b7}}',
            ':root[data-theme="dark"]{--bg:#0d0d0d;--ink:#ffffff;--ink2:#c3c2b7}',
            f'body{{margin:0;padding:24px 16px;background:var(--bg);color:var(--ink);{FONT}}}',
            'h1{font-size:20px;margin:0 0 6px}p{color:var(--ink2);margin:0 0 20px;max-width:760px;font-size:14px}',
            'figure{margin:0 0 28px;overflow-x:auto}figure svg{max-width:100%;height:auto;border-radius:6px}</style></head><body>',
            '<h1>Phase 2E.0 — Stage B frozen cross-play</h1>',
            '<p>Evaluation only; all five Stage-A exports frozen. Charts are plotted on a light surface in both themes. Hover a cell or dot for its value and 95% CI. No chart ranks the algorithms.</p>']
    for name, svg in svgs:
        page.append(f"<figure>{svg}</figure>")
    page.append("</body></html>")
    (out / "index.html").write_text("".join(page), encoding="utf-8")
    print(f"wrote {len(svgs)} charts to {out}")


if __name__ == "__main__":
    main(sys.argv[1])
