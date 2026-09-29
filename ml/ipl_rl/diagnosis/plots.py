"""Phase 2E.1 plots — every chart from the frozen Phase 2E.0 data (or its
digest-verified replays), via the analysis JSON files.

    python plots.py <report_dir>     → <report_dir>/plots/*.svg + index.html
"""
import json
import sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "crossplay"))
import plots as P  # Phase 2E.0 chart helpers (heatmap, dotplot, strips, palette)  # noqa: E402

ALGOS = P.ALGOS
NAMES = P.NAMES
FONT = P.FONT
INK, INK2, MUTED, GRID, AXIS, SURF = P.INK, P.INK2, P.MUTED, P.GRID, P.AXIS, P.SURF
COND_COL = {"A": "#52514e", "C1": "#2a78d6", "C4": "#eb6834", "S4": "#1baf7a"}
ALGO_COL = {"ppo": "#2a78d6", "a2c": "#eb6834", "d3qn": "#1baf7a", "qrdqn": "#eda100", "es": "#e87ba4"}
CONDS = ["A", "C1", "C4", "S4"]


def svg_open(w, h, title, sub):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" style="{FONT};background:{SURF}">',
            f'<text x="16" y="24" font-size="15" font-weight="600" fill="{INK}">{escape(title)}</text>',
            f'<text x="16" y="44" font-size="12" fill="{INK2}">{escape(sub)}</text>']


def legend(o, items, x, y):
    for label, col in items:
        o.append(f'<circle cx="{x}" cy="{y}" r="4.5" fill="{col}"/><text x="{x + 9}" y="{y + 4}" font-size="11" fill="{INK2}">{escape(label)}</text>')
        x += 24 + 6.6 * len(label)


def slope(title, sub, series_by_algo, conds, ylabel, fmt="{:.2f}", y_pad=0.3):
    """One small panel per algorithm: value across conditions (A → C1 → C4 → S4)."""
    pw, ph, left, top, gap = 150, 170, 50, 76, 22
    w = left + 5 * (pw + gap) + 10
    h = top + ph + 58
    vals = [v for a in ALGOS for v in series_by_algo[a] if v is not None]
    lo, hi = min(vals) - y_pad, max(vals) + y_pad
    Y = lambda v: top + ph - (v - lo) / (hi - lo) * ph
    o = svg_open(w, h, title, sub)
    for t in range(5):
        v = lo + (hi - lo) * t / 4
        o.append(f'<line x1="{left - 4}" x2="{w - 10}" y1="{Y(v)}" y2="{Y(v)}" stroke="{GRID}"/><text x="{left - 8}" y="{Y(v) + 4}" font-size="10" fill="{MUTED}" text-anchor="end">{fmt.format(v)}</text>')
    for i, a in enumerate(ALGOS):
        x0 = left + i * (pw + gap)
        X = lambda j: x0 + 14 + j * (pw - 28) / (len(conds) - 1)
        ys = series_by_algo[a]
        pts = [(X(j), Y(v)) for j, v in enumerate(ys) if v is not None]
        o.append(f'<text x="{x0 + pw / 2}" y="{top - 12}" font-size="12" font-weight="600" fill="{INK2}" text-anchor="middle">{NAMES[a]}</text>')
        o.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="none" stroke="{ALGO_COL[a]}" stroke-width="2"/>')
        for j, v in enumerate(ys):
            if v is None:
                continue
            o.append(f'<g><title>{NAMES[a]} {conds[j]}: {fmt.format(v)}</title><circle cx="{X(j)}" cy="{Y(v)}" r="4.5" fill="{COND_COL.get(conds[j], ALGO_COL[a])}" stroke="{SURF}" stroke-width="2"/></g>')
            o.append(f'<text x="{X(j)}" y="{Y(v) - 9}" font-size="9.5" fill="{INK2}" text-anchor="middle">{fmt.format(v)}</text>')
            o.append(f'<text x="{X(j)}" y="{top + ph + 16}" font-size="10" fill="{MUTED}" text-anchor="middle">{conds[j]}</text>')
    o.append(f'<text x="14" y="{top + ph / 2}" font-size="11" fill="{MUTED}" transform="rotate(-90 14 {top + ph / 2})" text-anchor="middle">{escape(ylabel)}</text>')
    legend(o, [(c, COND_COL[c]) for c in conds if c in COND_COL], left, top + ph + 42)
    o.append("</svg>")
    return "".join(o)


def lines(title, sub, curves, ylabel, fmt="{:.2f}", ymin=None, ymax=None):
    """curves: {algo: {cond: [11 values]}} — small multiples, A vs S4 over auction progress."""
    pw, ph, left, top, gap = 150, 160, 50, 76, 22
    w = left + 5 * (pw + gap) + 10
    h = top + ph + 62
    vals = [v for a in curves for c in curves[a] for v in curves[a][c] if v is not None]
    lo = min(vals) if ymin is None else ymin
    hi = max(vals) if ymax is None else ymax
    Y = lambda v: top + ph - (v - lo) / ((hi - lo) or 1) * ph
    labels = ["0.1", "", "", "", "0.5", "", "", "", "", "1.0", "RA"]
    o = svg_open(w, h, title, sub)
    for t in range(5):
        v = lo + (hi - lo) * t / 4
        o.append(f'<line x1="{left - 4}" x2="{w - 10}" y1="{Y(v)}" y2="{Y(v)}" stroke="{GRID}"/><text x="{left - 8}" y="{Y(v) + 4}" font-size="10" fill="{MUTED}" text-anchor="end">{fmt.format(v)}</text>')
    for i, a in enumerate(ALGOS):
        if a not in curves:
            continue
        x0 = left + i * (pw + gap)
        X = lambda j: x0 + 6 + j * (pw - 12) / 10
        o.append(f'<text x="{x0 + pw / 2}" y="{top - 12}" font-size="12" font-weight="600" fill="{INK2}" text-anchor="middle">{NAMES[a]}</text>')
        for j, lab in enumerate(labels):
            if lab:
                o.append(f'<text x="{X(j)}" y="{top + ph + 16}" font-size="9.5" fill="{MUTED}" text-anchor="middle">{lab}</text>')
        for c, ys in curves[a].items():
            pts = " ".join(f"{X(j):.1f},{Y(v):.1f}" for j, v in enumerate(ys))
            o.append(f'<polyline points="{pts}" fill="none" stroke="{COND_COL[c]}" stroke-width="2"/>')
            for j, v in enumerate(ys):
                o.append(f'<circle cx="{X(j)}" cy="{Y(v)}" r="6" fill="transparent"><title>{NAMES[a]} {c} @ {labels[j] or f"0.{j + 1}"}: {fmt.format(v)}</title></circle>')
            o.append(f'<text x="{X(10) + 3}" y="{Y(ys[-1]) + (11 if c == "A" else -3)}" font-size="9.5" fill="{COND_COL[c]}">{c}</text>')
    o.append(f'<text x="{left + (w - left) / 2}" y="{top + ph + 32}" font-size="11" fill="{MUTED}" text-anchor="middle">main-round progress (RA = end of the re-auction)</text>')
    o.append(f'<text x="14" y="{top + ph / 2}" font-size="11" fill="{MUTED}" transform="rotate(-90 14 {top + ph / 2})" text-anchor="middle">{escape(ylabel)}</text>')
    legend(o, [(c, COND_COL[c]) for c in ("A", "S4")], left, top + ph + 50)
    o.append("</svg>")
    return "".join(o)


def grouped(title, sub, data, conds, xlabel, fmt="{:.2f}", zero=None):
    """data: {algo: {cond: (mean, lo, hi) or value}} → dot plot rows per algorithm."""
    rows = []
    for a in ALGOS:
        vals = {}
        for c in conds:
            v = data[a].get(c)
            if v is None:
                continue
            vals[c] = v if isinstance(v, tuple) else (v, v, v)
        rows.append((NAMES[a], vals))
    xs = [x for _, v in rows for t in v.values() for x in t if x is not None]
    lo, hi = min(xs), max(xs)
    pad = (hi - lo) * 0.08 or 0.1
    old = dict(P.SERIES)
    P.SERIES.update(COND_COL)
    s = P.dotplot(title, sub, rows, conds, lo - pad, hi + pad, zero=zero, xlabel=xlabel)
    P.SERIES.clear()
    P.SERIES.update(old)
    return s


def rel_heatmap(title, sub, rows, cols, M, fmt="{:+.0%}", vmax=None):
    cw, ch, left, top = 118, 30, 230, 78
    w, h = left + cw * len(cols) + 20, top + ch * len(rows) + 30
    vals = [abs(M[r][c]) for r in rows for c in cols if M[r][c] is not None]
    vmax = vmax or max(vals)
    o = svg_open(w, h, title, sub)
    for j, c in enumerate(cols):
        o.append(f'<text x="{left + cw * j + cw / 2}" y="{top - 10}" font-size="12" fill="{INK2}" text-anchor="middle">{NAMES.get(c, c)}</text>')
    for i, r in enumerate(rows):
        o.append(f'<text x="{left - 10}" y="{top + ch * i + ch / 2 + 4}" font-size="11.5" fill="{INK2}" text-anchor="end">{escape(r)}</text>')
        for j, c in enumerate(cols):
            v = M[r][c]
            x, y = left + cw * j, top + ch * i
            if v is None:
                o.append(f'<rect x="{x + 1}" y="{y + 1}" width="{cw - 2}" height="{ch - 2}" rx="4" fill="{SURF}" stroke="{GRID}"/>')
                continue
            col = P.diverging(v, vmax)
            o.append(f'<g><title>{escape(r)} · {NAMES.get(c, c)}: {fmt.format(v)}</title><rect x="{x + 1}" y="{y + 1}" width="{cw - 2}" height="{ch - 2}" rx="4" fill="{col}"/>'
                     f'<text x="{x + cw / 2}" y="{y + ch / 2 + 4}" font-size="11.5" fill="{P.text_on(col)}" text-anchor="middle">{fmt.format(v)}</text></g>')
    o.append("</svg>")
    return "".join(o)


def keeper_timeline(K):
    rows = K["findings"]
    rows = sorted(rows, key=lambda x: (x["mechanism"][:2], x["affected"], x["seed"]))
    left, right, top, rh = 290, 30, 84, 26
    w = 980
    h = top + rh * len(rows) + 70
    pw = w - left - right
    X = lambda p: left + min(p, 1.1) / 1.1 * pw
    o = svg_open(w, h, "The 20 incomplete-XI findings (C4): keeper opportunities of the affected seat",
                 "gray line = purse share left at the seat's decisions · ○ passed a keeper while SAFE · ● keeper bid lost · ◆ forced keeper bid (lost) · x-axis: progress (1.0–1.1 = re-auction)")
    for p in (0, 0.25, 0.5, 0.75, 1.0):
        o.append(f'<line x1="{X(p)}" x2="{X(p)}" y1="{top - 8}" y2="{top + rh * len(rows)}" stroke="{GRID}"/><text x="{X(p)}" y="{top + rh * len(rows) + 16}" font-size="10" fill="{MUTED}" text-anchor="middle">{p:g}</text>')
    o.append(f'<text x="{X(1.05)}" y="{top + rh * len(rows) + 16}" font-size="10" fill="{MUTED}" text-anchor="middle">re-auction</text>')
    for i, x in enumerate(rows):
        y = top + rh * i + rh / 2
        mech = x["mechanism"][:2]
        o.append(f'<text x="{left - 10}" y="{y + 4}" font-size="11" fill="{INK2}" text-anchor="end">{mech} · {x["affected"]} ({"learner" if x["affectedType"] == "learner" else "opp"}) · {x["seed"]}</text>')
        tr = x["_trace"]
        pts = " ".join(f"{X(p):.1f},{y + rh * 0.4 - s * rh * 0.8:.1f}" for p, s in tr)
        o.append(f'<polyline points="{pts}" fill="none" stroke="{AXIS}" stroke-width="1.5"/>')
        for ev in x["_events"]:
            p, kind = ev
            if kind == "pass":
                o.append(f'<circle cx="{X(p)}" cy="{y}" r="3.5" fill="none" stroke="#2a78d6" stroke-width="1.5"/>')
            elif kind == "lost":
                o.append(f'<circle cx="{X(p)}" cy="{y}" r="3.5" fill="#52514e"/>')
            elif kind == "forced":
                o.append(f'<rect x="{X(p) - 4}" y="{y - 4}" width="8" height="8" fill="#d03b3b" transform="rotate(45 {X(p)} {y})"/>')
    o.append("</svg>")
    return "".join(o)


def main(rep):
    rep = Path(rep)
    L = lambda n: json.loads((rep / f"{n}.json").read_text(encoding="utf-8"))
    TA, OE, CC, BS, KF, SH, PA, RA, SA, SS = (L(x) for x in ("transfer-analysis", "opponent-effects", "c1-c4-analysis", "behavior-shifts", "keeper-failures",
                                                              "shield-analysis", "purse-analysis", "requirement-analysis", "stratum-analysis", "seed-stability"))
    MR = json.loads((rep.parents[0] / "phase2e0" / "matchup-results.json").read_text(encoding="utf-8"))
    out = []
    by = TA["byAlgorithm"]
    # 1 XI by condition
    out.append(("01_xi_by_condition.svg", slope("1 · Validation XI: Stage A → C1 → C4 → S4", "mean over all episodes of each algorithm (C1/C4: all opponents and seed pairings)",
                                                {a: [by[a]["xi"][c] for c in CONDS] for a in ALGOS}, CONDS, "Best XI")))
    # 2 transfer heatmaps (from Phase 2E.0 cells)
    for c in ("C1", "C4"):
        cells = {k: {"mean": v["transfer"]["mean"], "ci95": v["transfer"]["ci95"]} for k, v in MR[c].items()}
        vt = max(abs(MR[x][k]["transfer"]["mean"]) for x in ("C1", "C4") for k in MR[x])
        out.append((f"02_transfer_{c}.svg", P.heatmap(f"2 · Transfer Δ, {c}", "learner Stage-B XI − the same export's Stage-A XI on the same auctions", cells, vt)))
    # 3 opponent-induced degradation (C1) — matrix plus opponent / learner main effects
    mat = {k: {"mean": v["deltaXI"]["mean"], "ci95": v["deltaXI"]["ci95"]} for k, v in OE["C1matrix"].items()}
    s3 = P.heatmap("3 · Opponent-induced degradation (C1)", f"additive fit learner + opponent explains {OE['varianceExplained']['additiveLearnerPlusOpponent']:.0%} of the cell variance "
                   f"(opponent alone {OE['varianceExplained']['opponentOnly']:.0%}, learner alone {OE['varianceExplained']['learnerOnly']:.0%})", mat, max(abs(v["mean"]) for v in mat.values()))
    oe = OE["opponentMainEffect"]
    extra = "".join(f'<text x="{150 + 112 * j + 56}" y="{100 + 46 * 5 + 62}" font-size="11" fill="{INK2}" text-anchor="middle">mean {oe[b]:+.2f}</text>' for j, b in enumerate(ALGOS))
    s3 = s3.replace("</svg>", f'<text x="16" y="{100 + 46 * 5 + 62}" font-size="11" fill="{MUTED}">column means →</text>{extra}</svg>')
    s3 = s3.replace(f'height="{100 + 46 * 5 + 70}"', f'height="{100 + 46 * 5 + 80}"').replace(f'viewBox="0 0 734 {100 + 46 * 5 + 70}"', f'viewBox="0 0 734 {100 + 46 * 5 + 80}"')
    out.append(("03_opponent_degradation_C1.svg", s3))
    # 4 escalation
    esc = {k: {"mean": v["escalation"], "ci95": v["escalationCI95"]} for k, v in CC["pairs"].items()}
    out.append(("04_escalation_C4_minus_C1.svg", P.heatmap("4 · C1 → C4 escalation: C4 transfer − C1 transfer", "extra XI change when the opponent is multiplied from one RL seat to four (more red = disproportionate)",
                                                           esc, max(abs(v["mean"]) for v in esc.values()))))
    # 5, 6 bid rate and price/fair shifts
    out.append(("05_bid_rate_shift.svg", slope("5 · Bid rate by condition", "share of the learner's decisions that bid", {a: [by[a]["bidRate"][c] for c in CONDS] for a in ALGOS}, CONDS, "bid rate", "{:.2f}", 0.05)))
    out.append(("06_price_fair_shift.svg", slope("6 · Price paid / fair value by condition", "mean over the learner's purchases", {a: [by[a]["priceToFair"][c] for c in CONDS] for a in ALGOS}, CONDS, "price / fair", "{:.2f}", 0.05)))
    # 7, 8 purse and squad trajectories
    cur = PA["curves"]
    out.append(("07_purse_over_progress.svg", lines("7 · Purse remaining over auction progress: Stage A vs S4", "share of the starting purse left (learner), mean over 1,500 auctions per algorithm and condition",
                                                    {a: {c: [1 - v for v in cur[c][a]["spend"]] for c in ("A", "S4")} for a in ALGOS}, "purse left (share)", "{:.2f}", 0, 1)))
    out.append(("08_squad_over_progress.svg", lines("8 · Squad size over auction progress: Stage A vs S4", "players bought (learner), mean",
                                                    {a: {c: cur[c][a]["squad"] for c in ("A", "S4")} for a in ALGOS}, "squad size", "{:.0f}", 0, 25)))
    # 9 requirement timing (keeper median)
    raw = RA["fromRawRecords"]
    for q, label in (("keeper", "keeper"), ("indians", "Indian requirement")):
        out.append((f"09_{q}_completion.svg", grouped(f"9 · {label.capitalize()} completion timing", "mean auction progress at completion (0–1 main round; 1 = re-auction); bar = inter-quartile range",
                                                      {a: {c: (raw[a][c][q]["mean"], raw[a][c][q]["p25"], raw[a][c][q]["p75"]) for c in CONDS} for a in ALGOS}, CONDS, "progress at completion")))
    # 10 shield activation, 11 forced keeper
    g = SH["byAlgorithm"]
    out.append(("10_shield_forced_per_episode.svg", grouped("10 · Shield-forced bids per episode", "mean forced bids per learner episode",
                                                            {a: {c: g[a][c]["forcedPerEpisode"] for c in CONDS} for a in ALGOS}, CONDS, "forced bids / episode", zero=0)))
    out.append(("11_forced_keeper_purchases.svg", grouped("11 · Keeper requirement closed by a forced bid (per 500 episodes)", "learner",
                                                          {a: {c: g[a][c]["keeperClosedByForcedBidPer500"] for c in CONDS} for a in ALGOS}, CONDS, "per 500 episodes", zero=0)))
    # 12 stratum
    st = SA["byAlgorithmAndCondition"]
    old = dict(P.SERIES)
    P.SERIES.update({"low": "#2a78d6", "normal": "#eb6834", "high": "#1baf7a"})
    rows = []
    for a in ALGOS:
        for c in ("C1", "C4", "S4"):
            rows.append((f"{NAMES[a]} {c}", {s: (st[a][c][s]["transfer"]["mean"], *st[a][c][s]["transfer"]["ci95"]) for s in ("low", "normal", "high")}))
    xs = [v[1] for _, r in rows for v in r.values()] + [v[2] for _, r in rows for v in r.values()]
    out.append(("12_degradation_by_stratum.svg", P.dotplot("12 · Transfer Δ by purse stratum", "learner, mean with 95% CI", rows, ["low", "normal", "high"], min(xs) - 0.3, 0.3, zero=0, xlabel="transfer Δ XI")))
    P.SERIES.clear()
    P.SERIES.update(old)
    # 13 seed variability
    srows = []
    for c in ("C1", "C4"):
        for a in ALGOS:
            for b, cell in SS[c][a]["cells"].items():
                srows.append((f"{c} {NAMES[a]} vs {NAMES[b]}", [x for r in cell["matrix"] for x in r], COND_COL[c]))
    vv = [x for _, v, _ in srows for x in v]
    out.append(("13_seed_variability.svg", P.strips("13 · Seed variability of transfer Δ (9 seed pairings per cell)", "each dot = one learner-seed × opponent-seed pairing; C1 blue, C4 orange", srows, min(vv) - 0.3, 0.3)))
    # 14 behaviour comparison (relative change S4 vs A)
    metrics = [("bidRate", "bid rate"), ("priceToFair", "price/fair"), ("capToFair", "cap/fair"), ("stars", "stars"), ("marginalBuys", "marginal buys"), ("squadSize", "squad size"),
               ("overseas", "overseas"), ("purseLeftShare", "purse left"), ("keeperProg", "keeper completion"), ("indiansProg", "Indian completion"), ("forced", "forced bids"), ("xi", "Best XI")]
    M = {}
    for m, label in metrics:
        M[label] = {}
        for a in ALGOS:
            base, v = by[a][m]["A"], by[a][m]["S4"]
            M[label][a] = None if base in (None, 0) or v is None else (v - base) / abs(base) if m not in ("forced", "purseLeftShare") else (v - base)
    s14 = rel_heatmap("14 · Behaviour change S4 vs Stage A, by algorithm", "relative change (forced bids and purse left: absolute change); diverging colour = direction only",
                      [l for _, l in metrics], ALGOS, M, fmt="{:+.0%}", vmax=1.0)
    out.append(("14_behaviour_comparison.svg", s14))
    # 15 keeper failures
    for x in KF["findings"]:
        tr = x.get("_trace_points") or []
        x["_trace"] = tr
        x["_events"] = x.get("_event_points") or []
    out.append(("15_keeper_failures.svg", keeper_timeline(KF)))
    pdir = rep / "plots"
    pdir.mkdir(exist_ok=True)
    for name, svg in out:
        (pdir / name).write_text(svg, encoding="utf-8")
    page = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Phase 2E.1 diagnosis plots</title><style>',
            ':root{--bg:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e}@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0d0d0d;--ink:#fff;--ink2:#c3c2b7}}:root[data-theme="dark"]{--bg:#0d0d0d;--ink:#fff;--ink2:#c3c2b7}',
            f'body{{margin:0;padding:24px 16px;background:var(--bg);color:var(--ink);{FONT}}}h1{{font-size:20px;margin:0 0 6px}}p{{color:var(--ink2);max-width:780px;font-size:14px}}figure{{margin:0 0 28px;overflow-x:auto}}figure svg{{max-width:100%;height:auto;border-radius:6px}}</style></head><body>',
            '<h1>Phase 2E.1 — Stage-B robustness diagnosis</h1><p>All charts come from the frozen Phase 2E.0 records or their digest-verified read-only replays. Charts sit on a light surface in both themes; hover for values. Associations only — no chart ranks the algorithms.</p>']
    page += [f"<figure>{svg}</figure>" for _, svg in out]
    page.append("</body></html>")
    (pdir / "index.html").write_text("".join(page), encoding="utf-8")
    print(f"wrote {len(out)} charts")


if __name__ == "__main__":
    main(sys.argv[1])
