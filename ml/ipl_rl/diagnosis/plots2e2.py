"""Phase 2E.2 plots — every chart from the Phase 2E.2 analysis JSON files
(themselves from digest-verified read-only replays of Phase 2E.0 episodes).

    python plots2e2.py <report_dir>     → <report_dir>/plots/*.svg + index.html
"""
import json
import sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "crossplay"))
import plots as P  # Phase 2E.0 chart helpers / palette  # noqa: E402

FONT = P.FONT
INK, INK2, MUTED, GRID, AXIS, SURF = P.INK, P.INK2, P.MUTED, P.GRID, P.AXIS, P.SURF
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # validated categorical order (Phase 2E.0/2E.1)
COND_COL = {"A": "#52514e", "C1": "#2a78d6", "C4": "#eb6834", "S4": "#1baf7a"}
CHAN_COL = {"none": "#c3c2b7", "own": "#2a78d6", "supply": "#1baf7a", "direct": "#eb6834"}
ALGOS = ["ppo", "a2c", "d3qn", "qrdqn", "es"]
NAMES = {"ppo": "PPO", "a2c": "A2C", "d3qn": "D3QN", "qrdqn": "QR-DQN", "es": "OpenAI-ES"}
LABEL_COL = {"YES": "#1c5cab", "PARTIAL": "#eda100", "NO": "#b22e2e", "UNKNOWN": "#898781"}


def svg_open(w, h, title, sub):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" style="{FONT};background:{SURF}">',
            f'<text x="16" y="24" font-size="15" font-weight="600" fill="{INK}">{escape(title)}</text>',
            f'<text x="16" y="44" font-size="12" fill="{INK2}">{escape(sub)}</text>']


def legend(o, items, x, y):
    for label, col in items:
        o.append(f'<circle cx="{x}" cy="{y}" r="4.5" fill="{col}"/><text x="{x + 9}" y="{y + 4}" font-size="11" fill="{INK2}">{escape(label)}</text>')
        x += 24 + 6.4 * len(label)


def line_chart(title, sub, xlabels, series, ylabel, ymin=0.0, ymax=1.0, fmt="{:.0%}", w=760, h=330, end_labels=True):
    """series: [(label, colour, [values or None])]"""
    left, right, top, bottom = 58, 190 if end_labels else 24, 70, 64
    pw, ph = w - left - right, h - top - bottom
    X = lambda j: left + j * pw / max(1, len(xlabels) - 1)
    Y = lambda v: top + ph - (v - ymin) / ((ymax - ymin) or 1) * ph
    o = svg_open(w, h, title, sub)
    for t in range(5):
        v = ymin + (ymax - ymin) * t / 4
        o.append(f'<line x1="{left}" x2="{left + pw}" y1="{Y(v)}" y2="{Y(v)}" stroke="{GRID}"/><text x="{left - 8}" y="{Y(v) + 4}" font-size="10" fill="{MUTED}" text-anchor="end">{fmt.format(v)}</text>')
    for j, lab in enumerate(xlabels):
        o.append(f'<text x="{X(j)}" y="{top + ph + 16}" font-size="10" fill="{MUTED}" text-anchor="middle">{escape(lab)}</text>')
    used = []
    for label, col, ys in series:
        pts = [(X(j), Y(v)) for j, v in enumerate(ys) if v is not None]
        if not pts:
            continue
        o.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="none" stroke="{col}" stroke-width="2"/>')
        for j, v in enumerate(ys):
            if v is not None:
                o.append(f'<g><title>{escape(label)} @ {escape(xlabels[j])}: {fmt.format(v)}</title><circle cx="{X(j)}" cy="{Y(v)}" r="7" fill="transparent"/><circle cx="{X(j)}" cy="{Y(v)}" r="3" fill="{col}"/></g>')
        if end_labels:
            y = min(pts[-1][1] + 4, top + ph - 2)
            while any(abs(y - u) < 14 for u in used):
                y -= 14
            used.append(y)
            lx = left + pw + 34
            o.append(f'<line x1="{pts[-1][0] + 4}" x2="{lx - 3}" y1="{pts[-1][1]}" y2="{y - 4}" stroke="{col}" stroke-width="1" stroke-dasharray="2 2"/>'
                     f'<text x="{lx}" y="{y}" font-size="10.5" fill="{INK2}">{escape(label)}</text>')
    o.append(f'<text x="{left + pw / 2}" y="{top + ph + 34}" font-size="11" fill="{MUTED}" text-anchor="middle">main-round progress bin</text>')
    o.append(f'<text x="14" y="{top + ph / 2}" font-size="11" fill="{MUTED}" transform="rotate(-90 14 {top + ph / 2})" text-anchor="middle">{escape(ylabel)}</text>')
    legend(o, [(l, c) for l, c, _ in series], left, h - 14)
    o.append("</svg>")
    return "".join(o)


def hbars(title, sub, rows, xmax=1.0, fmt="{:.0%}", w=760, left=330, note_w=0):
    """rows: [(label, value, colour, note)]"""
    top, rh = 64, 26
    h = top + rh * len(rows) + 40
    pw = w - left - 70 - note_w
    X = lambda v: left + v / xmax * pw
    o = svg_open(w, h, title, sub)
    for t in range(5):
        v = xmax * t / 4
        o.append(f'<line x1="{X(v)}" x2="{X(v)}" y1="{top - 4}" y2="{top + rh * len(rows)}" stroke="{GRID}"/><text x="{X(v)}" y="{top + rh * len(rows) + 16}" font-size="10" fill="{MUTED}" text-anchor="middle">{fmt.format(v)}</text>')
    for i, (label, v, col, note) in enumerate(rows):
        y = top + rh * i
        o.append(f'<text x="{left - 10}" y="{y + rh / 2 + 4}" font-size="11.5" fill="{INK2}" text-anchor="end">{escape(label)}</text>')
        if v is None:
            o.append(f'<text x="{left + 4}" y="{y + rh / 2 + 4}" font-size="11" fill="{MUTED}">n/a</text>')
            continue
        bw = max(2.0, X(min(v, xmax)) - left)
        o.append(f'<g><title>{escape(label)}: {fmt.format(v)}{(" — " + escape(note)) if note else ""}</title><rect x="{left}" y="{y + 5}" width="{bw}" height="{rh - 10}" rx="4" fill="{col}"/></g>')
        o.append(f'<text x="{left + bw + 6}" y="{y + rh / 2 + 4}" font-size="11" fill="{INK}">{fmt.format(v)}{("  " + escape(note)) if note else ""}</text>')
    o.append("</svg>")
    return "".join(o)


def dots(title, sub, rows, series, xmin, xmax, xlabel, ref=None, fmt="{:.2f}", w=760, left=250):
    """rows: [(label, {series: value})]; series: [(name, colour)]"""
    top, rh = 70, 26
    h = top + rh * len(rows) + 60
    pw = w - left - 30
    X = lambda v: left + (v - xmin) / (xmax - xmin) * pw
    o = svg_open(w, h, title, sub)
    for t in range(6):
        v = xmin + (xmax - xmin) * t / 5
        o.append(f'<line x1="{X(v)}" x2="{X(v)}" y1="{top - 6}" y2="{top + rh * len(rows)}" stroke="{GRID}"/><text x="{X(v)}" y="{top + rh * len(rows) + 16}" font-size="10" fill="{MUTED}" text-anchor="middle">{fmt.format(v)}</text>')
    if ref is not None:
        o.append(f'<line x1="{X(ref)}" x2="{X(ref)}" y1="{top - 6}" y2="{top + rh * len(rows)}" stroke="{AXIS}" stroke-width="1.5"/>')
    offs = {s: (k - (len(series) - 1) / 2) * 6 for k, (s, _) in enumerate(series)}
    for i, (label, vals) in enumerate(rows):
        y = top + rh * i + rh / 2
        o.append(f'<text x="{left - 10}" y="{y + 4}" font-size="11.5" fill="{INK2}" text-anchor="end">{escape(label)}</text>')
        for s, col in series:
            v = vals.get(s)
            if v is None:
                continue
            o.append(f'<g><title>{escape(label)} · {escape(s)}: {fmt.format(v)}</title><circle cx="{X(max(xmin, min(xmax, v)))}" cy="{y + offs[s]}" r="5" fill="{col}" stroke="{SURF}" stroke-width="2"/></g>')
    o.append(f'<text x="{left + pw / 2}" y="{top + rh * len(rows) + 34}" font-size="11" fill="{MUTED}" text-anchor="middle">{escape(xlabel)}</text>')
    legend(o, series, left, 58)
    o.append("</svg>")
    return "".join(o)


def feature_map(inv):
    """80 tiles, grouped by block, coloured by how opponents can move the feature."""
    cols, cw, ch, left, top = 16, 44, 30, 16, 92
    blocks = ["global", "player", "self", "market", "rivals"]
    feats = inv["features"]
    rows_needed = sum((sum(1 for f in feats if f["block"] == b) + cols - 1) // cols for b in blocks)
    w = left * 2 + cols * cw
    h = top + rows_needed * ch + len(blocks) * 26 + 20
    o = svg_open(w, h, "1 · The 80 observation features, by how opponent behaviour can reach them",
                 "tile = one feature (index); colour = opponent channel from obsSpec.js — hover for the name and definition")
    legend(o, [("fixed by lot/room", CHAN_COL["none"]), ("own purchases only", CHAN_COL["own"]), ("which players went unsold", CHAN_COL["supply"]),
               ("direct: rival state / recent results", CHAN_COL["direct"])], left + 4, 62)
    y = top
    for b in blocks:
        fs = [f for f in feats if f["block"] == b]
        o.append(f'<text x="{left}" y="{y + 12}" font-size="12" font-weight="600" fill="{INK2}">{b} ({len(fs)})</text>')
        y += 18
        for i, f in enumerate(fs):
            x = left + (i % cols) * cw
            yy = y + (i // cols) * ch
            col = CHAN_COL[f["opponentChannel"]]
            o.append(f'<g><title>{f["index"]} {escape(f["name"])} — {escape(f["meaning"])} [{f["opponentChannel"]}, {f["temporalScope"]}]</title>'
                     f'<rect x="{x + 1}" y="{yy + 1}" width="{cw - 2}" height="{ch - 2}" rx="4" fill="{col}"/>'
                     f'<text x="{x + cw / 2}" y="{yy + ch / 2 + 4}" font-size="10.5" fill="{P.text_on(col)}" text-anchor="middle">{f["index"]}</text></g>')
        y += ((len(fs) + cols - 1) // cols) * ch + 8
    o.append("</svg>")
    return "".join(o)


def scorecard(sc):
    rows = sc["scorecard"]
    left, top, rh, w = 16, 64, 28, 760
    h = top + rh * len(rows) + 20
    o = svg_open(w, h, "11 · Information-sufficiency scorecard", "label per Stage-B problem (YES / PARTIAL / NO / UNKNOWN) — hover for the missing information")
    for i, r_ in enumerate(rows):
        y = top + rh * i
        col = LABEL_COL[r_["observable"]]
        o.append(f'<g><title>{escape(r_["problem"])}: {r_["observable"]} — missing: {escape(r_["missingInformation"])}</title>'
                 f'<rect x="{left}" y="{y + 3}" width="92" height="{rh - 6}" rx="4" fill="{col}"/>'
                 f'<text x="{left + 46}" y="{y + rh / 2 + 4}" font-size="11" font-weight="600" fill="{P.text_on(col)}" text-anchor="middle">{r_["observable"]}</text>'
                 f'<text x="{left + 104}" y="{y + rh / 2 + 4}" font-size="12" fill="{INK}">{escape(r_["problem"])}</text>'
                 f'<text x="{w - 16}" y="{y + rh / 2 + 4}" font-size="10.5" fill="{MUTED}" text-anchor="end">confidence: {escape(r_["confidence"])}</text></g>')
    o.append("</svg>")
    return "".join(o)


def categories(sc):
    rows = sc["mechanisms"]
    left, top, rh, w = 16, 64, 28, 760
    h = top + rh * len(rows) + 20
    o = svg_open(w, h, "12 · Failure mechanisms: observability category (Analysis B)",
                 "A direct · B indirect (inferable) · C partial (a critical component missing or coarsened) · D not observable — hover for the evidence")
    for i, r_ in enumerate(rows):
        y = top + rh * i
        col = LABEL_COL[r_["observable"]]
        o.append(f'<g><title>{escape(r_["mechanism"])}: {r_["category"]} — {escape(r_["quality"])}</title>'
                 f'<rect x="{left}" y="{y + 3}" width="40" height="{rh - 6}" rx="4" fill="{col}"/>'
                 f'<text x="{left + 20}" y="{y + rh / 2 + 4}" font-size="12" font-weight="600" fill="{P.text_on(col)}" text-anchor="middle">{r_["category"]}</text>'
                 f'<text x="{left + 54}" y="{y + rh / 2 + 4}" font-size="12" fill="{INK}">{escape(r_["mechanism"])}</text>'
                 f'<text x="{w - 16}" y="{y + rh / 2 + 4}" font-size="10.5" fill="{MUTED}" text-anchor="end">{escape(r_["missing"][:78] + ("…" if len(r_["missing"]) > 78 else ""))}</text></g>')
    o.append("</svg>")
    return "".join(o)


def main(rep):
    rep = Path(rep)
    L = lambda n: json.loads((rep / f"{n}.json").read_text(encoding="utf-8"))
    INV, CF, AL, KO, OO, MO, PS, SC, TA = (L(x) for x in ("observation-inventory", "counterfactual-pairs", "aliasing-analysis", "keeper-observability",
                                                          "opponent-observability", "multi-opponent-analysis", "purse-star-analysis", "information-scorecard", "temporal-analysis"))
    out = [("01_feature_channels.svg", feature_map(INV))]
    conds = ["C1", "C4", "S4"]
    labels = [b["progress"] for b in CF["conditions"]["C4"]["byProgress"]]
    out.append(("02_identical_obs_by_progress.svg", line_chart(
        "2 · Same lot, same own state: how often the Stage-B observation is bit-identical to Stage A",
        "learner decision pairs (same export, entry, lot; similar own purse/squad) — share with every feature equal", labels,
        [(c, COND_COL[c], [b["identicalObs"] for b in CF["conditions"][c]["byProgress"]]) for c in conds], "share of pairs with identical observation")))
    out.append(("03_material_aliased_by_progress.svg", line_chart(
        "3 · Materially different competition, (almost) the same observation",
        f"pairs where the price needed to win differs by ≥ {CF['meta']['thresholds']['materialPrice']} × fair value — share with L∞ ≤ {CF['meta']['thresholds']['aliasTight']} over all 80 features", labels,
        [(c, COND_COL[c], [b["material_linfLeTight"] for b in CF["conditions"][c]["byProgress"]]) for c in conds], "share of material pairs aliased", ymax=0.2)))
    sep = {x["feature"]: x for x in OO["separation"]}
    feats = ["mkt_recent_price_ratio", "riv_recent_spend", "mkt_recent_sold_share", "riv_purse_mean", "riv_purse_min"]
    out.append(("04_signal_separation.svg", line_chart(
        "4 · When do opponent signals in the observation separate a high-pressure room from Stage A?",
        "|2·AUC − 1| between C4 D3QN/QR-DQN rooms and Stage-A rooms, learner observations in each progress bin (0 = same distribution)", labels,
        [(f, CAT[i], [b["sep_C4high_vs_A"] for b in sep[f]["byProgress"]]) for i, f in enumerate(feats)], "separation", end_labels=True)))
    tr = OO["hiddenTracking"]
    out.append(("05_hidden_tracking.svg", hbars(
        "5 · How well the best single feature tracks what opponents actually did",
        "largest |Spearman ρ| between a hidden quantity and any of the 80 features, within progress bins (learner decisions, main round)",
        [(k, v["bestAbs"], CAT[0] if (v["bestAbs"] or 0) >= 0.5 else CAT[3] if (v["bestAbs"] or 0) >= 0.2 else CAT[1], v["top"][0]["feature"] if v["top"] else "") for k, v in tr.items()],
        xmax=1.0, fmt="{:.2f}", left=380, note_w=150)))
    km = KO["summary"]
    rows = [(n, {"observed feature": v}) for n, v in km["medianPercentileObserved"].items()] + [(n + " (hidden)", {"hidden rival state": v}) for n, v in km["medianPercentileHidden"].items()]
    out.append(("06_keeper_percentiles.svg", dots(
        "6 · Keeper-failure decisions vs matched safe keeper decisions",
        "median percentile of the dangerous decision within matched safe references (0.5 = typical); observed features vs hidden rival keeper state",
        rows, [("observed feature", CAT[0]), ("hidden rival state", CAT[1])], 0, 1, "percentile among matched safe decisions", ref=0.5, fmt="{:.1f}")))
    am = AL["mechanisms"]
    out.append(("07_aliasing_rates.svg", hbars(
        "7 · Aliasing: materially different situations with near-identical observations",
        f"share of materially different pairs (same entry, lot, phase) whose observations are within L∞ ≤ {AL['meta']['thresholds']['aliasTight']}",
        [(k.replace("_", " "), v["aliasRateTight"], CAT[1] if (v["aliasRateTight"] or 0) >= 0.05 else CAT[0], f"{v['aliasedTight']:,} / {v['materialPairs']:,}") for k, v in am.items()],
        xmax=max(0.05, max((v["aliasRateTight"] or 0) for v in am.values()) * 1.15), fmt="{:.1%}", left=240, note_w=150)))
    su = OO["stageASupport"]
    out.append(("08_stage_a_support.svg", dots(
        "8 · Stage-B observations outside the Stage-A range",
        "share of learner observations with at least one feature outside the Stage-A 0.5–99.5th percentile (same 40 entries)",
        [(k, {"any feature": v["anyFeatureOutside"], "opponent-direct feature": v["anyDirectFeatureOutside"], "own-state feature": v["anyOwnFeatureOutside"]}) for k, v in su.items()],
        [("any feature", CAT[0]), ("opponent-direct feature", CAT[1]), ("own-state feature", CAT[2])], 0, 1, "share of observations", fmt="{:.0%}")))
    pr = OO["policyResponse"]["C4"]["byLearnerAlgorithm"]
    out.append(("09_policy_response.svg", dots(
        "9 · What the frozen caps move with (C4 vs Stage A, same lot, similar own state)",
        "Spearman ρ between the change in the learner's cap/fair and the change in a feature or in the hidden price needed to win",
        [(NAMES[a], {"Δ recent price ratio (obs)": pr[a]["dCap_vs_dRecentPriceRatio"], "Δ own purse (obs)": pr[a]["dCap_vs_dOwnPurse"], "Δ price needed (hidden)": pr[a]["dCap_vs_dNeedToWin"]}) for a in ALGOS],
        [("Δ recent price ratio (obs)", CAT[1]), ("Δ own purse (obs)", CAT[0]), ("Δ price needed (hidden)", CAT[2])], -0.6, 0.6, "Spearman ρ", ref=0.0)))
    bo = MO["byOpponent"]
    out.append(("10_one_vs_four.svg", hbars(
        "10 · One aggressive copy vs four: does the rival block tell them apart?",
        "C1 vs C4 same-lot pairs with ≥ 2 more RL bidders: share whose 17 rival + window features are all within 0.02",
        [(NAMES[a], bo[a]["rlBiddersDiffer2plus_rivalBlockLeTight"], CAT[i], f"median rival-block L∞ {bo[a]['medianRivalBlockLinf']}") for i, a in enumerate(ALGOS)],
        xmax=1.0, fmt="{:.0%}", left=140, note_w=190)))
    out.append(("11_scorecard.svg", scorecard(SC)))
    out.append(("12_mechanism_categories.svg", categories(SC)))
    pdir = rep / "plots"
    pdir.mkdir(exist_ok=True)
    for name, svg in out:
        (pdir / name).write_text(svg, encoding="utf-8")
    page = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Phase 2E.2 observability plots</title><style>',
            ':root{--bg:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e}@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0d0d0d;--ink:#fff;--ink2:#c3c2b7}}:root[data-theme="dark"]{--bg:#0d0d0d;--ink:#fff;--ink2:#c3c2b7}',
            f'body{{margin:0;padding:24px 16px;background:var(--bg);color:var(--ink);{FONT}}}h1{{font-size:20px;margin:0 0 6px}}p{{color:var(--ink2);max-width:780px;font-size:14px}}figure{{margin:0 0 28px;overflow-x:auto}}figure svg{{max-width:100%;height:auto;border-radius:6px}}</style></head><body>',
            '<h1>Phase 2E.2 — observation sufficiency diagnosis</h1><p>All charts come from digest-verified read-only replays of recorded Phase 2E.0 episodes. Nothing was trained or changed. Charts sit on a light surface in both themes; hover for values and definitions.</p>']
    page += [f"<figure>{svg}</figure>" for _, svg in out]
    page.append("</body></html>")
    (pdir / "index.html").write_text("".join(page), encoding="utf-8")
    print(f"wrote {len(out)} charts")


if __name__ == "__main__":
    main(sys.argv[1])
