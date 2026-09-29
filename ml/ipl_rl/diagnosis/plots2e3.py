"""Phase 2E.3 plots — from the Phase 2E.3 JSON files (digest-verified read-only replays).

    python plots2e3.py <report_dir>     → <report_dir>/plots/*.svg + index.html
"""
import json
import sys
from html import escape
from pathlib import Path

import plots2e2 as Q

CAT, INK, INK2, MUTED, SURF, FONT = Q.CAT, Q.INK, Q.INK2, Q.MUTED, Q.SURF, Q.FONT
CLS_COL = {"KEEP": "#1c5cab", "TRANSFORM": "#eda100", "ADD": "#b22e2e", "DO NOT ADD": "#898781"}


def delta_table(md):
    rows = md["delta"]
    left, top, rh, w = 16, 64, 34, 760
    h = top + rh * len(rows) + 40
    o = Q.svg_open(w, h, "9 · Minimum observation delta (design proposal only)", "classification per diagnosed gap — ADD: 0 · TRANSFORM: 0; hover for the reason")
    for i, r_ in enumerate(rows):
        y = top + rh * i
        col = CLS_COL[r_["classification"]]
        o.append(f'<g><title>{escape(r_["item"])}: {r_["classification"]} — {escape(r_["reason"])}</title>'
                 f'<rect x="{left}" y="{y + 4}" width="104" height="{rh - 8}" rx="4" fill="{col}"/>'
                 f'<text x="{left + 52}" y="{y + rh / 2 + 4}" font-size="11" font-weight="600" fill="#ffffff" text-anchor="middle">{r_["classification"]}</text>'
                 f'<text x="{left + 116}" y="{y + rh / 2 + 4}" font-size="12" fill="{INK}">{escape(r_["item"])}</text></g>')
    o.append(f'<text x="{left}" y="{h - 14}" font-size="11" fill="{INK2}">Proposed observation delta: {md["proposedObservationDelta"]["size"]} features. obs-v2 (629b25783f833af7) unchanged.</text>')
    o.append("</svg>")
    return "".join(o)


def main(rep):
    rep = Path(rep)
    L = lambda n: json.loads((rep / f"{n}.json").read_text(encoding="utf-8"))
    EA, KG, MG, SA, CT, MD = (L(x) for x in ("early-aggression-analysis", "keeper-gap-analysis", "multi-opponent-gap-analysis", "saturation-analysis", "counterfactual-tests", "minimum-delta"))
    out = []
    for n, (lab, title) in enumerate([("C4 D3QN/QR-DQN room vs Stage-A room", "1 · Early auction: C4 D3QN/QR-DQN room vs Stage-A room"),
                                      ("future-aggressive vs future-passive room (realised star price/fair, lots 33–162, top vs bottom tercile)", "2 · Early auction: rooms that later become aggressive vs stay passive")]):
        bins = EA["separationByLotBin"][lab]
        xl = [b["bin"] for b in bins]
        best_c = [max(v for v in b["candidates"].values() if v is not None) if any(v is not None for v in b["candidates"].values()) else 0.0 for b in bins]
        add_c = [max(v for v in b["candidatesWithinBestExistingDeciles"].values() if v is not None) if any(v is not None for v in b["candidatesWithinBestExistingDeciles"].values()) else 0.0 for b in bins]
        out.append((f"0{n + 1}_early_separation.svg", Q.line_chart(title, "|2·AUC − 1| by lot bin: best of the 80 features vs best history candidate (lot 1: all identical)", xl,
                                                                   [("best existing", CAT[0], [b["bestExisting"]["separation"] for b in bins]),
                                                                    ("best candidate", CAT[1], best_c),
                                                                    ("cand. within deciles", CAT[2], add_c),
                                                                    ("identity (excluded)", "#898781", [b["identityOracle_rlRivals"] for b in bins])],
                                                                   "separation", fmt="{:.1f}")))
    cur = [x for x in KG["absorptionByNonNeedingContestants"] if x["rate"] is not None]
    out.append(("03_keeper_absorption.svg", Q.hbars("3 · Keeper lots taken by a rival that already had a keeper",
                                                    "keeper decisions while needing one, ≤ 1 rival needing: rate by number of non-needing rivals able to buy the lot",
                                                    [(f"{x['kContestNoNeed']} non-needing able rivals", x["rate"], CAT[1], f"n = {x['rows']:,}") for x in cur], xmax=0.4, fmt="{:.0%}", left=210, note_w=90)))
    fc = KG["fineCells_plus_rivAble_rivFreeSlots"]
    rows = [(c, v["withinCellSep"], CAT[1], "candidate") for c, v in fc["candidates"].items()] + [(f, v, CAT[0], "existing") for f, v in fc["existingOutsideCells"].items()]
    rows.sort(key=lambda z: -(z[1] or 0))
    out.append(("04_keeper_fine_cells.svg", Q.hbars("4 · Keeper: separation added inside cells of the existing features",
                                                    "outcome: keeper taken by a non-needing rival; cells fix progress, purse, value, supply, rival need and ability",
                                                    rows, xmax=0.6, fmt="{:.2f}", left=230, note_w=80)))
    cons = KG["consistency"]
    names = ["kContestNoNeed", "kHistExtraBuyers", "kStockExcess", "self_purse", "mkt_scarcity_keeper"]
    fr = []
    for f in KG["findings"]:
        fr.append((f"k{f['k']} {f['room'].split(' vs')[0]} ({f['mechanism']})", {n: f["candidatePercentiles"].get(n) for n in names}))
    out.append(("05_keeper_findings.svg", Q.dots("5 · The 20 keeper findings: where the dangerous decisions sit among matched safe ones",
                                                 "median percentile (0.5 = typical of safe; a separating signal sits near 0 or 1); orange/green/pink = candidates, blue/yellow = existing features",
                                                 fr, [("kContestNoNeed", CAT[1]), ("kHistExtraBuyers", CAT[2]), ("kStockExcess", CAT[4]), ("self_purse", CAT[0]), ("mkt_scarcity_keeper", CAT[3])],
                                                 0, 1, "percentile among matched safe keeper decisions", ref=0.5, fmt="{:.1f}", w=820, left=220)))
    mc = MG["fineCellCheck"]
    rows = [(c, v["withinCellSep_C4vsC1"], CAT[1], "candidate") for c, v in mc["candidates"].items()] + [(f, v["withinCellSep_C4vsC1"], CAT[0], "existing") for f, v in mc["existingNotInCells"].items()]
    rows.sort(key=lambda z: -(z[1] or 0))
    out.append(("06_multi_within_cells.svg", Q.hbars("6 · One vs four copies (C1 vs C4): separation inside cells of the rival block + price ratio",
                                                     "candidates (orange) add no more than existing features left out of the cells (blue)", rows, xmax=0.5, fmt="{:.2f}", left=230, note_w=80)))
    t = CT["tests"]["price needed to win differs ≥ 0.5 × fair"]["candidates"]
    out.append(("07_cf_candidate_differs.svg", Q.hbars("7 · When the current 80 features are near-identical, do candidates differ?",
                                                       f"{CT['similarPairs']:,} same-lot pairs with L∞ ≤ 0.05 — share where the candidate differs by at least its threshold",
                                                       sorted([(c, v["shareCandidateDiffers"], CAT[1], f"{v['pairsCandidateDiffers']:,} pairs") for c, v in t.items()], key=lambda z: -(z[1] or 0)),
                                                       xmax=0.05, fmt="{:.1%}", left=200, note_w=120)))
    rows = [("riv_capacity_1 at clip", SA["riv_capacity_1"]["shareAtClip"], CAT[3], ""), ("riv_capacity_3 at clip", SA["riv_capacity_3"]["shareAtClip"], CAT[3], ""),
            ("price needed > 5 × fair", SA["priceNeededToWin"]["share>5xFair"], CAT[0], "the range the clip removes"),
            ("mkt_scarcity_bowling at clip (own need)", SA["mkt_scarcity_bowling"]["shareAtClipWhenOwnNeed"], CAT[3], ""),
            ("bowling scarcity < 2 with own need", SA["mkt_scarcity_bowling"]["shareRawBelow2WhenOwnNeed"], CAT[0], "the range that matters"),
            ("mkt_scarcity_indians at clip (own need)", SA["mkt_scarcity_indians"]["shareAtClipWhenOwnNeed"], CAT[3], ""),
            ("Indians scarcity < 2 with own need", SA["mkt_scarcity_indians"]["shareRawBelow2WhenOwnNeed"], CAT[0], "the range that matters")]
    out.append(("08_saturation.svg", Q.hbars("8 · Saturated features: what the clip removes vs what matters",
                                             "yellow = share of decisions at the clip value 5; blue = share of decisions where the removed range would have mattered", rows, xmax=1.0, fmt="{:.1%}", left=290, note_w=150)))
    out.append(("09_minimum_delta.svg", delta_table(MD)))
    pdir = rep / "plots"
    pdir.mkdir(exist_ok=True)
    for name, svg in out:
        (pdir / name).write_text(svg, encoding="utf-8")
    page = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Phase 2E.3 gap-design plots</title><style>',
            ':root{--bg:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e}@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0d0d0d;--ink:#fff;--ink2:#c3c2b7}}:root[data-theme="dark"]{--bg:#0d0d0d;--ink:#fff;--ink2:#c3c2b7}',
            f'body{{margin:0;padding:24px 16px;background:var(--bg);color:var(--ink);{FONT}}}h1{{font-size:20px;margin:0 0 6px}}p{{color:var(--ink2);max-width:780px;font-size:14px}}figure{{margin:0 0 28px;overflow-x:auto}}figure svg{{max-width:100%;height:auto;border-radius:6px}}</style></head><body>',
            '<h1>Phase 2E.3 — minimum observation-gap design</h1><p>Design analysis only: obs-v2 is unchanged, nothing was implemented or trained. All charts come from digest-verified read-only replays of recorded Phase 2E.0 episodes. Hover for values.</p>']
    page += [f"<figure>{svg}</figure>" for _, svg in out]
    page.append("</body></html>")
    (pdir / "index.html").write_text("".join(page), encoding="utf-8")
    print(f"wrote {len(out)} charts")


if __name__ == "__main__":
    main(sys.argv[1])
