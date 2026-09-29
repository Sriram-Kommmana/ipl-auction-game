"""Phase 2F plots (§29) — from ml/reports/phase2f/*.json.

    python -m ipl_rl.stage_b.plots_b      → ml/reports/phase2f/plots/*.svg + index.html
Intervals shown are the bootstrap 95% CIs over validation entries computed by
analyse_b (Phase 2E.0 method); no chart implies a significance test that was not run.
"""
import json
import sys
from html import escape
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ML_ROOT / "ipl_rl" / "diagnosis"))
sys.path.insert(0, str(ML_ROOT / "ipl_rl" / "crossplay"))
import plots as P  # noqa: E402
import plots2e2 as Q  # noqa: E402

REP = ML_ROOT / "reports/phase2f"
ALGOS = ["ppo", "a2c", "d3qn", "qrdqn", "es"]
NAMES = {"ppo": "PPO", "a2c": "A2C", "d3qn": "D3QN", "qrdqn": "QR-DQN", "es": "OpenAI-ES"}
CAT = Q.CAT
STAGE_COL = {"Stage A (same entries)": "#52514e", "Stage B pilot": "#2a78d6", "Stage A reference (§30)": "#c3c2b7"}
J = lambda n: json.loads((REP / n).read_text(encoding="utf-8"))


def dot_ci(title, sub, rows, xlabel, zero=None):
    """rows: [(label, {series: (m, lo, hi)})]"""
    xs = [x for _, v in rows for t in v.values() for x in t if x is not None]
    lo, hi = min(xs), max(xs)
    pad = (hi - lo) * 0.08 or 0.1
    old = dict(P.SERIES)
    P.SERIES.clear()
    P.SERIES.update(STAGE_COL)
    s = P.dotplot(title, sub, rows, list(STAGE_COL), lo - pad, hi + pad, zero=zero, xlabel=xlabel)
    P.SERIES.clear()
    P.SERIES.update(old)
    return s


def trio(m):
    if m is None:
        return None
    ci = m.get("ci95") or [m["mean"], m["mean"]]
    return (m["mean"], ci[0], ci[1])


def stage_rows(block, metric_fn, conds=("A", "C1", "C4", "S4")):
    rows = []
    for a in ALGOS:
        for c in conds:
            v = metric_fn(block, c, a)
            if v:
                rows.append((f"{NAMES[a]} · {c}", v))
    return rows


def main():
    AR, BM, KA, SA, SH, TA = (J(x) for x in ("algorithm-results.json", "behavioral-metrics.json", "keeper-analysis.json", "star-analysis.json", "shield-analysis.json", "transfer-analysis.json"))
    L5 = AR["levels"]["c500"]["transfer"]
    out = []
    out.append(("01_xi_stageA_vs_stageB.svg", dot_ci("1 · Validation XI: frozen Stage A vs Stage-B pilot (500k), same entries and cells",
                                                     "mean over seeds; bar = bootstrap 95% CI over entries of the paired Stage B − Stage A difference, drawn on Stage B",
                                                     stage_rows(L5, lambda b, c, a: {"Stage A (same entries)": (b[c][a]["xiStageA"]["mean"],) * 3,
                                                                                     "Stage B pilot": (b[c][a]["xiStageB"]["mean"], b[c][a]["xiStageB"]["mean"] + (b[c][a]["xiDiffBminusA"]["ci95"] or [0, 0])[0] - b[c][a]["xiDiffBminusA"]["mean"],
                                                                                                       b[c][a]["xiStageB"]["mean"] + (b[c][a]["xiDiffBminusA"]["ci95"] or [0, 0])[1] - b[c][a]["xiDiffBminusA"]["mean"])}), "validation XI")))
    for n, c in enumerate(("C1", "C4", "S4")):
        rows = [(NAMES[a], {"Stage A reference (§30)": (L5[c][a]["historicalReference"],) * 3, "Stage A (same entries)": (L5[c][a]["transferStageA"]["mean"],) * 3,
                            "Stage B pilot": trio(L5[c][a]["transferStageB"])}) for a in ALGOS]
        out.append((f"0{n + 2}_{c.lower()}_transfer.svg", dot_ci(f"{n + 2} · {c} transfer Δ (XI in {c} − the same policy's Stage-A XI on the same entry)",
                                                                  "Stage-B pilot at 500k (CI over entries) vs the frozen Stage-A exports on the same 500 entries and the §30 reference", rows, "transfer Δ (XI)", zero=0)))
    labels = ["100k", "250k", "500k"]
    for n, c in enumerate(("C1", "C4", "S4")):
        series = [(NAMES[a], CAT[i], [TA["trend"][lv][c][a]["transferImprovement"]["mean"] for lv in ("c100", "c250", "c500_first100")]) for i, a in enumerate(ALGOS)]
        out.append((f"05{'abc'[n]}_transfer_trend_{c.lower()}.svg", Q.line_chart(f"5{'abc'[n]} · {c}: transfer improvement across checkpoints (Stage B − Stage A, same entries 0–99)",
                                                                                 "positive = smaller transfer loss than the frozen Stage-A policy; all three checkpoints on validation entries 0–99",
                                                                                 labels, series, "transfer improvement (XI)", ymin=min(min(s[2]) for s in series) - 0.1, ymax=max(max(s[2]) for s in series) + 0.1, fmt="{:+.2f}")))
    D = BM["decisionLevel_c500_40entries"]
    def ab(metric, conds=("A", "C4", "S4"), src=D):
        return [(f"{NAMES[a]} · {c}", {"Stage A (same entries)": (src[c][a][metric]["stageA"],) * 3, "Stage B pilot": (src[c][a][metric]["stageB"],) * 3})
                for a in ALGOS for c in conds if a in src[c] and src[c][a][metric]["stageB"] is not None]
    out.append(("06_early_stars.svg", dot_ci("6 · Stars won in the first 30% of the main round (per episode)", "40 Phase 2E.2 entries, paired cells; Stage A = frozen export", ab("starsBy30"), "stars won by 30% progress")))
    out.append(("07_keeper_timing.svg", dot_ci("7 · Keeper acquisition timing", "main-round progress of the first keeper won (re-auction = 1.0); 40 entries", ab("firstKeeperWinProgress", ("A", "C1", "C4", "S4")), "progress")))
    RL = BM["recordLevel"]["c500"]
    fails = []
    for a in ALGOS:
        for c in ("C1", "C4", "S4"):
            v = RL[c][a]
            fails.append((f"{NAMES[a]} · {c}", {"Stage A (same entries)": (v["learnerFinding"]["total_stageA"],) * 3, "Stage B pilot": (v["learnerFinding"]["total_stageB"],) * 3}))
    out.append(("08_keeper_failures.svg", dot_ci("8 · Learner incomplete-XI findings (all are keeper-type in Phase 2E.0)", "count over the 500-entry evaluation cells; M1/M2 split in keeper-analysis.json", fails, "learner incomplete-XI episodes")))
    def rec(metric, conds=("A", "C1", "C4", "S4")):
        return [(f"{NAMES[a]} · {c}", {"Stage A (same entries)": (RL[c][a][metric]["stageA"],) * 3, "Stage B pilot": (RL[c][a][metric]["stageB"],) * 3}) for a in ALGOS for c in conds]
    for num, (metric, title, xl) in enumerate([("shieldRate", "Shield intervention rate (share of decisions)", "share"), ("finalPath", "Final-path forced bids per episode", "per episode"),
                                                ("bidRate", "Bid rate", "share of decisions with a bid"), ("priceToFair", "Price paid / fair value", "price / fair"),
                                                ("purseLeftShare", "Purse remaining at the end", "share of purse"), ("squadSize", "Squad size", "players"),
                                                ("overseas", "Overseas players", "players")], start=9):
        out.append((f"{num:02d}_{metric}.svg", dot_ci(f"{num} · {title}", "500-entry evaluation, mean per episode; Stage A = frozen export on the same cells", rec(metric), xl)))
    cells = {}
    bo = AR["levels"]["c500"]
    byo = J("c4-results.json")["c500"]["byOpponent"]
    for k, v in byo.items():
        a, b = k.split("|")
        cells[f"{a}>{b}"] = {"mean": v["improvement"], "ci95": v["improvementCi95"] or [v["improvement"], v["improvement"]]}
    vmax = max(abs(v["mean"]) for v in cells.values()) or 1
    out.append(("16_algorithm_x_opponent_c4.svg", P.heatmap("16 · C4 transfer improvement by learner × opponent algorithm (Stage B − Stage A)",
                                                            "500k checkpoints, 500 entries, 3 seeds × 3 opponent seeds; positive = smaller loss; bracket = 95% CI over entries", cells, vmax)))
    pdir = REP / "plots"
    pdir.mkdir(exist_ok=True)
    for name, svg in out:
        (pdir / name).write_text(svg, encoding="utf-8")
    page = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Phase 2F pilot plots</title><style>',
            ':root{--bg:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e}@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0d0d0d;--ink:#fff;--ink2:#c3c2b7}}:root[data-theme="dark"]{--bg:#0d0d0d;--ink:#fff;--ink2:#c3c2b7}',
            f'body{{margin:0;padding:24px 16px;background:var(--bg);color:var(--ink);{Q.FONT}}}h1{{font-size:20px;margin:0 0 6px}}p{{color:var(--ink2);max-width:780px;font-size:14px}}figure{{margin:0 0 28px;overflow-x:auto}}figure svg{{max-width:100%;height:auto;border-radius:6px}}</style></head><body>',
            '<h1>Phase 2F — Stage-B training pilot</h1><p>Experimental stage_b_pilot checkpoints vs the frozen Stage-A exports on the same validation entries and cells. Intervals are bootstrap 95% CIs over entries where shown; nothing here is a ranking.</p>']
    page += [f"<figure>{svg}</figure>" for _, svg in out]
    page.append("</body></html>")
    (pdir / "index.html").write_text("".join(page), encoding="utf-8")
    print(f"wrote {len(out)} charts")


if __name__ == "__main__":
    main()
