"""Phase 2D.4 (OpenAI-ES) learning curves as a self-contained HTML page (inline SVG small multiples, no dependencies).

Three seeds per panel in validated categorical slots 1-3 with distinct markers
(secondary encoding), one y-axis per panel, legend always present, hover
tooltips via <title> on every point, light + dark surfaces.
"""
import json

import numpy as np
from html import escape
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "openai_es_2d4_curves.html"
CKPTS = [40, 81, 122, 163, 203, 244, 285, 325]
SEEDS = (1, 2, 3)
W, H, PAD_L, PAD_R, PAD_T, PAD_B = 330, 190, 48, 12, 10, 30


def updates(seed, prefix="openai-es-2d4"):
    return [json.loads(l) for l in open(RUNS / f"{prefix}-s{seed}" / "metrics.jsonl") if '"type": "update"' in l]


def _val_row(x, eps):
    n = len(eps)
    legal, strong = sum(e["legalXI"] for e in eps), sum(e["strongXI"] for e in eps)
    forced, dec = sum(e["shield"]["forced"] for e in eps), sum(e["decisions"] for e in eps)
    return {"x": x, "xi": sum(e["xi"] for e in eps) / n, "legal": 100 * legal / n, "strong": 100 * strong / n,
            "legal_txt": f"{legal}/{n}", "strong_txt": f"{strong}/{n}", "forced": 100 * forced / dec, "forced_txt": f"{forced}/{dec}",
            "ptf": sum(e["priceToFair"] for e in eps if e.get("priceToFair") is not None) / max(1, sum(1 for e in eps if e.get("priceToFair") is not None)),
            "bid": sum(e["bidRate"] for e in eps) / n}


def validation(seed, prefix="openai-es-2d4", name="policy:es"):
    if prefix == "openai-es-2d4":
        rows = []
        for l in open(RUNS / f"{prefix}-s{seed}" / "metrics.jsonl"):
            if '"type": "evaluation"' in l:
                e = json.loads(l)
                eps = json.loads((RUNS / f"{prefix}-s{seed}" / "checkpoints" / f"gen_{e['generation']:04d}" / "validation" / "episodes.json").read_text())["episodes"][name]
                rows.append(_val_row(e["decisions"], eps))
        return rows
    rows = []
    for u in CKPTS:
        eps = json.loads((RUNS / f"{prefix}-s{seed}" / "checkpoints" / f"update_{u:04d}" / "validation" / "episodes.json").read_text())["episodes"][name]
        rows.append(_val_row(u * 6144, eps))
    return rows


DATA = {s: (updates(s), validation(s)) for s in SEEDS}
PPO_VAL = [validation(s, "ppo-2c3", "policy:ppo") for s in SEEDS]
PPO_XI = [(PPO_VAL[0][i]["x"], sum(v[i]["xi"] for v in PPO_VAL) / 3) for i in range(len(CKPTS))]
A2C_VAL = [validation(s, "a2c-2d1", "policy:a2c") for s in SEEDS]
A2C_XI = [(A2C_VAL[0][i]["x"], sum(v[i]["xi"] for v in A2C_VAL) / 3) for i in range(len(CKPTS))]
D3_VAL = [validation(s, "d3qn-2d2", "policy:d3qn") for s in SEEDS]
D3_XI = [(D3_VAL[0][i]["x"], sum(v[i]["xi"] for v in D3_VAL) / 3) for i in range(len(CKPTS))]
QR_VAL = [validation(s, "qr-dqn-2d3", "policy:qrdqn") for s in SEEDS]
QR_XI = [(QR_VAL[0][i]["x"], sum(v[i]["xi"] for v in QR_VAL) / 3) for i in range(len(CKPTS))]
REFS = [("PPO 2C.3", PPO_XI, "ref"), ("A2C 2D.1", A2C_XI, "ref2"), ("D3QN 2D.2", D3_XI, "ref3"), ("QR-DQN 2D.3", QR_XI, "ref4")]
XMAX = max(u["decisions"] for s_ in SEEDS for u in DATA[s_][0])


def all_full(key):
    return all(r[f"{key}_txt"].split("/")[0] == r[f"{key}_txt"].split("/")[1] for s in SEEDS for r in DATA[s][1])


def pct_title(label, key):
    return (f"Validation {label} XI — 500/500 for every seed at every checkpoint (lines coincide)" if all_full(key)
            else f"Validation {label} XI % (exact n/500 on hover)")
TR = lambda k: (lambda u: u["train"].get(k) if u["train"].get("updates_in_window") else None)
TR = lambda k: (lambda u: u["train"].get(k))
PANELS = [
    ("Validation XI (500 seeds, masked argmax) vs decisions — references: dashed PPO, dotted A2C, dash-dot D3QN, long-dash QR-DQN (three-seed means, 0–2M)", "val", "xi", "{:.3f}"),
    (pct_title("legal", "legal"), "val", "legal", "{:.1f}%"),
    (pct_title("strong", "strong"), "val", "strong", "{:.1f}%"),
    ("Validation shield-forced decisions, % (exact n/d on hover)", "val", "forced", "{:.3f}%"),
    ("Validation price / fair value", "val", "ptf", "{:.3f}"),
    ("Validation bid rate", "val", "bid", "{:.3f}"),
    ("Population mean fitness = mean episode return of the 64 perturbed policies", "train", TR("fitness_mean"), "{:.4f}"),
    ("Population best fitness (max of 64)", "train", TR("fitness_max"), "{:.4f}"),
    ("Population worst fitness (min of 64)", "train", TR("fitness_min"), "{:.4f}"),
    ("Mean |f+ − f−| within antithetic pairs (CRN)", "train", TR("pair_diff_abs_mean"), "{:.4f}"),
    ("Antithetic pairs with f+ = f− (of 32)", "train", TR("pair_ties"), "{:.0f}"),
    ("|ĝ| — ES gradient estimate norm", "train", TR("grad_norm"), "{:.3f}"),
    ("|θ| — parameter norm", "train", TR("theta_norm"), "{:.3f}"),
    ("Adam step / |θ| (log scale)", "train", TR("update_ratio"), "{:.2e}"),
    ("Training XI (sampled policies, per generation)", "train", lambda u: u["episode"].get("xi"), "{:.2f}"),
    ("Training bid rate (sampled policies)", "train", lambda u: u["actions"]["bid_share"], "{:.3f}"),
    ("Decisions per generation (64 episodes)", "train", TR("decisions_in_generation"), "{:,.0f}"),
    ("Purse left at episode end, ₹L (sampled policies)", "train", lambda u: u["episode"].get("purseLeft"), "{:.1f}"),
]
MARKERS = {1: "circle", 2: "square", 3: "triangle"}


def marker(kind, x, y, cls, tip):
    t = f"<title>{escape(tip)}</title>"
    if kind == "circle":
        return f'<circle class="{cls} mk" cx="{x:.1f}" cy="{y:.1f}" r="4">{t}</circle>'
    if kind == "square":
        return f'<rect class="{cls} mk" x="{x - 4:.1f}" y="{y - 4:.1f}" width="8" height="8" rx="1">{t}</rect>'
    return f'<path class="{cls} mk" d="M{x:.1f},{y - 5:.1f} L{x + 5:.1f},{y + 4:.1f} L{x - 5:.1f},{y + 4:.1f} Z">{t}</path>'


def nice_ticks(lo, hi, n=4):
    if hi == lo:
        hi = lo + 1
    step = (hi - lo) / n
    return [lo + i * step for i in range(n + 1)]


def panel(title, kind, key, fmt):
    series = {}
    for s in SEEDS:
        ups, val = DATA[s]
        if kind == "train":
            pts = [(u["decisions"], key(u)) for u in ups if key(u) is not None]
            tips = [f"seed {s} · {d:,} decisions · {fmt.format(v)}" for d, v in pts]
        else:
            pts = [(r["x"], r[key]) for r in val]
            extra = {"legal": "legal_txt", "strong": "strong_txt", "forced": "forced_txt"}.get(key)
            tips = [f"seed {s} · {r['x']:,} decisions · {fmt.format(r[key])}" + (f" ({r[extra]})" if extra else "") for r in val]
        series[s] = (pts, tips)
    import math
    log = "log scale" in title
    tf = (lambda v: math.log10(v)) if log else (lambda v: v)
    series = {s: ([(x, tf(v)) for x, v in pts], tips) for s, (pts, tips) in series.items()}
    ys = [v for pts, _ in series.values() for _, v in pts] + ([v for _, ref, _ in REFS for _, v in ref] if key == "xi" else [])
    lo, hi = min(ys), max(ys)
    ticks = None
    if key in ("legal", "strong"):
        lo0 = min(97.6, min(ys) - 0.4)
        lo, hi = lo0, 100.4
        ticks = [98.0, 99.0, 100.0] if lo0 == 97.6 else None
    else:
        pad = (hi - lo) * 0.08 or 0.5
        lo, hi = (lo - pad if lo - pad >= 0 or min(ys) < 0 or log else 0), hi + pad
    x0, x1 = 0, XMAX
    sx = lambda x: PAD_L + (x - x0) / (x1 - x0) * (W - PAD_L - PAD_R)
    sy = lambda y: PAD_T + (hi - y) / (hi - lo) * (H - PAD_T - PAD_B)
    g = []
    for t in ticks or nice_ticks(lo, hi):
        label = f"{10 ** t:.0e}" if log else fmt.format(t).replace("+", "")
        g.append(f'<line class="grid" x1="{PAD_L}" x2="{W - PAD_R}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/>'
                 f'<text class="tick" x="{PAD_L - 6}" y="{sy(t) + 3:.1f}" text-anchor="end">{escape(label)}</text>')
    for m in [x / 1e6 for x in np.linspace(0, XMAX, 6)]:
        g.append(f'<text class="tick" x="{sx(m * 1e6):.1f}" y="{H - 12}" text-anchor="middle">{"0" if m == 0 else f"{m:.1f}M"}</text>')
    if key == "xi":
        for rname, pts_ref, cls in REFS:
            ref = " ".join(f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for i, (x, y) in enumerate(pts_ref))
            g.append(f'<path class="{cls}" d="{ref}"/>')
            for x, y in pts_ref:
                g.append(f'<circle class="refpt" cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3"><title>{escape(f"{rname} three-seed mean · {x:,} decisions · {y:.3f}")}</title></circle>')
    for s in SEEDS:
        pts, tips = series[s]
        path = " ".join(f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for i, (x, y) in enumerate(pts))
        g.append(f'<path class="s{s} line" d="{path}"/>')
        step = 1 if kind == "val" else 250
        for i in range(0, len(pts), step):
            g.append(marker(MARKERS[s], sx(pts[i][0]), sy(pts[i][1]), f"s{s}", tips[i]))
        # invisible wide hit targets for the training lines (every 5th point)
        if kind == "train":
            for i in range(0, len(pts), 10):
                g.append(f'<circle class="hit" cx="{sx(pts[i][0]):.1f}" cy="{sy(pts[i][1]):.1f}" r="6"><title>{escape(tips[i])}</title></circle>')
    return (f'<figure class="panel"><figcaption>{escape(title)}</figcaption>'
            f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{escape(title)}">{"".join(g)}</svg></figure>')


def legend():
    items = []
    for s in SEEDS:
        items.append(f'<span class="key"><svg width="26" height="12" viewBox="0 0 26 12"><line class="s{s} line" x1="1" x2="25" y1="6" y2="6"/>'
                     f'{marker(MARKERS[s], 13, 6, f"s{s}", f"seed {s}")}</svg>seed {s}</span>')
    items.append('<span class="key"><svg width="26" height="12" viewBox="0 0 26 12"><line class="ref" x1="1" x2="25" y1="6" y2="6"/></svg>PPO 2C.3 mean (validation XI)</span>')
    items.append('<span class="key"><svg width="26" height="12" viewBox="0 0 26 12"><line class="ref2" x1="1" x2="25" y1="6" y2="6"/></svg>A2C 2D.1 mean (validation XI)</span>')
    items.append('<span class="key"><svg width="26" height="12" viewBox="0 0 26 12"><line class="ref3" x1="1" x2="25" y1="6" y2="6"/></svg>D3QN 2D.2 mean (validation XI)</span>')
    items.append('<span class="key"><svg width="26" height="12" viewBox="0 0 26 12"><line class="ref4" x1="1" x2="25" y1="6" y2="6"/></svg>QR-DQN 2D.3 mean (validation XI)</span>')
    return "".join(items)


html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OpenAI-ES 2D.4 learning curves</title>
<style>
:root {{ --surface:#fcfcfb; --ink:#1a1a19; --muted:#6b6a63; --grid:#e6e5df; --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --surface:#1a1a19; --ink:#ffffff; --muted:#c3c2b7; --grid:#34332f; --s1:#3987e5; --s2:#d95926; --s3:#199e70; }} }}
:root[data-theme="dark"] {{ --surface:#1a1a19; --ink:#ffffff; --muted:#c3c2b7; --grid:#34332f; --s1:#3987e5; --s2:#d95926; --s3:#199e70; }}
body {{ margin:0; padding:16px; background:var(--surface); color:var(--ink); font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; }}
h1 {{ font-size:18px; margin:0 0 4px; }} p.sub {{ color:var(--muted); margin:0 0 12px; }}
.legend {{ display:flex; flex-wrap:wrap; gap:18px; margin:0 0 12px; color:var(--ink); }} .key {{ display:inline-flex; align-items:center; gap:6px; white-space:nowrap; }}
.legend svg {{ width:26px; height:12px; flex:none; }}
.grid-wrap {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:14px; }}
.panel {{ margin:0; }} figcaption {{ font-weight:600; font-size:13px; margin-bottom:4px; }}
svg {{ width:100%; height:auto; display:block; overflow:visible; }}
.grid {{ stroke:var(--grid); stroke-width:1; }} .tick {{ fill:var(--muted); font-size:10px; }}
.line {{ fill:none; stroke-width:2; stroke-linejoin:round; }}
.s1.line {{ stroke:var(--s1); }} .s2.line {{ stroke:var(--s2); }} .s3.line {{ stroke:var(--s3); }}
.mk {{ stroke:var(--surface); stroke-width:2; }} .s1.mk {{ fill:var(--s1); }} .s2.mk {{ fill:var(--s2); }} .s3.mk {{ fill:var(--s3); }}
.hit {{ fill:transparent; }} .ref {{ fill:none; stroke:var(--muted); stroke-width:1.5; stroke-dasharray:4 3; }} .ref2 {{ fill:none; stroke:var(--muted); stroke-width:1.5; stroke-dasharray:1 3; stroke-linecap:round; }} .ref3 {{ fill:none; stroke:var(--muted); stroke-width:1.5; stroke-dasharray:6 2 1 2; }} .ref4 {{ fill:none; stroke:var(--muted); stroke-width:1.5; stroke-dasharray:10 3; }} .refpt {{ fill:var(--muted); }}
</style></head><body>
<h1>Phase 2D.4 — OpenAI-ES × 3 seeds × Stage A × act-v3</h1>
<p class="sub">2,000 generations × 64 perturbed policies (32 antithetic pairs, one episode each) per seed; x-axis = cumulative learner decisions. Training panels: one unsmoothed point per generation. Validation: the fixed 500-seed manifest, masked argmax, at the decision-matched and every-250-generation checkpoints. The other algorithms trained to 1,996,800 decisions (their reference lines end there). Hover a point for its value.</p>
<div class="legend">{legend()}</div>
<div class="grid-wrap">{"".join(panel(*p) for p in PANELS)}</div>
</body></html>"""
OUT.write_text(html, encoding="utf-8")
print(OUT)
