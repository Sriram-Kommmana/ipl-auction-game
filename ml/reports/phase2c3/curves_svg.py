"""Phase 2C.3 learning curves as a self-contained HTML page (inline SVG small multiples, no dependencies).

Three seeds per panel in validated categorical slots 1-3 with distinct markers
(secondary encoding), one y-axis per panel, legend always present, hover
tooltips via <title> on every point, light + dark surfaces.
"""
import json
from html import escape
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "ppo_2c3_curves.html"
CKPTS = [40, 81, 122, 163, 203, 244, 285, 325]
SEEDS = (1, 2, 3)
W, H, PAD_L, PAD_R, PAD_T, PAD_B = 330, 190, 48, 12, 10, 30


def updates(seed):
    return [json.loads(l) for l in open(RUNS / f"ppo-2c3-s{seed}" / "metrics.jsonl") if '"type": "update"' in l]


def validation(seed):
    rows = []
    for u in CKPTS:
        eps = json.loads((RUNS / f"ppo-2c3-s{seed}" / "checkpoints" / f"update_{u:04d}" / "validation" / "episodes.json").read_text())["episodes"]["policy:ppo"]
        n = len(eps)
        legal, strong = sum(e["legalXI"] for e in eps), sum(e["strongXI"] for e in eps)
        rows.append({"x": u * 6144, "xi": sum(e["xi"] for e in eps) / n, "legal": 100 * legal / n, "strong": 100 * strong / n,
                     "legal_txt": f"{legal}/{n}", "strong_txt": f"{strong}/{n}"})
    return rows


DATA = {s: (updates(s), validation(s)) for s in SEEDS}
PANELS = [
    ("Training return (per rollout)", "train", lambda u: u["episode"].get("return"), "{:.3f}"),
    ("Validation XI (500 seeds, T = 0.3)", "val", "xi", "{:.3f}"),
    ("Validation legal XI — 500/500 for every seed at every checkpoint (lines coincide)", "val", "legal", "{:.1f}%"),
    ("Validation strong XI — 500/500 for every seed at every checkpoint (lines coincide)", "val", "strong", "{:.1f}%"),
    ("Policy entropy (nats)", "train", lambda u: u["train"]["entropy"], "{:.3f}"),
    ("Explained variance — γ = 1, reward telescopes: not evidence of learning", "train", lambda u: u["train"]["explained_variance"], "{:.4f}"),
    ("Policy loss", "train", lambda u: u["train"]["policy_loss"], "{:+.4f}"),
    ("Value loss (log scale)", "train", lambda u: u["train"]["value_loss"], "{:.5f}"),
    ("Bid rate (share of learner decisions with a bid)", "train", lambda u: u["actions"]["bid_share"], "{:.3f}"),
    ("Purse left at episode end, ₹L (training rollouts)", "train", lambda u: u["episode"].get("purseLeft"), "{:.1f}"),
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
            extra = {"legal": "legal_txt", "strong": "strong_txt"}.get(key)
            tips = [f"seed {s} · {r['x']:,} decisions · {fmt.format(r[key])}" + (f" ({r[extra]})" if extra else "") for r in val]
        series[s] = (pts, tips)
    import math
    log = "log scale" in title
    tf = (lambda v: math.log10(v)) if log else (lambda v: v)
    series = {s: ([(x, tf(v)) for x, v in pts], tips) for s, (pts, tips) in series.items()}
    ys = [v for pts, _ in series.values() for _, v in pts]
    lo, hi = min(ys), max(ys)
    ticks = None
    if key in ("legal", "strong"):
        lo, hi, ticks = 97.6, 100.4, [98.0, 99.0, 100.0]
    else:
        pad = (hi - lo) * 0.08 or 0.5
        lo, hi = (lo - pad if lo - pad >= 0 or min(ys) < 0 or log else 0), hi + pad
    x0, x1 = 0, 2_000_000
    sx = lambda x: PAD_L + (x - x0) / (x1 - x0) * (W - PAD_L - PAD_R)
    sy = lambda y: PAD_T + (hi - y) / (hi - lo) * (H - PAD_T - PAD_B)
    g = []
    for t in ticks or nice_ticks(lo, hi):
        label = f"{10 ** t:.0e}" if log else fmt.format(t).replace("+", "")
        g.append(f'<line class="grid" x1="{PAD_L}" x2="{W - PAD_R}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/>'
                 f'<text class="tick" x="{PAD_L - 6}" y="{sy(t) + 3:.1f}" text-anchor="end">{escape(label)}</text>')
    for m in (0, 0.5, 1.0, 1.5, 2.0):
        g.append(f'<text class="tick" x="{sx(m * 1e6):.1f}" y="{H - 12}" text-anchor="middle">{"0" if m == 0 else f"{m:g}M"}</text>')
    for s in SEEDS:
        pts, tips = series[s]
        path = " ".join(f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for i, (x, y) in enumerate(pts))
        g.append(f'<path class="s{s} line" d="{path}"/>')
        step = 1 if kind == "val" else 40
        for i in range(0, len(pts), step):
            g.append(marker(MARKERS[s], sx(pts[i][0]), sy(pts[i][1]), f"s{s}", tips[i]))
        # invisible wide hit targets for the training lines (every 5th point)
        if kind == "train":
            for i in range(0, len(pts), 5):
                g.append(f'<circle class="hit" cx="{sx(pts[i][0]):.1f}" cy="{sy(pts[i][1]):.1f}" r="6"><title>{escape(tips[i])}</title></circle>')
    return (f'<figure class="panel"><figcaption>{escape(title)}</figcaption>'
            f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{escape(title)}">{"".join(g)}</svg></figure>')


def legend():
    items = []
    for s in SEEDS:
        items.append(f'<span class="key"><svg width="26" height="12" viewBox="0 0 26 12"><line class="s{s} line" x1="1" x2="25" y1="6" y2="6"/>'
                     f'{marker(MARKERS[s], 13, 6, f"s{s}", f"seed {s}")}</svg>seed {s}</span>')
    return "".join(items)


html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PPO 2C.3 learning curves</title>
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
.hit {{ fill:transparent; }}
</style></head><body>
<h1>Phase 2C.3 — PPO × 3 seeds × Stage A × act-v3</h1>
<p class="sub">1,996,800 learner decisions per seed; validation on the fixed 500-seed manifest at 8 checkpoints (temperature 0.3). Hover a point for its value.</p>
<div class="legend">{legend()}</div>
<div class="grid-wrap">{"".join(panel(*p) for p in PANELS)}</div>
</body></html>"""
OUT.write_text(html, encoding="utf-8")
print(OUT)
