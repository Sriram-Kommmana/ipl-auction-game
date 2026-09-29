"""Phase 2D.3 analysis — QR-DQN × 3 seeds vs PPO 2C.3, A2C 2D.1, QR-DQN 2D.2 and the locked baselines.
Reads run outputs only (no training, no evaluation)."""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
RUNS = Path(__file__).resolve().parents[1]
BASE = RUNS.parents[1] / "packages" / "shared" / "data" / "rl-baselines" / "validation.episodes.json"
SEEDS = (1, 2, 3)
CKPTS = [40, 81, 122, 163, 203, 244, 285, 325]
BASELINES = ["moneyball", "starChaser", "balancedBuilder", "opportunist", "productFallback", "randomLegal", "fairValue", "plannerGreedy"]
ALGOS = {"qr": ("qr-dqn-2d3-s{}", "policy:qrdqn"), "d3qn": ("d3qn-2d2-s{}", "policy:d3qn"), "a2c": ("a2c-2d1-s{}", "policy:a2c"), "ppo": ("ppo-2c3-s{}", "policy:ppo")}


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def val(algo, seed, u):
    run, name = ALGOS[algo]
    d = RUNS / run.format(seed) / "checkpoints" / f"update_{u:04d}" / "validation"
    if not (d / "episodes.json").exists():
        return None
    return load(d / "report.json"), load(d / "episodes.json")["episodes"][name]


def mean(rows, k):
    xs = [float(r[k]) for r in rows if r.get(k) is not None]
    return float(np.mean(xs)) if xs else float("nan")


def boot(d, it=2000, seed=7):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    return np.percentile(d[rng.integers(0, len(d), (it, len(d)))].mean(1), [2.5, 97.5])


def forced(rows):
    return sum(r["shield"]["forced"] for r in rows), sum(r["decisions"] for r in rows)


def sd(a):
    return float(np.std(a, ddof=1)) if len(a) > 1 else float("nan")


V = {(a, s, u): val(a, s, u) for a in ALGOS for s in SEEDS for u in CKPTS}
V = {k: v for k, v in V.items() if v}
A = {(s, u): V[("qr", s, u)] for s in SEEDS for u in CKPTS if ("qr", s, u) in V}
REFS = {r: {(s, u): V[(r, s, u)] for s in SEEDS for u in CKPTS if (r, s, u) in V} for r in ("ppo", "a2c", "d3qn")}
P = REFS["ppo"]
out = {"checkpoints": {}, "final": {}, "vsPpo": {}, "vsA2c": {}, "vsD3qn": {}, "vsBaselines": {}}
COLS = ["xi", "strongXI", "purseLeft", "squadSize", "overseas", "stars", "priceToFair", "capToFair", "bidRate", "reauctionBuys", "marginalBuys", "xiGainPer1000", "rank"]

print("=" * 30, "QR-DQN per-seed checkpoint table (500 validation seeds, act-v3, ε = 0 masked argmax)")
for s in SEEDS:
    print(f"-- seed {s}")
    print(f"{'upd':>4} {'decisions':>10} {'legal':>9} {'strong':>9} " + " ".join(f"{c[:9]:>9}" for c in COLS) + "  forced/decisions")
    for u in CKPTS:
        if (s, u) not in A:
            continue
        rep, rows = A[(s, u)]
        n = len(rows)
        lg, st = sum(r["legalXI"] for r in rows), sum(r["strongXI"] for r in rows)
        f, d = forced(rows)
        print(f"{u:>4} {u * 6144:>10,} {lg:>4}/{n:<4} {st:>4}/{n:<4} " + " ".join(f"{mean(rows, c):9.3f}" for c in COLS) + f"  {f}/{d} = {100 * f / d:.3f}%")
        out["checkpoints"].setdefault(str(u), {})[str(s)] = {"legal": [lg, n], "strong": [st, n], **{c: mean(rows, c) for c in COLS}, "forced": [f, d]}

STAT = ["xi", "legalXI", "strongXI", "purseLeft", "squadSize", "overseas", "stars", "priceToFair", "bidRate", "reauctionBuys"]
print("=" * 30, "QR-DQN three-seed statistics per checkpoint (mean ± std [min, max])")
for u in CKPTS:
    have = [s for s in SEEDS if (s, u) in A]
    if not have:
        continue
    parts = []
    for c in STAT:
        a = np.array([mean(A[(s, u)][1], c) for s in have])
        parts.append(f"{c} {a.mean():.3f}±{sd(a):.3f} [{a.min():.3f},{a.max():.3f}]")
    print(f"u{u} ({u * 6144:,}; seeds {have}): " + " | ".join(parts))

last = max((u for u in CKPTS if all((s, u) in A for s in SEEDS)), default=None)
if last:
    print("=" * 30, f"QR-DQN final checkpoint u{last} ({last * 6144:,} decisions): per seed, mean, std, min, max, CV")
    for c in STAT + ["capToFair", "marginalBuys", "xiGainPer1000", "rank"]:
        a = np.array([mean(A[(s, last)][1], c) for s in SEEDS])
        m, v = a.mean(), sd(a)
        cv = 100 * v / abs(m) if m else float("nan")
        print(f"  {c:15s} " + "  ".join(f"s{s} {x:.3f}" for s, x in zip(SEEDS, a)) + f"   mean {m:.3f} std {v:.3f} min {a.min():.3f} max {a.max():.3f} CV {cv:.2f}%")
        out["final"][c] = {"perSeed": a.tolist(), "mean": m, "std": v, "min": a.min(), "max": a.max(), "cvPct": cv}
    for s in SEEDS:
        rows = A[(s, last)][1]
        print(f"  seed {s}: legal {sum(r['legalXI'] for r in rows)}/{len(rows)}  strong {sum(r['strongXI'] for r in rows)}/{len(rows)}")
    for i in range(3):
        for j in range(i + 1, 3):
            d = [x["xi"] - y["xi"] for x, y in zip(A[(SEEDS[i], last)][1], A[(SEEDS[j], last)][1])]
            lo, hi = boot(d)
            print(f"  paired ΔXI qr-dqn seed{SEEDS[i]} − seed{SEEDS[j]}: {np.mean(d):+.3f} [{lo:+.3f}, {hi:+.3f}]")

for RNAME, R in (("PPO 2C.3", REFS["ppo"]), ("A2C 2D.1", REFS["a2c"]), ("D3QN 2D.2", REFS["d3qn"])):
    key = {"PPO": "vsPpo", "A2C": "vsA2c", "D3Q": "vsD3qn"}[RNAME[:3]]
    print("=" * 30, f"QR-DQN vs {RNAME} per checkpoint (same 500 validation seeds; seed s vs seed s, and seed-mean vs seed-mean)")
    for u in CKPTS:
        have = [s for s in SEEDS if (s, u) in A and (s, u) in R]
        if not have:
            continue
        print(f"-- u{u} ({u * 6144:,} decisions)")
        for s in have:
            a, p = A[(s, u)][1], R[(s, u)][1]
            assert [r["seed"] for r in a] == [r["seed"] for r in p]
            d = [x["xi"] - y["xi"] for x, y in zip(a, p)]
            lo, hi = boot(d)
            fa, da = forced(a)
            fp, dp = forced(p)
            n = len(a)
            print(f"  s{s}: ΔXI {np.mean(d):+.3f} [{lo:+.3f},{hi:+.3f}] W/L {sum(x > 0 for x in d)}/{sum(x < 0 for x in d)} | "
                  f"legal {sum(r['legalXI'] for r in a)}/{n} vs {sum(r['legalXI'] for r in p)}/{n} | strong {sum(r['strongXI'] for r in a)}/{n} vs {sum(r['strongXI'] for r in p)}/{n} | "
                  f"purse {mean(a, 'purseLeft'):.1f} vs {mean(p, 'purseLeft'):.1f} | squad {mean(a, 'squadSize'):.2f} vs {mean(p, 'squadSize'):.2f} | "
                  f"stars {mean(a, 'stars'):.2f} vs {mean(p, 'stars'):.2f} | price/fair {mean(a, 'priceToFair'):.3f} vs {mean(p, 'priceToFair'):.3f} | "
                  f"bid {mean(a, 'bidRate'):.3f} vs {mean(p, 'bidRate'):.3f} | forced {fa}/{da} vs {fp}/{dp}")
            out[key].setdefault(str(u), {})[str(s)] = {"dxi": float(np.mean(d)), "ci": [float(lo), float(hi)]}
        if len(have) == 3:
            am = np.mean([[r["xi"] for r in A[(s, u)][1]] for s in SEEDS], axis=0)
            pm = np.mean([[r["xi"] for r in R[(s, u)][1]] for s in SEEDS], axis=0)
            lo, hi = boot(am - pm)
            print(f"  seed-mean: QR-DQN {am.mean():.3f} vs {RNAME} {pm.mean():.3f}  ΔXI {np.mean(am - pm):+.3f} [{lo:+.3f},{hi:+.3f}]")
            out[key].setdefault(str(u), {})["seedMean"] = {"qr": am.mean(), "ref": pm.mean(), "dxi": float(np.mean(am - pm)), "ci": [float(lo), float(hi)]}

    if last and all((s, 325) in R for s in SEEDS):
        print("=" * 30, f"QR-DQN u{last} vs {RNAME} u325 — all 9 seed pairs (ΔXI [95% CI])")
        for s in SEEDS:
            line = f"  QR-DQN s{s}:"
            for t in SEEDS:
                d = [x["xi"] - y["xi"] for x, y in zip(A[(s, last)][1], R[(t, 325)][1])]
                lo, hi = boot(d)
                line += f"  vs s{t} {np.mean(d):+.3f} [{lo:+.3f},{hi:+.3f}]"
            print(line)
        am = np.mean([[r["xi"] for r in A[(s, last)][1]] for s in SEEDS], axis=0)
        pm = np.mean([[r["xi"] for r in R[(s, 325)][1]] for s in SEEDS], axis=0)
        ref = A[(1, last)][1]
        for strat in ("low", "normal", "high"):
            idx = [i for i, r in enumerate(ref) if r["stratum"] == strat]
            d = (am - pm)[idx]
            lo, hi = boot(d)
            print(f"  stratum {strat:6s} seed-mean ΔXI {d.mean():+.3f} [{lo:+.3f},{hi:+.3f}] (n={len(idx)})")
        print(f"  {RNAME} u325 three-seed reference: " + " | ".join(
            f"{c} {np.mean([mean(R[(s, 325)][1], c) for s in SEEDS]):.3f}±{sd([mean(R[(s, 325)][1], c) for s in SEEDS]):.3f}" for c in STAT))

if last:
    print("=" * 30, f"QR-DQN u{last} vs the eight locked baselines (paired by validation seed, unchanged baseline file)")
    locked = load(BASE)["episodes"]
    for b in BASELINES:
        ref = locked[b]
        line = f"  {b:16s} baseline XI {mean(ref, 'xi'):.3f} |"
        pooled = []
        for s in SEEDS:
            rows = A[(s, last)][1]
            assert [r["seed"] for r in rows] == [r["seed"] for r in ref]
            d = [x["xi"] - y["xi"] for x, y in zip(rows, ref)]
            pooled.append(d)
            lo, hi = boot(d)
            line += f" s{s} {np.mean(d):+.3f} [{lo:+.3f},{hi:+.3f}] W/L {sum(x > 0 for x in d)}/{sum(x < 0 for x in d)} |"
        m = np.mean(pooled, axis=0)
        lo, hi = boot(m)
        line += f" seed-mean {m.mean():+.3f} [{lo:+.3f},{hi:+.3f}]"
        out["vsBaselines"][b] = {"baselineXi": mean(ref, "xi"), "seedMean": float(m.mean()), "ci": [float(lo), float(hi)]}
        print(line)
        for strat in ("low", "normal", "high"):
            idx = [i for i, r in enumerate(ref) if r["stratum"] == strat]
            dm = np.mean([[pooled[k][i] for i in idx] for k in range(3)], axis=0)
            lo, hi = boot(dm)
            print(f"      {strat:6s} seed-mean ΔXI {dm.mean():+.3f} [{lo:+.3f},{hi:+.3f}]  (n={len(idx)})")

print("=" * 30, "QR-DQN behaviour / shield audit per checkpoint (validation)")
for s in SEEDS:
    print(f"-- seed {s}")
    for u in CKPTS:
        if (s, u) not in A:
            continue
        rows = A[(s, u)][1]
        purse = np.array([r["purseLeft"] for r in rows])
        byreq = {}
        for r in rows:
            for k, v in r["shield"]["forcedByRequirement"].items():
                byreq[k] = byreq.get(k, 0) + v
        states = {k: sum(r["shield"]["states"][k] for r in rows) for k in ("SAFE", "WARNING", "CRITICAL", "IMPOSSIBLE")}
        f, d = forced(rows)
        print(f"  u{u:<3} purse mean {purse.mean():.1f} p10 {np.percentile(purse, 10):.0f} p50 {np.percentile(purse, 50):.0f} p90 {np.percentile(purse, 90):.0f} max {purse.max():.0f} | "
              f"=0: {int((purse == 0).sum())}/500 ≤20: {int((purse <= 20).sum())}/500 | forced {f}/{d} ({100 * f / d:.3f}%) won {sum(r['shieldWins'] for r in rows)} "
              f"re-auction {sum(r['shield']['reauctionForced'] for r in rows)} | states {states} | all-bids-removed lots {sum(r['shield']['decisionsRemoved'] for r in rows)} | "
              f"margin trims {sum(r['shield']['marginTrims'] for r in rows)} | alreadyInfeasible {sum(r['shield']['alreadyInfeasible'] for r in rows)} | byReq {byreq}")

print("=" * 30, "QR-DQN safety / optimisation budget / diagnostics / throughput / reproducibility per run")
for s in SEEDS:
    run = RUNS / f"qr-dqn-2d3-s{s}"
    if not (run / "summary.json").exists():
        print(f"seed {s}: running / not started")
        continue
    sm = load(run / "summary.json")
    te = json.loads((run / "training_episodes.json").read_text())
    ups = [json.loads(l) for l in open(run / "metrics.jsonl") if '"type": "update"' in l]
    evs = [json.loads(l) for l in open(run / "metrics.jsonl") if '"type": "evaluation"' in l]
    inc = sum(1 for e in te if not e["legalXI"])
    inv = sum(e["invariantViolations"] for e in te)
    tm = sm["timers"]
    ob = sm["optimisationBudget"]
    print(f"seed {s}: status {sm['status']} failure {sm['failure']} | decisions {sm['totalDecisions']:,} cycles {sm['cycles']:,} episodes {sm['episodes']:,} | "
          f"training: incomplete XI {inc}/{len(te)}, invariant violations {inv}, consistency checks {sm['episodeConsistencyChecks']}/{len(te)}, illegal {sm['illegalActions']} | "
          f"actSpec {sm['actSpec']['version']} {sm['actSpec']['hash']}")
    print(f"   digests: initial online {sm['initialParamsDigest']} target {sm['initialTargetDigest']} | final online {sm['finalParamsDigest']} target {sm['finalTargetDigest']}")
    print(f"   budget: {ob}")
    print(f"   wall {sm['wallSeconds']:.0f}s | waiting on simulators {tm['wait']:.0f} + gradient updates {tm['update']:.0f} + acting {tm['act']:.0f} + replay insert {tm['replay']:.0f} + "
          f"validation/export/parity {tm['eval']:.0f} + logging/audits {tm['log']:.0f} | overall {sm['totalDecisions'] / sm['wallSeconds']:.0f} dec/s, {sm['episodes'] / sm['wallSeconds']:.2f} eps/s | "
          f"excluding validation {sm['totalDecisions'] / (sm['wallSeconds'] - tm['eval']):.0f} dec/s | {1000 * tm['update'] / max(1, ob['gradientUpdates']):.2f} ms/update")
    for e in evs:
        p = e["parity"]
        top = max(e["actionShares"].values()) if isinstance(e["actionShares"], dict) else max(e["actionShares"])
        print(f"   eval u{e['update']}: n={e['validationEpisodes']} parity {p['maxAbsScoreDiff']:.1e}/{p['argmaxAgreement']:.3f} {p.get('byMask')} states {p['states']} {p.get('composition')} "
              f"safety {e['safetyProblems'] or 'OK'} legal {e['metrics']['legalXI']} secs {e['seconds']:.0f} updates {e['updates']:,} target syncs {e['targetUpdates']} max action share {top:.3f}")
    t = [x["train"] for x in ups if x["train"]["updates_in_window"]]
    f = lambda k, fn=max: fn(x[k] for x in t)
    print(f"   diagnostics over {len(t)} learning windows: quantile loss first {t[0]['loss']:.5f} last {t[-1]['loss']:.6f} | |TD| (mean_j|y_j-θ_j|) first {t[0]['td_abs_mean']:.4f} last {t[-1]['td_abs_mean']:.4f} | "
          f"Q mean first {t[0]['q_mean']:.3f} last {t[-1]['q_mean']:.3f} | Q range [{f('q_min', min):.3f}, {f('q_max'):.3f}] |Z|max {f('z_absmax'):.3f} | "
          f"online/target distance max-of-means {f('target_distance'):.4f} max {f('target_distance_max'):.4f} | grad norm max {f('grad_norm_max'):.3f} median-of-means {np.median([x['grad_norm'] for x in t]):.4f}")
    print(f"   quantiles (taken action, batch means) first → last: τ.016 {t[0]['z_q01']:.3f}→{t[-1]['z_q01']:.3f}  τ.25 {t[0]['z_q25']:.3f}→{t[-1]['z_q25']:.3f}  τ.5 {t[0]['z_q50']:.3f}→{t[-1]['z_q50']:.3f}  "
          f"τ.75 {t[0]['z_q75']:.3f}→{t[-1]['z_q75']:.3f}  τ.984 {t[0]['z_q99']:.3f}→{t[-1]['z_q99']:.3f} | spread {t[0]['z_spread']:.3f}→{t[-1]['z_spread']:.3f} IQR {t[0]['z_iqr']:.3f}→{t[-1]['z_iqr']:.3f} | "
          f"target IQR last {t[-1]['y_iqr']:.3f} | crossing rate {t[0]['crossing_rate']:.3f}→{t[-1]['crossing_rate']:.4f} (max after warm-up {max(x['crossing_rate'] for x in t if x['updates'] >= 20000):.4f}) "
          f"rows with a crossing {t[-1]['crossing_rows']:.3f} size {t[-1]['crossing_size']:.4f}")
    print(f"   priority mean last {t[-1]['priority_mean']:.4f} max {f('priority_max'):.3f} | replay size last {t[-1]['replay_size']:,}")
    shares = [max(x["actions"]["action_shares"]) for x in ups]
    print(f"   max single-action share in any window {max(shares):.3f} | explored decisions {sum(x['train']['explored'] for x in ups):,} vs expected {sum(x['train']['explored_expected'] for x in ups):,.1f}")
    for x in ups:
        if x["update"] in CKPTS:  # window 1 has no gradient updates yet (learning starts at 10,000 transitions)
            tr, ep, ac = x["train"], x["episode"], x["actions"]
            print(f"   w{x['update']:3d} dec {x['decisions']:>9,} train return {ep.get('return', float('nan')):.4f} XI {ep.get('xi', float('nan')):.2f} purse {ep.get('purseLeft', float('nan')):.1f} "
                  f"bid {ac['bid_share']:.3f} greedy-bid {ac['greedy_bid_share']:.3f} explored {ac['explored_share']:.3f} | eps {tr['epsilon']:.3f} beta {tr['beta']:.3f} "
                  f"upd {tr['updates']:,} tgt {tr['target_updates']} | loss {tr['loss']:.6f} |TD| {tr['td_abs_mean']:.4f} Q {tr['q_mean']:.3f} [{tr['q_min']:.2f},{tr['q_max']:.2f}] "
                  f"Z τ.016/.25/.5/.75/.984 {tr['z_q01']:.2f}/{tr['z_q25']:.2f}/{tr['z_q50']:.2f}/{tr['z_q75']:.2f}/{tr['z_q99']:.2f} IQR {tr['z_iqr']:.3f} "
                  f"cross {tr['crossing_rate']:.4f} dist {tr['target_distance']:.4f} gn {tr['grad_norm']:.4f} | {x['perf']['decisions_per_sec_overall']:.0f} dec/s cumulative")
for RN, pre in (("PPO 2C.3", "ppo-2c3"), ("A2C 2D.1", "a2c-2d1"), ("D3QN 2D.2", "d3qn-2d2")):
    print(f"-- {RN} throughput reference")
    for s in SEEDS:
        sm = load(RUNS / f"{pre}-s{s}" / "summary.json")
        ups = [json.loads(l) for l in open(RUNS / f"{pre}-s{s}" / "metrics.jsonl") if '"type": "update"' in l]
        evs = [json.loads(l) for l in open(RUNS / f"{pre}-s{s}" / "metrics.jsonl") if '"type": "evaluation"' in l]
        if pre.startswith("d3qn"):
            t = sm["timers"]
            print(f"  {RN} s{s}: wall {sm['wallSeconds']:.0f}s waiting {t['wait']:.0f} update {t['update']:.0f} act {t['act']:.0f} validation {t['eval']:.0f} | "
                  f"overall {sm['totalDecisions'] / sm['wallSeconds']:.0f} dec/s {sm['episodes'] / sm['wallSeconds']:.2f} eps/s | gradient steps {sm['optimisationBudget']['gradientUpdates']:,} | episodes {sm['episodes']:,}")
            continue
        roll = sum(x["perf"]["rollout_seconds"] for x in ups)
        upd = sum(x["perf"]["update_seconds"] for x in ups)
        print(f"  {RN} s{s}: wall {sm['wallSeconds']:.0f}s collection {roll:.0f} update {upd:.1f} validation {sum(e['seconds'] for e in evs):.0f} | "
              f"overall {sm['totalDecisions'] / sm['wallSeconds']:.0f} dec/s {sm['episodes'] / sm['wallSeconds']:.2f} eps/s | collection {sm['totalDecisions'] / roll:.0f} dec/s | "
              f"gradient steps {len(ups) * (96 if pre.startswith('ppo') else 1):,} | episodes {sm['episodes']:,}")
Path(RUNS / "_2d3" / "analysis.json").write_text(json.dumps(out, indent=1, default=float))
