"""Phase 2D.1 analysis — A2C × 3 seeds vs the PPO 2C.3 anchor and the locked baselines.
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
ALGOS = {"a2c": ("a2c-2d1-s{}", "policy:a2c"), "ppo": ("ppo-2c3-s{}", "policy:ppo")}


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
A = {(s, u): V[("a2c", s, u)] for s in SEEDS for u in CKPTS if ("a2c", s, u) in V}
P = {(s, u): V[("ppo", s, u)] for s in SEEDS for u in CKPTS if ("ppo", s, u) in V}
out = {"checkpoints": {}, "final": {}, "vsPpo": {}, "vsBaselines": {}}
COLS = ["xi", "strongXI", "purseLeft", "squadSize", "overseas", "stars", "priceToFair", "capToFair", "bidRate", "reauctionBuys", "marginalBuys", "xiGainPer1000", "rank"]

print("=" * 30, "A2C per-seed checkpoint table (500 validation seeds, act-v3, T=0.3)")
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
print("=" * 30, "A2C three-seed statistics per checkpoint (mean ± std [min, max])")
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
    print("=" * 30, f"A2C final checkpoint u{last} ({last * 6144:,} decisions): per seed, mean, std, min, max, CV")
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
            print(f"  paired ΔXI a2c seed{SEEDS[i]} − seed{SEEDS[j]}: {np.mean(d):+.3f} [{lo:+.3f}, {hi:+.3f}]")

print("=" * 30, "A2C vs PPO 2C.3 per checkpoint (same 500 validation seeds; seed s vs seed s, and seed-mean vs seed-mean)")
for u in CKPTS:
    have = [s for s in SEEDS if (s, u) in A and (s, u) in P]
    if not have:
        continue
    print(f"-- u{u} ({u * 6144:,} decisions)")
    for s in have:
        a, p = A[(s, u)][1], P[(s, u)][1]
        assert [r["seed"] for r in a] == [r["seed"] for r in p]
        d = [x["xi"] - y["xi"] for x, y in zip(a, p)]
        lo, hi = boot(d)
        fa, da = forced(a)
        fp, dp = forced(p)
        n = len(a)
        print(f"  s{s}: ΔXI {np.mean(d):+.3f} [{lo:+.3f},{hi:+.3f}] W/L {sum(x > 0 for x in d)}/{sum(x < 0 for x in d)} | "
              f"legal {sum(r['legalXI'] for r in a)}/{n} vs {sum(r['legalXI'] for r in p)}/{n} | strong {sum(r['strongXI'] for r in a)}/{n} vs {sum(r['strongXI'] for r in p)}/{n} | "
              f"purse {mean(a, 'purseLeft'):.1f} vs {mean(p, 'purseLeft'):.1f} (Δ {mean(a, 'purseLeft') - mean(p, 'purseLeft'):+.1f}) | "
              f"squad {mean(a, 'squadSize'):.2f} vs {mean(p, 'squadSize'):.2f} | stars {mean(a, 'stars'):.2f} vs {mean(p, 'stars'):.2f} | "
              f"price/fair {mean(a, 'priceToFair'):.3f} vs {mean(p, 'priceToFair'):.3f} | bid {mean(a, 'bidRate'):.3f} vs {mean(p, 'bidRate'):.3f} | "
              f"forced {fa}/{da} vs {fp}/{dp}")
        out["vsPpo"].setdefault(str(u), {})[str(s)] = {"dxi": float(np.mean(d)), "ci": [float(lo), float(hi)]}
    if len(have) == 3:
        am = np.mean([[r["xi"] for r in A[(s, u)][1]] for s in SEEDS], axis=0)
        pm = np.mean([[r["xi"] for r in P[(s, u)][1]] for s in SEEDS], axis=0)
        lo, hi = boot(am - pm)
        print(f"  seed-mean: A2C {am.mean():.3f} vs PPO {pm.mean():.3f}  ΔXI {np.mean(am - pm):+.3f} [{lo:+.3f},{hi:+.3f}]")
        out["vsPpo"].setdefault(str(u), {})["seedMean"] = {"a2c": am.mean(), "ppo": pm.mean(), "dxi": float(np.mean(am - pm)), "ci": [float(lo), float(hi)]}

if last and all((s, 325) in P for s in SEEDS):
    print("=" * 30, f"A2C u{last} vs PPO u325 — all 9 seed pairs (ΔXI [95% CI])")
    for s in SEEDS:
        line = f"  A2C s{s}:"
        for t in SEEDS:
            d = [x["xi"] - y["xi"] for x, y in zip(A[(s, last)][1], P[(t, 325)][1])]
            lo, hi = boot(d)
            line += f"  vs PPO s{t} {np.mean(d):+.3f} [{lo:+.3f},{hi:+.3f}]"
        print(line)
    am = np.mean([[r["xi"] for r in A[(s, last)][1]] for s in SEEDS], axis=0)
    pm = np.mean([[r["xi"] for r in P[(s, 325)][1]] for s in SEEDS], axis=0)
    ref = A[(1, last)][1]
    for strat in ("low", "normal", "high"):
        idx = [i for i, r in enumerate(ref) if r["stratum"] == strat]
        d = (am - pm)[idx]
        lo, hi = boot(d)
        print(f"  stratum {strat:6s} seed-mean ΔXI {d.mean():+.3f} [{lo:+.3f},{hi:+.3f}] (n={len(idx)})")
    print("  PPO u325 three-seed reference: " + " | ".join(
        f"{c} {np.mean([mean(P[(s, 325)][1], c) for s in SEEDS]):.3f}±{sd([mean(P[(s, 325)][1], c) for s in SEEDS]):.3f}" for c in STAT))

if last:
    print("=" * 30, f"A2C u{last} vs the eight locked baselines (paired by validation seed, unchanged baseline file)")
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

print("=" * 30, "A2C behaviour / shield audit per checkpoint (validation)")
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

print("=" * 30, "A2C safety / health / throughput / reproducibility per run")
for s in SEEDS:
    run = RUNS / f"a2c-2d1-s{s}"
    if not (run / "summary.json").exists():
        print(f"seed {s}: running / not started")
        continue
    sm = load(run / "summary.json")
    te = json.loads((run / "training_episodes.json").read_text())
    ups = [json.loads(l) for l in open(run / "metrics.jsonl") if '"type": "update"' in l]
    evs = [json.loads(l) for l in open(run / "metrics.jsonl") if '"type": "evaluation"' in l]
    roll = sum(x["perf"]["rollout_seconds"] for x in ups)
    upd = sum(x["perf"]["update_seconds"] for x in ups)
    evs_s = sum(e["seconds"] for e in evs)
    inc = sum(1 for e in te if not e["legalXI"])
    inv = sum(e["invariantViolations"] for e in te)
    print(f"seed {s}: status {sm['status']} failure {sm['failure']} | decisions {sm['totalDecisions']:,} updates {len(ups)} episodes {sm['episodes']:,} | "
          f"training: incomplete XI {inc}/{len(te)}, invariant violations {inv}, consistency checks {sm['episodeConsistencyChecks']}/{len(te)}, illegal {sm['illegalActions']} | "
          f"actSpec {sm['actSpec']['version']} {sm['actSpec']['hash']} | initial {sm['initialParamsDigest']} final {sm['finalParamsDigest']}")
    print(f"   wall {sm['wallSeconds']:.0f}s = collection {roll:.0f} + update {upd:.1f} + validation {evs_s:.0f} (+other {sm['wallSeconds'] - roll - upd - evs_s:.0f}) | "
          f"overall {sm['totalDecisions'] / sm['wallSeconds']:.0f} dec/s, {sm['episodes'] / sm['wallSeconds']:.2f} eps/s | "
          f"collection {sm['totalDecisions'] / roll:.0f} dec/s, {sm['episodes'] / roll:.2f} eps/s | update {1000 * upd / len(ups):.0f} ms/update")
    for e in evs:
        print(f"   eval u{e['update']}: n={e['validationEpisodes']} parity {e['parity']['maxAbsScoreDiff']:.1e}/{e['parity']['argmaxAgreement']:.3f} "
              f"safety {e['safetyProblems'] or 'OK'} legal {e['metrics']['legalXI']} secs {e['seconds']:.0f}")
    t = [x["train"] for x in ups]
    shares = [max(x["actions"]["action_shares"]) for x in ups]
    nonfinite = sum(1 for x in t if not all(math.isfinite(x[k]) for k in ("loss", "policy_loss", "value_loss", "entropy", "grad_norm")) or not x["grads_finite"] or not x["params_finite"])
    print(f"   health: non-finite updates {nonfinite}/{len(t)} | normalised entropy min {min(x['entropy_normalised'] for x in t):.3f} last {t[-1]['entropy_normalised']:.3f} | "
          f"entropy first {t[0]['entropy']:.3f} last {t[-1]['entropy']:.3f} | value loss first {t[0]['value_loss']:.4f} max(after u20) {max(x['value_loss'] for x in t[20:]):.4f} last {t[-1]['value_loss']:.5f} | "
          f"grad norm (pre-clip) median {np.median([x['grad_norm'] for x in t]):.2f} max {max(x['grad_norm'] for x in t):.2f} | post-update KL median {np.median([x['post_update_kl'] for x in t]):.5f} max {max(x['post_update_kl'] for x in t):.5f} | "
          f"logp drift max {max(x['logp_drift'] for x in t):.1e} | max single-action share {max(shares):.3f} | adv std first {t[0]['adv_std']:.3f} last {t[-1]['adv_std']:.4f} absmax max {max(x['adv_absmax'] for x in t):.3f}")
    for x in ups:
        if x["update"] in CKPTS or x["update"] == 1:
            tr, ep = x["train"], x["episode"]
            print(f"   u{x['update']:3d} train return {ep.get('return', float('nan')):.4f} XI {ep.get('xi', float('nan')):.2f} legal {ep.get('legalXI', float('nan')):.3f} "
                  f"purse {ep.get('purseLeft', float('nan')):.1f} bid {x['actions']['bid_share']:.3f} maxshare {max(x['actions']['action_shares']):.3f} | ent {tr['entropy']:.3f} (norm {tr['entropy_normalised']:.3f}) "
                  f"EV {tr['explained_variance']:.4f} pl {tr['policy_loss']:+.4f} vl {tr['value_loss']:.5f} gn {tr['grad_norm']:.2f} kl {tr['post_update_kl']:.5f} | "
                  f"{x['perf']['rollout_decisions_per_sec']:.0f} dec/s")
print("-- PPO 2C.3 throughput reference")
for s in SEEDS:
    sm = load(RUNS / f"ppo-2c3-s{s}" / "summary.json")
    ups = [json.loads(l) for l in open(RUNS / f"ppo-2c3-s{s}" / "metrics.jsonl") if '"type": "update"' in l]
    evs = [json.loads(l) for l in open(RUNS / f"ppo-2c3-s{s}" / "metrics.jsonl") if '"type": "evaluation"' in l]
    roll = sum(x["perf"]["rollout_seconds"] for x in ups)
    upd = sum(x["perf"]["update_seconds"] for x in ups)
    print(f"  PPO s{s}: wall {sm['wallSeconds']:.0f}s collection {roll:.0f} update {upd:.1f} validation {sum(e['seconds'] for e in evs):.0f} | overall {sm['totalDecisions'] / sm['wallSeconds']:.0f} dec/s "
          f"{sm['episodes'] / sm['wallSeconds']:.2f} eps/s | collection {sm['totalDecisions'] / roll:.0f} dec/s | update {1000 * upd / len(ups):.0f} ms/update | episodes {sm['episodes']:,}")
Path(RUNS / "_2d1" / "analysis.json").write_text(json.dumps(out, indent=1, default=float))
