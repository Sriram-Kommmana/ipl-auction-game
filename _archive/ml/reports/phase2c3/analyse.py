"""Phase 2C.3 analysis — reads run outputs only (no training, no evaluation)."""
import json
import sys
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parents[1]
BASE = RUNS.parents[1] / "packages" / "shared" / "data" / "rl-baselines" / "validation.episodes.json"
SEEDS = [s for s in (1, 2, 3) if (RUNS / f"ppo-2c3-s{s}").exists()]
CKPTS = [40, 81, 122, 163, 203, 244, 285, 325]
BASELINES = ["moneyball", "starChaser", "balancedBuilder", "opportunist", "productFallback", "randomLegal", "fairValue", "plannerGreedy"]


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def val(seed, u):
    d = RUNS / f"ppo-2c3-s{seed}" / "checkpoints" / f"update_{u:04d}" / "validation"
    if not (d / "episodes.json").exists():
        return None
    return load(d / "report.json"), load(d / "episodes.json")["episodes"]["policy:ppo"]


def mean(rows, k):
    xs = [float(r[k]) for r in rows if r.get(k) is not None]
    return float(np.mean(xs)) if xs else float("nan")


def boot(d, it=2000, seed=7):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    return np.percentile(d[rng.integers(0, len(d), (it, len(d)))].mean(1), [2.5, 97.5])


def forced(rows):
    return sum(r["shield"]["forced"] for r in rows), sum(r["decisions"] for r in rows)


V = {(s, u): val(s, u) for s in SEEDS for u in CKPTS}
V = {k: v for k, v in V.items() if v}
out = {"seeds": SEEDS, "checkpoints": {}}

print("=" * 30, "per-seed checkpoint table (500 validation seeds, act-v3, T=0.3)")
COLS = ["xi", "strongXI", "purseLeft", "squadSize", "overseas", "stars", "priceToFair", "capToFair", "bidRate", "reauctionBuys", "marginalBuys", "xiGainPer1000", "rank"]
for s in SEEDS:
    print(f"-- seed {s}")
    print(f"{'upd':>4} {'decisions':>10} {'legal':>9} {'strong':>9} " + " ".join(f"{c[:9]:>9}" for c in COLS) + "  forced/decisions")
    for u in CKPTS:
        if (s, u) not in V:
            continue
        rep, rows = V[(s, u)]
        n = len(rows)
        lg = sum(r["legalXI"] for r in rows)
        st = sum(r["strongXI"] for r in rows)
        f, d = forced(rows)
        print(f"{u:>4} {u * 6144:>10,} {lg:>4}/{n:<4} {st:>4}/{n:<4} " + " ".join(f"{mean(rows, c):9.3f}" for c in COLS) + f"  {f}/{d} = {100 * f / d:.3f}%")
        out["checkpoints"].setdefault(str(u), {})[str(s)] = {"legal": f"{lg}/{n}", "strong": f"{st}/{n}", **{c: mean(rows, c) for c in COLS}, "forced": f, "decisions": d}

print("=" * 30, "three-seed statistics per checkpoint (mean ± std [min, max])")
STAT = ["xi", "legalXI", "strongXI", "purseLeft", "squadSize", "overseas", "stars", "priceToFair", "bidRate", "reauctionBuys"]
for u in CKPTS:
    have = [s for s in SEEDS if (s, u) in V]
    if not have:
        continue
    parts = []
    for c in STAT:
        a = np.array([mean(V[(s, u)][1], c) for s in have])
        sd = a.std(ddof=1) if len(a) > 1 else float("nan")
        parts.append(f"{c} {a.mean():.3f}±{sd:.3f} [{a.min():.3f},{a.max():.3f}]")
    print(f"u{u} ({u * 6144:,}; seeds {have}): " + " | ".join(parts))

last = max((u for u in CKPTS if all((s, u) in V for s in SEEDS)), default=None)
if last:
    print("=" * 30, f"final checkpoint u{last} ({last * 6144:,} decisions)")
    for c in STAT + ["capToFair", "marginalBuys", "xiGainPer1000", "rank"]:
        a = np.array([mean(V[(s, last)][1], c) for s in SEEDS])
        sd = a.std(ddof=1) if len(a) > 1 else float("nan")
        cv = 100 * sd / abs(a.mean()) if a.mean() else float("nan")
        print(f"  {c:15s} " + "  ".join(f"s{s} {v:.3f}" for s, v in zip(SEEDS, a)) + f"   mean {a.mean():.3f} std {sd:.3f} var {sd ** 2:.5f} min {a.min():.3f} max {a.max():.3f} CV {cv:.2f}%")
    # seed-to-seed paired differences in final XI
    for i in range(len(SEEDS)):
        for j in range(i + 1, len(SEEDS)):
            d = [x["xi"] - y["xi"] for x, y in zip(V[(SEEDS[i], last)][1], V[(SEEDS[j], last)][1])]
            lo, hi = boot(d)
            print(f"  paired ΔXI seed{SEEDS[i]} − seed{SEEDS[j]}: {np.mean(d):+.3f} [{lo:+.3f}, {hi:+.3f}]")

    print("=" * 30, "vs the eight locked baselines (paired by validation seed, unchanged baseline file)")
    locked = load(BASE)["episodes"]
    for b in BASELINES:
        ref = locked[b]
        line = f"  {b:16s} baseline XI {mean(ref, 'xi'):.3f} |"
        pooled = []
        for s in SEEDS:
            rows = V[(s, last)][1]
            assert [r["seed"] for r in rows] == [r["seed"] for r in ref]
            d = [x["xi"] - y["xi"] for x, y in zip(rows, ref)]
            pooled.append(d)
            lo, hi = boot(d)
            w = sum(x > 0 for x in d); l = sum(x < 0 for x in d)
            line += f" s{s} {np.mean(d):+.3f} [{lo:+.3f},{hi:+.3f}] W/L {w}/{l} |"
        m = np.mean(pooled, axis=0)
        lo, hi = boot(m)
        line += f" seed-mean {m.mean():+.3f} [{lo:+.3f},{hi:+.3f}]"
        print(line)
        for strat in ("low", "normal", "high"):
            idx = [i for i, r in enumerate(ref) if r["stratum"] == strat]
            dm = np.mean([[pooled[k][i] for i in idx] for k in range(len(SEEDS))], axis=0)
            lo, hi = boot(dm)
            print(f"      {strat:6s} seed-mean ΔXI {dm.mean():+.3f} [{lo:+.3f},{hi:+.3f}]  (n={len(idx)})")

print("=" * 30, "behaviour / shield audit per checkpoint (validation)")
for s in SEEDS:
    print(f"-- seed {s}")
    for u in CKPTS:
        if (s, u) not in V:
            continue
        rows = V[(s, u)][1]
        purse = np.array([r["purseLeft"] for r in rows])
        sh = {}
        for r in rows:
            for k, v in r["shield"]["forcedByRequirement"].items():
                sh[k] = sh.get(k, 0) + v
        states = {k: sum(r["shield"]["states"][k] for r in rows) for k in ("SAFE", "WARNING", "CRITICAL", "IMPOSSIBLE")}
        f, d = forced(rows)
        won = sum(r["shieldWins"] for r in rows)
        reauc_forced = sum(r["shield"]["reauctionForced"] for r in rows)
        removed = sum(r["shield"]["decisionsRemoved"] for r in rows)
        print(f"  u{u:<3} purse left mean {purse.mean():.1f} p50 {np.percentile(purse, 50):.0f} p90 {np.percentile(purse, 90):.0f} max {purse.max():.0f} | "
              f"=0: {int((purse == 0).sum())}/500 ≤20: {int((purse <= 20).sum())}/500 | forced {f}/{d} ({100 * f / d:.3f}%) won {won} re-auction {reauc_forced} | "
              f"states {states} | lots with all bids removed {removed} | byReq {sh}")

print("=" * 30, "safety / throughput / reproducibility per run")
for s in SEEDS:
    run = RUNS / f"ppo-2c3-s{s}"
    if not (run / "summary.json").exists():
        print(f"seed {s}: running")
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
    print(f"seed {s}: status {sm['status']} failure {sm['failure']} | decisions {sm['totalDecisions']:,} episodes {sm['episodes']:,} | "
          f"training: incomplete XI {inc}/{len(te)}, invariant violations {inv}, consistency checks {sm['episodeConsistencyChecks']}/{len(te)}, illegal {sm['illegalActions']} | "
          f"actSpec {sm['actSpec']['version']} {sm['actSpec']['hash']} | initial {sm['initialParamsDigest']} final {sm['finalParamsDigest']}")
    print(f"   wall {sm['wallSeconds']:.0f}s = rollout {roll:.0f} + update {upd:.0f} + eval {evs_s:.0f} (+other) | overall {sm['totalDecisions'] / sm['wallSeconds']:.0f} dec/s, "
          f"{sm['episodes'] / sm['wallSeconds']:.2f} eps/s | collection {sm['totalDecisions'] / roll:.0f} dec/s, {sm['episodes'] / roll:.2f} eps/s")
    for e in evs:
        print(f"   eval u{e['update']}: n={e['validationEpisodes']} parity {e['parity']['maxAbsScoreDiff']:.1e}/{e['parity']['argmaxAgreement']:.3f} "
              f"safety {e['safetyProblems'] or 'OK'} legal {e['metrics']['legalXI']} secs {e['seconds']:.0f}")
    # learning curve rows at checkpoints
    for x in ups:
        if x["update"] in CKPTS or x["update"] == 1:
            t, ep = x["train"], x["episode"]
            print(f"   u{x['update']:3d} train return {ep.get('return', float('nan')):.4f} XI {ep.get('xi', float('nan')):.2f} legal {ep.get('legalXI', float('nan')):.3f} "
                  f"purse {ep.get('purseLeft', float('nan')):.1f} bid {x['actions']['bid_share']:.3f} | ent {t['entropy']:.3f} (norm {t['entropy_normalised']:.3f}) "
                  f"EV {t['explained_variance']:.4f} pl {t['policy_loss']:+.4f} vl {t['value_loss']:.5f} kl {t['approx_kl']:.4f} clip {t['clip_fraction']:.3f} gn {t['grad_norm']:.2f} | "
                  f"{x['perf']['rollout_decisions_per_sec']:.0f} dec/s")
Path(RUNS / "_2c3" / "analysis.json").write_text(json.dumps(out, indent=1))
