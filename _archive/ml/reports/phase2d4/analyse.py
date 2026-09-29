"""Phase 2D.4 analysis — OpenAI-ES × 3 seeds vs PPO / A2C / D3QN / QR-DQN and the locked baselines.
Reads run outputs only (no training, no evaluation).

ES validation checkpoints are generation boundaries: the first generation at or
after each of the eight decision checkpoints (the equal-decision comparison
points) and every 250 generations (the ES budget points, up to 2,000)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
RUNS = Path(__file__).resolve().parents[1]
BASE = RUNS.parents[1] / "packages" / "shared" / "data" / "rl-baselines" / "validation.episodes.json"
SEEDS = (1, 2, 3)
DC = [245_760, 497_664, 749_568, 1_001_472, 1_247_232, 1_499_136, 1_751_040, 1_996_800]
UPD = [40, 81, 122, 163, 203, 244, 285, 325]
BASELINES = ["moneyball", "starChaser", "balancedBuilder", "opportunist", "productFallback", "randomLegal", "fairValue", "plannerGreedy"]
REF = {"PPO": ("ppo-2c3-s{}", "policy:ppo"), "A2C": ("a2c-2d1-s{}", "policy:a2c"), "D3QN": ("d3qn-2d2-s{}", "policy:d3qn"), "QR-DQN": ("qr-dqn-2d3-s{}", "policy:qrdqn")}
COLS = ["xi", "strongXI", "purseLeft", "squadSize", "overseas", "stars", "priceToFair", "capToFair", "bidRate", "reauctionBuys", "marginalBuys", "xiGainPer1000", "rank"]
STAT = ["xi", "legalXI", "strongXI", "purseLeft", "squadSize", "overseas", "stars", "priceToFair", "bidRate", "reauctionBuys"]


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def mean(rows, k):
    xs = [float(r[k]) for r in rows if r.get(k) is not None]
    return float(np.mean(xs)) if xs else float("nan")


def sd(a):
    return float(np.std(a, ddof=1)) if len(a) > 1 else float("nan")


def boot(d, it=2000, seed=7):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    return np.percentile(d[rng.integers(0, len(d), (it, len(d)))].mean(1), [2.5, 97.5])


def forced(rows):
    return sum(r["shield"]["forced"] for r in rows), sum(r["decisions"] for r in rows)


def es_evals(s):
    run = RUNS / f"openai-es-2d4-s{s}"
    if not (run / "metrics.jsonl").exists():
        return []
    out = []
    for l in open(run / "metrics.jsonl"):
        if '"type": "evaluation"' in l:
            e = json.loads(l)
            e["episodes_rows"] = load(run / "checkpoints" / f"gen_{e['generation']:04d}" / "validation" / "episodes.json")["episodes"]["policy:es"]
            out.append(e)
    return out


E = {s: es_evals(s) for s in SEEDS}
# decision-matched ES checkpoint for target T: the evaluation whose reasons include "≥T decisions"
DM = {(s, T): next((e for e in E[s] if f"≥{T:,} decisions" in e["reasons"]), None) for s in SEEDS for T in DC}
GEN = {(s, e["generation"]): e for s in SEEDS for e in E[s] if any(r.startswith("generation") for r in e["reasons"])}
GENS = sorted({g for (_, g) in GEN})
FINAL_GEN = max((g for g in GENS if all((s, g) in GEN for s in SEEDS)), default=None)


def ref_rows(name, s, u):
    run, key = REF[name]
    p = RUNS / run.format(s) / "checkpoints" / f"update_{u:04d}" / "validation" / "episodes.json"
    return load(p)["episodes"][key] if p.exists() else None


def table_row(rows):
    n = len(rows)
    lg, st = sum(r["legalXI"] for r in rows), sum(r["strongXI"] for r in rows)
    f, d = forced(rows)
    return f"{lg:>4}/{n:<4} {st:>4}/{n:<4} " + " ".join(f"{mean(rows, c):9.3f}" for c in COLS) + f"  {f}/{d} = {100 * f / max(1, d):.3f}%"


print("=" * 30, "OpenAI-ES per-seed validation checkpoints (500 validation seeds, act-v3, masked argmax), in generation order")
for s in SEEDS:
    print(f"-- seed {s}")
    print(f"{'gen':>5} {'decisions':>11} {'trigger':>28} {'legal':>9} {'strong':>9} " + " ".join(f"{c[:9]:>9}" for c in COLS) + "  forced/decisions")
    for e in E[s]:
        print(f"{e['generation']:>5} {e['decisions']:>11,} {'; '.join(e['reasons'])[:28]:>28} {table_row(e['episodes_rows'])}")


def stat_line(rowsets):
    parts = []
    for c in STAT:
        a = np.array([mean(r, c) for r in rowsets])
        parts.append(f"{c} {a.mean():.3f}±{sd(a):.3f} [{a.min():.3f},{a.max():.3f}]")
    return " | ".join(parts)


print("=" * 30, "three-seed statistics at the decision-matched checkpoints (mean ± std [min, max]); actual decisions per seed")
for T in DC:
    have = [s for s in SEEDS if DM[(s, T)]]
    if len(have) < 3:
        continue
    print(f"≥{T:,}: gens {[DM[(s, T)]['generation'] for s in SEEDS]} decisions {[DM[(s, T)]['decisions'] for s in SEEDS]} | " + stat_line([DM[(s, T)]["episodes_rows"] for s in SEEDS]))
print("=" * 30, "three-seed statistics at the generation checkpoints")
for g in GENS:
    if all((s, g) in GEN for s in SEEDS):
        print(f"gen {g}: decisions {[GEN[(s, g)]['decisions'] for s in SEEDS]} | " + stat_line([GEN[(s, g)]["episodes_rows"] for s in SEEDS]))


def final_block(label, rowsets):
    print("=" * 30, f"{label}: per seed, mean, std, min, max, CV")
    for c in STAT + ["capToFair", "marginalBuys", "xiGainPer1000", "rank"]:
        a = np.array([mean(r, c) for r in rowsets])
        m, v = a.mean(), sd(a)
        print(f"  {c:15s} " + "  ".join(f"s{s} {x:.3f}" for s, x in zip(SEEDS, a)) + f"   mean {m:.3f} std {v:.3f} min {a.min():.3f} max {a.max():.3f} CV {100 * v / abs(m) if m else float('nan'):.2f}%")
    for s, r in zip(SEEDS, rowsets):
        print(f"  seed {s}: legal {sum(x['legalXI'] for x in r)}/{len(r)}  strong {sum(x['strongXI'] for x in r)}/{len(r)}")
    for i in range(3):
        for j in range(i + 1, 3):
            d = [x["xi"] - y["xi"] for x, y in zip(rowsets[i], rowsets[j])]
            lo, hi = boot(d)
            print(f"  paired ΔXI es seed{SEEDS[i]} − seed{SEEDS[j]}: {np.mean(d):+.3f} [{lo:+.3f}, {hi:+.3f}]")


EQ = [DM[(s, DC[-1])]["episodes_rows"] for s in SEEDS] if all(DM[(s, DC[-1])] for s in SEEDS) else None
FIN = [GEN[(s, FINAL_GEN)]["episodes_rows"] for s in SEEDS] if FINAL_GEN else None
if EQ:
    final_block(f"ES at the ≥1,996,800-decision checkpoint (equal decisions to the other algorithms' finals)", EQ)
if FIN:
    final_block(f"ES at generation {FINAL_GEN} (the ES budget)", FIN)


def compare(label, es_sets, ref_sets):
    for s in SEEDS:
        a, p = es_sets[s], ref_sets[s]
        assert [r["seed"] for r in a] == [r["seed"] for r in p]
        d = [x["xi"] - y["xi"] for x, y in zip(a, p)]
        lo, hi = boot(d)
        fa, da = forced(a)
        fp, dp = forced(p)
        n = len(a)
        print(f"  s{s}: ΔXI {np.mean(d):+.3f} [{lo:+.3f},{hi:+.3f}] W/L {sum(x > 0 for x in d)}/{sum(x < 0 for x in d)} | legal {sum(r['legalXI'] for r in a)}/{n} vs "
              f"{sum(r['legalXI'] for r in p)}/{n} | strong {sum(r['strongXI'] for r in a)}/{n} vs {sum(r['strongXI'] for r in p)}/{n} | purse {mean(a, 'purseLeft'):.1f} vs "
              f"{mean(p, 'purseLeft'):.1f} | stars {mean(a, 'stars'):.2f} vs {mean(p, 'stars'):.2f} | price/fair {mean(a, 'priceToFair'):.3f} vs {mean(p, 'priceToFair'):.3f} | "
              f"bid {mean(a, 'bidRate'):.3f} vs {mean(p, 'bidRate'):.3f} | marginal {mean(a, 'marginalBuys'):.3f} vs {mean(p, 'marginalBuys'):.3f} | re-auction {mean(a, 'reauctionBuys'):.3f} vs {mean(p, 'reauctionBuys'):.3f} | forced {fa}/{da} vs {fp}/{dp}")
    am = np.mean([[r["xi"] for r in es_sets[s]] for s in SEEDS], axis=0)
    pm = np.mean([[r["xi"] for r in ref_sets[s]] for s in SEEDS], axis=0)
    lo, hi = boot(am - pm)
    print(f"  seed-mean: ES {am.mean():.3f} vs {label} {pm.mean():.3f}  ΔXI {np.mean(am - pm):+.3f} [{lo:+.3f},{hi:+.3f}]")
    ref0 = es_sets[1]
    for strat in ("low", "normal", "high"):
        idx = [i for i, r in enumerate(ref0) if r["stratum"] == strat]
        dd = (am - pm)[idx]
        lo, hi = boot(dd)
        print(f"    stratum {strat:6s} ΔXI {dd.mean():+.3f} [{lo:+.3f},{hi:+.3f}] (n={len(idx)})")
    return float(np.mean(am - pm))


for name in REF:
    print("=" * 30, f"ES vs {name} at the eight decision-matched checkpoints (same 500 validation seeds)")
    for T, u in zip(DC, UPD):
        if not all(DM[(s, T)] for s in SEEDS) or not all(ref_rows(name, s, u) for s in SEEDS):
            continue
        print(f"-- {T:,} decisions (ES gens {[DM[(s, T)]['generation'] for s in SEEDS]}, decisions {[DM[(s, T)]['decisions'] for s in SEEDS]}; {name} update {u})")
        compare(name, {s: DM[(s, T)]["episodes_rows"] for s in SEEDS}, {s: ref_rows(name, s, u) for s in SEEDS})
    if FIN and all(ref_rows(name, s, 325) for s in SEEDS):
        print(f"-- ES generation {FINAL_GEN} (its full budget) vs {name} final (1,996,800 decisions)")
        compare(name, {s: GEN[(s, FINAL_GEN)]["episodes_rows"] for s in SEEDS}, {s: ref_rows(name, s, 325) for s in SEEDS})
        print(f"   all 9 seed pairs (ES gen {FINAL_GEN} vs {name} final):")
        for s in SEEDS:
            line = f"   ES s{s}:"
            for t in SEEDS:
                d = [x["xi"] - y["xi"] for x, y in zip(GEN[(s, FINAL_GEN)]["episodes_rows"], ref_rows(name, t, 325))]
                lo, hi = boot(d)
                line += f"  vs s{t} {np.mean(d):+.3f} [{lo:+.3f},{hi:+.3f}]"
            print(line)

locked = load(BASE)["episodes"]
for label, sets in (("ES at ≥1,996,800 decisions", EQ), (f"ES at generation {FINAL_GEN}", FIN)):
    if not sets:
        continue
    print("=" * 30, f"{label} vs the eight locked baselines (paired, unchanged baseline file)")
    for b in BASELINES:
        ref = locked[b]
        line = f"  {b:16s} baseline XI {mean(ref, 'xi'):.3f} |"
        pooled = []
        for s, rows in zip(SEEDS, sets):
            assert [r["seed"] for r in rows] == [r["seed"] for r in ref]
            d = [x["xi"] - y["xi"] for x, y in zip(rows, ref)]
            pooled.append(d)
            lo, hi = boot(d)
            line += f" s{s} {np.mean(d):+.3f} [{lo:+.3f},{hi:+.3f}] W/L {sum(x > 0 for x in d)}/{sum(x < 0 for x in d)} |"
        m = np.mean(pooled, axis=0)
        lo, hi = boot(m)
        print(line + f" seed-mean {m.mean():+.3f} [{lo:+.3f},{hi:+.3f}]")
        for strat in ("low", "normal", "high"):
            idx = [i for i, r in enumerate(ref) if r["stratum"] == strat]
            dm = m[idx]
            lo, hi = boot(dm)
            print(f"      {strat:6s} ΔXI {dm.mean():+.3f} [{lo:+.3f},{hi:+.3f}]  (n={len(idx)})")

print("=" * 30, "ES behaviour / shield per validation checkpoint")
for s in SEEDS:
    print(f"-- seed {s}")
    for e in E[s]:
        rows = e["episodes_rows"]
        purse = np.array([r["purseLeft"] for r in rows])
        byreq = {}
        for r in rows:
            for k, v in r["shield"]["forcedByRequirement"].items():
                byreq[k] = byreq.get(k, 0) + v
        states = {k: sum(r["shield"]["states"][k] for r in rows) for k in ("SAFE", "WARNING", "CRITICAL", "IMPOSSIBLE")}
        f, d = forced(rows)
        print(f"  gen {e['generation']:<5} purse mean {purse.mean():.1f} p50 {np.percentile(purse, 50):.0f} p90 {np.percentile(purse, 90):.0f} max {purse.max():.0f} | "
              f"=0: {int((purse == 0).sum())}/500 ≤20: {int((purse <= 20).sum())}/500 | forced {f}/{d} won {sum(r['shieldWins'] for r in rows)} re-auction "
              f"{sum(r['shield']['reauctionForced'] for r in rows)} | states {states} | byReq {byreq}")

print("=" * 30, "ES stability: paired ΔXI between consecutive validation checkpoints (generation order); * = significant decrease")
for s in SEEDS:
    ev = E[s]
    drops, line = 0, f" s{s}:"
    for a, b in zip(ev, ev[1:]):
        d = np.array([y["xi"] - x["xi"] for x, y in zip(a["episodes_rows"], b["episodes_rows"])])
        lo, hi = boot(d)
        neg = hi < 0
        drops += neg
        line += f" g{b['generation']} {d.mean():+.3f}[{lo:+.2f},{hi:+.2f}]{'*' if neg else ''}"
    print(line)
    xs = {e["generation"]: mean(e["episodes_rows"], "xi") for e in ev}
    if xs:
        gb = max(xs, key=xs.get)
        last = ev[-1]
        best = next(e for e in ev if e["generation"] == gb)
        d = np.array([x["xi"] - y["xi"] for x, y in zip(last["episodes_rows"], best["episodes_rows"])])
        lo, hi = boot(d)
        print(f"   significant decreases {drops}/{len(ev) - 1} | best checkpoint gen {gb} XI {xs[gb]:.3f} | last (gen {last['generation']}) − best {d.mean():+.3f} [{lo:+.3f},{hi:+.3f}]")

print("=" * 30, "ES safety / budget / throughput / reproducibility per run")
for s in SEEDS:
    run = RUNS / f"openai-es-2d4-s{s}"
    if not (run / "summary.json").exists():
        print(f"seed {s}: running / not started")
        continue
    sm = load(run / "summary.json")
    te = json.loads((run / "training_episodes.json").read_text())
    ups = [json.loads(l) for l in open(run / "metrics.jsonl") if '"type": "update"' in l]
    evs = [json.loads(l) for l in open(run / "metrics.jsonl") if '"type": "evaluation"' in l]
    tm = sm["timers"]
    print(f"seed {s}: status {sm['status']} failure {sm['failure']} | generations {sm['generations']:,} decisions {sm['totalDecisions']:,} episodes {sm['episodes']:,} "
          f"policy evaluations {sm['policyEvaluations']:,} optimizer updates {sm['optimizerUpdates']:,} | training: incomplete XI {sum(1 for e in te if not e['legalXI'])}/{len(te)}, "
          f"invariant violations {sum(e['invariantViolations'] for e in te)}, consistency checks {sm['episodeConsistencyChecks']}/{len(te)}, illegal {sm['illegalActions']} | "
          f"act {sm['actSpec']['hash']} obs {sm['obsSpec']['hash']}")
    print(f"   digests: initial {sm['initialParamsDigest']} final {sm['finalParamsDigest']}")
    print(f"   wall {sm['wallSeconds']:.0f}s = episodes {tm['episodes']:.0f} (of which waiting on simulators {tm['wait']:.0f}, acting {tm['act']:.0f}) + noise {tm['noise']:.0f} + update {tm['update']:.1f} "
          f"+ validation {tm['eval']:.0f} | overall {sm['totalDecisions'] / sm['wallSeconds']:.0f} dec/s, {sm['episodes'] / sm['wallSeconds']:.2f} eps/s, "
          f"{sm['generations'] / sm['wallSeconds'] * 3600:.0f} generations/h | excluding validation {sm['totalDecisions'] / (sm['wallSeconds'] - tm['eval']):.0f} dec/s | "
          f"{1000 * tm['update'] / max(1, sm['optimizerUpdates']):.2f} ms/update")
    for e in evs:
        p = e["parity"]
        top = max(e["actionShares"].values()) if isinstance(e["actionShares"], dict) else max(e["actionShares"])
        print(f"   eval gen {e['generation']} ({e['decisions']:,}): parity {p['maxAbsScoreDiff']:.1e}/{p['argmaxAgreement']:.3f} {p.get('byMask')} "
              f"safety {e['safetyProblems'] or 'OK'} legal {e['metrics']['legalXI']} XI {e['metrics']['xi']:.3f} secs {e['seconds']:.0f} max action share {top:.3f}")
    t = [u["train"] for u in ups]
    print(f"   training: fitness mean first {t[0]['fitness_mean']:.4f} last {t[-1]['fitness_mean']:.4f} max-of-means {max(x['fitness_mean'] for x in t):.4f} | "
          f"|Δpair| first {t[0]['pair_diff_abs_mean']:.4f} last {t[-1]['pair_diff_abs_mean']:.4f} | pair ties mean {np.mean([x['pair_ties'] for x in t]):.2f}/32 | "
          f"|g| first {t[0]['grad_norm']:.2f} last {t[-1]['grad_norm']:.2f} | |θ| first {t[0]['theta_norm']:.2f} last {t[-1]['theta_norm']:.2f} | "
          f"step/θ first {t[0]['update_ratio']:.2e} last {t[-1]['update_ratio']:.2e} | decisions/generation first {t[0]['decisions_in_generation']} last {t[-1]['decisions_in_generation']}")
    for gsel in (1, 50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000):
        u = next((x for x in ups if x["generation"] == gsel), None)
        if u:
            ep, tr = u["episode"], u["train"]
            print(f"   g{gsel:5d} dec {u['decisions']:>11,} fitness {tr['fitness_mean']:.4f} [{tr['fitness_min']:.3f},{tr['fitness_max']:.3f}] XI(train, sampled) {ep.get('xi', float('nan')):.2f} "
                  f"purse {ep.get('purseLeft', float('nan')):.1f} bid {u['actions']['bid_share']:.3f} | |Δpair| {tr['pair_diff_abs_mean']:.4f} ties {tr['pair_ties']} |g| {tr['grad_norm']:.2f} "
                  f"|θ| {tr['theta_norm']:.2f} step/θ {tr['update_ratio']:.2e} | {u['perf']['decisions_per_sec_overall']:.0f} dec/s cumulative")
print("-- throughput references (overall decisions/s): PPO 894–898 · A2C 588–696 · D3QN 536–556 · QR-DQN 437–523")
