"""Paired ΔXI between consecutive validation checkpoints (same 500 seeds) for QR-DQN, D3QN, A2C and PPO."""
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parents[1]
C = [40, 81, 122, 163, 203, 244, 285, 325]
RUN = {"qr": ("qr-dqn-2d3", "policy:qrdqn"), "d3qn": ("d3qn-2d2", "policy:d3qn"), "a2c": ("a2c-2d1", "policy:a2c"), "ppo": ("ppo-2c3", "policy:ppo")}


def ep(a, s, u):
    run, name = RUN[a]
    return np.array([e["xi"] for e in json.loads((R / f"{run}-s{s}/checkpoints/update_{u:04d}/validation/episodes.json").read_text())["episodes"][name]])


def boot(d):
    rng = np.random.default_rng(7)
    return np.percentile(d[rng.integers(0, len(d), (2000, len(d)))].mean(1), [2.5, 97.5])


for a in RUN:
    print(f"== {a}: paired ΔXI between consecutive checkpoints; * = significant decrease (CI < 0)")
    drops = 0
    for s in (1, 2, 3):
        line = f" s{s}:"
        for u0, u1 in zip(C, C[1:]):
            d = ep(a, s, u1) - ep(a, s, u0)
            lo, hi = boot(d)
            neg = hi < 0
            drops += neg
            line += f" u{u1} {d.mean():+.3f}[{lo:+.2f},{hi:+.2f}]{'*' if neg else ''}"
        print(line)
    print(f"  significant decreases: {drops}/21")
    for s in (1, 2, 3):
        xs = {u: ep(a, s, u).mean() for u in C}
        b = max(xs, key=xs.get)
        d = ep(a, s, 325) - ep(a, s, b)
        lo, hi = boot(d)
        print(f"  s{s} best checkpoint u{b} {xs[b]:.3f}; u325 − best {d.mean():+.3f} [{lo:+.3f},{hi:+.3f}]")
