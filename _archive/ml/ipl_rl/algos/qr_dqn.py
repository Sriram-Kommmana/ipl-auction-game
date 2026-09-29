"""Quantile-Regression DQN for IplAuctionEnv-v2 (Phase 2D.3) — one file.

    cd ml
    .venv/Scripts/python -m ipl_rl.algos.qr_dqn --config ipl_rl/configs/qr_dqn_2d3.json --set seed=1

Network (standard QR-DQN — no dueling streams): the deployable
PolicyNet("qrdqn"): trunk 80 → 128 → 128 (tanh), linear head 128 → 640,
reshaped to [batch, 20 actions, 32 quantiles] (action-major, exactly the
rl-policy-v2 "quantiles" layout). Orthogonal init as D3QN (√2 hidden, 1.0
output layer), zero biases, drawn from a QR-DQN-specific seed
("qr_dqn", run seed).

Quantiles: N = 32 fixed fractions τ_i = (i − 0.5)/32, i = 1..32
(0.015625 … 0.984375); never learned, never sorted or clamped.

Action value: Q(s,a) = mean_i Z_i(s,a). Greedy action (acting and
evaluation) = argmax over the LEGAL act-v3 actions of that mean; never a
single quantile, never an unmasked mean.

Target (Double DQN): a* = masked argmax_a mean_i Z_online(s',a) over the
stored next-state mask; target distribution y_j = G + γ^n Z_target(s',a*)_j,
or y_j = G for every j at termination (n = 16, γ = 1).

Loss (quantile Huber, κ = 1): δ_ij = y_j − θ_i;
ρ_ij = |τ_i − 1{δ_ij < 0}| · H_κ(δ_ij) / κ with H_κ(δ) = ½δ² (|δ| ≤ κ),
κ(|δ| − ½κ) otherwise; per-sample loss = mean over the 32 × 32 pairs;
batch loss = mean(w_b · loss_b) with the normalised PER weights.

PER priority (frozen before training): p_b = mean_j |y_j − θ_j| + 1e-6 —
target and predicted quantiles paired by index (same τ_j).

Replay, n-step, ε-greedy, hard target copy every 2,500 updates, Adam, clip
and the deterministic asynchronous collector are exactly the Phase 2D.2
D3QN ones (the train loop below is that loop with the QR-DQN network,
update, diagnostics and export substituted).

Export: rl-policy-v2 with algorithm "qrdqn" — the frozen production
contract's identifier for QR-DQN (a file stamped "qr_dqn" would be refused
by the loader and fall back); meta.algorithmName = "qr_dqn".
"""

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..bridge import run_node_script
from ..common.checkpoint import save_checkpoint
from ..common.config import config_hash, load_config
from ..common.evaluation import headline, node_evaluate, safety_problems
from ..common.logger import RunLogger
from ..common.metadata import run_metadata
from ..common.replay import NStepBuilder, PrioritizedReplay, ReplayCorruption
from ..common.seeding import seed_everything
from ..common.stats import EpisodeLog, action_stats, summarise_episodes
from ..common.vec_env import VecIplAuctionEnv
from ..stage_b import hooks as stage_b  # Phase 2F (default-off in Stage A)
from ..common.win_qos import disable_throttling
from ..env import ACTION_COUNT, OBS_SIZE, PASS
from ..export import to_policy_json as _policy_json, write_policy
from ..nets import QUANTILES, PolicyNet
from .d3qn import (EpisodeConsistencyError, InstabilityError, SafetyError, beta_at, epsilon_at, epsilon_problem, layer_init,
                   masked_argmax, params_digest, sync_target)

ML_ROOT = Path(__file__).resolve().parents[2]
N_QUANTILES = QUANTILES            # 32, the frozen production quantile count
TAUS = torch.tensor([(i - 0.5) / N_QUANTILES for i in range(1, N_QUANTILES + 1)], dtype=torch.float64)
# Representative quantile levels: exact grid points at the ends; 0.25 / 0.5 /
# 0.75 lie exactly midway between two grid fractions and are reported as the
# mean of that pair (e.g. 0.25 = ½(τ_8 + τ_9) = ½(0.234375 + 0.265625)).
LEVELS = {"q01": (0,), "q25": (7, 8), "q50": (15, 16), "q75": (23, 24), "q99": (31,)}

DEFAULTS = {
    "algorithm": "qr_dqn",
    "seed": 1,
    "total_decisions": 1_996_800,
    "num_envs": 12,
    "gamma": 1.0,
    "n_step": 16,
    "n_quantiles": 32,
    "huber_kappa": 1.0,
    "replay_capacity": 500_000,
    "learning_starts": 10_000,
    "batch_size": 256,
    "per_alpha": 0.6,
    "per_beta_start": 0.4,
    "per_beta_final": 1.0,
    "per_eps": 1e-6,
    "learning_rate": 3e-4,
    "adam_eps": 1e-5,
    "max_grad_norm": 0.5,
    "target_update_every": 2_500,
    "eps_start": 1.0,
    "eps_final": 0.05,
    "eps_decay_decisions": 1_000_000,
    "actor_lag_cycles": 64,
    "log_every_cycles": 512,
    "split": "train",
    "stage": "A",
    "tremble": 0.01,
    "snapshot_share": 0.0,
    "eval_decisions": [],
    "eval_limit": 500,
    "final_eval_limit": 500,
    "eval_workers": 12,
    "parity_states": 256,
    "torch_threads": 2,
    "watchdog_seconds": 120,
    "expected_act_spec": None,
    "run_dir": "runs",
    "run_name": None,
}

# Algorithm-health alarms (Phase 2D.3 §19), fixed before any long run. Every
# return lies in about [−2, 10] (rewards ΔBestXI / 110, γ = 1).
HEALTH = {
    "max_grad_norm_preclip": 100.0,        # any update
    "max_abs_quantile": 50.0,              # any quantile of a legal action in a sampled state, of Z(s',a*) or of the target
    "max_target_distance": 5.0,            # mean_j |Z_online(s',a*)_j − Z_target(s',a*)_j| over a batch
    "max_crossing_rate": 0.5,              # adjacent-pair crossing rate of Z(s,a) — 0.5 = no better than random order …
    "crossing_windows": 3,                 # … in this many consecutive log windows …
    "crossing_after_updates": 20_000,      # … once this many gradient updates have been made
    "max_action_share": 0.95,
    "action_share_windows": 3,
    "epsilon_sigma": 6.0,
}


def _derived(tag, seed, bytes_=8):
    return int.from_bytes(hashlib.sha256(f"{tag}/{int(seed)}".encode()).digest()[:bytes_], "little")


def init_seed(seed):
    """Network-initialisation seed of QR-DQN run `seed`, derived from ("qr_dqn", seed)."""
    return _derived("qr_dqn/init", seed, 4)


def replay_seed(seed):
    return _derived("qr_dqn/replay", seed)


def explore_seed(seed, env_index):
    return _derived(f"qr_dqn/explore/{int(env_index)}", seed)


def make_net():
    """PolicyNet("qrdqn"): 80 → 128 → 128 (tanh) → 640 linear, viewed as [B, 20, 32]."""
    net = PolicyNet("qrdqn")
    for m in net.body:
        if isinstance(m, nn.Linear):
            layer_init(m)
    layer_init(net.head, std=1.0)
    return net


class MeanQ(nn.Module):
    """Q(s,a) = mean over the 32 quantiles — the only quantity actions are selected by."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, obs):
        return self.net(obs).mean(dim=-1)


def select_action(net, obs_row, mask_row, eps, gen):
    """ε-greedy over the LEGAL actions: draw u; explore (uniform legal action)
    when u < ε, else masked argmax of the mean quantile value."""
    legal = np.flatnonzero(mask_row)
    if not len(legal):
        raise AssertionError("no legal action")
    if gen.random() < eps:
        return int(legal[gen.integers(len(legal))]), True
    with torch.inference_mode():
        q = net(torch.from_numpy(np.asarray(obs_row, np.float32)).unsqueeze(0)).mean(dim=-1)[0]
    return int(masked_argmax(q, torch.from_numpy(np.asarray(mask_row, bool)))), False


def quantile_huber_loss(theta, target, taus, kappa=1.0):
    """Per-sample quantile-Huber loss. theta [B, N] (predicted, fraction τ_i),
    target [B, N'] (target samples y_j). δ_ij = y_j − θ_i;
    ρ_ij = |τ_i − 1{δ_ij < 0}| · H_κ(δ_ij) / κ; returns mean over (i, j), shape [B]."""
    delta = target.unsqueeze(1) - theta.unsqueeze(2)                     # [B, N(i), N'(j)]
    abs_d = delta.abs()
    huber = torch.where(abs_d <= kappa, 0.5 * delta ** 2, kappa * (abs_d - 0.5 * kappa))
    tau = taus.to(theta.dtype).view(1, -1, 1)
    weight = (tau - (delta.detach() < 0).to(theta.dtype)).abs()
    return (weight * huber / kappa).mean(dim=(1, 2))


def qr_targets(online, target, ret, next_obs, next_mask, mult):
    """Double-DQN distributional target: a* chosen by the ONLINE mean over quantiles
    under the next-state mask; its distribution taken from the TARGET network."""
    with torch.no_grad():
        z_on = online(next_obs)                                          # [B, 20, 32]
        a_star = masked_argmax(z_on.mean(dim=-1), next_mask)
        rows = torch.arange(len(a_star))
        z_on_star = z_on[rows, a_star]                                   # [B, 32]
        z_tg_star = target(next_obs)[rows, a_star]                       # [B, 32]
        boot = mult > 0
        y = ret.unsqueeze(1) + torch.where(boot.unsqueeze(1), mult.unsqueeze(1) * z_tg_star, torch.zeros_like(z_tg_star))
    return y, a_star, z_on_star, z_tg_star, boot


def level_means(theta):
    """Batch means of the representative quantile levels of [B, 32] quantiles."""
    return {k: float(theta[:, list(ix)].mean(dim=1).mean()) for k, ix in LEVELS.items()}


def qr_update(online, target, optimizer, batch, weights, cfg):
    """One optimiser step on the quantile-Huber loss. Returns (stats, priority base [B])."""
    obs = torch.from_numpy(batch["obs"])
    act = torch.from_numpy(batch["action"])
    ret = torch.from_numpy(batch["ret"]).float()
    next_obs = torch.from_numpy(batch["next_obs"])
    next_mask = torch.from_numpy(batch["next_mask"])
    mult = torch.from_numpy(batch["mult"]).float()
    w = torch.from_numpy(weights).float()
    y, a_star, z_on_star, z_tg_star, boot = qr_targets(online, target, ret, next_obs, next_mask, mult)
    z_all = online(obs)
    rows = torch.arange(len(act))
    theta = z_all[rows, act]                                             # [B, 32]
    per_sample = quantile_huber_loss(theta, y, TAUS, cfg["huber_kappa"])
    loss = (w * per_sample).mean()
    optimizer.zero_grad()
    loss.backward()
    grads_finite = all(bool(torch.isfinite(p.grad).all()) for p in online.parameters() if p.grad is not None)
    grad_norm = nn.utils.clip_grad_norm_(online.parameters(), cfg["max_grad_norm"]).item()
    optimizer.step()
    with torch.no_grad():
        th = theta.detach()
        pri = (y - th).abs().mean(dim=1)                                 # index-paired |y_j − θ_j|
        q = th.mean(dim=1)
        diffs = th[:, 1:] - th[:, :-1]
        crossed = diffs < 0
        legal_z = z_all.detach()[torch.from_numpy(batch["mask"])]        # [#legal, 32]
        dist = (z_on_star - z_tg_star).abs().mean(dim=1)[boot]
        lv = level_means(th)
        ylv = level_means(y)
    stats = {
        "loss": loss.item(), "loss_unweighted": per_sample.detach().mean().item(),
        "td_abs_mean": pri.mean().item(), "td_abs_max": pri.max().item(),
        "td_mean_abs": (y.mean(dim=1) - q).abs().mean().item(),
        "q_mean": q.mean().item(), "q_min": q.min().item(), "q_max": q.max().item(),
        **{f"z_{k}": v for k, v in lv.items()}, **{f"y_{k}": v for k, v in ylv.items()},
        "z_spread": (th[:, -1] - th[:, 0]).mean().item(),
        "z_iqr": lv["q75"] - lv["q25"], "y_iqr": ylv["q75"] - ylv["q25"],
        "crossing_rate": crossed.float().mean().item(),
        "crossing_rows": crossed.any(dim=1).float().mean().item(),
        "crossing_size": (-diffs).clamp(min=0).sum(dim=1).mean().item(),
        "z_legal_absmax": legal_z.abs().max().item() if legal_z.numel() else 0.0,
        "z_next_absmax": max(z_on_star.abs().max().item(), z_tg_star.abs().max().item()),
        "y_absmax": y.abs().max().item(), "theta_absmax": th.abs().max().item(),
        "target_distance": dist.mean().item() if dist.numel() else 0.0,
        "bootstrap_share": boot.float().mean().item(),
        "grad_norm": grad_norm, "grads_finite": grads_finite,
        "params_finite": all(bool(torch.isfinite(p).all()) for p in online.parameters()),
        "values_finite": bool(torch.isfinite(z_all).all() and torch.isfinite(y).all()),
        "is_weight_mean": float(weights.mean()),
    }
    return stats, pri.numpy().astype(np.float64)


def update_problems(stats):
    """Per-update alarms (§19)."""
    out = []
    keys = ("loss", "td_abs_mean", "q_mean", "q_min", "q_max", "z_legal_absmax", "z_next_absmax", "y_absmax", "grad_norm", "target_distance")
    if not all(math.isfinite(stats[k]) for k in keys) or not (stats["grads_finite"] and stats["params_finite"] and stats["values_finite"]):
        out.append("non-finite loss / quantile / gradient / parameter: " + json.dumps({k: stats[k] for k in keys + ("grads_finite", "params_finite", "values_finite")}, default=str))
    if stats["grad_norm"] > HEALTH["max_grad_norm_preclip"]:
        out.append(f"exploding gradient: pre-clip norm {stats['grad_norm']:.3g}")
    zmax = max(stats["z_legal_absmax"], stats["z_next_absmax"], stats["y_absmax"], stats["theta_absmax"])
    if zmax > HEALTH["max_abs_quantile"]:
        out.append(f"quantile divergence: |Z| {zmax:.3g}")
    if stats["target_distance"] > HEALTH["max_target_distance"]:
        out.append(f"online/target distribution distance {stats['target_distance']:.3g}")
    return out


def to_policy_json(net, obs_spec, act_spec, meta=None):
    return _policy_json(net, obs_spec, act_spec, meta={
        "exporter": "ml/ipl_rl/algos/qr_dqn.py", "algorithmName": "qr_dqn",
        "quantiles": N_QUANTILES, "quantileFractions": [float(t) for t in TAUS],
        "quantileLayout": "last layer = 20 actions x 32 quantiles, action-major (index a*32 + i)",
        "selection": "masked argmax over actions of the mean of the 32 quantiles",
        "trainingArchitecture": {"trunk": [OBS_SIZE, 128, 128], "head": [128, ACTION_COUNT * N_QUANTILES], "activation": "tanh",
                                 "reshape": [ACTION_COUNT, N_QUANTILES], "dueling": False},
        "inference": "640 outputs -> [20, 32] -> mean per action -> act-v3 mask -> argmax; no exploration, replay or target network",
        **(meta or {}),
    })


def parity(net, policy, obs, masks):
    """Python (float64 copy) vs production JavaScript: all 640 raw quantile
    outputs, the 20 per-action means, and masked-argmax agreement under the
    real masks, PASS only, one single legal action and all 20 legal."""
    obs = np.asarray(obs, dtype=np.float32)
    masks = np.asarray(masks, dtype=bool)
    js = run_node_script("rl-policy-scores.js", {"policy": policy, "observations": obs.tolist()})
    if not js["ok"]:
        raise RuntimeError(f"production loader rejected the export: {js['error']}")
    net64 = copy.deepcopy(net).double()
    with torch.no_grad():
        z = net64(torch.from_numpy(obs).double())
    py_raw = z.reshape(len(obs), -1).numpy()
    py = z.mean(dim=-1).numpy()
    js_raw = np.asarray(js["raw"], dtype=np.float64)
    jsn = np.asarray(js["scores"], dtype=np.float64)
    if js_raw.shape != (len(obs), ACTION_COUNT * N_QUANTILES):
        raise RuntimeError(f"JS raw output shape {js_raw.shape}")
    single = np.zeros_like(masks)
    single[np.arange(len(masks)), np.array([np.flatnonzero(m)[-1] for m in masks])] = True
    pass_only = np.zeros_like(masks)
    pass_only[:, PASS] = True
    variants = {"real": masks, "passOnly": pass_only, "single": single, "allLegal": np.ones_like(masks)}
    agree = {name: float((np.where(m, py, -np.inf).argmax(1) == np.where(m, jsn, -np.inf).argmax(1)).mean()) for name, m in variants.items()}
    return {"states": int(len(obs)), "maxAbsRawDiff": float(np.abs(py_raw - js_raw).max()), "maxAbsScoreDiff": float(np.abs(py - jsn).max()),
            "argmaxAgreement": min(agree.values()), "byMask": agree}


def train(cfg):
    cfg_hash = config_hash(cfg)
    if cfg["algorithm"] != "qr_dqn":
        raise ValueError("this trainer is QR-DQN only")
    if cfg["n_quantiles"] != N_QUANTILES:
        raise ValueError(f"QR-DQN uses exactly {N_QUANTILES} quantiles (the frozen production head)")
    stage_b.check_stage(cfg, "Phase 2D.3 trains against Stage A only (no RL snapshots)")
    N, L = cfg["num_envs"], cfg["actor_lag_cycles"]
    total_cycles = cfg["total_decisions"] // N
    eval_cycles = sorted({d // N for d in cfg["eval_decisions"] if d % N == 0 and 0 < d // N <= total_cycles} | {total_cycles})
    run_name = cfg["run_name"] or f"qr-dqn-s{cfg['seed']}-{cfg_hash[:8]}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = (ML_ROOT / cfg["run_dir"] / run_name).resolve()

    seed_everything(cfg["seed"], cfg["torch_threads"])
    envs = VecIplAuctionEnv(N, cfg["seed"], split=cfg["split"], tremble=cfg["tremble"],
                            snapshot_share=cfg["snapshot_share"], watchdog_seconds=cfg["watchdog_seconds"], **stage_b.env_kwargs(cfg))
    expected = cfg["expected_act_spec"]
    if expected and envs.act_spec["hash"] != expected:
        envs.close()
        raise ValueError(f"action-spec mismatch: bridge {envs.act_spec}, expected {expected}")
    if envs.gamma != cfg["gamma"]:
        envs.close()
        raise ValueError(f"config gamma {cfg['gamma']} ≠ environment gamma {envs.gamma} (frozen)")

    log = RunLogger(run_dir)
    meta = run_metadata(cfg, cfg_hash, envs.obs_spec, envs.act_spec, envs.gamma, extra={
        "runName": run_name, "totalCycles": total_cycles, "plannedDecisions": total_cycles * N,
        "evalDecisions": [c * N for c in eval_cycles], "health": HEALTH,
        "quantiles": {"count": N_QUANTILES, "fractions": [float(t) for t in TAUS], "kappa": cfg["huber_kappa"],
                      "priority": "mean_j |y_j - theta_j| + per_eps (index-paired)", "levels": {k: list(v) for k, v in LEVELS.items()}},
        "seeds": {"init": init_seed(cfg["seed"]), "replay": replay_seed(cfg["seed"]), "explore": [explore_seed(cfg["seed"], i) for i in range(N)]}})
    log.write_json("metadata.json", meta)
    print(f"[qr_dqn] {run_name}: {total_cycles:,} cycles × {N} = {total_cycles * N:,} decisions  config {cfg_hash}  obs {envs.obs_spec['hash']}  act {envs.act_spec['hash']}", flush=True)

    torch.manual_seed(init_seed(cfg["seed"]))
    online = make_net()
    stage_b.warm_start(cfg, {"online": online}, run_dir)
    target = copy.deepcopy(online)
    for p in target.parameters():
        p.requires_grad_(False)
    initial_online, initial_target = params_digest(online), params_digest(target)
    optimizer = torch.optim.Adam(online.parameters(), lr=cfg["learning_rate"], eps=cfg["adam_eps"])
    replay = PrioritizedReplay(cfg["replay_capacity"], cfg["per_alpha"], cfg["per_eps"], np.random.default_rng(replay_seed(cfg["seed"])))
    explore = [np.random.default_rng(explore_seed(cfg["seed"], i)) for i in range(N)]
    builders = [NStepBuilder(cfg["n_step"], cfg["gamma"]) for _ in range(N)]
    # Online-network versions for the actors: ring[v % (L + 1)] holds version v
    # (= the network after v cycles have been processed).
    ring = [copy.deepcopy(online) for _ in range(L + 1)]
    ring_version = [0] * (L + 1)
    for net in ring:
        for p in net.parameters():
            p.requires_grad_(False)

    # Per-environment state.
    obs = np.zeros((N, OBS_SIZE), np.float32)
    mask = np.zeros((N, ACTION_COUNT), bool)
    ready = [False] * N          # has a current decision state and nothing in flight
    k_next = [0] * N             # decisions dispatched
    inflight = [None] * N        # (k, obs, mask, action) of the step in flight
    ep_ret = np.zeros(N, np.float64)
    ep_steps = np.zeros(N, np.int64)
    ep_actions = np.zeros((N, ACTION_COUNT), np.int64)
    consistency_checks = 0
    pending = {}                 # cycle → {env: [transitions]}
    completed = {}               # cycle → decisions received
    episodes_by_cycle = {}       # cycle → [(env, summary)]
    win = {}                     # window → per-window records (filled by dispatch)
    W = cfg["log_every_cycles"]

    def window(w):
        if w not in win:
            win[w] = {"acts": np.zeros((W, N), np.int64), "masks": np.zeros((W, N, ACTION_COUNT), bool), "explored": np.zeros((W, N), bool),
                      "eps": np.full((W, N), np.nan)}
        return win[w]

    for i in range(N):
        envs.send_reset(i)

    episodes = EpisodeLog()
    all_episodes = []
    evaluations = []
    target_syncs = []
    updates = 0
    samples_consumed = 0
    processed = 0
    t_start = time.perf_counter()
    timers = {"wait": 0.0, "update": 0.0, "act": 0.0, "replay": 0.0, "eval": 0.0, "log": 0.0}
    upd_stats = []
    sample_hash = hashlib.sha256()
    concentrated = 0
    last_sync_digest = initial_target
    first_update = [None]
    crossing = [0]
    status, failure = "completed", None

    def dispatch(i):
        k = k_next[i]
        v = max(0, k - L)
        net = ring[v % (L + 1)]
        if ring_version[v % (L + 1)] != v:
            raise RuntimeError(f"actor version {v} not available (ring holds {ring_version[v % (L + 1)]})")
        d = k * N + i
        eps = epsilon_at(d, cfg)
        a, explored = select_action(net, obs[i], mask[i], eps, explore[i])  # masked mean-quantile argmax / ε
        if not mask[i, a]:
            raise AssertionError(f"env {i}: selected masked action {a}")
        rec = window(k // W)
        rec["acts"][k % W, i] = a
        rec["masks"][k % W, i] = mask[i]
        rec["explored"][k % W, i] = explored
        rec["eps"][k % W, i] = eps
        inflight[i] = (k, obs[i].copy(), mask[i].copy(), a)
        ep_actions[i, a] += 1
        envs.send_step(i, a)
        ready[i] = False
        k_next[i] = k + 1

    def on_reply(i, reply):
        nonlocal consistency_checks
        if inflight[i] is None:  # reset reply
            obs[i] = reply["obs"]
            mask[i] = reply["mask"]
            ready[i] = True
            return
        k, s, m, a = inflight[i]
        inflight[i] = None
        r, done = float(reply["reward"]), bool(reply["done"])
        ep_ret[i] += r
        ep_steps[i] += 1
        nxt_obs = None if done else np.asarray(reply["obs"], np.float32)
        nxt_mask = None if done else np.asarray(reply["mask"], bool)
        out = builders[i].push(s, m, a, r, nxt_obs, nxt_mask, done, tag=k * N + i)
        pending.setdefault(k, {})[i] = out
        completed[k] = completed.get(k, 0) + 1
        if done:
            ep = reply["info"]["episode"]
            problems = []
            if abs(ep_ret[i] - ep["return"]) > 1e-9:
                problems.append(f"reward sum {ep_ret[i]!r} ≠ episode return {ep['return']!r}")
            if ep_steps[i] != ep["decisions"]:
                problems.append(f"{ep_steps[i]} steps ≠ {ep['decisions']} decisions")
            if ep_actions[i].tolist() != list(ep["actionCounts"]):
                problems.append("action counts differ")
            if problems:
                raise EpisodeConsistencyError(f"env {i} seed {ep['seed']}: " + "; ".join(problems))
            consistency_checks += 1
            ep_ret[i], ep_steps[i] = 0.0, 0
            ep_actions[i] = 0
            if ep.get("invariantViolations", 0):
                raise SafetyError([f"training episode invariant violation (seed {ep['seed']}): {ep.get('violations')}"])
            if not ep["legalXI"] and stage_b.incomplete_is_stop(cfg):
                raise SafetyError([f"training episode ended without a legal XI: seed {ep['seed']}"])
            episodes_by_cycle.setdefault(k, []).append((i, ep))
            envs.send_reset(i)
        else:
            obs[i] = nxt_obs
            mask[i] = nxt_mask
            ready[i] = True

    def process_cycle(c):
        nonlocal updates, samples_consumed, last_sync_digest
        t0 = time.perf_counter()
        replay.add_many([t for i in range(N) for t in pending[c].get(i, [])])
        del pending[c], completed[c]
        timers["replay"] += time.perf_counter() - t0
        if len(replay) >= cfg["learning_starts"]:
            t0 = time.perf_counter()
            beta = beta_at((c + 1) * N, cfg)
            idx, w, _ = replay.sample(cfg["batch_size"], beta)
            sample_hash.update(idx.tobytes())
            stats, td = qr_update(online, target, optimizer, replay.batch(idx), w, cfg)
            replay.update_priorities(idx, td)
            updates += 1
            samples_consumed += len(idx)
            if first_update[0] is None:
                first_update[0] = (c + 1) * N
            stats["beta"] = beta
            upd_stats.append(stats)
            problems = update_problems(stats)
            if problems:
                raise InstabilityError(problems)
            dt = sync_target(updates, online, target, cfg["target_update_every"])
            if dt is not None:
                last_sync_digest = dt
                target_syncs.append({"update": updates, "decisions": (c + 1) * N, "digest": dt})
            timers["update"] += time.perf_counter() - t0
        v = c + 1
        ring[v % (L + 1)].load_state_dict(online.state_dict())
        ring_version[v % (L + 1)] = v

    def log_window(w):
        nonlocal concentrated, upd_stats
        t0 = time.perf_counter()
        rec = win.pop(w)
        c_end = (w + 1) * W  # cycles processed
        decisions = c_end * N
        finished = []
        for c in range(w * W, c_end):
            for i, ep in sorted(episodes_by_cycle.pop(c, []), key=lambda x: x[0]):
                ep["envIndex"] = i
                episodes.add(ep)
                finished.append(ep)
        episodes.pending = []
        all_episodes.extend(finished)
        ep_stats = summarise_episodes(finished)
        acts = action_stats(rec["acts"], rec["masks"], ACTION_COUNT)
        legal = rec["masks"].sum(-1)
        acts["legal_count_mean"] = float(legal.mean())
        acts["masked_count_mean"] = float(ACTION_COUNT - legal.mean())
        explored = int(rec["explored"].sum())
        acts["explored_share"] = explored / rec["explored"].size
        greedy = ~rec["explored"]
        acts["greedy_bid_share"] = float((rec["acts"][greedy] != PASS).mean()) if greedy.any() else float("nan")
        problems = []
        expected_eps = np.array([[epsilon_at((w * W + r) * N + i, cfg) for i in range(N)] for r in range(W)])
        if not np.array_equal(rec["eps"], expected_eps):
            problems.append(f"epsilon schedule mismatch in window {w + 1}: max |Δ| {np.nanmax(np.abs(rec['eps'] - expected_eps)):.3g}")
        ep_problem = epsilon_problem(explored, float(rec["eps"].sum()), float((rec["eps"] * (1 - rec["eps"])).sum()))
        if ep_problem:
            problems.append(ep_problem)
        eps_last = float(expected_eps[-1, -1])
        if max(acts["action_shares"]) >= HEALTH["max_action_share"]:
            concentrated += 1
        else:
            concentrated = 0
        if concentrated >= HEALTH["action_share_windows"]:
            problems.append(f"one action ≥ {HEALTH['max_action_share']:.0%} of decisions in {concentrated} consecutive windows")
        audit = replay.audit(cfg["gamma"], cfg["n_step"])
        if audit:
            problems.append(f"replay corruption: {audit}")
        td_now = params_digest(target)
        if td_now != last_sync_digest:
            problems.append(f"target network changed between synchronisations ({td_now} ≠ {last_sync_digest})")
        pri = replay.raw_priorities()
        us = upd_stats
        agg = lambda k, f=np.mean: float(f([s[k] for s in us])) if us else float("nan")
        numeric = [k for k, v in (us[0].items() if us else []) if isinstance(v, float)]
        train_stats = {"updates": updates, "updates_in_window": len(us), **{k: agg(k) for k in numeric}}
        if us:
            train_stats.update({"td_abs_max": agg("td_abs_max", np.max), "q_min": agg("q_min", np.min), "q_max": agg("q_max", np.max),
                                "z_absmax": max(agg("z_legal_absmax", np.max), agg("z_next_absmax", np.max), agg("y_absmax", np.max)),
                                "target_distance_max": agg("target_distance", np.max), "grad_norm_max": agg("grad_norm", np.max),
                                "crossing_rate_max": agg("crossing_rate", np.max)})
        train_stats.update({
            "epsilon": eps_last, "explored": explored, "explored_expected": float(rec["eps"].sum()), "beta": beta_at(decisions, cfg), "replay_size": len(replay), "replay_added": replay.added,
            "priority_mean": float(pri.mean()) if len(pri) else float("nan"), "priority_max": float(pri.max()) if len(pri) else float("nan"),
            "max_priority_seen": replay.max_priority, "target_updates": len(target_syncs), "samples_consumed": samples_consumed,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        # Quantile-crossing alarm (§19/§21): the window's adjacent-pair crossing
        # rate no better than random order, for consecutive windows, after warm-up.
        if us and updates >= HEALTH["crossing_after_updates"] and train_stats["crossing_rate"] >= HEALTH["max_crossing_rate"]:
            crossing[0] += 1
        else:
            crossing[0] = 0
        if crossing[0] >= HEALTH["crossing_windows"]:
            problems.append(f"pathological quantile crossing: rate {train_stats['crossing_rate']:.3f} in {crossing[0]} consecutive windows")
        upd_stats = []
        elapsed = time.perf_counter() - t_start
        perf = {"decisions_per_sec_overall": decisions / elapsed, **{f"{k}_seconds": v for k, v in timers.items()}}
        digests = {"actionsDigest": hashlib.sha256(rec["acts"].tobytes()).hexdigest()[:16],
                   "exploreDigest": hashlib.sha256(rec["explored"].tobytes()).hexdigest()[:16],
                   "samplesDigest": sample_hash.hexdigest()[:16],
                   "priorityDigest": hashlib.sha256(replay.tree.leaves().tobytes()).hexdigest()[:16],
                   "onlineDigest": params_digest(online), "targetDigest": td_now}
        row = {"type": "update", "update": w + 1, "cycles": c_end, "decisions": decisions, "episodes": episodes.total, **digests,
               "train": train_stats, "episode": ep_stats, "actions": acts, "perf": perf}
        log.record(row)
        log.scalars("train", {k: v for k, v in train_stats.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}, decisions)
        log.scalars("episode", ep_stats, decisions)
        log.scalars("actions", {k: v for k, v in acts.items() if k != "action_shares"}, decisions)
        log.histogram_shares("action_share", acts["action_shares"], envs.action_names, decisions)
        print(f"[qr_dqn] win {w + 1:3d}  dec {decisions:>9,}  eps {episodes.total:5d}  ret {ep_stats.get('return', float('nan')):6.3f}  "
              f"XI {ep_stats.get('xi', float('nan')):6.2f}  ε {eps_last:.3f}  β {train_stats['beta']:.3f}  upd {updates:,}  tgt {len(target_syncs)}  "
              f"loss {train_stats.get('loss', float('nan')):.5f}  |td| {train_stats.get('td_abs_mean', float('nan')):.4f}  Q {train_stats.get('q_mean', float('nan')):.3f} "
              f"[{train_stats.get('q_min', float('nan')):.2f},{train_stats.get('q_max', float('nan')):.2f}]  Z .01/.5/.99 {train_stats.get('z_q01', float('nan')):.2f}/"
              f"{train_stats.get('z_q50', float('nan')):.2f}/{train_stats.get('z_q99', float('nan')):.2f}  cross {train_stats.get('crossing_rate', float('nan')):.3f}  "
              f"dist {train_stats.get('target_distance', float('nan')):.4f}  gn {train_stats.get('grad_norm', float('nan')):.3f}  bid {acts['bid_share']:.3f}  replay {len(replay):,}  "
              f"{perf['decisions_per_sec_overall']:.0f} dec/s", flush=True)
        timers["log"] += time.perf_counter() - t0
        if problems:
            raise InstabilityError(problems)

    def parity_states():
        n = len(replay)
        idx = np.unique(np.linspace(0, n - 1, min(cfg["parity_states"], n)).astype(np.int64))
        reauc = np.flatnonzero(replay.obs[:n, 0] > 0.5)[:64]            # phase_reauction
        low_purse = np.argsort(replay.obs[:n, 15], kind="stable")[:32]   # self_purse, lowest
        high_purse = np.argsort(-replay.obs[:n, 15], kind="stable")[:32]
        sel = np.concatenate([idx, reauc, low_purse, high_purse])
        return replay.obs[sel], replay.mask[sel], {"regular": int(len(idx)), "reauction": int(len(reauc)), "lowPurse": int(len(low_purse)), "highPurse": int(len(high_purse))}

    def evaluate_checkpoint(c, final):
        t0 = time.perf_counter()
        decisions = c * N
        tag = f"update_{c // W:04d}" if c % W == 0 else f"cycle_{c:07d}"
        ckpt_dir = run_dir / "checkpoints" / tag
        save_checkpoint(ckpt_dir / "checkpoint.pt", modules={"online": online, "target": target}, optimizer=optimizer,
                        step_state={"cycles": c, "decisions": decisions, "updates": updates, "targetUpdates": len(target_syncs),
                                    "episodeIndex": envs.episode_index.tolist(), "replaySize": len(replay)},
                        cfg=cfg, cfg_hash=cfg_hash)
        policy = to_policy_json(online, envs.obs_spec, envs.act_spec, meta={
            "trainedSteps": int(decisions), "seed": cfg["seed"], "config": cfg, "configHash": cfg_hash,
            "cycles": c, "updates": updates, "stage": cfg["stage"], "runName": run_name})
        write_policy(ckpt_dir / "policy.json", policy)
        p_obs, p_mask, composition = parity_states()
        par = parity(online, policy, p_obs, p_mask)
        par["composition"] = composition
        t1 = time.perf_counter()
        limit = cfg["final_eval_limit"] if final else cfg["eval_limit"]
        report, eps = node_evaluate(ckpt_dir / "policy.json", ckpt_dir / "validation", limit=limit, workers=cfg["eval_workers"])
        problems = safety_problems(report, eps)
        if expected and report.get("actSpecHash") != expected:
            problems.append(f"action-spec mismatch in validation: {report.get('actSpecHash')} != {expected}")
        name = "policy:qrdqn"
        head = headline(report, name)
        shares = report["report"][name]["actionShares"]
        row = {"update": c // W, "cycles": c, "decisions": decisions, "final": final, "validationEpisodes": report["n"],
               "seconds": time.perf_counter() - t1, "parity": par, "safetyProblems": problems, "metrics": head, "actionShares": shares,
               "updates": updates, "targetUpdates": len(target_syncs)}
        evaluations.append(row)
        log.scalars("validation", head, decisions)
        log.record({"type": "evaluation", **row})
        print(f"[qr_dqn] eval @ {decisions:,}: validation n={report['n']} XI {head['xi']:.2f} (Δ vs moneyball {head.get('xiDiffVsMoneyball', float('nan')):+.2f})  "
              f"legal {100 * head['legalXI']:.0f}%  purse left {head['purseLeftShare']:.2f}  price/fair {head['priceToFair']:.2f}  "
              f"parity {par['maxAbsRawDiff']:.1e}/{par['maxAbsScoreDiff']:.1e}/{par['argmaxAgreement']:.3f}  safety {'OK' if not problems else problems[:3]}", flush=True)
        timers["eval"] += time.perf_counter() - t0
        if problems:
            raise SafetyError(problems)
        if par["maxAbsScoreDiff"] > 1e-6 or par["maxAbsRawDiff"] > 1e-6 or par["argmaxAgreement"] < 1.0:
            raise SafetyError([f"export parity failed: {par}"])
        top = max(shares.values()) if isinstance(shares, dict) else max(shares)
        if top >= HEALTH["max_action_share"]:
            raise InstabilityError([f"validation action concentration: one action takes {top:.3f} of decisions"])

    try:
        while processed < total_cycles:
            for i in range(N):
                if ready[i] and k_next[i] < total_cycles and processed >= k_next[i] - L:
                    t0 = time.perf_counter()
                    dispatch(i)
                    timers["act"] += time.perf_counter() - t0
            if completed.get(processed, 0) == N:
                process_cycle(processed)
                processed += 1
                if processed % W == 0:
                    log_window(processed // W - 1)
                if processed in eval_cycles:
                    evaluate_checkpoint(processed, final=(processed == total_cycles))
                continue
            t0 = time.perf_counter()
            i, reply = envs.next_reply()
            timers["wait"] += time.perf_counter() - t0
            on_reply(i, reply)
        if processed % W:
            log_window(processed // W)
    except SafetyError as err:
        status, failure = "stopped: safety", err.problems
        print(f"[qr_dqn] STOPPED — safety invariant failed: {err.problems[:5]}", flush=True)
    except (InstabilityError, ReplayCorruption) as err:
        status, failure = "stopped: instability", getattr(err, "problems", [str(err)])
        print(f"[qr_dqn] STOPPED — algorithm instability: {failure[:5]}", flush=True)
    except Exception as err:
        status, failure = f"stopped: {type(err).__name__}", [str(err)]
        print(f"[qr_dqn] STOPPED — {type(err).__name__}: {err}", flush=True)
        raise
    finally:
        wall = time.perf_counter() - t_start
        decisions = processed * N
        summary = {
            "runName": run_name, "algorithm": "qr_dqn", "exportAlgorithm": "qrdqn", "quantiles": N_QUANTILES, "status": status, "failure": failure, "configHash": cfg_hash,
            "seed": cfg["seed"], "obsHash": envs.obs_spec["hash"], "actHash": envs.act_spec["hash"], "actSpec": envs.act_spec,
            "numEnvs": N, "actorLagCycles": L, "cycles": processed, "totalDecisions": decisions, "episodes": episodes.total,
            "wallSeconds": wall, "timers": timers, "decisionsPerSecOverall": decisions / wall if wall else None,
            "initialParamsDigest": initial_online, "initialTargetDigest": initial_target,
            "finalParamsDigest": params_digest(online), "finalTargetDigest": params_digest(target),
            "optimisationBudget": {
                "environmentDecisions": decisions, "gradientUpdates": updates, "replayTransitionsGenerated": replay.added,
                "replaySamplesConsumed": samples_consumed, "updatesPerDecision": updates / decisions if decisions else None,
                "samplesPerTransition": samples_consumed / replay.added if replay.added else None,
                "targetUpdates": len(target_syncs), "firstUpdateAtDecision": first_update[0],
                "epsilon": {"start": cfg["eps_start"], "final": cfg["eps_final"], "decayDecisions": cfg["eps_decay_decisions"]},
            },
            "targetSyncs": target_syncs,
            "episodeConsistencyChecks": consistency_checks,
            "illegalActions": 0 if status == "completed" else None,
            "incompleteXiTrainingEpisodes": sum(1 for e in all_episodes if not e["legalXI"]),
            "trainingEpisodes": {"all": summarise_episodes(all_episodes), "first100": summarise_episodes(all_episodes[:100]),
                                 "last100": summarise_episodes(all_episodes[-100:])},
            "evaluations": evaluations,
        }
        log.write_json("summary.json", summary)
        (run_dir / "training_episodes.json").write_text(json.dumps(all_episodes), encoding="utf-8")
        log.close()
        envs.close()
    return run_dir, summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", help="JSON config (keys of DEFAULTS)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=JSON", help="override config keys, e.g. seed=2")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    disable_throttling()
    overrides = {k: json.loads(v) for k, v in (s.split("=", 1) for s in args.set)}
    cfg = load_config(args.config, DEFAULTS, overrides)
    _, summary = train(cfg)
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
