"""Dueling Double DQN for IplAuctionEnv-v2 (Phase 2D.2) — one file.

    cd ml
    .venv/Scripts/python -m ipl_rl.algos.d3qn --config ipl_rl/configs/d3qn_2d2.json --set seed=1

Network (training): shared trunk 80 → 128 → 128 (tanh); value stream
128 → 128 → 1 and advantage stream 128 → 128 → 20 (tanh hidden);
Q = V + A − mean(A). Orthogonal init (√2 hidden, 1.0 output layers), drawn
from a D3QN-specific seed (init_seed). Exported to rl-policy-v2 EXACTLY as a
plain tanh MLP 80 → 128 → 128 → 256 → 20: the two stream hidden layers stack
into the 256-wide layer and V + A − mean(A) folds into the last linear layer
(head "q", selection argmax — no production-code change).

Learning:
  - n-step (n = 16, γ = 1) transitions per environment, never bootstrapping
    across an episode end (common/replay.NStepBuilder)
  - prioritised replay (α 0.6, β 0.4 → 1.0 linearly over the decision
    budget, ε 1e-6, new transitions at the max priority seen so far)
  - Double-DQN target: a* = masked argmax_a Q_online(s', a) over the stored
    next-state act-v3 mask, evaluated by Q_target(s', a*);
    y = G + γ^n Q_target(s', a*), or y = G at termination
  - Huber loss (δ = 1) weighted by the normalised importance weights; Adam
    (lr 3e-4, eps 1e-5); global gradient-norm clip 0.5; ONE optimiser step
    per learner cycle (= one decision from each of the 12 environments)
  - hard target copy every 2,500 gradient updates (no Polyak)
  - ε-greedy over the legal actions only: ε 1.0 → 0.05 linearly over the
    first 1,000,000 decisions, then 0.05. Evaluation: ε = 0, masked argmax.

Collection (deterministic and asynchronous): cycle c = every environment's
c-th decision. When all N decisions of cycle c are back, their n-step
transitions enter the replay in environment order and exactly one gradient
update runs. Environment i makes its k-th decision with the online network
as it was after cycle k − L (L = actor_lag_cycles), so environments run up
to L cycles apart without waiting for the slowest one — and nothing depends
on which simulator answers first. The replay content and every sampled
batch are functions of the seed only.

Everything else is the shared Phase 2C infrastructure: JavaScript auction,
act-v3 masks from JavaScript, per-episode consistency checks, checkpoints,
rl-policy-v2 export, JS/Python export parity, Node validation against the
locked baselines, safety invariants. Algorithm-health alarms (HEALTH) stop
a run; they are never "fixed" by changing hyperparameters.
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
import torch.nn.functional as F
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
from ..export import FORMAT, SELECTION, write_policy

ML_ROOT = Path(__file__).resolve().parents[2]

DEFAULTS = {
    "algorithm": "d3qn",
    "seed": 1,
    "total_decisions": 1_996_800,     # rounded DOWN to whole cycles of num_envs decisions
    "num_envs": 12,
    # returns / replay
    "gamma": 1.0,                     # must equal the environment's (frozen) γ
    "n_step": 16,
    "replay_capacity": 500_000,
    "learning_starts": 10_000,        # transitions in the replay before the first update
    "batch_size": 256,
    "per_alpha": 0.6,
    "per_beta_start": 0.4,
    "per_beta_final": 1.0,            # linear over the whole decision budget
    "per_eps": 1e-6,
    # optimisation
    "learning_rate": 3e-4,
    "adam_eps": 1e-5,
    "max_grad_norm": 0.5,
    "huber_delta": 1.0,
    "target_update_every": 2_500,     # gradient updates between hard target copies
    # exploration
    "eps_start": 1.0,
    "eps_final": 0.05,
    "eps_decay_decisions": 1_000_000,
    # collection
    "actor_lag_cycles": 64,
    "log_every_cycles": 512,          # one metrics row per 512 cycles (6,144 decisions at 12 envs)
    # environment / curriculum
    "split": "train",
    "stage": "A",
    "tremble": 0.01,
    "snapshot_share": 0.0,
    # evaluation / checkpoints
    "eval_decisions": [],             # decision counts to checkpoint + validate at (and always at the end)
    "eval_limit": 500,
    "final_eval_limit": 500,
    "eval_workers": 12,
    "parity_states": 256,
    # runtime
    "torch_threads": 2,
    "watchdog_seconds": 120,
    "expected_act_spec": None,
    "run_dir": "runs",
    "run_name": None,
}

# Algorithm-health alarms (Phase 2D.2 §17), fixed before any long run.
# Returns are bounded: rewards are ΔBestXI / 110 with γ = 1, so every return
# lies in about [−2, 10] (XI total ≤ ~1,100 → ≤ 10; −2 only for an empty
# XI slot, which the mask prevents).
HEALTH = {
    "max_grad_norm_preclip": 100.0,        # any update (Huber keeps real gradients O(1))
    "max_abs_q": 50.0,                     # any |Q| in a sampled batch (5 × the largest possible return)
    "max_target_divergence": 5.0,          # mean |Q_online(s',a*) − Q_target(s',a*)| over a batch
    "max_action_share": 0.95,              # one action ≥ 95 % of decisions …
    "action_share_windows": 3,             # … in this many consecutive log windows, or in a validation run
    "epsilon_sigma": 6.0,                  # explored-decision count vs Σε: |Δ| ≤ 6σ + 1
    "replay_tree_rel_error": 1e-9,
}


def _derived(tag, seed, bytes_=8):
    return int.from_bytes(hashlib.sha256(f"{tag}/{int(seed)}".encode()).digest()[:bytes_], "little")


def init_seed(seed):
    """Network-initialisation seed of D3QN run `seed`, derived from ("d3qn", seed)."""
    return _derived("d3qn/init", seed, 4)


def replay_seed(seed):
    return _derived("d3qn/replay", seed)


def explore_seed(seed, env_index):
    return _derived(f"d3qn/explore/{int(env_index)}", seed)


def epsilon_at(d, cfg):
    """ε for global decision index d (0-based): linear, then exactly eps_final."""
    if d >= cfg["eps_decay_decisions"]:
        return float(cfg["eps_final"])
    return cfg["eps_start"] + (cfg["eps_final"] - cfg["eps_start"]) * (d / cfg["eps_decay_decisions"])


def beta_at(decisions, cfg):
    """PER β after `decisions` decisions: linear over the whole budget, then exactly per_beta_final."""
    if decisions >= cfg["total_decisions"]:
        return float(cfg["per_beta_final"])
    return cfg["per_beta_start"] + (cfg["per_beta_final"] - cfg["per_beta_start"]) * (decisions / cfg["total_decisions"])


def layer_init(layer, std=math.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class DuelingQNet(nn.Module):
    algorithm = "d3qn"

    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(layer_init(nn.Linear(OBS_SIZE, 128)), nn.Tanh(), layer_init(nn.Linear(128, 128)), nn.Tanh())
        self.value = nn.Sequential(layer_init(nn.Linear(128, 128)), nn.Tanh(), layer_init(nn.Linear(128, 1), std=1.0))
        self.advantage = nn.Sequential(layer_init(nn.Linear(128, 128)), nn.Tanh(), layer_init(nn.Linear(128, ACTION_COUNT), std=1.0))

    def streams(self, obs):
        z = self.trunk(obs)
        return self.value(z), self.advantage(z)

    def forward(self, obs):
        v, a = self.streams(obs)
        return v + a - a.mean(dim=-1, keepdim=True)

    def action_scores(self, obs):
        return self.forward(obs)

    def export_layers(self):
        """The whole network as a plain tanh MLP 80 → 128 → 128 → 256 → 20
        (exact; the fold is computed in float64)."""
        t1, t2 = self.trunk[0], self.trunk[2]
        v1, v2 = self.value[0], self.value[2]
        a1, a2 = self.advantage[0], self.advantage[2]
        d = lambda x: x.detach().double()
        w3 = torch.cat([d(v1.weight), d(a1.weight)], dim=0)                   # 256 × 128
        b3 = torch.cat([d(v1.bias), d(a1.bias)], dim=0)
        wa = d(a2.weight)
        w4 = torch.cat([d(v2.weight).expand(ACTION_COUNT, -1), wa - wa.mean(dim=0, keepdim=True)], dim=1)  # 20 × 256
        b4 = d(v2.bias) + d(a2.bias) - d(a2.bias).mean()
        return [(d(t1.weight), d(t1.bias)), (d(t2.weight), d(t2.bias)), (w3, b3), (w4, b4)]


EXPORT_HIDDEN = [128, 128, 256]


def to_policy_json(net, obs_spec, act_spec, meta=None):
    layers = net.export_layers()
    return {
        "format": FORMAT,
        "algorithm": "d3qn",
        "modelType": "mlp",
        "architecture": {"input": OBS_SIZE, "hidden": list(EXPORT_HIDDEN), "activation": "tanh", "head": "q", "actions": ACTION_COUNT},
        "obsSpec": {"version": obs_spec["version"], "hash": obs_spec["hash"], "size": obs_spec["size"]},
        "actSpec": {"version": act_spec["version"], "hash": act_spec["hash"], "count": act_spec["count"]},
        "selection": dict(SELECTION["d3qn"]),
        "layers": [{"weight": w.double().tolist(), "bias": b.double().tolist()} for w, b in layers],
        "meta": {
            "trainedSteps": 0, "seed": None, "config": {},
            "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "exporter": "ml/ipl_rl/algos/d3qn.py",
            "trainingArchitecture": {"trunk": [OBS_SIZE, 128, 128], "value": [128, 128, 1], "advantage": [128, 128, ACTION_COUNT],
                                     "combine": "Q = V + A - mean(A), folded into the last exported layer",
                                     "exportedAs": "plain tanh MLP 80-128-128-256-20 (value|advantage hidden layers stacked)"},
            "inference": "masked argmax Q; no exploration, no replay, no target network",
            **(meta or {}),
        },
    }


def masked_argmax(q, mask):
    return torch.where(mask, q, torch.full_like(q, -math.inf)).argmax(dim=-1)


def select_action(net, obs_row, mask_row, eps, gen):
    """ε-greedy over the LEGAL actions only. Always draws one uniform u from
    gen; explores (uniform legal action, one more draw) when u < ε, else
    masked argmax Q. Returns (action, explored)."""
    legal = np.flatnonzero(mask_row)
    if not len(legal):
        raise AssertionError("no legal action")
    if gen.random() < eps:
        return int(legal[gen.integers(len(legal))]), True
    with torch.inference_mode():
        q = net(torch.from_numpy(np.asarray(obs_row, np.float32)).unsqueeze(0))[0]
    return int(masked_argmax(q, torch.from_numpy(np.asarray(mask_row, bool)))), False


def sync_target(updates, online, target, every):
    """Hard copy online → target when `updates` is a multiple of `every`.
    Returns the new target digest, or None when no copy was due."""
    if updates % every:
        return None
    target.load_state_dict(online.state_dict())
    dt, do = params_digest(target), params_digest(online)
    if dt != do:
        raise InstabilityError([f"target-network synchronisation failure: {dt} ≠ {do}"])
    return dt


def double_dqn_targets(online, target, ret, next_obs, next_mask, mult):
    """y = G + mult · Q_target(s', argmax_{a legal in s'} Q_online(s', a)); mult = 0 at termination."""
    with torch.no_grad():
        q_on = online(next_obs)
        a_star = masked_argmax(q_on, next_mask)
        q_tg = target(next_obs)
        q_tg_star = q_tg.gather(1, a_star.unsqueeze(1)).squeeze(1)
        boot = mult > 0
        y = ret + torch.where(boot, mult * q_tg_star, torch.zeros_like(ret))
        q_on_star = q_on.gather(1, a_star.unsqueeze(1)).squeeze(1)
    return y, a_star, q_on_star, q_tg_star, boot


def dqn_update(online, target, optimizer, batch, weights, cfg):
    """One optimiser step. Returns (stats, TD errors as numpy)."""
    obs = torch.from_numpy(batch["obs"])
    act = torch.from_numpy(batch["action"])
    ret = torch.from_numpy(batch["ret"]).float()
    next_obs = torch.from_numpy(batch["next_obs"])
    next_mask = torch.from_numpy(batch["next_mask"])
    mult = torch.from_numpy(batch["mult"]).float()
    w = torch.from_numpy(weights).float()
    y, a_star, q_on_star, q_tg_star, boot = double_dqn_targets(online, target, ret, next_obs, next_mask, mult)
    q_all = online(obs)
    q = q_all.gather(1, act.unsqueeze(1)).squeeze(1)
    td = y - q
    loss = (w * F.huber_loss(q, y, reduction="none", delta=cfg["huber_delta"])).mean()
    optimizer.zero_grad()
    loss.backward()
    grads_finite = all(bool(torch.isfinite(p.grad).all()) for p in online.parameters() if p.grad is not None)
    grad_norm = nn.utils.clip_grad_norm_(online.parameters(), cfg["max_grad_norm"]).item()
    optimizer.step()
    tdd = td.detach()
    with torch.no_grad():
        div = (q_on_star - q_tg_star).abs()[boot]
        legal_q = q_all.detach()[torch.from_numpy(batch["mask"])]
    stats = {
        "loss": loss.item(), "td_abs_mean": tdd.abs().mean().item(), "td_abs_max": tdd.abs().max().item(),
        "q_mean": q.detach().mean().item(), "q_min": q.detach().min().item(), "q_max": q.detach().max().item(),
        "q_legal_absmax": legal_q.abs().max().item() if legal_q.numel() else 0.0,
        "q_next_absmax": max(q_on_star.abs().max().item(), q_tg_star.abs().max().item()),
        "target_divergence": div.mean().item() if div.numel() else 0.0,
        "target_mean": y.mean().item(), "bootstrap_share": boot.float().mean().item(),
        "grad_norm": grad_norm, "grads_finite": grads_finite,
        "params_finite": all(bool(torch.isfinite(p).all()) for p in online.parameters()),
        "is_weight_mean": float(weights.mean()),
    }
    return stats, tdd.numpy().astype(np.float64)


def params_digest(module):
    h = hashlib.sha256()
    for name, p in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def update_problems(stats):
    """Per-update alarms (§17)."""
    out = []
    keys = ("loss", "td_abs_mean", "td_abs_max", "q_mean", "q_min", "q_max", "q_legal_absmax", "q_next_absmax", "grad_norm", "target_divergence")
    if not all(math.isfinite(stats[k]) for k in keys) or not stats["grads_finite"] or not stats["params_finite"]:
        out.append("non-finite loss / TD error / Q-value / gradient / parameter: " + json.dumps({k: stats[k] for k in keys + ("grads_finite", "params_finite")}, default=str))
    if stats["grad_norm"] > HEALTH["max_grad_norm_preclip"]:
        out.append(f"exploding gradient: pre-clip norm {stats['grad_norm']:.3g}")
    qmax = max(stats["q_legal_absmax"], stats["q_next_absmax"], abs(stats["q_min"]), abs(stats["q_max"]))
    if qmax > HEALTH["max_abs_q"]:
        out.append(f"Q divergence: |Q| {qmax:.3g}")
    if stats["target_divergence"] > HEALTH["max_target_divergence"]:
        out.append(f"online/target divergence {stats['target_divergence']:.3g}")
    return out


def epsilon_problem(explored, eps_sum, eps_var):
    """Explored-decision count vs its expectation under the schedule (binomial)."""
    sd = math.sqrt(max(eps_var, 0.0))
    if abs(explored - eps_sum) > HEALTH["epsilon_sigma"] * sd + 1.0:
        return f"epsilon schedule mismatch: {explored} explored decisions vs expected {eps_sum:.1f} ± {sd:.1f}"
    return None


def parity(net, policy, obs, masks):
    """Python (float64 copy of the network) vs production JavaScript Q-values,
    and masked-argmax agreement under the real masks and three synthetic ones:
    PASS only, one single legal action, and all 20 actions legal."""
    obs = np.asarray(obs, dtype=np.float32)
    masks = np.asarray(masks, dtype=bool)
    js = run_node_script("rl-policy-scores.js", {"policy": policy, "observations": obs.tolist()})
    if not js["ok"]:
        raise RuntimeError(f"production loader rejected the export: {js['error']}")
    net64 = copy.deepcopy(net).double()
    with torch.no_grad():
        py = net64(torch.from_numpy(obs).double()).numpy()
    jsn = np.asarray(js["scores"], dtype=np.float64)
    single = np.zeros_like(masks)
    rows = np.arange(len(masks))
    single[rows, np.array([np.flatnonzero(m)[-1] for m in masks])] = True
    pass_only = np.zeros_like(masks)
    pass_only[:, PASS] = True
    variants = {"real": masks, "passOnly": pass_only, "single": single, "allLegal": np.ones_like(masks)}
    agree = {}
    for name, m in variants.items():
        agree[name] = float((np.where(m, py, -np.inf).argmax(1) == np.where(m, jsn, -np.inf).argmax(1)).mean())
    return {"states": int(len(obs)), "maxAbsScoreDiff": float(np.abs(py - jsn).max()), "argmaxAgreement": min(agree.values()), "byMask": agree}


class SafetyError(RuntimeError):
    def __init__(self, problems):
        super().__init__("; ".join(map(str, problems[:5])))
        self.problems = problems


class InstabilityError(RuntimeError):
    def __init__(self, problems):
        super().__init__("; ".join(map(str, problems[:5])))
        self.problems = problems


class EpisodeConsistencyError(AssertionError):
    pass


def train(cfg):
    cfg_hash = config_hash(cfg)
    if cfg["algorithm"] != "d3qn":
        raise ValueError("this trainer is D3QN only")
    stage_b.check_stage(cfg, "Phase 2D.2 trains against Stage A only (no RL snapshots)")
    N, L = cfg["num_envs"], cfg["actor_lag_cycles"]
    total_cycles = cfg["total_decisions"] // N
    eval_cycles = sorted({d // N for d in cfg["eval_decisions"] if d % N == 0 and 0 < d // N <= total_cycles} | {total_cycles})
    run_name = cfg["run_name"] or f"d3qn-s{cfg['seed']}-{cfg_hash[:8]}-{time.strftime('%Y%m%d-%H%M%S')}"
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
        "seeds": {"init": init_seed(cfg["seed"]), "replay": replay_seed(cfg["seed"]), "explore": [explore_seed(cfg["seed"], i) for i in range(N)]}})
    log.write_json("metadata.json", meta)
    print(f"[d3qn] {run_name}: {total_cycles:,} cycles × {N} = {total_cycles * N:,} decisions  config {cfg_hash}  obs {envs.obs_spec['hash']}  act {envs.act_spec['hash']}", flush=True)

    torch.manual_seed(init_seed(cfg["seed"]))
    online = DuelingQNet()
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
    status, failure = "completed", None

    def dispatch(i):
        k = k_next[i]
        v = max(0, k - L)
        net = ring[v % (L + 1)]
        if ring_version[v % (L + 1)] != v:
            raise RuntimeError(f"actor version {v} not available (ring holds {ring_version[v % (L + 1)]})")
        d = k * N + i
        eps = epsilon_at(d, cfg)
        a, explored = select_action(net, obs[i], mask[i], eps, explore[i])
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
            stats, td = dqn_update(online, target, optimizer, replay.batch(idx), w, cfg)
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
        train_stats = {
            "updates": updates, "updates_in_window": len(us), "loss": agg("loss"), "td_abs_mean": agg("td_abs_mean"), "td_abs_max": agg("td_abs_max", np.max),
            "q_mean": agg("q_mean"), "q_min": agg("q_min", np.min), "q_max": agg("q_max", np.max), "q_absmax": max(agg("q_legal_absmax", np.max), agg("q_next_absmax", np.max)) if us else float("nan"),
            "target_divergence": agg("target_divergence"), "target_divergence_max": agg("target_divergence", np.max),
            "target_mean": agg("target_mean"), "bootstrap_share": agg("bootstrap_share"),
            "grad_norm": agg("grad_norm"), "grad_norm_max": agg("grad_norm", np.max), "is_weight_mean": agg("is_weight_mean"),
            "epsilon": eps_last, "explored": explored, "explored_expected": float(rec["eps"].sum()), "beta": beta_at(decisions, cfg), "replay_size": len(replay), "replay_added": replay.added,
            "priority_mean": float(pri.mean()) if len(pri) else float("nan"), "priority_max": float(pri.max()) if len(pri) else float("nan"),
            "max_priority_seen": replay.max_priority, "target_updates": len(target_syncs), "samples_consumed": samples_consumed,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
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
        print(f"[d3qn] win {w + 1:3d}  dec {decisions:>9,}  eps {episodes.total:5d}  ret {ep_stats.get('return', float('nan')):6.3f}  "
              f"XI {ep_stats.get('xi', float('nan')):6.2f}  ε {eps_last:.3f}  β {train_stats['beta']:.3f}  upd {updates:,}  tgt {len(target_syncs)}  "
              f"loss {train_stats['loss']:.5f}  |td| {train_stats['td_abs_mean']:.4f}  Q {train_stats['q_mean']:.3f} [{train_stats['q_min']:.2f},{train_stats['q_max']:.2f}]  "
              f"div {train_stats['target_divergence']:.4f}  gn {train_stats['grad_norm']:.3f}  bid {acts['bid_share']:.3f}  replay {len(replay):,}  "
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
        name = "policy:d3qn"
        head = headline(report, name)
        shares = report["report"][name]["actionShares"]
        row = {"update": c // W, "cycles": c, "decisions": decisions, "final": final, "validationEpisodes": report["n"],
               "seconds": time.perf_counter() - t1, "parity": par, "safetyProblems": problems, "metrics": head, "actionShares": shares,
               "updates": updates, "targetUpdates": len(target_syncs)}
        evaluations.append(row)
        log.scalars("validation", head, decisions)
        log.record({"type": "evaluation", **row})
        print(f"[d3qn] eval @ {decisions:,}: validation n={report['n']} XI {head['xi']:.2f} (Δ vs moneyball {head.get('xiDiffVsMoneyball', float('nan')):+.2f})  "
              f"legal {100 * head['legalXI']:.0f}%  purse left {head['purseLeftShare']:.2f}  price/fair {head['priceToFair']:.2f}  "
              f"parity {par['maxAbsScoreDiff']:.1e}/{par['argmaxAgreement']:.3f}  safety {'OK' if not problems else problems[:3]}", flush=True)
        timers["eval"] += time.perf_counter() - t0
        if problems:
            raise SafetyError(problems)
        if par["maxAbsScoreDiff"] > 1e-6 or par["argmaxAgreement"] < 1.0:
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
        print(f"[d3qn] STOPPED — safety invariant failed: {err.problems[:5]}", flush=True)
    except (InstabilityError, ReplayCorruption) as err:
        status, failure = "stopped: instability", getattr(err, "problems", [str(err)])
        print(f"[d3qn] STOPPED — algorithm instability: {failure[:5]}", flush=True)
    except Exception as err:
        status, failure = f"stopped: {type(err).__name__}", [str(err)]
        print(f"[d3qn] STOPPED — {type(err).__name__}: {err}", flush=True)
        raise
    finally:
        wall = time.perf_counter() - t_start
        decisions = processed * N
        summary = {
            "runName": run_name, "algorithm": "d3qn", "status": status, "failure": failure, "configHash": cfg_hash,
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
