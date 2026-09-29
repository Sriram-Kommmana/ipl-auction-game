"""OpenAI-ES (antithetic, centered-rank) for IplAuctionEnv-v2 (Phase 2D.4).

    cd ml
    .venv/Scripts/python -m ipl_rl.algos.openai_es --config ipl_rl/configs/openai_es_2d4.json --set seed=1

Policy: the deployable PolicyNet("es") — 80 → 64 → 64 → 20 logits, tanh
hidden, linear head; no value head, no critic, no replay. Parameters θ are
the network's parameters flattened in PyTorch order (body.0.weight,
body.0.bias, body.2.weight, body.2.bias, head.weight, head.bias; d = 10,644).
Initialisation: orthogonal (√2 hidden, 0.01 head), zero biases, from the
seed derived from ("openai_es", run seed).

Acting during training: categorical over the LEGAL actions only (masked
softmax of the logits, masked actions probability 0, renormalised), one
uniform draw per decision from the job's own action stream. Evaluation and
production: masked argmax of the logits (rl-policy-v2 algorithm "es").

One generation g (frozen):
  1. for pair i = 0..N−1 (N = 32): ε_i ~ N(0, I_d) from the stream
     ("openai_es/noise", seed, g, i)
  2. two policies θ + σε_i and θ − σε_i; each plays ONE complete episode on
     the same auction — training seed episode_seed(seed, i, g) — with the
     same action stream ("openai_es/act", seed, g, i): common random numbers,
     only the parameters differ
  3. fitness f = the episode return (frozen reward, γ = 1)
  4. centered ranks over the 2N fitnesses: s = rank/(2N − 1) − 0.5, rank
     0-based with AVERAGE ranks for ties (so f+ = f− ⇒ s+ = s−)
  5. g_hat = 1/(2N·σ) · Σ_i (s_i+ − s_i−) ε_i  — the shaped paired difference
     (not the raw one), 2N = 64 = the population size in perturbations
  6. Adam ascent (lr 0.01, β 0.9/0.999, eps 1e-8, no weight decay):
     θ.grad = −g_hat, one optimizer step per generation

Episodes run asynchronously on the existing vector of Node simulators, but
every episode's trajectory is a function of (parameters, auction seed,
action stream) only, so nothing depends on which simulator runs it or when.
Checkpoint + Node validation (masked argmax, the locked baselines, the
safety invariants) at frozen points: the first generation boundary at or
after each of the eight PPO/A2C/DQN decision checkpoints, and every
`eval_every_generations` generations.
"""

import argparse
import hashlib
import json
import math
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from ..common.checkpoint import export_policy, save_checkpoint
from ..common.config import config_hash, load_config
from ..common.evaluation import headline, node_evaluate, safety_problems
from ..common.logger import RunLogger
from ..common.metadata import run_metadata
from ..common.seeding import episode_seed, seed_everything
from ..common.stats import EpisodeLog, action_stats, summarise_episodes
from ..common.vec_env import VecIplAuctionEnv
from ..stage_b import hooks as stage_b  # Phase 2F (default-off in Stage A)
from ..common.win_qos import disable_throttling
from ..env import ACTION_COUNT, OBS_SIZE, PASS
from ..nets import PolicyNet

ML_ROOT = Path(__file__).resolve().parents[2]
DECISION_CHECKPOINTS = [245_760, 497_664, 749_568, 1_001_472, 1_247_232, 1_499_136, 1_751_040, 1_996_800]

DEFAULTS = {
    "algorithm": "openai_es",
    "seed": 1,
    "generations": 2_000,
    "pairs": 32,                      # antithetic pairs per generation (64 perturbed policies)
    "episodes_per_perturbation": 1,
    "sigma": 0.05,
    "learning_rate": 0.01,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_eps": 1e-8,
    "weight_decay": 0.0,
    "fitness_shaping": "centered_rank",
    "num_envs": 12,
    "stage": "A",
    "tremble": 0.01,
    "snapshot_share": 0.0,
    "split": "train",
    "decision_checkpoints": DECISION_CHECKPOINTS,
    "eval_every_generations": 250,
    "eval_limit": 500,
    "eval_workers": 12,
    "parity_states": 256,
    "torch_threads": 2,
    "watchdog_seconds": 120,
    "expected_act_spec": None,
    "expected_obs_spec": None,
    "run_dir": "runs",
    "run_name": None,
}


def _derived(tag, *parts, bytes_=8):
    key = "/".join([tag, *(str(int(p)) for p in parts)])
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:bytes_], "little")


def init_seed(seed):
    """Initialisation seed of ES run `seed`, derived from ("openai_es", seed)."""
    return _derived("openai_es/init", seed, bytes_=4)


def noise_seed(seed, generation, pair):
    return _derived("openai_es/noise", seed, generation, pair)


def action_seed(seed, generation, pair):
    """The action-sampling stream of pair i — shared by θ+σε and θ−σε (CRN)."""
    return _derived("openai_es/act", seed, generation, pair)


def pair_episode_seed(run_seed, generation, pair):
    """The auction of pair i in generation g — one training seed from the frozen
    schedule, shared by both members of the pair (CRN)."""
    return int(episode_seed(run_seed, pair, generation))


def layer_init(layer, std=math.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


def make_policy(seed):
    torch.manual_seed(init_seed(seed))
    net = PolicyNet("es")
    for m in net.body:
        if isinstance(m, nn.Linear):
            layer_init(m)
    layer_init(net.head, std=0.01)
    return net


def noise(seed, generation, pair, dim):
    return np.random.default_rng(noise_seed(seed, generation, pair)).standard_normal(dim, dtype=np.float32)


def centered_ranks(x):
    """Centered ranks in [−0.5, 0.5]: 0-based ranks (ties get the AVERAGE rank)
    divided by (n − 1), minus 0.5."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    if n < 2:
        return np.zeros(n)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    sx = x[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks / (n - 1) - 0.5


def es_gradient(shaped_plus, shaped_minus, eps, sigma):
    """g_hat = 1/(2N·σ) · Σ_i (s_i+ − s_i−) ε_i ; eps [N, d]."""
    diff = np.asarray(shaped_plus, np.float64) - np.asarray(shaped_minus, np.float64)
    n_pert = 2 * len(diff)
    return (diff @ np.asarray(eps, np.float64)) / (n_pert * sigma)


class FlatPolicy:
    """Numpy forward of one parameter vector (PolicyNet("es") layout)."""

    SHAPES = [(64, OBS_SIZE), (64,), (64, 64), (64,), (ACTION_COUNT, 64), (ACTION_COUNT,)]

    def __init__(self, theta):
        theta = np.asarray(theta, dtype=np.float32)
        parts, k = [], 0
        for shp in self.SHAPES:
            size = int(np.prod(shp))
            parts.append(theta[k:k + size].reshape(shp))
            k += size
        if k != len(theta):
            raise ValueError(f"parameter vector of length {len(theta)}, expected {k}")
        self.w1, self.b1, self.w2, self.b2, self.w3, self.b3 = parts

    def logits(self, obs):
        h = np.tanh(self.w1 @ obs + self.b1)
        h = np.tanh(self.w2 @ h + self.b2)
        return self.w3 @ h + self.b3


def masked_probs(logits, mask):
    """Categorical over the legal actions: masked actions exactly 0, renormalised."""
    z = np.where(mask, np.asarray(logits, np.float64), -np.inf)
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def sample_action(logits, mask, rng):
    p = masked_probs(logits, mask)
    u = rng.random()
    a = int(np.searchsorted(np.cumsum(p), u, side="right"))
    a = min(a, ACTION_COUNT - 1)
    while not mask[a]:          # guard against a cumulative-sum rounding edge
        a -= 1
    return a


def masked_argmax(logits, mask):
    return int(np.argmax(np.where(mask, logits, -np.inf)))


def digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


class SafetyError(RuntimeError):
    def __init__(self, problems):
        super().__init__("; ".join(map(str, problems[:5])))
        self.problems = problems


class InstabilityError(RuntimeError):
    def __init__(self, problems):
        super().__init__("; ".join(map(str, problems[:5])))
        self.problems = problems


class EpisodeRunner:
    """Plays episode jobs on the vector of Node simulators, asynchronously.

    A job = (key, auction seed, parameter vector, action-stream seed). Its
    trajectory depends on nothing else, so results are identical whatever
    simulator runs it and in whatever order replies arrive. Every finished
    episode is checked against the JavaScript summary (return, decisions,
    action counts) and the safety invariants.
    """

    def __init__(self, venv):
        self.venv = venv
        self.consistency_checks = 0
        self.allow_incomplete = False  # Phase 2F: True only in Stage B (§26 finding, not a stop)
        self.wait_seconds = 0.0
        self.act_seconds = 0.0

    def run(self, jobs):
        venv = self.venv
        n = venv.num_envs
        queue = deque(jobs)
        slot = [None] * n
        results = {}

        def start(i):
            job = queue.popleft()
            slot[i] = {"job": job, "policy": FlatPolicy(job["theta"]), "rng": np.random.default_rng(job["act_seed"]),
                       "ret": 0.0, "steps": 0, "actions": np.zeros(ACTION_COUNT, np.int64), "masks": 0, "pending": "reset"}
            venv.current_seed[i] = job["seed"]
            venv.bridges[i].send("reset", seed=int(job["seed"]), split="train")

        def act(i, obs, mask):
            t0 = time.perf_counter()
            s = slot[i]
            a = sample_action(s["policy"].logits(np.asarray(obs, np.float32)), np.asarray(mask, bool), s["rng"])
            self.act_seconds += time.perf_counter() - t0
            if not mask[a]:
                raise SafetyError([f"masked action {a} selected"])
            s["actions"][a] += 1
            s["pending"] = "step"
            venv.send_step(i, a)

        for i in range(min(n, len(queue))):
            start(i)
        in_flight = sum(1 for s in slot if s is not None)
        while in_flight:
            t0 = time.perf_counter()
            i, reply = venv.next_reply()
            self.wait_seconds += time.perf_counter() - t0
            s = slot[i]
            if s["pending"] == "reset":
                act(i, reply["obs"], reply["mask"])
                continue
            s["ret"] += reply["reward"]
            s["steps"] += 1
            if not reply["done"]:
                act(i, reply["obs"], reply["mask"])
                continue
            ep = reply["info"]["episode"]
            problems = []
            if abs(s["ret"] - ep["return"]) > 1e-9:
                problems.append(f"reward sum {s['ret']!r} ≠ episode return {ep['return']!r}")
            if s["steps"] != ep["decisions"]:
                problems.append(f"{s['steps']} steps ≠ {ep['decisions']} decisions")
            if s["actions"].tolist() != list(ep["actionCounts"]):
                problems.append("action counts differ")
            if ep["seed"] != s["job"]["seed"]:
                problems.append(f"played seed {ep['seed']} ≠ scheduled {s['job']['seed']}")
            if problems:
                raise SafetyError([f"episode consistency (job {s['job']['key']}): " + "; ".join(problems)])
            if ep.get("invariantViolations", 0):
                raise SafetyError([f"invariant violation (seed {ep['seed']}): {ep.get('violations')}"])
            if not ep["legalXI"] and not self.allow_incomplete:
                raise SafetyError([f"episode ended without a legal XI: seed {ep['seed']}"])
            self.consistency_checks += 1
            results[s["job"]["key"]] = {"return": float(s["ret"]), "episode": ep, "actions": s["actions"].copy()}
            slot[i] = None
            in_flight -= 1
            if queue:
                start(i)
                in_flight += 1
        return results


def generation_jobs(seed, generation, theta, sigma, pairs, dim):
    """The 2N jobs of one generation and the noise used. θ and ε in float32."""
    eps = np.stack([noise(seed, generation, i, dim) for i in range(pairs)])
    jobs = []
    for i in range(pairs):
        ep_seed = pair_episode_seed(seed, generation, i)
        a_seed = action_seed(seed, generation, i)
        for sign in (+1, -1):
            jobs.append({"key": (i, sign), "seed": ep_seed, "act_seed": a_seed,
                         "theta": (theta + np.float32(sign * sigma) * eps[i]).astype(np.float32)})
    return jobs, eps


def es_update(net, optimizer, grad):
    """One Adam ascent step on the flattened parameters. Returns the update vector."""
    before = parameters_to_vector(net.parameters()).detach().clone()
    g = torch.from_numpy(np.asarray(-grad, np.float32))
    k = 0
    for p in net.parameters():
        n = p.numel()
        p.grad = g[k:k + n].view_as(p).clone()
        k += n
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return (parameters_to_vector(net.parameters()).detach() - before).numpy()


def optimizer_state_finite(optimizer):
    for st in optimizer.state.values():
        for v in st.values():
            if torch.is_tensor(v) and not bool(torch.isfinite(v).all()):
                return False
    return True


def optimizer_digest(optimizer):
    h = hashlib.sha256()
    for st in optimizer.state.values():
        for k in sorted(st):
            v = st[k]
            h.update(k.encode())
            h.update(v.detach().cpu().numpy().tobytes() if torch.is_tensor(v) else str(v).encode())
    return h.hexdigest()[:16]


def params_digest(net):
    return digest(parameters_to_vector(net.parameters()).detach().numpy())


def parity(net, policy, obs, masks):
    """Python (float64 forward on the exported float32 weights) vs production
    JavaScript logits, and masked-argmax agreement under the real masks, PASS
    only, one single legal action and all 20 legal."""
    import copy
    from ..bridge import run_node_script
    obs = np.asarray(obs, dtype=np.float32)
    masks = np.asarray(masks, dtype=bool)
    js = run_node_script("rl-policy-scores.js", {"policy": policy, "observations": obs.tolist()})
    if not js["ok"]:
        raise RuntimeError(f"production loader rejected the export: {js['error']}")
    with torch.no_grad():
        py = copy.deepcopy(net).double()(torch.from_numpy(obs).double()).numpy()
    jsn = np.asarray(js["scores"], dtype=np.float64)
    single = np.zeros_like(masks)
    single[np.arange(len(masks)), np.array([np.flatnonzero(m)[-1] for m in masks])] = True
    pass_only = np.zeros_like(masks)
    pass_only[:, PASS] = True
    variants = {"real": masks, "passOnly": pass_only, "single": single, "allLegal": np.ones_like(masks)}
    agree = {k: float((np.where(m, py, -np.inf).argmax(1) == np.where(m, jsn, -np.inf).argmax(1)).mean()) for k, m in variants.items()}
    return {"states": int(len(obs)), "maxAbsScoreDiff": float(np.abs(py - jsn).max()), "argmaxAgreement": min(agree.values()), "byMask": agree}


def train(cfg):
    cfg_hash = config_hash(cfg)
    if cfg["algorithm"] != "openai_es":
        raise ValueError("this trainer is OpenAI-ES only")
    stage_b.check_stage(cfg, "Phase 2D.4 trains against Stage A only (no RL snapshots)")
    if cfg["fitness_shaping"] != "centered_rank" or cfg["weight_decay"] != 0.0 or cfg["episodes_per_perturbation"] != 1:
        raise ValueError("frozen ES configuration: centered-rank shaping, no weight decay, one episode per perturbation")
    run_name = cfg["run_name"] or f"openai-es-s{cfg['seed']}-{cfg_hash[:8]}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = (ML_ROOT / cfg["run_dir"] / run_name).resolve()
    seed, N, sigma = cfg["seed"], cfg["pairs"], cfg["sigma"]

    seed_everything(seed, cfg["torch_threads"])
    envs = VecIplAuctionEnv(cfg["num_envs"], seed, split=cfg["split"], tremble=cfg["tremble"],
                            snapshot_share=cfg["snapshot_share"], watchdog_seconds=cfg["watchdog_seconds"], **stage_b.env_kwargs(cfg))
    for key, spec in (("expected_act_spec", envs.act_spec), ("expected_obs_spec", envs.obs_spec)):
        if cfg[key] and spec["hash"] != cfg[key]:
            envs.close()
            raise ValueError(f"spec mismatch: bridge {spec}, expected {cfg[key]}")
    if envs.gamma != 1.0:
        envs.close()
        raise ValueError(f"environment gamma {envs.gamma} ≠ 1 (frozen)")

    log = RunLogger(run_dir)
    net = make_policy(seed)
    stage_b.warm_start(cfg, {"policy": net}, run_dir)
    dim = sum(p.numel() for p in net.parameters())
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg["learning_rate"], betas=(cfg["adam_beta1"], cfg["adam_beta2"]),
                                 eps=cfg["adam_eps"], weight_decay=cfg["weight_decay"])
    initial_digest = params_digest(net)
    log.write_json("metadata.json", run_metadata(cfg, cfg_hash, envs.obs_spec, envs.act_spec, envs.gamma, extra={
        "runName": run_name, "parameters": dim, "initSeed": init_seed(seed),
        "estimator": "g = 1/(2N sigma) * sum_i (s_i+ - s_i-) eps_i with s = centered ranks (average ties) of all 2N returns",
        "checkpointRule": "first generation boundary at or after each decision checkpoint, plus every eval_every_generations"}))
    print(f"[es] {run_name}: {cfg['generations']:,} generations × {2 * N} perturbations (d = {dim:,})  σ {sigma}  lr {cfg['learning_rate']}  "
          f"config {cfg_hash}  obs {envs.obs_spec['hash']}  act {envs.act_spec['hash']}", flush=True)

    runner = EpisodeRunner(envs)
    runner.allow_incomplete = not stage_b.incomplete_is_stop(cfg)
    episodes = EpisodeLog()
    all_episodes, evaluations = [], []
    decisions = 0
    pending_decision_ckpts = list(cfg["decision_checkpoints"])
    t_start = time.perf_counter()
    timers = {"episodes": 0.0, "update": 0.0, "noise": 0.0, "eval": 0.0}
    last_states = None
    done = [0]
    status, failure = "completed", None

    def evaluate(generation, reasons):
        t0 = time.perf_counter()
        ckpt_dir = run_dir / "checkpoints" / f"gen_{generation:04d}"
        save_checkpoint(ckpt_dir / "checkpoint.pt", modules={"policy": net}, optimizer=optimizer,
                        step_state={"generation": generation, "decisions": decisions, "episodes": episodes.total}, cfg=cfg, cfg_hash=cfg_hash)
        policy = export_policy(ckpt_dir / "policy.json", net, envs.obs_spec, envs.act_spec, trained_steps=decisions, seed=seed, cfg=cfg,
                               cfg_hash=cfg_hash, extra_meta={"algorithmName": "openai_es", "generation": generation, "stage": cfg["stage"],
                                                              "runName": run_name, "parameterDigest": params_digest(net),
                                                              "selection": "masked argmax of the 20 logits",
                                                              "esHyperparameters": {k: cfg[k] for k in ("pairs", "sigma", "learning_rate", "adam_beta1", "adam_beta2",
                                                                                                        "adam_eps", "weight_decay", "fitness_shaping",
                                                                                                        "episodes_per_perturbation")},
                                                              "trainingBudget": {"generations": generation, "decisions": decisions, "episodes": episodes.total,
                                                                                 "policyEvaluations": generation * 2 * N, "optimizerUpdates": generation}})
        obs, masks = last_states
        par = parity(net, policy, obs, masks)
        report, eps = node_evaluate(ckpt_dir / "policy.json", ckpt_dir / "validation", limit=cfg["eval_limit"], workers=cfg["eval_workers"])
        problems = safety_problems(report, eps)
        for key, name in (("expected_act_spec", "actSpecHash"), ("expected_obs_spec", "obsSpecHash")):
            if cfg[key] and report.get(name) not in (None, cfg[key]):
                problems.append(f"{name} mismatch in validation: {report.get(name)} != {cfg[key]}")
        head = headline(report, "policy:es")
        row = {"generation": generation, "decisions": decisions, "episodes": episodes.total, "reasons": reasons,
               "validationEpisodes": report["n"], "seconds": time.perf_counter() - t0, "parity": par, "safetyProblems": problems,
               "metrics": head, "actionShares": report["report"]["policy:es"]["actionShares"], "parameterDigest": params_digest(net)}
        evaluations.append(row)
        log.scalars("validation", head, decisions)
        log.record({"type": "evaluation", **row})
        print(f"[es] eval gen {generation} @ {decisions:,} ({', '.join(reasons)}): validation n={report['n']} XI {head['xi']:.2f} "
              f"(Δ vs moneyball {head.get('xiDiffVsMoneyball', float('nan')):+.2f})  legal {100 * head['legalXI']:.0f}%  "
              f"price/fair {head['priceToFair']:.2f}  parity {par['maxAbsScoreDiff']:.1e}/{par['argmaxAgreement']:.3f}  "
              f"safety {'OK' if not problems else problems[:3]}", flush=True)
        timers["eval"] += time.perf_counter() - t0
        if problems:
            raise SafetyError(problems)
        if par["maxAbsScoreDiff"] > 1e-9 or par["argmaxAgreement"] < 1.0:
            raise SafetyError([f"export parity failed: {par}"])

    try:
        for gen in range(1, cfg["generations"] + 1):
            t0 = time.perf_counter()
            theta = parameters_to_vector(net.parameters()).detach().numpy().astype(np.float32)
            jobs, eps = generation_jobs(seed, gen, theta, sigma, N, dim)
            # Determinism / CRN self-checks (§20): regenerate one noise vector,
            # and every pair's two members share auction and action streams.
            if digest(noise(seed, gen, 0, dim)) != digest(eps[0]):
                raise InstabilityError(["ES perturbation not deterministic"])
            for i in range(N):
                a, b = jobs[2 * i], jobs[2 * i + 1]
                if a["seed"] != b["seed"] or a["act_seed"] != b["act_seed"] or a["key"][0] != b["key"][0]:
                    raise SafetyError([f"antithetic pair {i}: seeds differ"])
            timers["noise"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            res = runner.run(jobs)
            timers["episodes"] += time.perf_counter() - t0
            f_plus = np.array([res[(i, +1)]["return"] for i in range(N)])
            f_minus = np.array([res[(i, -1)]["return"] for i in range(N)])
            fit = np.stack([f_plus, f_minus], axis=1).ravel()       # [+0, −0, +1, −1, …]
            shaped = centered_ranks(fit).reshape(N, 2)
            t0 = time.perf_counter()
            grad = es_gradient(shaped[:, 0], shaped[:, 1], eps, sigma)
            if not np.all(np.isfinite(fit)) or not np.all(np.isfinite(grad)):
                raise InstabilityError(["non-finite fitness or gradient"])
            step = es_update(net, optimizer, grad)
            theta_new = parameters_to_vector(net.parameters()).detach().numpy()
            if not np.all(np.isfinite(theta_new)) or not optimizer_state_finite(optimizer):
                raise InstabilityError(["non-finite parameters or optimizer state"])
            timers["update"] += time.perf_counter() - t0
            done[0] = gen

            gen_eps = [res[(i, s)]["episode"] for i in range(N) for s in (+1, -1)]
            for k, e in enumerate(gen_eps):
                e = dict(e)
                e["envIndex"] = k
                episodes.add(e)
            finished = episodes.drain()
            all_episodes.extend(finished)
            gen_decisions = int(sum(e["decisions"] for e in gen_eps))
            decisions += gen_decisions
            acts = np.array([res[(i, s)]["actions"] for i in range(N) for s in (+1, -1)]).sum(0)
            ep_stats = summarise_episodes(finished)
            dfit = f_plus - f_minus
            stats = {
                "fitness_mean": float(fit.mean()), "fitness_std": float(fit.std()), "fitness_min": float(fit.min()), "fitness_max": float(fit.max()),
                "fitness_plus_mean": float(f_plus.mean()), "fitness_minus_mean": float(f_minus.mean()),
                "pair_diff_abs_mean": float(np.abs(dfit).mean()), "pair_ties": int((dfit == 0).sum()),
                "grad_norm": float(np.linalg.norm(grad)), "step_norm": float(np.linalg.norm(step)), "theta_norm": float(np.linalg.norm(theta_new)),
                "update_ratio": float(np.linalg.norm(step) / max(1e-12, np.linalg.norm(theta))),
                "decisions_in_generation": gen_decisions,
            }
            row = {"type": "update", "update": gen, "generation": gen, "decisions": decisions, "episodes": episodes.total,
                   "noiseDigest": digest(eps), "episodeSeeds": [jobs[2 * i]["seed"] for i in range(N)], "fitnessDigest": digest(fit),
                   "shapedDigest": digest(shaped), "gradDigest": digest(grad), "paramsDigest": digest(theta_new),
                   "optimizerDigest": optimizer_digest(optimizer), "actionsDigest": digest(np.array([res[(i, s)]["actions"] for i in range(N) for s in (+1, -1)])),
                   "train": stats, "episode": ep_stats,
                   "actions": {"action_shares": (acts / max(1, acts.sum())).tolist(), "bid_share": float(1 - acts[PASS] / max(1, acts.sum()))},
                   "perf": {"decisions_per_sec_overall": decisions / (time.perf_counter() - t_start), **{f"{k}_seconds": v for k, v in timers.items()},
                            "wait_seconds": runner.wait_seconds, "act_seconds": runner.act_seconds}}
            log.record(row)
            log.scalars("train", stats, decisions)
            log.scalars("episode", ep_stats, decisions)
            if gen == 1 or gen % 10 == 0:
                print(f"[es] gen {gen:4d}  dec {decisions:>10,}  eps {episodes.total:6d}  fit {stats['fitness_mean']:.4f} "
                      f"[{stats['fitness_min']:.3f},{stats['fitness_max']:.3f}]  |Δpair| {stats['pair_diff_abs_mean']:.4f} ties {stats['pair_ties']}  "
                      f"XI {ep_stats.get('xi', float('nan')):.2f}  bid {row['actions']['bid_share']:.3f}  |g| {stats['grad_norm']:.2f}  "
                      f"|θ| {stats['theta_norm']:.2f}  step/θ {stats['update_ratio']:.2e}  {row['perf']['decisions_per_sec_overall']:.0f} dec/s", flush=True)

            reasons = []
            while pending_decision_ckpts and decisions >= pending_decision_ckpts[0]:
                reasons.append(f"≥{pending_decision_ckpts.pop(0):,} decisions")
            if gen % cfg["eval_every_generations"] == 0 or gen == cfg["generations"]:
                reasons.append(f"generation {gen}")
            if reasons:
                last_states = parity_states(envs, net, seed, cfg["parity_states"])
                evaluate(gen, reasons)
            sb_stop = (cfg.get("stage_b") or {}).get("stop_decisions")  # Phase 2F decision budget (Stage B only)
            if sb_stop is not None and decisions >= sb_stop:
                if not reasons:
                    last_states = parity_states(envs, net, seed, cfg["parity_states"])
                    evaluate(gen, ["stage-b decision budget"])
                break
    except SafetyError as err:
        status, failure = "stopped: safety", err.problems
        print(f"[es] STOPPED — safety: {err.problems[:5]}", flush=True)
    except InstabilityError as err:
        status, failure = "stopped: instability", err.problems
        print(f"[es] STOPPED — instability: {err.problems[:5]}", flush=True)
    except Exception as err:
        status, failure = f"stopped: {type(err).__name__}", [str(err)]
        print(f"[es] STOPPED — {type(err).__name__}: {err}", flush=True)
        raise
    finally:
        wall = time.perf_counter() - t_start
        summary = {
            "runName": run_name, "algorithm": "openai_es", "exportAlgorithm": "es", "status": status, "failure": failure, "configHash": cfg_hash,
            "seed": seed, "obsSpec": envs.obs_spec, "actSpec": envs.act_spec, "parameters": dim,
            "generations": done[0], "totalDecisions": decisions, "episodes": episodes.total,
            "policyEvaluations": done[0] * 2 * N, "optimizerUpdates": done[0],
            "wallSeconds": wall, "timers": {**timers, "wait": runner.wait_seconds, "act": runner.act_seconds},
            "decisionsPerSecOverall": decisions / wall if wall else None,
            "initialParamsDigest": initial_digest, "finalParamsDigest": params_digest(net),
            "episodeConsistencyChecks": runner.consistency_checks,
            "illegalActions": 0 if status == "completed" else None,
            "incompleteXiTrainingEpisodes": sum(1 for e in all_episodes if not e["legalXI"]),
            "trainingEpisodes": {"all": summarise_episodes(all_episodes), "first500": summarise_episodes(all_episodes[:500]),
                                 "last500": summarise_episodes(all_episodes[-500:])},
            "evaluations": evaluations,
        }
        log.write_json("summary.json", summary)
        (run_dir / "training_episodes.json").write_text(json.dumps(all_episodes), encoding="utf-8")
        log.close()
        envs.close()
    return run_dir, summary


def parity_states(venv, net, seed, count):
    """Real decision states for export parity: one deterministic episode per call
    (a fixed training seed, the current policy's masked argmax), all its states."""
    runner_seed = pair_episode_seed(seed, 10**6, 0)
    b = venv.bridges[0]
    policy = FlatPolicy(parameters_to_vector(net.parameters()).detach().numpy())
    b.send("reset", seed=int(runner_seed), split="train")
    obs, masks = [], []
    i, r = venv.next_reply()
    while True:
        obs.append(r["obs"])
        masks.append(r["mask"])
        a = masked_argmax(policy.logits(np.asarray(r["obs"], np.float32)), np.asarray(r["mask"], bool))
        venv.send_step(0, a)
        i, r = venv.next_reply()
        if r["done"]:
            break
    idx = np.unique(np.linspace(0, len(obs) - 1, min(count, len(obs))).astype(int))
    return np.asarray(obs, np.float32)[idx], np.asarray(masks, bool)[idx]


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
