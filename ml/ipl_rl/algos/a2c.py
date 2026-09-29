"""Masked A2C for IplAuctionEnv-v2 (Phase 2D.1) — synchronous advantage actor-critic, one file.

    cd ml
    .venv/Scripts/python -m ipl_rl.algos.a2c --config ipl_rl/configs/a2c_2d1.json --set seed=1

Networks (same conventions as the PPO anchor, independent parameters):
  actor   the deployable PolicyNet("a2c"): 80 → 128 → 128 → 20 logits (tanh),
          exported as rl-policy-v2 with algorithm "a2c"
  critic  a separate 80 → 128 → 128 → 1 tanh MLP (training only, never exported)
Orthogonal init (√2 hidden, 0.01 policy head, 1.0 value head), drawn from an
A2C-specific seed (init_seed) — never PPO's initial or trained weights.

Algorithm (standard synchronous A2C): collect num_envs × num_steps on-policy
decisions with masked sampling; advantages by GAE(γ, λ) with
termination-only episodes (γ = 1 from the environment); then exactly ONE
gradient step on the whole rollout:

    loss = −mean(Â · log π(a|s)) − ent_coef · mean(H[π(·|s)]) + vf_coef · ½ mean((V(s) − R)²)

Â = normalised advantages, R = Â_raw + V_old. No importance ratio, no clipping,
no epochs, no minibatches, no KL target — the rollout is discarded after the
step. Adam (eps 1e-5), global gradient-norm clipping, constant learning rate.
The log-probabilities recomputed for the loss are checked against the ones
used when sampling (the update is exactly on-policy).

Masking: the act-v3 mask comes from JavaScript with every observation.
Illegal actions get logit −1e8 (common/masking.py), i.e. probability 0 in
sampling, in the loss and in the entropy; a masked action reaching the
environment is a hard stop (assert_legal before the update, the Python env
wrapper and the JavaScript step all refuse it).

Rollout collection, checkpointing, export, export parity, Node validation
against the locked baselines, the safety invariants and the per-episode
consistency checks are the shared Phase 2C infrastructure, unchanged.

Algorithm-health stops (reported, never "fixed" by changing hyperparameters):
non-finite loss / gradients / parameters; entropy collapse; value-loss
explosion; a single action taking almost every decision (see HEALTH).
"""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..common.checkpoint import export_policy, save_checkpoint
from ..common.config import config_hash, load_config
from ..common.evaluation import export_parity, headline, node_evaluate, safety_problems
from ..common.logger import RunLogger
from ..common.masking import assert_legal, masked_distribution, normalised_entropy
from ..common.metadata import run_metadata
from ..common.rollout import AsyncRolloutState, collect_async
from ..common.seeding import seed_everything
from ..common.stats import EpisodeLog, action_stats, summarise_episodes
from ..common.vec_env import VecIplAuctionEnv
from ..stage_b import hooks as stage_b  # Phase 2F (default-off in Stage A)
from ..common.win_qos import disable_throttling
from ..env import ACTION_COUNT, OBS_SIZE
from ..nets import PolicyNet

ML_ROOT = Path(__file__).resolve().parents[2]

DEFAULTS = {
    "algorithm": "a2c",
    "seed": 1,
    # budget
    "total_decisions": 250_000,       # upper bound; rounded DOWN to whole updates
    "num_envs": 12,
    "num_steps": 512,                 # rollout length per environment; one update per rollout
    # A2C
    "learning_rate": 3e-4,            # constant
    "gamma": 1.0,                     # must equal the environment's (frozen) γ
    "gae_lambda": 0.95,
    "norm_adv": True,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "adam_eps": 1e-5,
    # environment / curriculum
    "split": "train",
    "stage": "A",                     # A = frozen rule bots + human proxy, no RL snapshots
    "tremble": 0.01,
    "snapshot_share": 0.0,
    # evaluation / checkpoints
    "eval_interval_updates": 10,
    "eval_updates": None,             # explicit list of updates to evaluate (overrides the interval)
    "eval_limit": 100,
    "final_eval_limit": 500,
    "eval_workers": 12,
    "parity_states": 256,
    # runtime
    "torch_threads": 4,
    "watchdog_seconds": 120,
    "expected_act_spec": None,        # hard stop unless the bridge + every validation report run this act spec
    "run_dir": "runs",
    "run_name": None,
}

# Algorithm-health thresholds (Phase 2D.1 §19). Crossing one stops the run
# with status "stopped: instability" — they are alarms, not tuning knobs.
HEALTH = {
    "min_normalised_entropy": 0.05,   # rollout-mean entropy / log(#legal); PPO 2C.3 ended at 0.44–0.47
    # Value-loss explosion = rising far above the run's own best, not merely
    # slow to fall (A2C takes one optimiser step per rollout, so the critic
    # starts at ~3 and descends slowly): value loss > max(floor, factor × min so far).
    "value_loss_explosion_factor": 10.0,
    "value_loss_explosion_floor": 1.0,
    "value_loss_warmup_updates": 20,
    "max_action_share": 0.95,         # one action index taking ≥ 95 % of a rollout's decisions
    "max_logp_drift": 1e-4,           # recomputed vs sampling log-prob (on-policy check)
}


def layer_init(layer, std=math.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class A2CAgent(nn.Module):
    """Separate actor (deployable) and critic; no shared layers."""

    def __init__(self):
        super().__init__()
        self.actor = PolicyNet("a2c")
        for m in self.actor.body:
            if isinstance(m, nn.Linear):
                layer_init(m)
        layer_init(self.actor.head, std=0.01)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(OBS_SIZE, 128)), nn.Tanh(),
            layer_init(nn.Linear(128, 128)), nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    def evaluate(self, obs, mask, action):
        """log π(a|s), H[π(·|s)] over legal actions, V(s)."""
        dist = masked_distribution(self.actor(obs), mask)
        return dist.log_prob(action), dist.entropy(), self.value(obs)


def compute_gae(rewards, values, dones, next_value, gamma, lam):
    """GAE over a [T, N] rollout; dones[t] = 1 when step t ended its episode
    (termination only — there is no truncation in an auction).
    Returns (advantages, returns = advantages + values)."""
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(next_value)
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        nv = next_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * nv * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv, adv + values


def a2c_loss(agent, obs, mask, act, adv, ret, cfg):
    """The A2C objective on one batch. Returns (loss, parts dict of tensors)."""
    logp, entropy, value = agent.evaluate(obs, mask, act)
    a = (adv - adv.mean()) / (adv.std() + 1e-8) if cfg["norm_adv"] else adv
    pg_loss = -(a.detach() * logp).mean()
    v_loss = 0.5 * ((value - ret) ** 2).mean()
    ent = entropy.mean()
    loss = pg_loss - cfg["ent_coef"] * ent + cfg["vf_coef"] * v_loss
    return loss, {"pg_loss": pg_loss, "v_loss": v_loss, "entropy": entropy, "logp": logp, "value": value}


def a2c_update(agent, optimizer, obs, mask, act, adv, ret, cfg, sample_logp=None):
    """Exactly one optimiser step on the whole rollout. Returns training stats."""
    loss, p = a2c_loss(agent, obs, mask, act, adv, ret, cfg)
    optimizer.zero_grad()
    loss.backward()
    grads_finite = all(torch.isfinite(q.grad).all() for q in agent.parameters() if q.grad is not None)
    grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), cfg["max_grad_norm"]).item()
    optimizer.step()
    with torch.no_grad():
        # Diagnostics only (not part of the objective): how far the single
        # step moved the policy on this batch.
        new_logp, _, _ = agent.evaluate(obs, mask, act)
        logratio = new_logp - p["logp"].detach()
        post_kl = ((logratio.exp() - 1) - logratio).mean().item()
    stats = {
        "loss": loss.item(), "policy_loss": p["pg_loss"].item(), "value_loss": p["v_loss"].item(),
        "entropy": p["entropy"].mean().item(),
        "entropy_normalised": normalised_entropy(p["entropy"].detach(), mask).mean().item(),
        "grad_norm": grad_norm, "grads_finite": bool(grads_finite),
        "adv_mean": adv.mean().item(), "adv_std": adv.std().item(), "adv_absmax": adv.abs().max().item(),
        "value_mean": p["value"].mean().item(), "return_mean": ret.mean().item(),
        "post_update_kl": post_kl,
        "params_finite": all(torch.isfinite(q).all() for q in agent.parameters()),
    }
    if sample_logp is not None:
        stats["logp_drift"] = (p["logp"].detach() - sample_logp).abs().max().item()
    return stats


def init_seed(seed):
    """The parameter-initialisation seed of A2C run `seed`: derived from
    ("a2c", seed) so A2C never draws the same initial weights as the PPO run
    with the same seed (the auction schedule and per-environment sampling
    streams still follow `seed`, as for every algorithm)."""
    return int.from_bytes(hashlib.sha256(f"a2c/init/{int(seed)}".encode()).digest()[:4], "little")


def explained_variance(pred, target):
    var = np.var(target)
    return float("nan") if var == 0 else float(1 - np.var(target - pred) / var)


def params_digest(module):
    """Order-stable fingerprint of all parameters (reproducibility checks)."""
    h = hashlib.sha256()
    for name, p in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def health_problems(update, stats, acts, value_loss_min):
    problems = []
    finite = [stats["loss"], stats["policy_loss"], stats["value_loss"], stats["entropy"], stats["grad_norm"]]
    if not all(math.isfinite(x) for x in finite) or not stats["grads_finite"] or not stats["params_finite"]:
        problems.append(f"non-finite loss/gradient/parameters: {({k: stats[k] for k in ('loss', 'policy_loss', 'value_loss', 'entropy', 'grad_norm', 'grads_finite', 'params_finite')})}")
    if stats["entropy_normalised"] < HEALTH["min_normalised_entropy"]:
        problems.append(f"entropy collapse: normalised entropy {stats['entropy_normalised']:.4f}")
    limit = max(HEALTH["value_loss_explosion_floor"], HEALTH["value_loss_explosion_factor"] * value_loss_min)
    if update > HEALTH["value_loss_warmup_updates"] and stats["value_loss"] > limit:
        problems.append(f"value-loss explosion: {stats['value_loss']:.4f} > {limit:.4f} (best so far {value_loss_min:.4f})")
    if max(acts["action_shares"]) >= HEALTH["max_action_share"]:
        problems.append(f"action concentration: one action takes {max(acts['action_shares']):.3f} of decisions")
    if stats.get("logp_drift", 0.0) > HEALTH["max_logp_drift"]:
        problems.append(f"update not on-policy: log-prob drift {stats['logp_drift']:.2e}")
    return problems


class SafetyError(RuntimeError):
    def __init__(self, problems):
        super().__init__("; ".join(map(str, problems[:5])))
        self.problems = problems


class InstabilityError(RuntimeError):
    def __init__(self, problems):
        super().__init__("; ".join(map(str, problems[:5])))
        self.problems = problems


def train(cfg):
    cfg_hash = config_hash(cfg)
    if cfg["algorithm"] != "a2c":
        raise ValueError("this trainer is A2C only")
    stage_b.check_stage(cfg, "Phase 2D.1 trains against Stage A only (no RL snapshots)")
    batch = cfg["num_envs"] * cfg["num_steps"]
    num_updates = cfg["total_decisions"] // batch
    run_name = cfg["run_name"] or f"a2c-s{cfg['seed']}-{cfg_hash[:8]}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = (ML_ROOT / cfg["run_dir"] / run_name).resolve()

    seed_everything(cfg["seed"], cfg["torch_threads"])
    envs = VecIplAuctionEnv(cfg["num_envs"], cfg["seed"], split=cfg["split"], tremble=cfg["tremble"],
                            snapshot_share=cfg["snapshot_share"], watchdog_seconds=cfg["watchdog_seconds"], **stage_b.env_kwargs(cfg))
    expected = cfg["expected_act_spec"]
    if expected and envs.act_spec["hash"] != expected:
        envs.close()
        raise ValueError(f"action-spec mismatch: bridge {envs.act_spec}, expected {expected}")
    if envs.gamma != cfg["gamma"]:
        envs.close()
        raise ValueError(f"config gamma {cfg['gamma']} ≠ environment gamma {envs.gamma} (frozen)")

    log = RunLogger(run_dir)
    meta = run_metadata(cfg, cfg_hash, envs.obs_spec, envs.act_spec, envs.gamma,
                        extra={"runName": run_name, "numUpdates": num_updates, "decisionsPerUpdate": batch,
                               "plannedDecisions": num_updates * batch, "gradientStepsPerUpdate": 1, "health": HEALTH})
    log.write_json("metadata.json", meta)
    print(f"[a2c] {run_name}: {num_updates} updates × {batch} decisions = {num_updates * batch:,}  config {cfg_hash}  obs {envs.obs_spec['hash']}  act {envs.act_spec['hash']}", flush=True)

    torch.manual_seed(init_seed(cfg["seed"]))
    agent = A2CAgent()
    stage_b.warm_start(cfg, {"agent": agent}, run_dir)
    initial_digest = params_digest(agent)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg["learning_rate"], eps=cfg["adam_eps"])

    N, T = cfg["num_envs"], cfg["num_steps"]
    obs_buf = torch.zeros((T, N, OBS_SIZE))
    mask_buf = torch.zeros((T, N, ACTION_COUNT), dtype=torch.bool)
    act_buf = torch.zeros((T, N), dtype=torch.long)
    logp_buf = torch.zeros((T, N))
    rew_buf = torch.zeros((T, N))
    done_buf = torch.zeros((T, N))
    val_buf = torch.zeros((T, N))
    buffers = {"obs": obs_buf, "mask": mask_buf, "act": act_buf, "logp": logp_buf, "val": val_buf, "rew": rew_buf, "done": done_buf}

    episodes = EpisodeLog()
    all_episodes = []
    evaluations = []
    t_start = time.perf_counter()
    env_seconds = 0.0
    rollout_state = AsyncRolloutState(envs, cfg["seed"])
    global_decisions = 0
    value_loss_min = math.inf
    status = "completed"
    failure = None

    def evaluate_checkpoint(update, final=False):
        tag = f"update_{update:04d}"
        ckpt_dir = run_dir / "checkpoints" / tag
        save_checkpoint(ckpt_dir / "checkpoint.pt", modules={"agent": agent}, optimizer=optimizer,
                        step_state={"update": update, "decisions": global_decisions, "episodeIndex": envs.episode_index.tolist()},
                        cfg=cfg, cfg_hash=cfg_hash)
        policy = export_policy(ckpt_dir / "policy.json", agent.actor, envs.obs_spec, envs.act_spec,
                               trained_steps=global_decisions, seed=cfg["seed"], cfg=cfg, cfg_hash=cfg_hash,
                               extra_meta={"update": update, "stage": cfg["stage"], "runName": run_name})
        flat_obs = obs_buf.reshape(-1, OBS_SIZE)
        flat_mask = mask_buf.reshape(-1, ACTION_COUNT)
        idx = torch.linspace(0, flat_obs.shape[0] - 1, min(cfg["parity_states"], flat_obs.shape[0])).long()
        parity = export_parity(agent.actor, policy, flat_obs[idx].numpy(), flat_mask[idx].numpy())
        t0 = time.perf_counter()
        limit = cfg["final_eval_limit"] if final else cfg["eval_limit"]
        report, eps = node_evaluate(ckpt_dir / "policy.json", ckpt_dir / "validation", limit=limit, workers=cfg["eval_workers"])
        problems = safety_problems(report, eps)
        if expected and report.get("actSpecHash") != expected:
            problems.append(f"action-spec mismatch in validation: {report.get('actSpecHash')} != {expected}")
        name = "policy:a2c"
        head = headline(report, name)
        row = {"update": update, "decisions": global_decisions, "final": final, "validationEpisodes": report["n"],
               "seconds": time.perf_counter() - t0, "parity": parity, "safetyProblems": problems, "metrics": head,
               "actionShares": report["report"][name]["actionShares"]}
        evaluations.append(row)
        log.scalars("validation", head, global_decisions)
        log.scalars("parity", parity, global_decisions)
        log.record({"type": "evaluation", **row})
        print(f"[a2c] eval @ {global_decisions:,}: validation n={report['n']} XI {head['xi']:.2f} (Δ vs moneyball {head.get('xiDiffVsMoneyball', float('nan')):+.2f})  "
              f"legal {100 * head['legalXI']:.0f}%  purse left {head['purseLeftShare']:.2f}  price/fair {head['priceToFair']:.2f}  "
              f"parity {parity['maxAbsScoreDiff']:.1e}/{parity['argmaxAgreement']:.3f}  safety {'OK' if not problems else problems[:3]}", flush=True)
        if problems:
            raise SafetyError(problems)
        if parity["maxAbsScoreDiff"] > 1e-4 or parity["argmaxAgreement"] < 1.0:
            raise SafetyError([f"export parity failed: {parity}"])

    try:
        for update in range(1, num_updates + 1):
            t_roll = time.perf_counter()
            env_time = collect_async(rollout_state, agent.actor, agent.critic, T, buffers, episodes.add)
            assert_legal(act_buf.reshape(-1), mask_buf.reshape(-1, ACTION_COUNT))
            obs = torch.from_numpy(rollout_state.obs.copy())
            global_decisions += batch
            episodes.pending.sort(key=lambda e: e["envIndex"])
            rollout_seconds = time.perf_counter() - t_roll
            env_seconds += env_time
            if episodes.violations:
                raise SafetyError([f"training episode invariant violation: {episodes.violations[:3]}"])
            incomplete = [e["seed"] for e in episodes.pending if not e["legalXI"]]
            if incomplete and stage_b.incomplete_is_stop(cfg):
                raise SafetyError([f"training episode ended without a legal XI: seeds {incomplete[:5]}"])

            with torch.no_grad():
                adv, returns = compute_gae(rew_buf, val_buf, done_buf, agent.value(obs), cfg["gamma"], cfg["gae_lambda"])

            b_obs, b_mask = obs_buf.reshape(-1, OBS_SIZE), mask_buf.reshape(-1, ACTION_COUNT)
            b_act, b_logp = act_buf.reshape(-1), logp_buf.reshape(-1)
            b_adv, b_ret, b_val = adv.reshape(-1), returns.reshape(-1), val_buf.reshape(-1)

            t_upd = time.perf_counter()
            train_stats = a2c_update(agent, optimizer, b_obs, b_mask, b_act, b_adv, b_ret, cfg, sample_logp=b_logp)
            update_seconds = time.perf_counter() - t_upd
            train_stats["explained_variance"] = explained_variance(b_val.numpy(), b_ret.numpy())
            train_stats["learning_rate"] = optimizer.param_groups[0]["lr"]

            finished = episodes.drain()
            all_episodes.extend(finished)
            ep_stats = summarise_episodes(finished)
            acts = action_stats(b_act.numpy(), b_mask.numpy(), ACTION_COUNT)
            perf = {
                "rollout_decisions_per_sec": batch / rollout_seconds, "env_decisions_per_sec": batch / env_time,
                "update_seconds": update_seconds, "rollout_seconds": rollout_seconds,
                "decisions_per_sec_overall": global_decisions / (time.perf_counter() - t_start),
            }
            actions_digest = hashlib.sha256(act_buf.numpy().tobytes()).hexdigest()[:16]
            row = {"type": "update", "update": update, "decisions": global_decisions, "episodes": episodes.total, "actionsDigest": actions_digest,
                   "train": train_stats, "episode": ep_stats, "actions": acts, "perf": perf}
            log.record(row)
            log.scalars("train", {k: v for k, v in train_stats.items() if not isinstance(v, bool)}, global_decisions)
            log.scalars("episode", ep_stats, global_decisions)
            log.scalars("actions", {k: v for k, v in acts.items() if k != "action_shares"}, global_decisions)
            log.histogram_shares("action_share", acts["action_shares"], envs.action_names, global_decisions)
            log.scalars("perf", perf, global_decisions)
            print(f"[a2c] upd {update:3d}/{num_updates}  dec {global_decisions:>7,}  eps {episodes.total:4d}  "
                  f"ret {ep_stats.get('return', float('nan')):6.3f}  XI {ep_stats.get('xi', float('nan')):6.2f}  "
                  f"legal {ep_stats.get('legalXI', float('nan')):.2f}  ent {train_stats['entropy']:.3f} ({train_stats['entropy_normalised']:.3f})  "
                  f"vl {train_stats['value_loss']:.4f}  gn {train_stats['grad_norm']:.2f}  kl {train_stats['post_update_kl']:.5f}  "
                  f"EV {train_stats['explained_variance']:.3f}  bid {acts['bid_share']:.3f}  {perf['rollout_decisions_per_sec']:.0f} dec/s", flush=True)

            value_loss_min = min(value_loss_min, train_stats["value_loss"])
            unhealthy = health_problems(update, train_stats, acts, value_loss_min)
            if unhealthy:
                raise InstabilityError(unhealthy)

            due = update in cfg["eval_updates"] if cfg["eval_updates"] else update % cfg["eval_interval_updates"] == 0
            if due or update == num_updates:
                evaluate_checkpoint(update, final=(update == num_updates))
    except SafetyError as err:
        status, failure = "stopped: safety", err.problems
        print(f"[a2c] STOPPED — safety invariant failed: {err.problems[:5]}", flush=True)
    except InstabilityError as err:
        status, failure = "stopped: instability", err.problems
        print(f"[a2c] STOPPED — algorithm instability: {err.problems[:5]}", flush=True)
    except Exception as err:  # crash / deadlock / masked action → stop and record
        status, failure = f"stopped: {type(err).__name__}", [str(err)]
        print(f"[a2c] STOPPED — {type(err).__name__}: {err}", flush=True)
        raise
    finally:
        wall = time.perf_counter() - t_start
        summary = {
            "runName": run_name, "algorithm": "a2c", "status": status, "failure": failure, "configHash": cfg_hash,
            "seed": cfg["seed"], "obsHash": envs.obs_spec["hash"], "actHash": envs.act_spec["hash"],
            "numEnvs": N, "totalDecisions": global_decisions, "episodes": episodes.total,
            "wallSeconds": wall, "envSeconds": env_seconds,
            "decisionsPerSecOverall": global_decisions / wall if wall else None,
            "initialParamsDigest": initial_digest,
            "finalParamsDigest": params_digest(agent),
            "actSpec": envs.act_spec,
            "rolloutMode": "async",
            "episodeConsistencyChecks": rollout_state.consistency_checks,
            "illegalActions": 0 if status == "completed" else None,
            "incompleteXiTrainingEpisodes": sum(1 for e in all_episodes if not e["legalXI"]),
            "simulatorEpisodesPerSec": envs.episodes / (time.perf_counter() - t_start),
            "trainingEpisodes": {"all": summarise_episodes(all_episodes),
                                 "first100": summarise_episodes(all_episodes[:100]),
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
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=JSON", help="override config keys, e.g. seed=2 num_envs=4")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    disable_throttling()  # Windows EcoQoS: speed only, results unchanged
    overrides = {k: json.loads(v) for k, v in (s.split("=", 1) for s in args.set)}
    cfg = load_config(args.config, DEFAULTS, overrides)
    _, summary = train(cfg)
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
