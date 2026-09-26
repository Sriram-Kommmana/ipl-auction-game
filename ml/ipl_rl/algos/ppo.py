"""Masked PPO for IplAuctionEnv-v2 (Phase 2C) — CleanRL-style, one file.

    cd ml
    .venv/Scripts/python -m ipl_rl.algos.ppo --config ipl_rl/configs/ppo_pilot.json

Actor: the deployable PolicyNet("ppo") (80 → 128 → 128 → 20 logits, tanh) —
exactly what rl-policy-v2 exports. Critic: a separate 80 → 128 → 128 → 1
tanh MLP (training only, never exported). Orthogonal init (√2 hidden, 0.01
policy head, 1.0 value head), as in CleanRL.

Per update: num_envs × num_steps decisions with masked sampling; GAE(γ, λ)
with termination-only episodes (γ = 1 from the environment); update_epochs
passes over shuffled minibatches of the clipped surrogate objective, value
loss and entropy bonus; one Adam optimiser over both networks; global
gradient-norm clipping.

Masking: illegal actions get logit −1e8 in sampling, in the stored log-prob
and in the update, so the policy is a distribution over legal actions only.

Every eval_interval updates (and at the end): checkpoint, rl-policy-v2
export, export/inference parity, Node validation evaluation paired against
the locked baselines, and the safety invariants. Any invariant failure, a
masked action, a crash or a deadlock stops training.
"""

import argparse
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
from ..common.seeding import seed_everything
from ..common.rollout import AsyncRolloutState, collect_async
from ..common.stats import EpisodeLog, action_stats, summarise_episodes
from ..common.vec_env import VecIplAuctionEnv
from ..common.win_qos import disable_throttling
from ..env import ACTION_COUNT, OBS_SIZE
from ..nets import PolicyNet

ML_ROOT = Path(__file__).resolve().parents[2]

DEFAULTS = {
    "algorithm": "ppo",
    "seed": 1,
    # budget
    "total_decisions": 250_000,       # upper bound; rounded DOWN to whole updates
    "num_envs": 12,
    "num_steps": 512,                 # rollout length per environment
    # PPO
    "minibatch_size": 256,
    "update_epochs": 4,
    "learning_rate": 3e-4,
    "anneal_lr": False,
    "gamma": 1.0,                     # must equal the environment's (frozen) γ
    "gae_lambda": 0.95,
    "clip_coef": 0.2,
    "clip_vloss": False,
    "norm_adv": True,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": None,
    "adam_eps": 1e-5,
    # environment / curriculum
    "split": "train",
    "stage": "A",                     # A = frozen rule bots + human proxy, no RL snapshots
    "tremble": 0.01,
    "snapshot_share": 0.0,
    # evaluation / checkpoints
    "eval_interval_updates": 10,
    "eval_updates": None,             # explicit list of updates to evaluate (overrides the interval)
    "eval_limit": 100,                # validation episodes at intermediate evaluations
    "final_eval_limit": 500,          # the full validation manifest at the end
    "eval_workers": 12,
    "parity_states": 256,
    # runtime
    "torch_threads": 4,
    "watchdog_seconds": 120,
    # "async": each environment runs at its own pace (common/rollout.py);
    # "sync": lock-step vector steps (every step waits for the slowest env).
    "rollout_mode": "async",
    "run_dir": "runs",
    "run_name": None,
}


def layer_init(layer, std=math.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = PolicyNet("ppo")
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

    def act(self, obs, mask, action=None):
        dist = masked_distribution(self.actor(obs), mask)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.value(obs)


def explained_variance(pred, target):
    var = np.var(target)
    return float("nan") if var == 0 else float(1 - np.var(target - pred) / var)


def params_digest(module):
    """Order-stable fingerprint of all parameters (reproducibility checks)."""
    import hashlib
    h = hashlib.sha256()
    for name, p in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def train(cfg):
    cfg_hash = config_hash(cfg)
    if cfg["stage"] != "A" or cfg["snapshot_share"] != 0.0:
        raise ValueError("Phase 2C.0 trains against Stage A only (no RL snapshots)")
    batch = cfg["num_envs"] * cfg["num_steps"]
    if batch % cfg["minibatch_size"]:
        raise ValueError("num_envs × num_steps must be a multiple of minibatch_size")
    num_updates = cfg["total_decisions"] // batch
    run_name = cfg["run_name"] or f"ppo-s{cfg['seed']}-{cfg_hash[:8]}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = (ML_ROOT / cfg["run_dir"] / run_name).resolve()

    seed_everything(cfg["seed"], cfg["torch_threads"])
    envs = VecIplAuctionEnv(cfg["num_envs"], cfg["seed"], split=cfg["split"], tremble=cfg["tremble"],
                            snapshot_share=cfg["snapshot_share"], watchdog_seconds=cfg["watchdog_seconds"])
    if envs.gamma != cfg["gamma"]:
        envs.close()
        raise ValueError(f"config gamma {cfg['gamma']} ≠ environment gamma {envs.gamma} (frozen)")

    log = RunLogger(run_dir)
    meta = run_metadata(cfg, cfg_hash, envs.obs_spec, envs.act_spec, envs.gamma,
                        extra={"runName": run_name, "numUpdates": num_updates, "decisionsPerUpdate": batch, "plannedDecisions": num_updates * batch})
    log.write_json("metadata.json", meta)
    print(f"[ppo] {run_name}: {num_updates} updates × {batch} decisions = {num_updates * batch:,}  config {cfg_hash}  obs {envs.obs_spec['hash']}  act {envs.act_spec['hash']}", flush=True)

    agent = Agent()
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg["learning_rate"], eps=cfg["adam_eps"])
    shuffle_gen = torch.Generator().manual_seed(cfg["seed"])

    N, T = cfg["num_envs"], cfg["num_steps"]
    obs_buf = torch.zeros((T, N, OBS_SIZE))
    mask_buf = torch.zeros((T, N, ACTION_COUNT), dtype=torch.bool)
    act_buf = torch.zeros((T, N), dtype=torch.long)
    logp_buf = torch.zeros((T, N))
    rew_buf = torch.zeros((T, N))
    done_buf = torch.zeros((T, N))
    val_buf = torch.zeros((T, N))

    episodes = EpisodeLog()
    all_episodes = []
    evaluations = []
    history = []
    t_start = time.perf_counter()
    env_seconds = 0.0
    if cfg["rollout_mode"] == "async":
        rollout_state = AsyncRolloutState(envs, cfg["seed"])
        buffers = {"obs": obs_buf, "mask": mask_buf, "act": act_buf, "logp": logp_buf, "val": val_buf, "rew": rew_buf, "done": done_buf}
    elif cfg["rollout_mode"] == "sync":
        obs_np, mask_np = envs.reset()
        obs, mask = torch.from_numpy(obs_np), torch.from_numpy(mask_np)
    else:
        raise ValueError(f"rollout_mode {cfg['rollout_mode']}")
    global_decisions = 0
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
        # Parity on real states from the latest rollout.
        flat_obs = obs_buf.reshape(-1, OBS_SIZE)
        flat_mask = mask_buf.reshape(-1, ACTION_COUNT)
        idx = torch.linspace(0, flat_obs.shape[0] - 1, min(cfg["parity_states"], flat_obs.shape[0])).long()
        parity = export_parity(agent.actor, policy, flat_obs[idx].numpy(), flat_mask[idx].numpy())
        t0 = time.perf_counter()
        limit = cfg["final_eval_limit"] if final else cfg["eval_limit"]
        report, eps = node_evaluate(ckpt_dir / "policy.json", ckpt_dir / "validation", limit=limit, workers=cfg["eval_workers"])
        problems = safety_problems(report, eps)
        name = "policy:ppo"
        head = headline(report, name)
        row = {"update": update, "decisions": global_decisions, "final": final, "validationEpisodes": report["n"],
               "seconds": time.perf_counter() - t0, "parity": parity, "safetyProblems": problems, "metrics": head,
               "actionShares": report["report"][name]["actionShares"]}
        evaluations.append(row)
        log.scalars("validation", head, global_decisions)
        log.scalars("parity", parity, global_decisions)
        log.record({"type": "evaluation", **row})
        print(f"[ppo] eval @ {global_decisions:,}: validation n={report['n']} XI {head['xi']:.2f} (Δ vs moneyball {head.get('xiDiffVsMoneyball', float('nan')):+.2f})  "
              f"legal {100 * head['legalXI']:.0f}%  purse left {head['purseLeftShare']:.2f}  price/fair {head['priceToFair']:.2f}  "
              f"parity {parity['maxAbsScoreDiff']:.1e}/{parity['argmaxAgreement']:.3f}  safety {'OK' if not problems else problems[:3]}", flush=True)
        if problems:
            raise SafetyError(problems)
        if parity["maxAbsScoreDiff"] > 1e-4 or parity["argmaxAgreement"] < 1.0:
            raise SafetyError([f"export parity failed: {parity}"])

    try:
        for update in range(1, num_updates + 1):
            if cfg["anneal_lr"]:
                optimizer.param_groups[0]["lr"] = cfg["learning_rate"] * (1 - (update - 1) / num_updates)
            t_roll = time.perf_counter()
            env_time = 0.0
            if cfg["rollout_mode"] == "async":
                env_time = collect_async(rollout_state, agent.actor, agent.critic, T, buffers, episodes.add)
                assert_legal(act_buf.reshape(-1), mask_buf.reshape(-1, ACTION_COUNT))
                obs = torch.from_numpy(rollout_state.obs.copy())
                global_decisions += batch
                # Deterministic episode order whatever simulator finished first.
                episodes.pending.sort(key=lambda e: e["envIndex"])
            for t in range(T if cfg["rollout_mode"] == "sync" else 0):
                with torch.no_grad():
                    action, logp, _, value = agent.act(obs, mask)
                assert_legal(action, mask)
                obs_buf[t], mask_buf[t], act_buf[t], logp_buf[t], val_buf[t] = obs, mask, action, logp, value
                t_env = time.perf_counter()
                obs_np, rew_np, done_np, mask_np, infos = envs.step(action.numpy())
                env_time += time.perf_counter() - t_env
                rew_buf[t] = torch.from_numpy(rew_np.astype(np.float32))
                done_buf[t] = torch.from_numpy(done_np.astype(np.float32))
                obs, mask = torch.from_numpy(obs_np), torch.from_numpy(mask_np)
                for i in np.flatnonzero(done_np):
                    episodes.add(infos[i]["episode"])
                global_decisions += N
            rollout_seconds = time.perf_counter() - t_roll
            env_seconds += env_time
            if episodes.violations:
                raise SafetyError([f"training episode invariant violation: {episodes.violations[:3]}"])
            incomplete = [e["seed"] for e in episodes.pending if not e["legalXI"]]
            if incomplete:
                raise SafetyError([f"training episode ended without a legal XI: seeds {incomplete[:5]}"])

            # GAE (termination only; γ from the environment).
            with torch.no_grad():
                next_value = agent.value(obs)
                adv = torch.zeros_like(rew_buf)
                last = torch.zeros(N)
                for t in reversed(range(T)):
                    nonterminal = 1.0 - done_buf[t]
                    nv = next_value if t == T - 1 else val_buf[t + 1]
                    delta = rew_buf[t] + cfg["gamma"] * nv * nonterminal - val_buf[t]
                    last = delta + cfg["gamma"] * cfg["gae_lambda"] * nonterminal * last
                    adv[t] = last
                returns = adv + val_buf

            b_obs, b_mask = obs_buf.reshape(-1, OBS_SIZE), mask_buf.reshape(-1, ACTION_COUNT)
            b_act, b_logp = act_buf.reshape(-1), logp_buf.reshape(-1)
            b_adv, b_ret, b_val = adv.reshape(-1), returns.reshape(-1), val_buf.reshape(-1)

            t_upd = time.perf_counter()
            clipfracs, grad_norms, kls = [], [], []
            pg_l = v_l = ent_l = ent_norm = 0.0
            stop_early = False
            for epoch in range(cfg["update_epochs"]):
                perm = torch.randperm(batch, generator=shuffle_gen)
                for start in range(0, batch, cfg["minibatch_size"]):
                    mb = perm[start:start + cfg["minibatch_size"]]
                    _, newlogp, entropy, newvalue = agent.act(b_obs[mb], b_mask[mb], b_act[mb])
                    logratio = newlogp - b_logp[mb]
                    ratio = logratio.exp()
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean().item()
                        kls.append(approx_kl)
                        clipfracs.append(((ratio - 1.0).abs() > cfg["clip_coef"]).float().mean().item())
                    mb_adv = b_adv[mb]
                    if cfg["norm_adv"]:
                        mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                    pg_loss = torch.max(-mb_adv * ratio, -mb_adv * ratio.clamp(1 - cfg["clip_coef"], 1 + cfg["clip_coef"])).mean()
                    if cfg["clip_vloss"]:
                        v_clipped = b_val[mb] + (newvalue - b_val[mb]).clamp(-cfg["clip_coef"], cfg["clip_coef"])
                        v_loss = 0.5 * torch.max((newvalue - b_ret[mb]) ** 2, (v_clipped - b_ret[mb]) ** 2).mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_ret[mb]) ** 2).mean()
                    ent = entropy.mean()
                    loss = pg_loss - cfg["ent_coef"] * ent + cfg["vf_coef"] * v_loss
                    optimizer.zero_grad()
                    loss.backward()
                    grad_norms.append(nn.utils.clip_grad_norm_(agent.parameters(), cfg["max_grad_norm"]).item())
                    optimizer.step()
                    pg_l, v_l, ent_l = pg_loss.item(), v_loss.item(), ent.item()
                    ent_norm = normalised_entropy(entropy.detach(), b_mask[mb]).mean().item()
                if cfg["target_kl"] is not None and np.mean(kls[-(batch // cfg["minibatch_size"]):]) > cfg["target_kl"]:
                    stop_early = True
                    break
            update_seconds = time.perf_counter() - t_upd

            finished = episodes.drain()
            all_episodes.extend(finished)
            ep_stats = summarise_episodes(finished)
            acts = action_stats(b_act.numpy(), b_mask.numpy(), ACTION_COUNT)
            train_stats = {
                "policy_loss": pg_l, "value_loss": v_l, "entropy": ent_l, "entropy_normalised": ent_norm,
                "approx_kl": float(np.mean(kls)), "clip_fraction": float(np.mean(clipfracs)),
                "grad_norm": float(np.mean(grad_norms)), "grad_norm_max": float(np.max(grad_norms)),
                "explained_variance": explained_variance(b_val.numpy(), b_ret.numpy()),
                "learning_rate": optimizer.param_groups[0]["lr"], "early_stop": stop_early,
            }
            perf = {
                "rollout_decisions_per_sec": batch / rollout_seconds, "env_decisions_per_sec": batch / env_time,
                "update_seconds": update_seconds, "rollout_seconds": rollout_seconds,
                "decisions_per_sec_overall": global_decisions / (time.perf_counter() - t_start),
            }
            row = {"type": "update", "update": update, "decisions": global_decisions, "episodes": episodes.total,
                   "train": train_stats, "episode": ep_stats, "actions": acts, "perf": perf}
            history.append(row)
            log.record(row)
            log.scalars("train", {k: v for k, v in train_stats.items() if not isinstance(v, bool)}, global_decisions)
            log.scalars("episode", ep_stats, global_decisions)
            log.scalars("actions", {k: v for k, v in acts.items() if k != "action_shares"}, global_decisions)
            log.histogram_shares("action_share", acts["action_shares"], envs.action_names, global_decisions)
            log.scalars("perf", perf, global_decisions)
            print(f"[ppo] upd {update:3d}/{num_updates}  dec {global_decisions:>7,}  eps {episodes.total:4d}  "
                  f"ret {ep_stats.get('return', float('nan')):6.3f}  XI {ep_stats.get('xi', float('nan')):6.2f}  "
                  f"legal {ep_stats.get('legalXI', float('nan')):.2f}  ent {ent_l:.3f}  kl {train_stats['approx_kl']:.4f}  "
                  f"clip {train_stats['clip_fraction']:.3f}  EV {train_stats['explained_variance']:.3f}  "
                  f"bid {acts['bid_share']:.3f}  {perf['rollout_decisions_per_sec']:.0f} dec/s", flush=True)

            due = update in cfg["eval_updates"] if cfg["eval_updates"] else update % cfg["eval_interval_updates"] == 0
            if due or update == num_updates:
                evaluate_checkpoint(update, final=(update == num_updates))
    except SafetyError as err:
        status, failure = "stopped: safety", err.problems
        print(f"[ppo] STOPPED — safety invariant failed: {err.problems[:5]}", flush=True)
    except Exception as err:  # crash / deadlock / masked action → stop and record
        status, failure = f"stopped: {type(err).__name__}", [str(err)]
        print(f"[ppo] STOPPED — {type(err).__name__}: {err}", flush=True)
        raise
    finally:
        wall = time.perf_counter() - t_start
        summary = {
            "runName": run_name, "status": status, "failure": failure, "configHash": cfg_hash,
            "seed": cfg["seed"], "obsHash": envs.obs_spec["hash"], "actHash": envs.act_spec["hash"],
            "numEnvs": N, "totalDecisions": global_decisions, "episodes": episodes.total,
            "wallSeconds": wall, "envSeconds": env_seconds,
            "decisionsPerSecOverall": global_decisions / wall if wall else None,
            "finalParamsDigest": params_digest(agent),
            "rolloutMode": cfg["rollout_mode"],
            # Every finished episode's summed rewards / steps / action counts
            # matched the JavaScript summary (async collector only).
            "episodeConsistencyChecks": rollout_state.consistency_checks if cfg["rollout_mode"] == "async" else None,
            # A masked action raises before it is sent, so a completed run has none.
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


class SafetyError(RuntimeError):
    def __init__(self, problems):
        super().__init__("; ".join(map(str, problems[:5])))
        self.problems = problems


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
