"""SB3-contrib MaskablePPO cross-check (Phase 2C.0, Step 9). NOT a production model.

    cd ml
    .venv/Scripts/python -m ipl_rl.crosscheck.sb3_maskable_ppo --config ipl_rl/configs/ppo_pilot.json

A trusted reference implementation run on exactly what our PPO sees:
  - the same environment: VecIplAuctionEnv (Node bridge, obs-v2, act-v2, the
    canonical mask, the frozen reward, γ = 1), wrapped as an SB3 VecEnv;
  - the same auctions: the same run seed gives the same train-seed schedule
    (episode k of environment i);
  - the same hyperparameters and network shapes (pi 128-128 tanh, vf 128-128
    tanh, orthogonal init, Adam eps 1e-5, normalised advantages, no value
    clipping, constant learning rate).

Logged like our PPO (JSON lines + TensorBoard): episode return / XI / legal
XI / purse per rollout, entropy, losses, action distribution, illegal-action
count. At the end the SB3 actor is copied into PolicyNet("ppo") (identical
shapes), exported to rl-policy-v2 and evaluated by the same Node evaluator
on the validation manifest — so both implementations are compared with the
same metrics.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from torch import nn

from ..algos.ppo import DEFAULTS, ML_ROOT
from ..common.checkpoint import export_policy
from ..common.config import config_hash, load_config
from ..common.evaluation import export_parity, headline, node_evaluate, safety_problems
from ..common.logger import RunLogger
from ..common.metadata import run_metadata
from ..common.seeding import seed_everything
from ..common.stats import EpisodeLog, action_stats, summarise_episodes
from ..common.vec_env import VecIplAuctionEnv
from ..common.win_qos import disable_throttling
from ..env import ACTION_COUNT, OBS_SIZE
from ..nets import PolicyNet


class Sb3VecAdapter(VecEnv):
    """VecIplAuctionEnv behind SB3's VecEnv interface (auto-reset semantics match)."""

    def __init__(self, venv):
        self.venv = venv
        super().__init__(venv.num_envs, spaces.Box(-1.0, 5.0, (OBS_SIZE,), np.float32), spaces.Discrete(ACTION_COUNT))
        self._actions = None
        self._masks = None
        self.illegal_actions = 0

    def reset(self):
        obs, self._masks = self.venv.reset()
        return obs

    def step_async(self, actions):
        self._actions = np.asarray(actions).reshape(-1)

    def step_wait(self):
        picked = self._masks[np.arange(self.num_envs), self._actions]
        self.illegal_actions += int((~picked).sum())  # the env would also raise
        obs, rewards, dones, self._masks, infos = self.venv.step(self._actions)
        out_infos = []
        for i, info in enumerate(infos):
            info = dict(info)
            if dones[i]:
                ep = info.pop("episode")
                info["ipl_episode"] = ep
                info["episode"] = {"r": ep["return"], "l": ep["decisions"], "t": 0.0}  # SB3 Monitor format
                info["terminal_observation"] = np.zeros(OBS_SIZE, dtype=np.float32)  # never bootstrapped (termination)
            out_infos.append(info)
        return obs, rewards.astype(np.float32), dones, out_infos

    def action_masks(self):
        return self._masks.copy()

    def env_method(self, method_name, *args, indices=None, **kwargs):
        if method_name != "action_masks":
            raise NotImplementedError(method_name)
        idx = self._get_indices(indices)
        return [self._masks[i].copy() for i in idx]

    def has_attr(self, attr_name):
        return attr_name == "action_masks" or hasattr(self.venv, attr_name)

    def get_attr(self, attr_name, indices=None):
        idx = self._get_indices(indices)
        if attr_name == "render_mode":
            return [None for _ in idx]
        return [getattr(self.venv, attr_name) for _ in idx]

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError(attr_name)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]

    def seed(self, seed=None):
        return [None] * self.num_envs  # the episode schedule is fixed by run_seed

    def close(self):
        self.venv.close()


class CrossCheckLogger(BaseCallback):
    def __init__(self, log, action_names):
        super().__init__()
        self.log = log
        self.action_names = action_names
        self.episodes = EpisodeLog()
        self.all_episodes = []
        self.rollout_actions, self.rollout_masks = [], []
        self.rows = []
        self.iteration = 0
        self.t_roll = None

    def _on_rollout_start(self):
        self.t_roll = time.perf_counter()
        self.rollout_actions, self.rollout_masks = [], []

    def _on_step(self):
        self.rollout_actions.append(np.asarray(self.locals["actions"]).reshape(-1).copy())
        self.rollout_masks.append(np.asarray(self.locals["action_masks"]).copy())
        for info in self.locals["infos"]:
            if "ipl_episode" in info:
                self.episodes.add(info["ipl_episode"])
        return not self.episodes.violations

    def _on_rollout_end(self):
        self.iteration += 1
        finished = self.episodes.drain()
        self.all_episodes.extend(finished)
        acts = action_stats(np.concatenate(self.rollout_actions), np.concatenate(self.rollout_masks), ACTION_COUNT)
        # Train metrics of the PREVIOUS update (SB3 records them after this rollout's predecessor).
        train = {k.split("/", 1)[1]: v for k, v in self.model.logger.name_to_value.items() if k.startswith("train/")}
        row = {"type": "rollout", "iteration": self.iteration, "decisions": int(self.num_timesteps),
               "episodes": self.episodes.total, "episode": summarise_episodes(finished), "actions": acts,
               "train_prev": train, "rollout_seconds": time.perf_counter() - self.t_roll}
        self.rows.append(row)
        self.log.record(row)
        self.log.scalars("episode", row["episode"], self.num_timesteps)
        self.log.scalars("actions", {k: v for k, v in acts.items() if k != "action_shares"}, self.num_timesteps)
        self.log.histogram_shares("action_share", acts["action_shares"], self.action_names, self.num_timesteps)
        ep = row["episode"]
        print(f"[sb3] it {self.iteration:3d}  dec {self.num_timesteps:>7,}  eps {self.episodes.total:4d}  "
              f"ret {ep.get('return', float('nan')):6.3f}  XI {ep.get('xi', float('nan')):6.2f}  legal {ep.get('legalXI', float('nan')):.2f}  "
              f"ent(prev) {-train.get('entropy_loss', float('nan')):.3f}  bid {acts['bid_share']:.3f}  "
              f"{len(self.rollout_actions) * self.training_env.num_envs / row['rollout_seconds']:.0f} dec/s", flush=True)


def sb3_actor_to_policynet(model):
    """Copy SB3's actor (policy_net + action_net) into the deployable PolicyNet('ppo')."""
    net = PolicyNet("ppo")
    pn = [m for m in model.policy.mlp_extractor.policy_net if isinstance(m, nn.Linear)]
    body = [m for m in net.body if isinstance(m, nn.Linear)]
    assert [tuple(m.weight.shape) for m in pn] == [tuple(m.weight.shape) for m in body]
    with torch.no_grad():
        for src, dst in zip(pn, body):
            dst.weight.copy_(src.weight)
            dst.bias.copy_(src.bias)
        net.head.weight.copy_(model.policy.action_net.weight)
        net.head.bias.copy_(model.policy.action_net.bias)
    return net


def run(cfg):
    cfg_hash = config_hash(cfg)
    batch = cfg["num_envs"] * cfg["num_steps"]
    num_updates = cfg["total_decisions"] // batch
    total = num_updates * batch
    run_name = cfg["run_name"] or f"sb3-maskableppo-s{cfg['seed']}-{cfg_hash[:8]}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = (ML_ROOT / cfg["run_dir"] / run_name).resolve()
    seed_everything(cfg["seed"], cfg["torch_threads"])
    venv = VecIplAuctionEnv(cfg["num_envs"], cfg["seed"], split=cfg["split"], tremble=cfg["tremble"],
                            snapshot_share=cfg["snapshot_share"], watchdog_seconds=cfg["watchdog_seconds"])
    env = Sb3VecAdapter(venv)
    log = RunLogger(run_dir)
    import sb3_contrib
    import stable_baselines3
    log.write_json("metadata.json", run_metadata(cfg, cfg_hash, venv.obs_spec, venv.act_spec, venv.gamma, extra={
        "runName": run_name, "reference": "sb3-contrib MaskablePPO (cross-check only, not a production model)",
        "sb3": stable_baselines3.__version__, "sb3Contrib": sb3_contrib.__version__, "plannedDecisions": total}))
    print(f"[sb3] {run_name}: {num_updates} rollouts × {batch} = {total:,} decisions  config {cfg_hash}", flush=True)

    model = MaskablePPO(
        "MlpPolicy", env,
        learning_rate=cfg["learning_rate"], n_steps=cfg["num_steps"], batch_size=cfg["minibatch_size"], n_epochs=cfg["update_epochs"],
        gamma=cfg["gamma"], gae_lambda=cfg["gae_lambda"], clip_range=cfg["clip_coef"], clip_range_vf=None,
        normalize_advantage=cfg["norm_adv"], ent_coef=cfg["ent_coef"], vf_coef=cfg["vf_coef"], max_grad_norm=cfg["max_grad_norm"],
        target_kl=cfg["target_kl"], seed=cfg["seed"], device="cpu", verbose=0, tensorboard_log=str(run_dir / "tb_sb3"),
        policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}, "activation_fn": nn.Tanh, "ortho_init": True},
    )
    cb = CrossCheckLogger(log, venv.action_names)
    t0 = time.perf_counter()
    status, failure = "completed", None
    try:
        model.learn(total_timesteps=total, callback=cb, progress_bar=False)
        if cb.episodes.violations:
            status, failure = "stopped: safety", cb.episodes.violations[:5]
    except Exception as err:
        status, failure = f"stopped: {type(err).__name__}", [str(err)]
        raise
    finally:
        wall = time.perf_counter() - t0
        final_train = {k.split("/", 1)[1]: v for k, v in model.logger.name_to_value.items() if k.startswith("train/")}
        summary = {"runName": run_name, "status": status, "failure": failure, "configHash": cfg_hash, "seed": cfg["seed"],
                   "obsHash": venv.obs_spec["hash"], "actHash": venv.act_spec["hash"], "numEnvs": cfg["num_envs"],
                   "totalDecisions": int(model.num_timesteps), "episodes": cb.episodes.total, "illegalActions": env.illegal_actions,
                   "wallSeconds": wall, "decisionsPerSecOverall": model.num_timesteps / wall if wall else None,
                   "finalTrain": final_train,
                   "trainingEpisodes": {"all": summarise_episodes(cb.all_episodes), "first100": summarise_episodes(cb.all_episodes[:100]),
                                        "last100": summarise_episodes(cb.all_episodes[-100:])}}
        (run_dir / "training_episodes.json").write_text(json.dumps(cb.all_episodes), encoding="utf-8")

    # Evaluate the SB3 actor with the production evaluator (not a production model).
    if status == "completed":
        net = sb3_actor_to_policynet(model)
        obs = torch.as_tensor(model.rollout_buffer.observations.reshape(-1, OBS_SIZE)[:256])
        with torch.no_grad():
            sb3_logits = model.policy.get_distribution(obs).distribution.logits  # masked-free logits (normalised)
            ours = torch.log_softmax(net(obs), dim=-1)
        summary["actorCopyMaxDiff"] = float((torch.log_softmax(sb3_logits, -1) - ours).abs().max())
        out = run_dir / "export"
        policy = export_policy(out / "policy.json", net, venv.obs_spec, venv.act_spec, trained_steps=model.num_timesteps, seed=cfg["seed"],
                               cfg=cfg, cfg_hash=cfg_hash, extra_meta={"crossCheck": "sb3-contrib MaskablePPO — NOT a production model"})
        masks = model.rollout_buffer.action_masks.reshape(-1, ACTION_COUNT)[:256].astype(bool)
        summary["exportParity"] = export_parity(net, policy, obs.numpy(), masks)
        report, eps = node_evaluate(out / "policy.json", out / "validation", limit=cfg["final_eval_limit"], workers=cfg["eval_workers"])
        summary["validation"] = {"metrics": headline(report, "policy:ppo"), "safetyProblems": safety_problems(report, eps),
                                 "actionShares": report["report"]["policy:ppo"]["actionShares"], "n": report["n"]}
        print(f"[sb3] validation n={report['n']}: {json.dumps(summary['validation']['metrics'])}", flush=True)
    log.write_json("summary.json", summary)
    log.close()
    env.close()
    return run_dir, summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="SB3 MaskablePPO cross-check")
    ap.add_argument("--config", default=str(ML_ROOT / "ipl_rl" / "configs" / "ppo_pilot.json"))
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=JSON")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    disable_throttling()  # Windows EcoQoS: speed only, results unchanged
    overrides = {k: json.loads(v) for k, v in (s.split("=", 1) for s in args.set)}
    cfg = load_config(args.config, DEFAULTS, overrides)
    _, summary = run(cfg)
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
