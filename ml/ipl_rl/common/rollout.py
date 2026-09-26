"""Asynchronous on-policy rollout collection (PPO, and later A2C).

Every environment runs at its own pace: as soon as a simulator replies, its
next action is chosen and sent, while the others keep computing. A rollout
ends when every environment has `num_steps` transitions; an environment that
finishes early waits, holding its current state for the next rollout.

Deterministic regardless of which simulator answers first:
  - each environment samples with its own torch.Generator, and
  - the policy runs on one observation at a time,
so environment i's trajectory depends only on the policy parameters, its
generator and its seed schedule. Buffers are laid out [step, env] exactly as
in a lock-step rollout, so the learner update is unchanged.
"""

import time

import numpy as np
import torch

from ..env import ACTION_COUNT, OBS_SIZE
from .masking import MASKED_LOGIT


class AsyncRolloutState:
    """Per-environment state carried from one rollout to the next."""

    def __init__(self, venv, seed):
        self.venv = venv
        n = venv.num_envs
        self.generators = [torch.Generator().manual_seed(int(seed) * 1_000_003 + 7919 * i + 1) for i in range(n)]
        self.obs = np.zeros((n, OBS_SIZE), dtype=np.float32)
        self.mask = np.zeros((n, ACTION_COUNT), dtype=bool)
        obs, mask = venv.reset()
        self.obs[:], self.mask[:] = obs, mask
        # Per-episode running totals, checked against the environment's own
        # summary when the episode ends (same reward, decisions, actions).
        self.ep_return = np.zeros(n, dtype=np.float64)
        self.ep_steps = np.zeros(n, dtype=np.int64)
        self.ep_actions = np.zeros((n, ACTION_COUNT), dtype=np.int64)
        self.consistency_checks = 0


class EpisodeConsistencyError(AssertionError):
    pass


_MASKED = torch.tensor(MASKED_LOGIT)


def policy_step(actor, critic, obs_row, mask_row, generator):
    """One masked sample for one observation: (action, log-prob, value).

    Lean on purpose (≈0.1 ms): no torch.distributions object per call. The
    log-prob is log_softmax over the masked logits — the same distribution
    masking.masked_distribution builds for the update.
    """
    with torch.inference_mode():
        o = torch.from_numpy(obs_row).unsqueeze(0)
        m = torch.from_numpy(mask_row).unsqueeze(0)
        logp_all = torch.log_softmax(torch.where(m, actor(o), _MASKED), dim=-1)
        action = torch.multinomial(logp_all.exp(), 1, generator=generator)
        return int(action), float(logp_all[0, action[0, 0]]), float(critic(o)[0, 0])


def collect_async(state, actor, critic, num_steps, buffers, on_episode):
    """Fill buffers (dict of [T, N] tensors: obs, mask, act, logp, val, rew, done).

    on_episode(summary) is called for every finished episode. Returns the
    time spent waiting for simulators (seconds) — the rest is Python/torch.
    """
    venv = state.venv
    n = venv.num_envs
    t = [0] * n
    waiting = [None] * n  # "step" / "reset" while a request is in flight

    def act(i):
        a, logp, v = policy_step(actor, critic, state.obs[i], state.mask[i], state.generators[i])
        k = t[i]
        buffers["obs"][k, i] = torch.from_numpy(state.obs[i])
        buffers["mask"][k, i] = torch.from_numpy(state.mask[i])
        buffers["act"][k, i] = a
        buffers["logp"][k, i] = logp
        buffers["val"][k, i] = v
        state.ep_actions[i, a] += 1
        venv.send_step(i, a)
        waiting[i] = "step"

    # One thread for single-observation inference (thread hand-offs cost more
    # than the arithmetic); restored for the update.
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _collect(state, venv, n, t, waiting, act, num_steps, buffers, lambda i, ep: _finish(state, i, ep, on_episode))
    finally:
        torch.set_num_threads(threads)


def _finish(state, i, ep, on_episode):
    """The episode's summed step rewards, step count and action counts must
    equal what the JavaScript environment reports."""
    problems = []
    if abs(state.ep_return[i] - ep["return"]) > 1e-9:
        problems.append(f"reward sum {state.ep_return[i]!r} ≠ episode return {ep['return']!r}")
    if state.ep_steps[i] != ep["decisions"]:
        problems.append(f"{state.ep_steps[i]} steps ≠ {ep['decisions']} decisions")
    if state.ep_actions[i].tolist() != list(ep["actionCounts"]):
        problems.append("action counts differ")
    if problems:
        raise EpisodeConsistencyError(f"env {i} seed {ep['seed']}: " + "; ".join(problems))
    state.consistency_checks += 1
    state.ep_return[i], state.ep_steps[i] = 0.0, 0
    state.ep_actions[i] = 0
    on_episode(ep)


def _collect(state, venv, n, t, waiting, act, num_steps, buffers, on_episode):
    wait_seconds = 0.0
    for i in range(n):
        act(i)
    in_flight = n
    while in_flight:
        t0 = time.perf_counter()
        i, reply = venv.next_reply()
        wait_seconds += time.perf_counter() - t0
        in_flight -= 1
        kind, waiting[i] = waiting[i], None
        if kind == "step":
            k = t[i]
            buffers["rew"][k, i] = float(reply["reward"])
            buffers["done"][k, i] = float(reply["done"])
            t[i] += 1
            state.ep_return[i] += reply["reward"]
            state.ep_steps[i] += 1
            if reply["done"]:
                on_episode(i, reply["info"]["episode"])
                venv.send_reset(i)
                waiting[i] = "reset"
                in_flight += 1
                continue
        state.obs[i] = reply["obs"]
        state.mask[i] = reply["mask"]
        if t[i] < num_steps:
            act(i)
            in_flight += 1
    return wait_seconds
