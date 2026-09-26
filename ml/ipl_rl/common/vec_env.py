"""VecIplAuctionEnv — N IplAuctionEnv-v2 simulators stepped in parallel.

Each environment is its own Node process (packages/shared/bin/rl-bridge-v2.js),
so the auction runs in JavaScript exactly as in single-env IplAuctionEnv.
A reader thread per process queues its replies, so the N Node processes
always compute concurrently.

Two ways to drive it:

  Lock-step (Gymnasium / SB3 style):
    obs, masks = venv.reset()                      # (N, 80) float32, (N, 20) bool
    obs, rewards, dones, masks, infos = venv.step(actions)
  Every environment waits for the slowest one each step. The last step of an
  episode is slow (the rest of the auction is simulated with the learner
  passing), so lock-step spends most of its time waiting on stragglers.

  Asynchronous (ipl_rl.common.rollout.collect_async):
    venv.send_step(i, action) / venv.send_reset(i) and venv.next_reply()
  Each environment runs at its own pace; used by our PPO.

Auto-reset (lock-step): when environment i finishes an episode, dones[i] is
True, infos[i]["episode"] holds its summary, and obs[i] / masks[i] are
already the first state of its next episode. With γ = 1 and termination
only (no truncation), nothing is bootstrapped across that boundary.

Seeds: episode k of environment i plays train seed episode_seed(run_seed, i, k).

Safety: a masked action raises MaskedActionError before it is sent; a
simulator that does not reply within `watchdog_seconds` is killed and the
call raises DeadlockError.
"""

import json
import queue
import threading

import numpy as np

from ..bridge import BridgeError, BridgeV2
from ..env import ACTION_COUNT, OBS_SIZE, MaskedActionError
from .seeding import episode_seed


class DeadlockError(RuntimeError):
    pass


class VecIplAuctionEnv:
    def __init__(self, num_envs, run_seed, split="train", tremble=0.01, snapshot_share=0.0, watchdog_seconds=120, node="node"):
        if split != "train":
            raise ValueError("training environments play train seeds only")
        self.num_envs = int(num_envs)
        self.run_seed = int(run_seed)
        self.split = split
        self.bridges = []
        try:
            for _ in range(self.num_envs):
                self.bridges.append(BridgeV2(node))
        except Exception:
            self.close()
            raise
        info = self.bridges[0].info
        self.obs_spec = {k: info["obsSpec"][k] for k in ("version", "hash", "size")}
        self.act_spec = {k: info["actSpec"][k] for k in ("version", "hash", "count")}
        self.action_names = list(info["actSpec"]["actions"])
        self.gamma = float(info["gamma"])
        self.lambda_rel = float(info["lambdaRel"])
        if self.obs_spec["size"] != OBS_SIZE or self.act_spec["count"] != ACTION_COUNT:
            raise RuntimeError("bridge specs do not match obs-v2 / act-v2")
        for b in self.bridges:
            b.call("configure", tremble=float(tremble), snapshotShare=float(snapshot_share))
        self.episode_index = np.zeros(self.num_envs, dtype=np.int64)
        self.current_seed = np.zeros(self.num_envs, dtype=np.int64)
        self.masks = np.zeros((self.num_envs, ACTION_COUNT), dtype=bool)
        self.decisions = 0
        self.episodes = 0
        self.watchdog_seconds = float(watchdog_seconds)

        # From here on replies are read by one thread per process.
        self._replies = queue.Queue()
        self._readers = [threading.Thread(target=self._read, args=(i,), daemon=True) for i in range(self.num_envs)]
        for t in self._readers:
            t.start()

    def _read(self, i):
        stdout = self.bridges[i].proc.stdout
        try:
            for line in stdout:
                self._replies.put((i, line))
        except (ValueError, OSError):
            pass
        self._replies.put((i, None))

    def _next(self):
        try:
            i, line = self._replies.get(timeout=self.watchdog_seconds)
        except queue.Empty:
            for b in self.bridges:
                b.kill()
            raise DeadlockError(f"no simulator reply within {self.watchdog_seconds}s") from None
        if line is None:
            raise BridgeError(f"simulator {i} closed its output")
        reply = json.loads(line)
        if not reply.pop("ok", False):
            raise BridgeError(f"simulator {i}: {reply.get('error', 'unknown error')}")
        return i, reply

    def _collect(self, indices):
        want = set(indices)
        got = {}
        while want - set(got):
            i, reply = self._next()
            if i not in want or i in got:
                raise BridgeError(f"unexpected reply from simulator {i}")
            got[i] = reply
        return [got[i] for i in indices]

    # ── asynchronous interface ─────────────────────────────────────────────
    def send_reset(self, i):
        seed = episode_seed(self.run_seed, i, int(self.episode_index[i]), self.split)
        self.current_seed[i] = seed
        self.episode_index[i] += 1
        self.bridges[i].send("reset", seed=int(seed), split=self.split)

    def send_step(self, i, action):
        action = int(action)
        if not (0 <= action < ACTION_COUNT) or not self.masks[i, action]:
            raise MaskedActionError(f"env {i}: action {action} is masked (legal {np.flatnonzero(self.masks[i]).tolist()})")
        self.bridges[i].send("step", action=action)
        self.decisions += 1

    def next_reply(self):
        """(env index, reply) of the next simulator to answer. Keeps masks current."""
        i, reply = self._next()
        if reply.get("mask") is not None:
            self.masks[i] = reply["mask"]
        if reply.get("done"):
            self.episodes += 1
            reply["info"]["episode"]["envIndex"] = i
        return i, reply

    # ── lock-step interface ────────────────────────────────────────────────
    def reset(self):
        for i in range(self.num_envs):
            self.send_reset(i)
        replies = self._collect(range(self.num_envs))
        obs = np.asarray([r["obs"] for r in replies], dtype=np.float32)
        self.masks = np.asarray([r["mask"] for r in replies], dtype=bool)
        return obs, self.masks.copy()

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int64).reshape(self.num_envs)
        for i, a in enumerate(actions):
            if not (0 <= a < ACTION_COUNT) or not self.masks[i, a]:
                raise MaskedActionError(f"env {i}: action {a} is masked (legal {np.flatnonzero(self.masks[i]).tolist()})")
        for i, a in enumerate(actions):
            self.send_step(i, a)
        replies = self._collect(range(self.num_envs))

        obs = np.zeros((self.num_envs, OBS_SIZE), dtype=np.float32)
        rewards = np.zeros(self.num_envs, dtype=np.float64)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos = [None] * self.num_envs
        finished = []
        for i, r in enumerate(replies):
            rewards[i] = r["reward"]
            infos[i] = r["info"]
            if r["done"]:
                dones[i] = True
                finished.append(i)
            else:
                obs[i] = r["obs"]
                self.masks[i] = r["mask"]
        if finished:
            self.episodes += len(finished)
            for i in finished:
                infos[i]["episode"]["envIndex"] = i
                self.send_reset(i)
            for i, r in zip(finished, self._collect(finished)):
                obs[i] = r["obs"]
                self.masks[i] = r["mask"]
        return obs, rewards, dones, self.masks.copy(), infos

    def close(self):
        for b in self.bridges:
            try:
                b.close()
            except Exception:
                b.kill()
        self.bridges = []
