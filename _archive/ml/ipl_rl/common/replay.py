"""Replay for the value-based learners (Phase 2D.2 D3QN; reusable by QR-DQN).

NStepBuilder      per-environment n-step transitions from the decisions of
                  ONE environment, in order. Stops at termination (never
                  bootstraps across an episode end).
SumTree           a binary sum tree over priority^alpha (vectorised numpy).
PrioritizedReplay a ring buffer of n-step transitions with proportional
                  prioritised sampling (Schaul et al. 2016).

A stored transition is (s_t, mask_t, a_t, G_t, s_{t+k}, mask_{t+k}, k, m):
  G_t   = r_t + γ r_{t+1} + … + γ^(k−1) r_{t+k−1}, k = min(n, steps to the end)
  m     = γ^k if s_{t+k} is a real decision state of the same episode, 0 if
          the episode terminated within the k steps (no bootstrap)
  mask_{t+k} is the act-v3 mask the environment reported WITH s_{t+k} — the
  mask of the actual next decision, never reconstructed later. For terminal
  transitions there is no next state: s_{t+k} = 0 and mask = PASS only (a
  placeholder that can never contribute, since m = 0).

Priorities: p_i = |δ_i| + ε; P(i) = p_i^α / Σ_j p_j^α; importance weights
w_i = (N · P(i))^(−β), normalised by the maximum weight in the sampled batch.
A new transition enters with the maximum priority seen so far (1.0 before
any update), so every transition is likely to be replayed at least once.
"""

from collections import deque

import numpy as np

from ..env import ACTION_COUNT, OBS_SIZE, PASS


class NStepBuilder:
    """n-step returns for one environment. push() one decision at a time;
    it returns the transitions that became complete."""

    def __init__(self, n, gamma):
        if n < 1:
            raise ValueError("n must be ≥ 1")
        self.n = int(n)
        self.gamma = float(gamma)
        self.queue = deque()

    def push(self, obs, mask, action, reward, next_obs, next_mask, done, tag=None):
        """One decision: (s_t, mask_t, a_t, r_t) and what followed it (s_{t+1},
        mask_{t+1}, terminal?). tag is carried through (e.g. the decision index)."""
        self.queue.append((np.asarray(obs, np.float32), np.asarray(mask, bool), int(action), float(reward), tag))
        out = []
        if done:
            while self.queue:  # every pending start state ends at the terminal step
                out.append(self._emit(None, None, terminal=True))
        elif len(self.queue) == self.n:
            out.append(self._emit(np.asarray(next_obs, np.float32), np.asarray(next_mask, bool), terminal=False))
        return out

    def _emit(self, next_obs, next_mask, terminal):
        k = len(self.queue)
        g = 0.0
        for j, item in enumerate(self.queue):
            g += (self.gamma ** j) * item[3]
        obs, mask, action, _, tag = self.queue.popleft()
        if terminal:
            next_obs = np.zeros(OBS_SIZE, np.float32)
            next_mask = np.zeros(ACTION_COUNT, bool)
            next_mask[PASS] = True
            mult = 0.0
        else:
            mult = self.gamma ** k
        return {"obs": obs, "mask": mask, "action": action, "ret": g, "next_obs": next_obs, "next_mask": next_mask,
                "steps": k, "mult": mult, "done": terminal, "tag": tag}

    def reset(self):
        self.queue.clear()


class SumTree:
    """Leaves hold priority^alpha; internal node i = node 2i + node 2i+1; root = 1."""

    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.size = 1
        while self.size < self.capacity:
            self.size *= 2
        self.depth = int(np.log2(self.size))
        self.tree = np.zeros(2 * self.size, dtype=np.float64)

    @property
    def total(self):
        return float(self.tree[1])

    def leaves(self, idx=None):
        leaf = self.tree[self.size:self.size + self.capacity]
        return leaf if idx is None else leaf[idx]

    def update(self, idx, values):
        """Set leaves; duplicates in idx are allowed (last write wins) —
        parents are recomputed from their children, never incremented."""
        idx = np.asarray(idx, dtype=np.int64)
        values = np.asarray(values, dtype=np.float64)
        if idx.size == 0:
            return
        node = idx + self.size
        self.tree[node] = values
        node = np.unique(node // 2)
        while node[0] >= 1:
            self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]
            if node[0] == 1:
                break
            node = np.unique(node // 2)

    def find(self, u):
        """Leaf index for each cumulative value u ∈ [0, total)."""
        u = np.asarray(u, dtype=np.float64).copy()
        node = np.ones(u.shape, dtype=np.int64)
        for _ in range(self.depth):
            left = 2 * node
            lv = self.tree[left]
            go_right = u >= lv
            u = np.where(go_right, u - lv, u)
            node = np.where(go_right, left + 1, left)
        return node - self.size

    def check(self):
        """Every internal node equals the sum of its children (max relative error)."""
        worst = 0.0
        lo = self.size
        level = self.tree[lo:2 * lo]
        while lo > 1:
            parent = level[0::2] + level[1::2]
            stored = self.tree[lo // 2:lo]
            err = np.abs(parent - stored) / np.maximum(1e-300, np.abs(parent) + 1e-12)
            worst = max(worst, float(err.max()))
            lo //= 2
            level = stored
        return worst


class PrioritizedReplay:
    def __init__(self, capacity, alpha, eps, rng):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.rng = rng
        c = self.capacity
        self.obs = np.zeros((c, OBS_SIZE), np.float32)
        self.mask = np.zeros((c, ACTION_COUNT), bool)
        self.action = np.zeros(c, np.int64)
        self.ret = np.zeros(c, np.float64)
        self.next_obs = np.zeros((c, OBS_SIZE), np.float32)
        self.next_mask = np.zeros((c, ACTION_COUNT), bool)
        self.steps = np.zeros(c, np.int16)
        self.mult = np.zeros(c, np.float64)
        self.done = np.zeros(c, bool)
        self.tag = np.full(c, -1, np.int64)
        self.tree = SumTree(c)
        self.pos = 0
        self.n = 0
        self.added = 0
        self.max_priority = 1.0  # raw p (before ^alpha)

    def __len__(self):
        return self.n

    def add(self, t):
        i = self.pos
        self.obs[i], self.mask[i], self.action[i] = t["obs"], t["mask"], t["action"]
        self.ret[i], self.next_obs[i], self.next_mask[i] = t["ret"], t["next_obs"], t["next_mask"]
        self.steps[i], self.mult[i], self.done[i] = t["steps"], t["mult"], t["done"]
        self.tag[i] = -1 if t.get("tag") is None else int(t["tag"])
        self.tree.update([i], [self.max_priority ** self.alpha])
        self.pos = (i + 1) % self.capacity
        self.n = min(self.n + 1, self.capacity)
        self.added += 1
        return i

    def add_many(self, transitions):
        """Add in order with ONE sum-tree update (same result as repeated add())."""
        if len(transitions) > self.capacity:
            raise ValueError("more transitions than capacity in one call")
        idx = []
        for t in transitions:
            i = self.pos
            self.obs[i], self.mask[i], self.action[i] = t["obs"], t["mask"], t["action"]
            self.ret[i], self.next_obs[i], self.next_mask[i] = t["ret"], t["next_obs"], t["next_mask"]
            self.steps[i], self.mult[i], self.done[i] = t["steps"], t["mult"], t["done"]
            self.tag[i] = -1 if t.get("tag") is None else int(t["tag"])
            idx.append(i)
            self.pos = (i + 1) % self.capacity
            self.n = min(self.n + 1, self.capacity)
            self.added += 1
        self.tree.update(idx, [self.max_priority ** self.alpha] * len(idx))
        return idx

    def sample(self, batch_size, beta):
        """Stratified proportional sampling. Returns (indices, normalised IS weights, P(i))."""
        total = self.tree.total
        seg = total / batch_size
        u = (np.arange(batch_size) + self.rng.random(batch_size)) * seg
        u = np.minimum(u, np.nextafter(total, 0))
        idx = self.tree.find(u)
        pri = self.tree.leaves(idx)
        if np.any(idx >= self.n) or np.any(pri <= 0):
            raise ReplayCorruption(f"sampled an empty slot: idx {idx[(idx >= self.n) | (pri <= 0)][:5]}")
        prob = pri / total
        w = (self.n * prob) ** (-beta)
        return idx, w / w.max(), prob

    def update_priorities(self, idx, td_errors):
        p = np.abs(np.asarray(td_errors, np.float64)) + self.eps
        if not np.all(np.isfinite(p)):
            raise ReplayCorruption("non-finite priority")
        self.tree.update(idx, p ** self.alpha)
        self.max_priority = max(self.max_priority, float(p.max()))
        return p

    def batch(self, idx):
        return {k: getattr(self, k)[idx] for k in ("obs", "mask", "action", "ret", "next_obs", "next_mask", "steps", "mult", "done")}

    def raw_priorities(self):
        return self.tree.leaves()[:self.n] ** (1.0 / self.alpha)

    def audit(self, gamma, n_step):
        """Replay-corruption checks. Returns a list of problems (empty = healthy)."""
        k = self.n
        problems = []
        if k > self.capacity:
            problems.append("size above capacity")
        leaves = self.tree.leaves()
        if not np.all(np.isfinite(leaves[:k])) or np.any(leaves[:k] <= 0):
            problems.append("non-positive or non-finite priority in a filled slot")
        if np.any(leaves[k:] != 0):
            problems.append("priority in an empty slot")
        tree_err = self.tree.check()
        if tree_err > 1e-9:
            problems.append(f"sum tree inconsistent (relative error {tree_err:.2e})")
        if not np.all(self.mask[np.arange(k), self.action[:k]]):
            problems.append("stored action not legal under its stored mask")
        boot = self.mult[:k] > 0
        if np.any(self.done[:k] & boot):
            problems.append("terminal transition with a bootstrap multiplier")
        if not np.all(self.next_mask[:k][boot].any(axis=1)):
            problems.append("bootstrapped transition without a legal next action")
        if np.any((self.steps[:k] < 1) | (self.steps[:k] > n_step)):
            problems.append("n-step length out of range")
        if np.any(boot & (self.steps[:k] != n_step)):
            problems.append("bootstrapped transition shorter than n")
        if np.any(np.abs(self.mult[:k][boot] - gamma ** self.steps[:k][boot]) > 1e-12):
            problems.append("bootstrap multiplier ≠ γ^n")
        if not (np.all(np.isfinite(self.ret[:k])) and np.all(np.isfinite(self.obs[:k])) and np.all(np.isfinite(self.next_obs[:k]))):
            problems.append("non-finite stored values")
        return problems


class ReplayCorruption(RuntimeError):
    pass
