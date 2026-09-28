"""Phase 2D.2 — Dueling Double DQN pre-flight: network + export, Double-DQN
targets, n-step returns, prioritised replay, ε-greedy, target network, the
update, config hygiene, alarms, act-v3 alignment, export parity and
deterministic evaluation. Randomly initialised networks and hand-made data
only (no training run here)."""

import copy
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

HAVE_TORCH = all(importlib.util.find_spec(m) for m in ("torch", "numpy", "gymnasium"))

if HAVE_TORCH:
    import numpy as np
    import torch

    from ipl_rl.algos import d3qn
    from ipl_rl.algos.d3qn import (DEFAULTS, DuelingQNet, beta_at, double_dqn_targets, dqn_update, epsilon_at, epsilon_problem,
                                   init_seed, masked_argmax, params_digest, select_action, sync_target, to_policy_json, update_problems)
    from ipl_rl.bridge import BridgeV2
    from ipl_rl.common.config import load_config
    from ipl_rl.common.replay import NStepBuilder, PrioritizedReplay, SumTree
    from ipl_rl.env import IplAuctionEnv

ACT_V3 = {"version": "act-v3", "hash": "5f72f510c48b1f46", "count": 20}
OBS_V2 = {"version": "obs-v2", "hash": "629b25783f833af7", "size": 80}
ACTION_NAMES = ["PASS", "BASE", "FV_0.20", "FV_0.30", "FV_0.40", "FV_0.50", "FV_0.60", "FV_0.70", "FV_0.80", "FV_0.90",
                "FV_1.00", "FV_1.10", "FV_1.25", "FV_1.40", "FV_1.60", "FV_1.80", "FV_2.00", "FV_2.50", "FV_3.00", "MAX_SAFE"]


def seeded_net(seed):
    torch.manual_seed(init_seed(seed))
    return DuelingQNet()


def cfg(**kw):
    return {**DEFAULTS, **kw}


def real_states(count=10_000, seed=1_000_097):
    """Real observations / act-v3 masks from one whole JavaScript episode of a
    learner that passes whenever it may (it keeps its purse, so the episode
    has ~470 decisions, ~150 of them in the re-auction, and purses from full
    to nearly empty)."""
    with BridgeV2() as b:
        info = b.info
        r = b.call("reset", seed=seed)
        obs, masks, t = [], [], 0
        while len(obs) < count and not r.get("done"):
            obs.append(r["obs"])
            masks.append(r["mask"])
            legal = [a for a, ok in enumerate(r["mask"]) if ok]
            r = b.call("step", action=legal[0])  # PASS, or the cheapest bid when the shield forces one
            t += 1
    return info, np.asarray(obs, np.float32), np.asarray(masks, bool)


def transition(i, n_actions=20, rng=None):
    rng = rng or np.random.default_rng(i)
    m = rng.random(n_actions) < 0.5
    m[0] = True
    nm = rng.random(n_actions) < 0.5
    nm[0] = True
    return {"obs": rng.standard_normal(80).astype(np.float32), "mask": m, "action": int(np.flatnonzero(m)[-1]), "ret": float(rng.random()),
            "next_obs": rng.standard_normal(80).astype(np.float32), "next_mask": nm, "steps": 16, "mult": 1.0, "done": False, "tag": i}


# ── network and export ─────────────────────────────────────────────────────
@unittest.skipUnless(HAVE_TORCH, "torch/numpy/gymnasium not installed")
class Network(unittest.TestCase):
    def test_shapes_streams_and_dueling_combination(self):
        net = seeded_net(1)
        x = torch.randn(9, 80)
        v, a = net.streams(x)
        self.assertEqual(tuple(v.shape), (9, 1))
        self.assertEqual(tuple(a.shape), (9, 20))
        q = net(x)
        self.assertEqual(tuple(q.shape), (9, 20))
        self.assertTrue(torch.allclose(q, v + a - a.mean(-1, keepdim=True), atol=1e-6))
        self.assertTrue(torch.allclose(q.mean(-1), v.squeeze(-1), atol=1e-5))  # mean over actions of Q = V
        sizes = lambda seq: [(m.in_features, m.out_features) for m in seq if isinstance(m, torch.nn.Linear)]
        self.assertEqual(sizes(net.trunk), [(80, 128), (128, 128)])
        self.assertEqual(sizes(net.value), [(128, 128), (128, 1)])
        self.assertEqual(sizes(net.advantage), [(128, 128), (128, 20)])
        self.assertTrue(all(isinstance(m, torch.nn.Tanh) for m in (net.trunk[1], net.trunk[3], net.value[1], net.advantage[1])))

    def test_orthogonal_init_and_seeds(self):
        net = seeded_net(2)
        w = net.trunk[2].weight.detach()
        self.assertTrue(torch.allclose(w @ w.T, 2.0 * torch.eye(128), atol=1e-4))
        h = net.advantage[2].weight.detach()      # 20 × 128, gain 1 → orthonormal rows
        self.assertTrue(torch.allclose(h @ h.T, torch.eye(20), atol=1e-5))
        self.assertAlmostEqual(float(net.value[2].weight.detach().norm()), 1.0, places=5)
        self.assertTrue(all(torch.all(m.bias == 0) for m in net.modules() if isinstance(m, torch.nn.Linear)))
        digests = {s: params_digest(seeded_net(s)) for s in (1, 2, 3)}
        self.assertEqual(digests, {s: params_digest(seeded_net(s)) for s in (1, 2, 3)})
        self.assertEqual(len(set(digests.values())), 3)
        self.assertEqual(len({init_seed(1), init_seed(2), init_seed(3), d3qn.replay_seed(1), d3qn.explore_seed(1, 0), d3qn.explore_seed(1, 1)}), 6)

    def test_online_target_separation(self):
        online = seeded_net(1)
        target = copy.deepcopy(online)
        self.assertEqual(params_digest(online), params_digest(target))
        self.assertFalse({id(p) for p in online.parameters()} & {id(p) for p in target.parameters()})
        with torch.no_grad():
            next(online.parameters()).add_(1.0)
        self.assertNotEqual(params_digest(online), params_digest(target))

    def test_export_is_an_exact_plain_mlp(self):
        net = seeded_net(3)
        with torch.no_grad():  # make the streams non-trivial
            for p in net.parameters():
                p.add_(0.05 * torch.randn_like(p))
        pol = to_policy_json(net, OBS_V2, ACT_V3)
        self.assertEqual(pol["algorithm"], "d3qn")
        self.assertEqual(pol["selection"], {"mode": "argmax"})
        self.assertEqual(pol["architecture"], {"input": 80, "hidden": [128, 128, 256], "activation": "tanh", "head": "q", "actions": 20})
        x = np.random.default_rng(0).standard_normal((50, 80))
        h = x
        for li, layer in enumerate(pol["layers"]):
            h = h @ np.asarray(layer["weight"]).T + np.asarray(layer["bias"])
            if li < len(pol["layers"]) - 1:
                h = np.tanh(h)
        with torch.no_grad():
            ref = copy.deepcopy(net).double()(torch.from_numpy(x)).numpy()
        self.assertLess(np.abs(h - ref).max(), 1e-12)


# ── Double DQN ─────────────────────────────────────────────────────────────
@unittest.skipUnless(HAVE_TORCH, "torch/numpy/gymnasium not installed")
class DoubleDqn(unittest.TestCase):
    def nets(self, online_pref, target_pref):
        """Nets whose Q depends on the action only through the advantage bias."""
        def make(pref):
            n = seeded_net(1)
            with torch.no_grad():
                n.advantage[2].weight.zero_()
                n.advantage[2].bias.copy_(torch.arange(20, dtype=torch.float32) * 0.01)
                n.advantage[2].bias[pref] = 5.0
            return n
        return make(online_pref), make(target_pref)

    def test_online_selects_target_evaluates(self):
        online, target = self.nets(online_pref=3, target_pref=7)
        s2 = torch.randn(4, 80)
        mask = torch.ones(4, 20, dtype=torch.bool)
        ret = torch.tensor([0.1, 0.2, 0.3, 0.4])
        mult = torch.ones(4)
        y, a_star, _, q_tg_star, _ = double_dqn_targets(online, target, ret, s2, mask, mult)
        self.assertTrue(torch.all(a_star == 3), "online network chooses a*")
        with torch.no_grad():
            expect = ret + target(s2)[:, 3]
            not_max = ret + target(s2).max(-1).values
        self.assertTrue(torch.allclose(y, expect))
        self.assertTrue(torch.allclose(q_tg_star, target(s2)[:, 3].detach()))
        self.assertFalse(torch.allclose(y, not_max), "the target network must not choose the action")

    def test_invalid_next_actions_are_never_selected(self):
        online, target = self.nets(online_pref=3, target_pref=3)
        s2 = torch.randn(4, 80)
        mask = torch.ones(4, 20, dtype=torch.bool)
        mask[:, 3] = False                       # the online favourite is illegal next state
        _, a_star, _, _, _ = double_dqn_targets(online, target, torch.zeros(4), s2, mask, torch.ones(4))
        self.assertTrue(torch.all(a_star == 19), "best LEGAL action (bias 0.19)")
        single = torch.zeros(4, 20, dtype=torch.bool)
        single[:, 5] = True
        _, a_star, _, _, _ = double_dqn_targets(online, target, torch.zeros(4), s2, single, torch.ones(4))
        self.assertTrue(torch.all(a_star == 5))
        q = torch.full((2, 20), 1e30)            # even huge masked values cannot win
        m = torch.zeros(2, 20, dtype=torch.bool)
        m[:, 0] = True
        self.assertTrue(torch.all(masked_argmax(q, m) == 0))

    def test_terminal_target_is_the_return(self):
        online, target = self.nets(3, 7)
        ret = torch.tensor([0.5, -0.25, 1.0])
        pass_only = torch.zeros(3, 20, dtype=torch.bool)
        pass_only[:, 0] = True
        y, _, _, _, boot = double_dqn_targets(online, target, ret, torch.zeros(3, 80), pass_only, torch.zeros(3))
        self.assertTrue(torch.equal(y, ret))
        self.assertFalse(bool(boot.any()))


# ── n-step ─────────────────────────────────────────────────────────────────
@unittest.skipUnless(HAVE_TORCH, "torch/numpy/gymnasium not installed")
class NStep(unittest.TestCase):
    def episode(self, T, n=16, gamma=1.0):
        rng = np.random.default_rng(T)
        obs = rng.standard_normal((T + 1, 80)).astype(np.float32)
        masks = rng.random((T + 1, 20)) < 0.5
        masks[:, 0] = True
        rewards = rng.random(T)
        b = NStepBuilder(n, gamma)
        out = []
        for t in range(T):
            done = t == T - 1
            out += b.push(obs[t], masks[t], 0, rewards[t], None if done else obs[t + 1], None if done else masks[t + 1], done, tag=t)
        return obs, masks, rewards, out, b

    def test_full_16_step_return_and_next_state_mask(self):
        obs, masks, r, out, _ = self.episode(40)
        self.assertEqual(len(out), 40)
        for t in range(40 - 16):
            tr = out[t]
            self.assertEqual(tr["tag"], t)
            self.assertAlmostEqual(tr["ret"], float(r[t:t + 16].sum()), places=12)
            self.assertEqual(tr["steps"], 16)
            self.assertEqual(tr["mult"], 1.0)          # γ^16 with γ = 1
            self.assertFalse(tr["done"])
            self.assertTrue(np.array_equal(tr["next_obs"], obs[t + 16]))
            self.assertTrue(np.array_equal(tr["next_mask"], masks[t + 16]), "next mask = the mask of the actual state t+16")
            self.assertTrue(np.array_equal(tr["obs"], obs[t]) and np.array_equal(tr["mask"], masks[t]))

    def test_terminal_tail_never_bootstraps(self):
        obs, masks, r, out, b = self.episode(40)
        for t in range(40 - 16, 40):
            tr = out[t]
            self.assertEqual(tr["tag"], t)
            self.assertTrue(tr["done"])
            self.assertEqual(tr["mult"], 0.0)
            self.assertEqual(tr["steps"], 40 - t)
            self.assertAlmostEqual(tr["ret"], float(r[t:].sum()), places=12)
            self.assertTrue(np.array_equal(tr["next_mask"], np.eye(20, dtype=bool)[0]))  # placeholder, never used
        self.assertEqual(len(b.queue), 0, "nothing carried into the next episode")
        last = out[-1]
        self.assertEqual(last["steps"], 1)
        self.assertAlmostEqual(last["ret"], float(r[-1]))

    def test_shorter_than_n_episode(self):
        obs, masks, r, out, _ = self.episode(5)
        self.assertEqual(len(out), 5)
        for t, tr in enumerate(out):
            self.assertEqual((tr["tag"], tr["steps"], tr["mult"], tr["done"]), (t, 5 - t, 0.0, True))
            self.assertAlmostEqual(tr["ret"], float(r[t:].sum()), places=12)

    def test_discounting_formula(self):
        obs, masks, r, out, _ = self.episode(30, n=4, gamma=0.9)
        tr = out[0]
        self.assertAlmostEqual(tr["ret"], sum(0.9 ** j * r[j] for j in range(4)), places=12)
        self.assertAlmostEqual(tr["mult"], 0.9 ** 4, places=12)
        self.assertAlmostEqual(out[-2]["ret"], r[-2] + 0.9 * r[-1], places=12)

    def test_episode_boundary_isolation(self):
        b = NStepBuilder(16, 1.0)
        z = np.zeros(80, np.float32)
        m = np.eye(20, dtype=bool)[0]
        for t in range(3):
            b.push(z, m, 0, 100.0, z, m, t == 2)
        out = []
        for t in range(20):
            out += b.push(z, m, 0, 1.0, z, m, False)
        self.assertEqual(out[0]["ret"], 16.0, "no reward from the previous episode")

    def test_real_environment_masks(self):
        """Transitions built from a real JavaScript episode carry exactly the
        masks and observations the environment reported at t and t+16."""
        env = IplAuctionEnv(split="train")
        try:
            o, _ = env.reset(options={"episode_seed": 1_000_313})
            states, masks, rewards, actions = [o], [env.action_masks()], [], []
            b = NStepBuilder(16, 1.0)
            out = []
            t = 0
            while True:
                legal = np.flatnonzero(masks[-1])
                a = int(legal[(5 * t) % len(legal)])
                o2, r, done, _, _ = env.step(a)
                m2 = None if done else env.action_masks()
                out += b.push(states[-1], masks[-1], a, r, None if done else o2, m2, done, tag=t)
                rewards.append(r)
                actions.append(a)
                t += 1
                if done:
                    break
                states.append(o2)
                masks.append(m2)
        finally:
            env.close()
        T = len(rewards)
        self.assertEqual(len(out), T)
        for tr in out:
            t = tr["tag"]
            self.assertTrue(np.array_equal(tr["obs"], states[t]) and np.array_equal(tr["mask"], masks[t]))
            self.assertTrue(tr["mask"][tr["action"]], "stored action legal under its stored mask")
            if t + 16 < T:
                self.assertTrue(np.array_equal(tr["next_mask"], masks[t + 16]) and np.array_equal(tr["next_obs"], states[t + 16]))
                self.assertEqual((tr["steps"], tr["mult"]), (16, 1.0))
            else:
                self.assertEqual((tr["done"], tr["mult"], tr["steps"]), (True, 0.0, T - t))
            self.assertAlmostEqual(tr["ret"], sum(rewards[t:t + 16]), places=12)


# ── prioritised replay ─────────────────────────────────────────────────────
@unittest.skipUnless(HAVE_TORCH, "torch/numpy/gymnasium not installed")
class Replay(unittest.TestCase):
    def filled(self, n=64, cap=64, seed=0):
        rep = PrioritizedReplay(cap, 0.6, 1e-6, np.random.default_rng(seed))
        rep.add_many([transition(i) for i in range(n)])
        return rep

    def test_priorities_alpha_probabilities_and_weights(self):
        rep = self.filled(16, 16)
        self.assertTrue(np.allclose(rep.tree.leaves(), 1.0))            # new → max priority (1.0)^α
        td = np.linspace(-2, 2, 16)
        rep.update_priorities(np.arange(16), td)
        p = np.abs(td) + 1e-6
        self.assertTrue(np.allclose(rep.tree.leaves(), p ** 0.6, rtol=1e-12))
        self.assertAlmostEqual(rep.tree.total, float((p ** 0.6).sum()), places=10)
        self.assertEqual(rep.max_priority, float(p.max()))
        idx, w, prob = rep.sample(8, beta=0.4)
        P = (p ** 0.6) / (p ** 0.6).sum()
        self.assertTrue(np.allclose(prob, P[idx]))
        raw = (16 * P[idx]) ** -0.4
        self.assertTrue(np.allclose(w, raw / raw.max()))
        self.assertAlmostEqual(float(w.max()), 1.0)
        rep.add(transition(99))                                           # new transition → max priority seen
        self.assertAlmostEqual(float(rep.tree.leaves([0])[0]), float(p.max()) ** 0.6)

    def test_sampling_frequencies_match_p_alpha(self):
        rep = self.filled(8, 8, seed=3)
        rep.update_priorities(np.arange(8), np.array([0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 0.05, 1.0]))
        P = rep.tree.leaves() / rep.tree.total
        counts = np.zeros(8)
        for _ in range(4000):
            idx, _, _ = rep.sample(32, 0.4)
            np.add.at(counts, idx, 1)
        freq = counts / counts.sum()
        self.assertLess(np.abs(freq - P).max(), 0.01)

    def test_beta_schedule(self):
        c = cfg()
        self.assertEqual(beta_at(0, c), 0.4)
        self.assertAlmostEqual(beta_at(c["total_decisions"] // 2, c), 0.7, places=12)
        self.assertEqual(beta_at(c["total_decisions"], c), 1.0)
        self.assertEqual(beta_at(c["total_decisions"] + 100, c), 1.0)
        xs = [beta_at(d, c) for d in range(0, c["total_decisions"] + 1, 99_840)]
        self.assertTrue(all(b2 > b1 for b1, b2 in zip(xs, xs[1:])))

    def test_priority_updates_duplicates_and_tree_consistency(self):
        rep = self.filled(64, 64, seed=1)
        rep.update_priorities([5, 5, 9], [0.3, 2.0, 0.1])                # duplicate index: last write wins
        self.assertAlmostEqual(float(rep.tree.leaves([5])[0]), (2.0 + 1e-6) ** 0.6)
        self.assertLess(rep.tree.check(), 1e-12)
        rep.update_priorities(np.arange(64), np.zeros(64))
        rep.update_priorities([7], [1e6])                                 # one dominant priority
        idx, w, _ = rep.sample(16, 0.4)
        self.assertGreater((idx == 7).sum(), 1, "repeated indices are possible and handled")
        net = seeded_net(1)
        tgt = copy.deepcopy(net)
        opt = torch.optim.Adam(net.parameters(), lr=3e-4, eps=1e-5)
        stats, td = dqn_update(net, tgt, opt, rep.batch(idx), w, cfg())
        rep.update_priorities(idx, td)                                    # duplicates carry identical TD errors
        dup = idx == 7
        self.assertTrue(np.allclose(td[dup], td[dup][0]))
        self.assertLess(rep.tree.check(), 1e-12)
        self.assertEqual(rep.audit(1.0, 16), [])

    def test_sampling_is_deterministic_and_ring_wraps(self):
        a, b = self.filled(40, 64, seed=5), self.filled(40, 64, seed=5)
        for _ in range(5):
            ia, wa, _ = a.sample(16, 0.5)
            ib, wb, _ = b.sample(16, 0.5)
            self.assertTrue(np.array_equal(ia, ib) and np.array_equal(wa, wb))
        rep = PrioritizedReplay(8, 0.6, 1e-6, np.random.default_rng(0))
        rep.add_many([transition(i) for i in range(8)])
        rep.add_many([transition(i) for i in range(8, 12)])
        self.assertEqual((len(rep), rep.pos, rep.added), (8, 4, 12))
        self.assertEqual(sorted(rep.tag.tolist()), [4, 5, 6, 7, 8, 9, 10, 11])
        idx, _, _ = rep.sample(8, 0.4)
        self.assertTrue(np.all(idx < 8))

    def test_audit_detects_corruption(self):
        rep = self.filled(32, 64)
        self.assertEqual(rep.audit(1.0, 16), [])
        rep.action[3] = int(np.flatnonzero(~rep.mask[3])[0])
        self.assertTrue(any("legal" in p for p in rep.audit(1.0, 16)))
        rep = self.filled(32, 64)
        rep.tree.tree[1] += 1.0
        self.assertTrue(any("sum tree" in p for p in rep.audit(1.0, 16)))
        rep = self.filled(32, 64)
        rep.done[2] = True
        self.assertTrue(any("terminal" in p for p in rep.audit(1.0, 16)))

    def test_sum_tree_find(self):
        t = SumTree(5)
        t.update(np.arange(5), [1.0, 2.0, 0.0, 3.0, 4.0])
        self.assertEqual(t.total, 10.0)
        self.assertEqual(t.find([0.0, 0.99, 1.0, 2.99, 3.0, 5.99, 6.0, 9.99]).tolist(), [0, 0, 1, 1, 3, 3, 4, 4])


# ── exploration, target network, update ────────────────────────────────────
@unittest.skipUnless(HAVE_TORCH, "torch/numpy/gymnasium not installed")
class ExplorationTargetUpdate(unittest.TestCase):
    def test_epsilon_schedule(self):
        c = cfg()
        self.assertEqual(epsilon_at(0, c), 1.0)
        self.assertAlmostEqual(epsilon_at(500_000, c), 0.525, places=12)
        self.assertEqual(epsilon_at(1_000_000, c), 0.05)
        self.assertEqual(epsilon_at(1_500_000, c), 0.05)
        self.assertEqual(epsilon_at(1_996_799, c), 0.05)
        xs = [epsilon_at(d, c) for d in range(0, 1_000_001, 10_000)]
        self.assertTrue(all(b < a for a, b in zip(xs, xs[1:])))
        self.assertIsNone(epsilon_problem(500, 500.0, 250.0))
        self.assertIsNotNone(epsilon_problem(700, 500.0, 250.0))

    def test_exploration_never_selects_masked_actions(self):
        net = seeded_net(1)
        g = np.random.default_rng(0)
        masks = g.random((300, 20)) < 0.3
        masks[np.arange(300), g.integers(0, 20, 300)] = True
        obs = g.standard_normal((300, 80)).astype(np.float32)
        gen = np.random.default_rng(1)
        for eps in (1.0, 0.5, 0.05, 0.0):
            for i in range(300):
                a, explored = select_action(net, obs[i], masks[i], eps, gen)
                self.assertTrue(masks[i, a])
                if eps == 0.0:
                    self.assertFalse(explored)
                    with torch.no_grad():
                        self.assertEqual(a, int(masked_argmax(net(torch.from_numpy(obs[i:i + 1]))[0], torch.from_numpy(masks[i]))))
        # ε = 1: uniform over the legal actions.
        m = np.zeros(20, bool)
        m[[0, 4, 9, 19]] = True
        picks = [select_action(net, obs[0], m, 1.0, gen)[0] for _ in range(4000)]
        freq = np.bincount(picks, minlength=20)[[0, 4, 9, 19]] / 4000
        self.assertLess(np.abs(freq - 0.25).max(), 0.03)
        self.assertEqual(int(np.bincount(picks, minlength=20).sum() - np.bincount(picks, minlength=20)[[0, 4, 9, 19]].sum()), 0)

    def test_target_initialised_equal_frozen_between_syncs_copied_exactly(self):
        online = seeded_net(2)
        target = copy.deepcopy(online)
        self.assertEqual(params_digest(online), params_digest(target))
        rep = PrioritizedReplay(128, 0.6, 1e-6, np.random.default_rng(0))
        rep.add_many([transition(i) for i in range(128)])
        opt = torch.optim.Adam(online.parameters(), lr=1e-3, eps=1e-5)
        frozen = params_digest(target)
        syncs = []
        for u in range(1, 13):
            idx, w, _ = rep.sample(32, 0.4)
            _, td = dqn_update(online, target, opt, rep.batch(idx), w, cfg())
            rep.update_priorities(idx, td)
            d = sync_target(u, online, target, 5)
            if d is None:
                self.assertEqual(params_digest(target), frozen, f"target changed at update {u}")
                self.assertNotEqual(params_digest(online), frozen)
            else:
                syncs.append(u)
                self.assertEqual(d, params_digest(online))
                frozen = d
        self.assertEqual(syncs, [5, 10])
        self.assertTrue(all(not p.requires_grad or p.grad is None for p in target.parameters()))

    def test_update_objective_one_step_clipping(self):
        online = seeded_net(3)
        target = copy.deepcopy(online)
        with torch.no_grad():
            for p in target.parameters():
                p.add_(0.1)
        rep = PrioritizedReplay(64, 0.6, 1e-6, np.random.default_rng(0))
        ts = [transition(i) for i in range(64)]
        for t in ts[:10]:
            t.update(done=True, mult=0.0, steps=3)
            t["next_mask"] = np.eye(20, dtype=bool)[0]
        rep.add_many(ts)
        idx, w, _ = rep.sample(64, 0.4)
        b = rep.batch(idx)
        # Manual Double-DQN / Huber / IS-weighted loss.
        with torch.no_grad():
            s2 = torch.from_numpy(b["next_obs"])
            a_star = masked_argmax(online(s2), torch.from_numpy(b["next_mask"]))
            qt = target(s2).gather(1, a_star.unsqueeze(1)).squeeze(1)
            y = torch.from_numpy(b["ret"]).float() + torch.where(torch.from_numpy(b["mult"]) > 0, qt, torch.zeros_like(qt))
            q = online(torch.from_numpy(b["obs"])).gather(1, torch.from_numpy(b["action"]).unsqueeze(1)).squeeze(1)
            manual = (torch.from_numpy(w).float() * torch.nn.functional.huber_loss(q, y, reduction="none", delta=1.0)).mean().item()
        tgt_before = params_digest(target)
        opt = torch.optim.Adam(online.parameters(), lr=3e-4, eps=1e-5)
        before = params_digest(online)
        stats, td = dqn_update(online, target, opt, b, w * 1e4, cfg())   # large weights → gradient must be clipped
        self.assertNotEqual(params_digest(online), before)
        self.assertEqual(params_digest(target), tgt_before, "the update never touches the target network")
        self.assertEqual({int(s["step"]) for s in opt.state.values()}, {1}, "one optimiser step")
        clipped = math.sqrt(sum(float(p.grad.norm() ** 2) for p in online.parameters()))
        self.assertLessEqual(clipped, 0.5 + 1e-5)
        self.assertGreater(stats["grad_norm"], 0.5)
        self.assertTrue(np.allclose(td, (y - q).numpy(), atol=1e-6))
        online2 = seeded_net(3)
        stats2, _ = dqn_update(online2, target, torch.optim.Adam(online2.parameters(), lr=3e-4, eps=1e-5), b, w, cfg())
        self.assertAlmostEqual(stats2["loss"], manual, places=5)

    def test_alarms(self):
        ok = {"loss": 0.1, "td_abs_mean": 0.2, "td_abs_max": 1.0, "q_mean": 3.0, "q_min": -1.0, "q_max": 9.0, "q_legal_absmax": 9.0,
              "q_next_absmax": 9.0, "grad_norm": 2.0, "target_divergence": 0.3, "grads_finite": True, "params_finite": True}
        self.assertEqual(update_problems(ok), [])
        for bad in ({"loss": float("nan")}, {"q_max": float("inf")}, {"grads_finite": False}, {"grad_norm": 150.0},
                    {"q_legal_absmax": 60.0}, {"target_divergence": 6.0}):
            self.assertTrue(update_problems({**ok, **bad}), bad)

    def test_config_is_value_learning_only_and_frozen(self):
        for key in ("clip_coef", "ent_coef", "vf_coef", "gae_lambda", "norm_adv", "update_epochs", "minibatch_size", "target_kl", "num_steps", "tau"):
            self.assertNotIn(key, DEFAULTS)
            with self.assertRaises(KeyError):
                load_config(None, DEFAULTS, {key: 1})
        c = load_config(Path(d3qn.__file__).parents[1] / "configs" / "d3qn_2d2.json", DEFAULTS)
        frozen = {"num_envs": 12, "gamma": 1.0, "n_step": 16, "replay_capacity": 500_000, "learning_starts": 10_000, "batch_size": 256,
                  "per_alpha": 0.6, "per_beta_start": 0.4, "per_beta_final": 1.0, "per_eps": 1e-6, "learning_rate": 3e-4, "adam_eps": 1e-5,
                  "max_grad_norm": 0.5, "huber_delta": 1.0, "target_update_every": 2_500, "eps_start": 1.0, "eps_final": 0.05,
                  "eps_decay_decisions": 1_000_000, "total_decisions": 1_996_800, "stage": "A", "expected_act_spec": ACT_V3["hash"],
                  "eval_decisions": [245_760, 497_664, 749_568, 1_001_472, 1_247_232, 1_499_136, 1_751_040, 1_996_800]}
        self.assertEqual({k: c[k] for k in frozen}, frozen)
        self.assertTrue(all(d % 12 == 0 for d in c["eval_decisions"]))


# ── specs, export parity, deterministic evaluation ─────────────────────────
@unittest.skipUnless(HAVE_TORCH, "torch/numpy/gymnasium not installed")
class SpecsParityEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.info, cls.obs, cls.masks = real_states()

    def test_act_v3_alignment(self):
        act = self.info["actSpec"]
        self.assertEqual({k: act[k] for k in ("version", "hash", "count")}, ACT_V3)
        self.assertEqual(list(act["actions"]), ACTION_NAMES)
        self.assertEqual({k: self.info["obsSpec"][k] for k in ("version", "hash", "size")}, OBS_V2)
        self.assertEqual(self.info["gamma"], 1)
        self.assertEqual(seeded_net(1)(torch.zeros(1, 80)).shape[-1], len(ACTION_NAMES))

    def test_export_parity_all_mask_kinds_real_reauction_and_extreme_purse_states(self):
        net = seeded_net(1)
        with torch.no_grad():
            for p in net.parameters():
                p.add_(0.05 * torch.randn_like(p))
        pol = to_policy_json(net, OBS_V2, ACT_V3)
        reauc = self.obs[:, 0] > 0.5
        self.assertTrue(reauc.any(), "the test episode reaches the re-auction")
        order = np.argsort(self.obs[:, 15])
        extreme = self.obs.copy()[:4]
        extreme[:2, 15] = 0.0                       # self_purse at the floor
        extreme[2:, 15] = 1.5                       # far above any real purse
        obs = np.concatenate([self.obs, self.obs[order[:5]], extreme])
        masks = np.concatenate([self.masks, self.masks[order[:5]], self.masks[:4]])
        par = d3qn.parity(net, pol, obs, masks)
        self.assertLess(par["maxAbsScoreDiff"], 1e-9)
        self.assertEqual(par["argmaxAgreement"], 1.0)
        self.assertEqual(set(par["byMask"]), {"real", "passOnly", "single", "allLegal"})
        self.assertTrue(all(v == 1.0 for v in par["byMask"].values()))

    def test_deterministic_evaluation(self):
        from ipl_rl.common.evaluation import node_evaluate
        from ipl_rl.export import write_policy
        net = seeded_net(2)
        with tempfile.TemporaryDirectory() as tmp:
            p = write_policy(Path(tmp) / "policy.json", to_policy_json(net, OBS_V2, ACT_V3))
            runs = [node_evaluate(p, Path(tmp) / f"eval{k}", limit=3, workers=3)[1]["episodes"]["policy:d3qn"] for k in range(2)]
        strip = lambda rows: [{k: v for k, v in r.items() if k not in ("ms", "seconds", "wallMs")} for r in rows]
        self.assertEqual(strip(runs[0]), strip(runs[1]))
        self.assertEqual(len(runs[0]), 3)
        self.assertTrue(all(r["legalXI"] for r in runs[0]))


if __name__ == "__main__":
    unittest.main()
