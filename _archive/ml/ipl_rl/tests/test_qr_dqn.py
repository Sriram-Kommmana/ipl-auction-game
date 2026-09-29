"""Phase 2D.3 — QR-DQN pre-flight: network, quantile fractions, mean-quantile
action selection, Double-DQN distributional targets, the quantile-Huber loss
(hand-computed), n-step returns (synthetic AND real JavaScript episodes
checked against an independent JavaScript replay), prioritised replay,
ε-greedy, target network, the update, config hygiene, alarms, act-v3
alignment, export parity (all 640 outputs) and deterministic evaluation.
Randomly initialised networks and hand-made data only (no training run)."""

import copy
import importlib.util
import json
import math
import subprocess
import tempfile
import unittest
from pathlib import Path

HAVE_TORCH = all(importlib.util.find_spec(m) for m in ("torch", "numpy", "gymnasium"))

if HAVE_TORCH:
    import numpy as np
    import torch

    from ipl_rl.algos import qr_dqn
    from ipl_rl.algos.qr_dqn import (DEFAULTS, LEVELS, TAUS, MeanQ, beta_at, epsilon_at, init_seed, make_net, masked_argmax, params_digest,
                                     qr_targets, qr_update, quantile_huber_loss, select_action, sync_target, to_policy_json, update_problems)
    from ipl_rl.bridge import BridgeV2
    from ipl_rl.common.config import load_config
    from ipl_rl.common.replay import NStepBuilder, PrioritizedReplay
    from ipl_rl.env import IplAuctionEnv

HERE = Path(__file__).resolve().parent
ACT_V3 = {"version": "act-v3", "hash": "5f72f510c48b1f46", "count": 20}
OBS_V2 = {"version": "obs-v2", "hash": "629b25783f833af7", "size": 80}
ACTION_NAMES = ["PASS", "BASE", "FV_0.20", "FV_0.30", "FV_0.40", "FV_0.50", "FV_0.60", "FV_0.70", "FV_0.80", "FV_0.90",
                "FV_1.00", "FV_1.10", "FV_1.25", "FV_1.40", "FV_1.60", "FV_1.80", "FV_2.00", "FV_2.50", "FV_3.00", "MAX_SAFE"]


def seeded_net(seed):
    torch.manual_seed(init_seed(seed))
    return make_net()


def cfg(**kw):
    return {**DEFAULTS, **kw}


def transition(i, rng=None, done=False):
    rng = rng or np.random.default_rng(i)
    m = rng.random(20) < 0.5
    m[0] = True
    nm = rng.random(20) < 0.5
    nm[0] = True
    t = {"obs": rng.standard_normal(80).astype(np.float32), "mask": m, "action": int(np.flatnonzero(m)[-1]), "ret": float(rng.random()),
         "next_obs": rng.standard_normal(80).astype(np.float32), "next_mask": nm, "steps": 16, "mult": 1.0, "done": False, "tag": i}
    if done:
        t.update(done=True, mult=0.0, steps=3, next_mask=np.eye(20, dtype=bool)[0], next_obs=np.zeros(80, np.float32))
    return t


def shaped_net(means_bias=None, seed=1):
    """A net whose quantiles ignore the input (zero head weights) and equal a given [20, 32] bias."""
    n = seeded_net(seed)
    with torch.no_grad():
        n.head.weight.zero_()
        n.head.bias.copy_(torch.as_tensor(means_bias, dtype=torch.float32).reshape(-1))
    return n


def probe(entry, actions):
    """Independent JavaScript replay (fresh process): ground truth per decision."""
    out = subprocess.run(["node", str(HERE / "replay_probe.mjs")], input=json.dumps({"entry": entry, "actions": actions}),
                         capture_output=True, text=True, encoding="utf-8", check=True)
    return json.loads(out.stdout)


# ── network, quantiles, action selection ──────────────────────────────────
@unittest.skipUnless(HAVE_TORCH, "torch/numpy/gymnasium not installed")
class NetworkAndSelection(unittest.TestCase):
    def test_output_640_reshaped_20_by_32(self):
        net = seeded_net(1)
        self.assertEqual(net.head.out_features, 640)
        self.assertEqual([(m.in_features, m.out_features) for m in net.body if isinstance(m, torch.nn.Linear)], [(80, 128), (128, 128)])
        self.assertTrue(all(isinstance(m, torch.nn.Tanh) for m in list(net.body)[1::2]))
        z = net(torch.randn(7, 80))
        self.assertEqual(tuple(z.shape), (7, 20, 32))
        self.assertTrue(torch.all(torch.isfinite(z)))
        flat = net.head(net.body(torch.randn(3, 80)))
        self.assertTrue(torch.equal(flat.view(3, 20, 32)[:, 5, 7], flat[:, 5 * 32 + 7]), "action-major layout a*32+i")
        self.assertFalse(hasattr(net, "value") or hasattr(net, "advantage"), "standard QR-DQN: no dueling streams")

    def test_quantile_fractions(self):
        self.assertEqual(len(TAUS), 32)
        self.assertEqual(float(TAUS[0]), 0.015625)
        self.assertEqual(float(TAUS[-1]), 0.984375)
        self.assertTrue(torch.allclose(TAUS, (torch.arange(1, 33, dtype=torch.float64) - 0.5) / 32))
        mid = lambda ix: sum(float(TAUS[i]) for i in ix) / len(ix)
        self.assertEqual({k: mid(v) for k, v in LEVELS.items()}, {"q01": 0.015625, "q25": 0.25, "q50": 0.5, "q75": 0.75, "q99": 0.984375})

    def test_orthogonal_init_seeds_and_separation(self):
        net = seeded_net(2)
        w = net.body[2].weight.detach()
        self.assertTrue(torch.allclose(w @ w.T, 2.0 * torch.eye(128), atol=1e-4))
        h = net.head.weight.detach()                   # 640 × 128, gain 1 → orthonormal columns
        self.assertTrue(torch.allclose(h.T @ h, torch.eye(128), atol=1e-4))
        self.assertTrue(all(torch.all(m.bias == 0) for m in net.modules() if isinstance(m, torch.nn.Linear)))
        digests = {s: params_digest(seeded_net(s)) for s in (1, 2, 3)}
        self.assertEqual(digests, {s: params_digest(seeded_net(s)) for s in (1, 2, 3)})
        self.assertEqual(len(set(digests.values())), 3)
        from ipl_rl.algos import d3qn
        self.assertNotIn(init_seed(1), {d3qn.init_seed(1), qr_dqn.init_seed(2)}, "QR-DQN-specific initialisation seed")
        target = copy.deepcopy(net)
        self.assertEqual(params_digest(net), params_digest(target))
        self.assertFalse({id(p) for p in net.parameters()} & {id(p) for p in target.parameters()})

    def test_greedy_uses_the_masked_mean_not_a_single_quantile(self):
        z = torch.zeros(20, 32)
        z[4] = torch.linspace(-10, 3, 32)              # best median/upper quantiles, mean −3.5
        z[9] = torch.full((32,), 1.0)                 # best MEAN (1.0)
        z[12] = torch.full((32,), 0.5)
        net = shaped_net(z)
        mask = np.ones(20, bool)
        gen = np.random.default_rng(0)
        self.assertEqual(select_action(net, np.zeros(80, np.float32), mask, 0.0, gen)[0], 9)
        m2 = mask.copy()
        m2[9] = False
        self.assertEqual(select_action(net, np.zeros(80, np.float32), m2, 0.0, gen)[0], 12, "best LEGAL mean")
        self.assertEqual(select_action(net, np.zeros(80, np.float32), np.eye(20, dtype=bool)[0], 0.0, gen)[0], 0, "PASS-only mask")
        self.assertEqual(select_action(net, np.zeros(80, np.float32), np.eye(20, dtype=bool)[17], 0.0, gen)[0], 17, "single-action mask")
        self.assertTrue(torch.equal(MeanQ(net)(torch.zeros(2, 80)), net(torch.zeros(2, 80)).mean(-1)))

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
                        q = net(torch.from_numpy(obs[i:i + 1])).mean(-1)[0]
                    self.assertEqual(a, int(masked_argmax(q, torch.from_numpy(masks[i]))))

    def test_epsilon_schedule(self):
        c = cfg()
        self.assertEqual(epsilon_at(0, c), 1.0)
        self.assertAlmostEqual(epsilon_at(500_000, c), 0.525, places=12)
        self.assertEqual(epsilon_at(1_000_000, c), 0.05)
        self.assertEqual(epsilon_at(1_000_001, c), 0.05)
        self.assertEqual(epsilon_at(1_996_799, c), 0.05)
        self.assertEqual(beta_at(0, c), 0.4)
        self.assertEqual(beta_at(c["total_decisions"], c), 1.0)


# ── quantile-Huber loss ────────────────────────────────────────────────────
@unittest.skipUnless(HAVE_TORCH, "torch/numpy/gymnasium not installed")
class QuantileHuberLoss(unittest.TestCase):
    def one(self, theta, target, tau, kappa=1.0):
        return float(quantile_huber_loss(torch.tensor([[theta]], dtype=torch.float64), torch.tensor([[target]], dtype=torch.float64),
                                         torch.tensor([tau], dtype=torch.float64), kappa)[0])

    def test_hand_computed_pairs(self):
        self.assertAlmostEqual(self.one(0.0, 0.5, 0.5), 0.5 * 0.5 * 0.25)           # δ = +0.5: weight τ, H = ½δ²
        self.assertAlmostEqual(self.one(0.0, -0.5, 0.25), 0.75 * 0.5 * 0.25)        # δ = −0.5: weight 1 − τ
        self.assertEqual(self.one(1.3, 1.3, 0.7), 0.0)                               # δ = 0
        self.assertAlmostEqual(self.one(0.0, 3.0, 0.2), 0.2 * (3.0 - 0.5))           # |δ| > κ: linear region
        self.assertAlmostEqual(self.one(0.0, -3.0, 0.2), 0.8 * (3.0 - 0.5))
        self.assertAlmostEqual(self.one(0.0, 1.0, 0.9), 0.9 * 0.5)                   # |δ| = κ boundary: ½κ² = κ(|δ| − ½κ)
        self.assertAlmostEqual(self.one(0.0, 2.0, 0.5, kappa=2.0), 0.5 * 0.5 * 4.0 / 2.0)  # κ scaling: H/κ
        self.assertAlmostEqual(self.one(0.0, 0.5, 0.9), 0.9 * 0.125)                 # asymmetric weighting
        self.assertAlmostEqual(self.one(0.0, -0.5, 0.9), 0.1 * 0.125)

    def test_32_by_32_pairwise_matches_brute_force(self):
        g = torch.Generator().manual_seed(4)
        theta = torch.randn(3, 32, generator=g, dtype=torch.float64) * 2
        target = torch.randn(3, 32, generator=g, dtype=torch.float64) * 2
        got = quantile_huber_loss(theta, target, TAUS, 1.0)
        for b in range(3):
            tot = 0.0
            for i in range(32):
                for j in range(32):
                    d = float(target[b, j] - theta[b, i])
                    h = 0.5 * d * d if abs(d) <= 1 else abs(d) - 0.5
                    tot += abs(float(TAUS[i]) - (1.0 if d < 0 else 0.0)) * h
            self.assertAlmostEqual(float(got[b]), tot / (32 * 32), places=12)

    def test_minimiser_is_the_tau_quantile_not_the_mean(self):
        """Not an ordinary TD loss: a lone θ trained against samples 0..99 goes to
        their τ-quantile (a squared / TD loss would go to the mean 49.5)."""
        target = torch.arange(100, dtype=torch.float64).unsqueeze(0)
        for tau, expect in ((0.25, 24.75), (0.9, 89.1)):
            th = torch.zeros(1, 1, dtype=torch.float64, requires_grad=True)
            opt = torch.optim.SGD([th], lr=50.0)
            for _ in range(4000):
                opt.zero_grad()
                quantile_huber_loss(th, target, torch.tensor([tau], dtype=torch.float64), 1.0).sum().backward()
                opt.step()
            self.assertLess(abs(th.item() - expect), 1.5, (tau, th.item()))


# ── Double DQN on distributions ────────────────────────────────────────────
@unittest.skipUnless(HAVE_TORCH, "torch/numpy/gymnasium not installed")
class DoubleDqnTargets(unittest.TestCase):
    def pair(self):
        zo = torch.zeros(20, 32)
        zo[3] = 5.0                                     # online prefers action 3
        zo[7] = 1.0
        zt = torch.zeros(20, 32)
        zt[7] = torch.linspace(8, 9, 32)                # target prefers action 7
        zt[3] = torch.linspace(-1, 2, 32)               # the distribution that must be used
        return shaped_net(zo), shaped_net(zt)

    def test_online_selects_target_evaluates(self):
        online, target = self.pair()
        s2 = torch.randn(4, 80)
        ret = torch.tensor([0.1, 0.2, 0.3, 0.4])
        y, a_star, _, z_tg_star, boot = qr_targets(online, target, ret, s2, torch.ones(4, 20, dtype=torch.bool), torch.ones(4))
        self.assertTrue(torch.all(a_star == 3))
        expect = ret.unsqueeze(1) + torch.linspace(-1, 2, 32).unsqueeze(0)
        self.assertTrue(torch.allclose(y, expect, atol=1e-6))
        self.assertFalse(torch.allclose(y, ret.unsqueeze(1) + torch.linspace(8, 9, 32)), "the target network must not choose")
        m = torch.ones(4, 20, dtype=torch.bool)
        m[:, 3] = False                                 # online favourite illegal next state → action 7 by the online mean
        _, a_star, _, _, _ = qr_targets(online, target, ret, s2, m, torch.ones(4))
        self.assertTrue(torch.all(a_star == 7))

    def test_terminal_target_is_the_return_for_every_quantile(self):
        online, target = self.pair()
        ret = torch.tensor([0.5, -0.25])
        y, _, _, _, boot = qr_targets(online, target, ret, torch.zeros(2, 80), torch.eye(20, dtype=torch.bool)[:1].repeat(2, 1), torch.zeros(2))
        self.assertTrue(torch.equal(y, ret.unsqueeze(1).expand(2, 32)))
        self.assertFalse(bool(boot.any()))


# ── update, priorities, target network ─────────────────────────────────────
@unittest.skipUnless(HAVE_TORCH, "torch/numpy/gymnasium not installed")
class UpdatePriorityTarget(unittest.TestCase):
    def replay(self, n=64, cap=64, seed=0, terminal=10):
        rep = PrioritizedReplay(cap, 0.6, 1e-6, np.random.default_rng(seed))
        rep.add_many([transition(i, done=i < terminal) for i in range(n)])
        return rep

    def test_update_objective_priorities_one_step_clip_and_target_untouched(self):
        online = seeded_net(3)
        target = copy.deepcopy(online)
        with torch.no_grad():
            for p in target.parameters():
                p.add_(0.05)
        rep = self.replay()
        idx, w, _ = rep.sample(64, 0.4)
        b = rep.batch(idx)
        with torch.no_grad():
            y, _, _, _, _ = qr_targets(online, target, torch.from_numpy(b["ret"]).float(), torch.from_numpy(b["next_obs"]),
                                       torch.from_numpy(b["next_mask"]), torch.from_numpy(b["mult"]).float())
            th = online(torch.from_numpy(b["obs"]))[torch.arange(64), torch.from_numpy(b["action"])]
            manual = (torch.from_numpy(w).float() * quantile_huber_loss(th, y, TAUS, 1.0)).mean().item()
            manual_pri = (y - th).abs().mean(1).numpy()
        tgt_before = params_digest(target)
        opt = torch.optim.Adam(online.parameters(), lr=3e-4, eps=1e-5)
        stats, pri = qr_update(online, target, opt, b, w, cfg())
        self.assertAlmostEqual(stats["loss"], manual, places=5)
        self.assertTrue(np.allclose(pri, manual_pri, atol=1e-6), "priority = mean_j |y_j − θ_j|")
        self.assertEqual({int(s["step"]) for s in opt.state.values()}, {1})
        self.assertEqual(params_digest(target), tgt_before)
        rep.update_priorities(idx, pri)
        stored = rep.tree.leaves(idx) ** (1 / 0.6)
        self.assertTrue(np.allclose(stored, pri + 1e-6, rtol=1e-9), "stored p = |δ| + ε (duplicates: identical values)")
        self.assertLess(rep.tree.check(), 1e-12)
        self.assertEqual(rep.audit(1.0, 16), [])
        big = qr_update(seeded_net(3), target, torch.optim.Adam(seeded_net(3).parameters(), lr=3e-4), b, w * 1e6, cfg())[0]
        self.assertGreater(big["grad_norm"], 0.5)
        o2 = seeded_net(3)
        opt2 = torch.optim.Adam(o2.parameters(), lr=3e-4, eps=1e-5)
        qr_update(o2, target, opt2, b, w * 1e6, cfg())
        clipped = math.sqrt(sum(float(p.grad.norm() ** 2) for p in o2.parameters()))
        self.assertLessEqual(clipped, 0.5 + 1e-5)
        for k in ("z_q01", "z_q25", "z_q50", "z_q75", "z_q99", "z_spread", "z_iqr", "crossing_rate", "crossing_size", "target_distance"):
            self.assertTrue(math.isfinite(stats[k]), k)

    def test_per_sampling_weights_beta_and_determinism(self):
        rep = self.replay(16, 16, terminal=0)
        td = np.linspace(0.0, 2.0, 16)
        rep.update_priorities(np.arange(16), td)
        p = td + 1e-6
        P = p ** 0.6 / (p ** 0.6).sum()
        idx, w, prob = rep.sample(8, 0.55)
        self.assertTrue(np.allclose(prob, P[idx]))
        raw = (16 * P[idx]) ** -0.55
        self.assertTrue(np.allclose(w, raw / raw.max()))
        counts = np.zeros(16)
        for _ in range(3000):
            np.add.at(counts, rep.sample(32, 0.4)[0], 1)
        self.assertLess(np.abs(counts / counts.sum() - P).max(), 0.01)
        a, b = self.replay(seed=9), self.replay(seed=9)
        for _ in range(4):
            self.assertTrue(np.array_equal(a.sample(32, 0.4)[0], b.sample(32, 0.4)[0]))

    def test_target_frozen_between_syncs_and_copied_exactly(self):
        online = seeded_net(2)
        target = copy.deepcopy(online)
        rep = self.replay(128, 128)
        opt = torch.optim.Adam(online.parameters(), lr=1e-3, eps=1e-5)
        frozen = params_digest(target)
        syncs = []
        for u in range(1, 13):
            idx, w, _ = rep.sample(32, 0.4)
            _, pri = qr_update(online, target, opt, rep.batch(idx), w, cfg())
            rep.update_priorities(idx, pri)
            d = sync_target(u, online, target, 5)
            if d is None:
                self.assertEqual(params_digest(target), frozen)
            else:
                syncs.append(u)
                self.assertEqual(d, params_digest(online))
                frozen = d
        self.assertEqual(syncs, [5, 10])

    def test_alarms_and_config(self):
        ok = {"loss": 0.1, "td_abs_mean": 0.2, "q_mean": 3.0, "q_min": -1.0, "q_max": 9.0, "z_legal_absmax": 9.5, "z_next_absmax": 9.5,
              "y_absmax": 9.5, "theta_absmax": 9.5, "grad_norm": 2.0, "target_distance": 0.3, "grads_finite": True, "params_finite": True, "values_finite": True}
        self.assertEqual(update_problems(ok), [])
        for bad in ({"loss": float("nan")}, {"values_finite": False}, {"grads_finite": False}, {"params_finite": False}, {"grad_norm": 150.0},
                    {"z_legal_absmax": 60.0}, {"y_absmax": 51.0}, {"target_distance": 6.0}):
            self.assertTrue(update_problems({**ok, **bad}), bad)
        for key in ("clip_coef", "ent_coef", "vf_coef", "gae_lambda", "norm_adv", "update_epochs", "minibatch_size", "target_kl", "num_steps", "tau", "huber_delta"):
            with self.assertRaises(KeyError):
                load_config(None, DEFAULTS, {key: 1})
        c = load_config(Path(qr_dqn.__file__).parents[1] / "configs" / "qr_dqn_2d3.json", DEFAULTS)
        frozen = {"algorithm": "qr_dqn", "num_envs": 12, "gamma": 1.0, "n_step": 16, "n_quantiles": 32, "huber_kappa": 1.0, "replay_capacity": 500_000,
                  "learning_starts": 10_000, "batch_size": 256, "per_alpha": 0.6, "per_beta_start": 0.4, "per_beta_final": 1.0, "per_eps": 1e-6,
                  "learning_rate": 3e-4, "adam_eps": 1e-5, "max_grad_norm": 0.5, "target_update_every": 2_500, "eps_start": 1.0, "eps_final": 0.05,
                  "eps_decay_decisions": 1_000_000, "total_decisions": 1_996_800, "actor_lag_cycles": 64, "stage": "A", "expected_act_spec": ACT_V3["hash"],
                  "eval_decisions": [245_760, 497_664, 749_568, 1_001_472, 1_247_232, 1_499_136, 1_751_040, 1_996_800]}
        self.assertEqual({k: c[k] for k in frozen}, frozen)


# ── n-step: synthetic and real JavaScript episodes ─────────────────────────
@unittest.skipUnless(HAVE_TORCH, "torch/numpy/gymnasium not installed")
class NStepRealEnvironment(unittest.TestCase):
    def synthetic(self, T):
        rng = np.random.default_rng(T)
        obs = rng.standard_normal((T + 1, 80)).astype(np.float32)
        masks = rng.random((T + 1, 20)) < 0.5
        masks[:, 0] = True
        r = rng.random(T)
        b = NStepBuilder(16, 1.0)
        out = []
        for t in range(T):
            done = t == T - 1
            out += b.push(obs[t], masks[t], 0, r[t], None if done else obs[t + 1], None if done else masks[t + 1], done, tag=t)
        return obs, masks, r, out, b

    def test_synthetic_16_step_short_terminal_one_step_and_isolation(self):
        obs, masks, r, out, _ = self.synthetic(40)
        self.assertAlmostEqual(out[0]["ret"], float(r[:16].sum()), places=12)
        self.assertTrue(np.array_equal(out[0]["next_mask"], masks[16]) and np.array_equal(out[0]["next_obs"], obs[16]))
        self.assertEqual((out[0]["steps"], out[0]["mult"], out[0]["done"]), (16, 1.0, False))
        self.assertEqual((out[-1]["steps"], out[-1]["mult"], out[-1]["done"]), (1, 0.0, True))       # 1-step terminal
        self.assertAlmostEqual(out[-1]["ret"], float(r[-1]))
        _, _, r5, out5, b5 = self.synthetic(5)                                                       # shorter than 16
        self.assertEqual([t["steps"] for t in out5], [5, 4, 3, 2, 1])
        self.assertTrue(all(t["done"] and t["mult"] == 0.0 for t in out5))
        self.assertEqual(len(b5.queue), 0)
        z = np.zeros(80, np.float32)
        m = np.eye(20, dtype=bool)[0]
        b = NStepBuilder(16, 1.0)
        for t in range(3):
            b.push(z, m, 0, 100.0, z, m, t == 2)
        nxt = []
        for t in range(20):
            nxt += b.push(z, m, 0, 1.0, z, m, False)
        self.assertEqual(nxt[0]["ret"], 16.0, "no reward leaks across the episode boundary")

    def run_python(self, env, entry, choose):
        o, _ = env.reset(options={"entry": entry})
        states, masks, actions, rewards = [o], [env.action_masks()], [], []
        b = NStepBuilder(16, 1.0)
        out = []
        t = 0
        while True:
            a = choose(t, o, masks[-1])
            o2, r, done, _, _ = env.step(a)
            m2 = None if done else env.action_masks()
            out += b.push(states[-1], masks[-1], a, r, None if done else o2, m2, done, tag=t)
            actions.append(a)
            rewards.append(r)
            t += 1
            if done:
                return out, actions
            o = o2
            states.append(o2)
            masks.append(m2)

    def test_real_transitions_match_an_independent_javascript_replay(self):
        """Every stored transition — s_t, mask_t, a_t, G_t, s_t+16, mask_t+16, terminal —
        equals an independent JavaScript replay of the same episode, over main-auction,
        re-auction, low- and high-purse, keeper / Indian / bowling-need and shield-forced
        states (and final-path states when the fixture episode has them)."""
        env = IplAuctionEnv(split="train")
        episodes = []
        try:
            with BridgeV2() as b:
                entries = [b.call("reset", seed=s)["info"]["entry"] for s in (1_000_401, 1_000_402, 1_000_403)]
            passive = lambda t, o, m: int(np.flatnonzero(m)[0])                       # PASS, cheapest when forced → long, re-auction
            aggressive = lambda t, o, m: int(np.flatnonzero(m)[-1])                  # MAX_SAFE → low purse quickly
            mixed = lambda t, o, m: int(np.flatnonzero(m)[(5 * t) % int(m.sum())])
            for entry, choose in zip(entries, (passive, aggressive, mixed)):
                out, actions = self.run_python(env, entry, choose)
                episodes.append((entry, actions, out))
            fixture = HERE / "fixtures" / "final_path_episode.json"
            if fixture.exists():
                fx = json.loads(fixture.read_text())
                queue = list(fx["actions"])
                out, actions = self.run_python(env, fx["entry"], lambda t, o, m: queue[t])
                self.assertEqual(actions, fx["actions"])
                episodes.append((fx["entry"], actions, out))
        finally:
            env.close()
        cover = {"main": 0, "reauction": 0, "lowPurse": 0, "highPurse": 0, "keeperNeed": 0, "indianNeed": 0, "bowlingNeed": 0, "forced": 0, "finalPath": 0}
        checked = 0
        for entry, actions, out in episodes:
            js = probe(entry, actions)
            T = len(actions)
            self.assertEqual(len(js["rewards"]), T)
            self.assertEqual(js["dones"], [False] * (T - 1) + [True])
            self.assertEqual(len(out), T)
            for tr in out:
                t = tr["tag"]
                self.assertTrue(np.array_equal(tr["obs"], np.asarray(js["obs"][t], np.float32)), f"obs t={t}")
                self.assertTrue(np.array_equal(tr["mask"], np.asarray(js["masks"][t], bool)), f"mask t={t}")
                self.assertEqual(tr["action"], actions[t])
                self.assertTrue(tr["mask"][tr["action"]])
                k = min(16, T - t)
                self.assertAlmostEqual(tr["ret"], float(sum(js["rewards"][t:t + k])), places=12)
                if t + 16 < T:
                    self.assertEqual((tr["steps"], tr["mult"], tr["done"]), (16, 1.0, False))
                    self.assertTrue(np.array_equal(tr["next_obs"], np.asarray(js["obs"][t + 16], np.float32)), f"obs t+16 at t={t}")
                    self.assertTrue(np.array_equal(tr["next_mask"], np.asarray(js["masks"][t + 16], bool)), f"mask t+16 at t={t}")
                else:
                    self.assertEqual((tr["steps"], tr["mult"], tr["done"]), (T - t, 0.0, True))
                s = js["states"][t]
                cover["main"] += s["phase"] == "main"
                cover["reauction"] += s["phase"] == "reauction"
                cover["lowPurse"] += s["purse"] <= 200
                cover["highPurse"] += s["purse"] >= 9000
                cover["keeperNeed"] += s["need"]["keeper"] > 0
                cover["indianNeed"] += s["need"]["indians"] > 0
                cover["bowlingNeed"] += s["need"]["bowling"] > 0
                cover["forced"] += s["forced"]
                cover["finalPath"] += s["finalPath"]
                checked += 1
        for k, v in cover.items():
            if k != "finalPath" or (HERE / "fixtures" / "final_path_episode.json").exists():
                self.assertGreater(v, 0, f"no {k} state covered")
        print(f"\n[real-env n-step] {checked} transitions checked against the JS replay; coverage {cover}")
        self.__class__.coverage = cover


# ── specs, export parity, deterministic evaluation ─────────────────────────
@unittest.skipUnless(HAVE_TORCH, "torch/numpy/gymnasium not installed")
class SpecsExportEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        obs, masks = [], []
        with BridgeV2() as b:
            cls.info = b.info
            for seed, pick in ((1_000_097, 0), (1_000_402, -1)):             # a passive (re-auction) and an aggressive (low purse) episode
                r = b.call("reset", seed=seed)
                while not r.get("done"):
                    obs.append(r["obs"])
                    masks.append(r["mask"])
                    legal = [a for a, ok in enumerate(r["mask"]) if ok]
                    r = b.call("step", action=legal[pick])
        cls.obs, cls.masks = np.asarray(obs, np.float32), np.asarray(masks, bool)

    def test_act_v3_alignment(self):
        act = self.info["actSpec"]
        self.assertEqual({k: act[k] for k in ("version", "hash", "count")}, ACT_V3)
        self.assertEqual(list(act["actions"]), ACTION_NAMES)
        self.assertEqual({k: self.info["obsSpec"][k] for k in ("version", "hash", "size")}, OBS_V2)
        self.assertEqual(self.info["gamma"], 1)

    def test_export_contract_and_parity_of_all_640_outputs(self):
        net = seeded_net(1)
        with torch.no_grad():
            for p in net.parameters():
                p.add_(0.05 * torch.randn_like(p))
        pol = to_policy_json(net, OBS_V2, ACT_V3)
        self.assertEqual(pol["algorithm"], "qrdqn")               # the frozen production identifier
        self.assertEqual(pol["meta"]["algorithmName"], "qr_dqn")
        self.assertEqual(pol["architecture"], {"input": 80, "hidden": [128, 128], "activation": "tanh", "head": "quantiles", "actions": 20, "quantiles": 32})
        self.assertEqual(pol["selection"], {"mode": "argmax"})
        self.assertEqual(pol["meta"]["quantileFractions"], [(i - 0.5) / 32 for i in range(1, 33)])
        self.assertEqual(len(pol["layers"][-1]["weight"]), 640)
        reauc, low, high = self.obs[:, 0] > 0.5, self.obs[:, 15] < 0.05, self.obs[:, 15] > 0.9
        self.assertTrue(reauc.any() and low.any() and high.any(), "re-auction, low- and high-purse states present")
        extreme = self.obs[:4].copy()
        extreme[:2, 15], extreme[2:, 15] = 0.0, 1.5
        par = qr_dqn.parity(net, pol, np.concatenate([self.obs, extreme]), np.concatenate([self.masks, self.masks[:4]]))
        self.assertLess(par["maxAbsRawDiff"], 1e-9)
        self.assertLess(par["maxAbsScoreDiff"], 1e-9)
        self.assertEqual(par["byMask"], {"real": 1.0, "passOnly": 1.0, "single": 1.0, "allLegal": 1.0})

    def test_deterministic_evaluation(self):
        from ipl_rl.common.evaluation import node_evaluate
        from ipl_rl.export import write_policy
        net = seeded_net(2)
        with tempfile.TemporaryDirectory() as tmp:
            p = write_policy(Path(tmp) / "policy.json", to_policy_json(net, OBS_V2, ACT_V3))
            runs = [node_evaluate(p, Path(tmp) / f"eval{k}", limit=3, workers=3)[1]["episodes"]["policy:qrdqn"] for k in range(2)]
        strip = lambda rows: [{k: v for k, v in r.items() if k not in ("ms", "seconds", "wallMs")} for r in rows]
        self.assertEqual(strip(runs[0]), strip(runs[1]))
        self.assertTrue(all(r["legalXI"] for r in runs[0]))


if __name__ == "__main__":
    unittest.main()
