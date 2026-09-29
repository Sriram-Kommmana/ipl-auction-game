"""Phase 2D.4 — OpenAI-ES pre-flight: ES mathematics (hand-computed), the
80→64→64→20 policy, masked categorical sampling and masked-argmax export,
real JavaScript episodes, common random numbers for antithetic pairs,
scheduling independence, seeds, and the rl-policy-v2 production contract.
No training run here."""

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

HAVE_TORCH = all(importlib.util.find_spec(m) for m in ("torch", "numpy", "gymnasium"))

if HAVE_TORCH:
    import numpy as np
    import torch
    from torch.nn.utils import parameters_to_vector

    from ipl_rl.algos import openai_es as es
    from ipl_rl.algos.openai_es import (DEFAULTS, EpisodeRunner, FlatPolicy, centered_ranks, es_gradient, es_update, generation_jobs,
                                        make_policy, masked_argmax, masked_probs, noise, pair_episode_seed, params_digest, sample_action)
    from ipl_rl.bridge import BridgeV2
    from ipl_rl.common.config import load_config
    from ipl_rl.common.vec_env import VecIplAuctionEnv
    from ipl_rl.env import SPLIT_RANGES, IplAuctionEnv

HERE = Path(__file__).resolve().parent
ACT_V3 = {"version": "act-v3", "hash": "5f72f510c48b1f46", "count": 20}
OBS_V2 = {"version": "obs-v2", "hash": "629b25783f833af7", "size": 80}


def theta_of(net):
    return parameters_to_vector(net.parameters()).detach().numpy().astype(np.float32)


# ── ES mathematics ─────────────────────────────────────────────────────────
@unittest.skipUnless(HAVE_TORCH, "torch/numpy/gymnasium not installed")
class EsMathematics(unittest.TestCase):
    def test_deterministic_gaussian_noise(self):
        a = noise(1, 7, 3, 10_000)
        self.assertEqual(a.dtype, np.float32)
        self.assertTrue(np.array_equal(a, noise(1, 7, 3, 10_000)))
        for other in (noise(1, 7, 4, 10_000), noise(1, 8, 3, 10_000), noise(2, 7, 3, 10_000)):
            self.assertFalse(np.array_equal(a, other))
        self.assertLess(abs(float(a.mean())), 0.05)
        self.assertLess(abs(float(a.std()) - 1.0), 0.05)

    def test_antithetic_pairs_and_common_random_numbers(self):
        theta = theta_of(make_policy(1))
        jobs, eps = generation_jobs(1, 5, theta, 0.05, 32, len(theta))
        self.assertEqual(len(jobs), 64)
        self.assertEqual(eps.shape, (32, len(theta)))
        for i in range(32):
            plus, minus = jobs[2 * i], jobs[2 * i + 1]
            self.assertEqual((plus["key"], minus["key"]), ((i, +1), (i, -1)))
            self.assertEqual(plus["seed"], minus["seed"], "seed_plus == seed_minus")
            self.assertEqual(plus["act_seed"], minus["act_seed"], "same action stream")
            self.assertTrue(np.allclose(plus["theta"] - theta, 0.05 * eps[i], atol=1e-6))
            self.assertTrue(np.allclose(minus["theta"] - theta, -0.05 * eps[i], atol=1e-6))
            self.assertTrue(np.allclose(plus["theta"] + minus["theta"], 2 * theta, atol=1e-6), "+ε / −ε symmetric about θ")
        self.assertEqual(len({jobs[2 * i]["seed"] for i in range(32)}), 32, "every pair plays a different auction")

    def test_centered_ranks_hand_computed(self):
        self.assertTrue(np.allclose(centered_ranks([3.0, 1.0, 2.0]), [0.5, -0.5, 0.0]))
        self.assertTrue(np.allclose(centered_ranks([1.0, 1.0, 2.0, 0.0]), [0.0, 0.0, 0.5, -0.5]))   # ties → average rank
        self.assertTrue(np.allclose(centered_ranks([10, 20, 30, 40, 50]), [-0.5, -0.25, 0.0, 0.25, 0.5]))
        x = np.random.default_rng(0).standard_normal(64)
        s = centered_ranks(x)
        self.assertAlmostEqual(float(s.sum()), 0.0, places=12)
        self.assertEqual((float(s.min()), float(s.max())), (-0.5, 0.5))
        self.assertTrue(np.array_equal(centered_ranks(np.exp(x)), s), "invariant to monotone transforms")
        self.assertTrue(np.allclose(centered_ranks([2.0, 2.0, 2.0]), 0.0))

    def test_gradient_estimator_hand_calculation(self):
        eps = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]])
        g = es_gradient([0.5, -0.1], [-0.5, 0.1], eps, 0.5)       # Σ(s+ − s−)ε = [1, −0.2, 2.2]; /(2·2·0.5)
        self.assertTrue(np.allclose(g, [0.5, -0.1, 1.1]))
        self.assertTrue(np.allclose(es_gradient([0.5, -0.1], [-0.5, 0.1], eps, 0.25), 2 * g), "∝ 1/σ")
        g2 = es_gradient([0.5, -0.1, 0.5, -0.1], [-0.5, 0.1, -0.5, 0.1], np.concatenate([eps, eps]), 0.5)
        self.assertTrue(np.allclose(g2, g), "population averaging: duplicated pairs leave g unchanged")
        self.assertTrue(np.all(es_gradient([0.3, 0.1], [0.3, 0.1], eps, 0.5) == 0), "zero difference → zero gradient")
        # equal + and − fitness → equal shaped values (average-rank ties) → zero contribution
        fit = np.array([1.0, 1.0, 3.0, 2.0])                       # pair 0 tied, pair 1 not
        s = centered_ranks(fit).reshape(2, 2)
        self.assertEqual(s[0, 0], s[0, 1])
        only_pair1 = es_gradient(s[:, 0], s[:, 1], eps, 0.5)
        self.assertTrue(np.allclose(only_pair1, (s[1, 0] - s[1, 1]) * eps[1] / 2.0))

    def test_estimator_recovers_a_linear_gradient_with_raw_differences(self):
        rng = np.random.default_rng(3)
        c = rng.standard_normal(5)
        eps = rng.standard_normal((20_000, 5))
        sigma = 0.05
        f_plus, f_minus = (0.7 + (sigma * eps) @ c), (0.7 - (sigma * eps) @ c)
        g = es_gradient(f_plus, f_minus, eps, sigma)
        se = np.linalg.norm(c) / np.sqrt(len(eps))                 # Monte-Carlo standard error per coordinate (≈ 0.024)
        self.assertLess(np.abs(g - c).max(), 4 * se)
        self.assertGreater(float(g @ c / (np.linalg.norm(g) * np.linalg.norm(c))), 0.999)

    def test_adam_ascent_step(self):
        net = make_policy(2)
        opt = torch.optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
        before = theta_of(net).astype(np.float64)
        g = np.random.default_rng(1).standard_normal(len(before))
        step = es_update(net, opt, g)
        expect = 0.01 * g / (np.abs(g) + 1e-8)                     # first Adam step, bias-corrected: lr·ĝ/(|ĝ|+eps); ascent
        self.assertTrue(np.allclose(step, expect, atol=1e-6))
        self.assertTrue(np.allclose(theta_of(net) - before, expect, atol=1e-6))
        self.assertEqual({int(s["step"]) for s in opt.state.values()}, {1})
        zero = make_policy(2)
        zopt = torch.optim.Adam(zero.parameters(), lr=0.01, eps=1e-8)
        d0 = params_digest(zero)
        es_update(zero, zopt, np.zeros(len(before)))
        self.assertEqual(params_digest(zero), d0, "zero ES gradient → no parameter change")

    def test_config_frozen_and_es_only(self):
        for key in ("clip_coef", "ent_coef", "vf_coef", "gae_lambda", "replay_capacity", "target_update_every", "n_step", "n_quantiles", "elite", "crossover"):
            with self.assertRaises(KeyError):
                load_config(None, DEFAULTS, {key: 1})
        c = load_config(Path(es.__file__).parents[1] / "configs" / "openai_es_2d4.json", DEFAULTS)
        frozen = {"algorithm": "openai_es", "generations": 2000, "pairs": 32, "episodes_per_perturbation": 1, "sigma": 0.05, "learning_rate": 0.01,
                  "adam_eps": 1e-8, "weight_decay": 0.0, "fitness_shaping": "centered_rank", "num_envs": 12, "stage": "A",
                  "expected_act_spec": ACT_V3["hash"], "expected_obs_spec": OBS_V2["hash"],
                  "decision_checkpoints": [245_760, 497_664, 749_568, 1_001_472, 1_247_232, 1_499_136, 1_751_040, 1_996_800]}
        self.assertEqual({k: c[k] for k in frozen}, frozen)


# ── policy ─────────────────────────────────────────────────────────────────
@unittest.skipUnless(HAVE_TORCH, "torch/numpy/gymnasium not installed")
class Policy(unittest.TestCase):
    def test_network_shape_init_and_seeds(self):
        net = make_policy(1)
        sizes = [(m.in_features, m.out_features) for m in net.body if isinstance(m, torch.nn.Linear)] + [(64, net.head.out_features)]
        self.assertEqual(sizes, [(80, 64), (64, 64), (64, 20)])
        self.assertTrue(all(isinstance(m, torch.nn.Tanh) for m in list(net.body)[1::2]))
        self.assertFalse(hasattr(net, "value") or hasattr(net, "critic"), "no value head")
        self.assertEqual(tuple(net(torch.zeros(3, 80)).shape), (3, 20))
        self.assertEqual(sum(p.numel() for p in net.parameters()), 10_644)
        d = {s: params_digest(make_policy(s)) for s in (1, 2, 3)}
        self.assertEqual(d, {s: params_digest(make_policy(s)) for s in (1, 2, 3)})
        self.assertEqual(len(set(d.values())), 3)
        w = net.body[2].weight.detach()
        self.assertTrue(torch.allclose(w @ w.T, 2.0 * torch.eye(64), atol=1e-4))

    def test_flat_forward_equals_the_network(self):
        net = make_policy(3)
        with torch.no_grad():
            for p in net.parameters():
                p.add_(0.1 * torch.randn_like(p))
        x = np.random.default_rng(0).standard_normal((20, 80)).astype(np.float32)
        fp = FlatPolicy(theta_of(net))
        with torch.no_grad():
            ref = net(torch.from_numpy(x)).numpy()
        self.assertLess(np.abs(np.stack([fp.logits(o) for o in x]) - ref).max(), 1e-5)

    def test_masked_categorical_sampling(self):
        rng = np.random.default_rng(0)
        logits = rng.standard_normal(20) * 3
        mask = np.zeros(20, bool)
        mask[[0, 3, 7, 19]] = True
        p = masked_probs(logits, mask)
        self.assertTrue(np.all(p[~mask] == 0))
        self.assertAlmostEqual(float(p.sum()), 1.0, places=12)
        e = np.exp(logits[mask] - logits[mask].max())
        self.assertTrue(np.allclose(p[mask], e / e.sum()), "renormalised over the legal actions")
        g = np.random.default_rng(5)
        picks = np.array([sample_action(logits, mask, g) for _ in range(20_000)])
        self.assertTrue(np.all(mask[picks]))
        self.assertLess(np.abs(np.bincount(picks, minlength=20) / len(picks) - p).max(), 0.01)
        for _ in range(500):                                  # random masks: never a masked action
            m = rng.random(20) < 0.3
            m[rng.integers(20)] = True
            self.assertTrue(m[sample_action(rng.standard_normal(20) * 5, m, g)])

    def test_masked_argmax_masks(self):
        logits = np.arange(20, dtype=np.float64)             # best raw = 19
        self.assertEqual(masked_argmax(logits, np.ones(20, bool)), 19)
        self.assertEqual(masked_argmax(logits, np.eye(20, dtype=bool)[0]), 0, "PASS-only")
        self.assertEqual(masked_argmax(logits, np.eye(20, dtype=bool)[6]), 6, "single action")
        m = np.ones(20, bool)
        m[19] = False
        self.assertEqual(masked_argmax(logits, m), 18)
        self.assertEqual(masked_argmax(np.full(20, 1e30), np.eye(20, dtype=bool)[4]), 4)


# ── real environment ───────────────────────────────────────────────────────
@unittest.skipUnless(HAVE_TORCH, "torch/numpy/gymnasium not installed")
class RealEnvironment(unittest.TestCase):
    def test_seed_schedule(self):
        lo, hi = SPLIT_RANGES["train"]
        seeds = [pair_episode_seed(s, g, i) for s in (1, 2) for g in range(1, 30) for i in range(32)]
        self.assertTrue(all(lo <= x <= hi for x in seeds), "training seeds only")
        self.assertFalse(any(100_000 <= x <= 100_499 or 200_000 <= x <= 200_999 for x in seeds), "no validation / test seeds")
        self.assertEqual(len(set(seeds)), len(seeds))
        self.assertEqual(seeds[:5], [pair_episode_seed(1, 1, i) for i in range(5)])

    def run_jobs(self, jobs, n_envs):
        venv = VecIplAuctionEnv(n_envs, 1)
        try:
            return EpisodeRunner(venv).run(jobs)
        finally:
            venv.close()

    def test_episodes_are_independent_of_scheduling_and_crn_isolates_parameters(self):
        net = make_policy(1)
        theta = theta_of(net)
        jobs, _ = generation_jobs(1, 3, theta, 0.05, 3, len(theta))
        a = self.run_jobs(jobs, 1)                              # one simulator, sequential
        b = self.run_jobs(list(reversed(jobs)), 3)              # three simulators, reversed order
        for k in a:
            self.assertEqual(a[k]["return"], b[k]["return"])
            self.assertTrue(np.array_equal(a[k]["actions"], b[k]["actions"]))
            self.assertEqual(a[k]["episode"]["xi"], b[k]["episode"]["xi"])
            self.assertTrue(a[k]["episode"]["legalXI"])
        # CRN: with σ = 0 the two members of a pair are the same policy on the same
        # auction with the same action stream → identical episodes.
        zero, _ = generation_jobs(1, 3, theta, 0.0, 2, len(theta))
        z = self.run_jobs(zero, 2)
        for i in range(2):
            self.assertEqual(z[(i, +1)]["return"], z[(i, -1)]["return"])
            self.assertTrue(np.array_equal(z[(i, +1)]["actions"], z[(i, -1)]["actions"]))
            self.assertEqual(z[(i, +1)]["episode"]["seed"], z[(i, -1)]["episode"]["seed"])

    def test_real_episode_obs_mask_reward_parity_with_javascript(self):
        """A sampled ES episode in the Python environment equals an independent
        JavaScript replay of the same actions (obs, masks, rewards, termination)."""
        fp = FlatPolicy(theta_of(make_policy(2)))
        rng = np.random.default_rng(9)
        env = IplAuctionEnv(split="train")
        try:
            o, info = env.reset(options={"episode_seed": pair_episode_seed(2, 1, 0)})
            obs, masks, actions, rewards = [o], [env.action_masks()], [], []
            while True:
                a = sample_action(fp.logits(o), masks[-1], rng)
                self.assertTrue(masks[-1][a])
                o, r, done, _, _ = env.step(a)
                actions.append(a)
                rewards.append(r)
                if done:
                    break
                obs.append(o)
                masks.append(env.action_masks())
        finally:
            env.close()
        js = json.loads(subprocess.run(["node", str(HERE / "replay_probe.mjs")], input=json.dumps({"entry": info["entry"], "actions": actions}),
                                       capture_output=True, text=True, encoding="utf-8", check=True).stdout)
        self.assertEqual(js["rewards"], rewards)
        self.assertEqual(js["dones"], [False] * (len(actions) - 1) + [True])
        for t in range(len(obs)):
            self.assertTrue(np.array_equal(obs[t], np.asarray(js["obs"][t], np.float32)))
            self.assertTrue(np.array_equal(masks[t], np.asarray(js["masks"][t], bool)))


# ── production contract ────────────────────────────────────────────────────
@unittest.skipUnless(HAVE_TORCH, "torch/numpy/gymnasium not installed")
class ProductionContract(unittest.TestCase):
    def test_export_loads_and_matches_javascript(self):
        from ipl_rl.common.checkpoint import export_policy
        net = make_policy(1)
        with torch.no_grad():
            for p in net.parameters():
                p.add_(0.1 * torch.randn_like(p))
        obs, masks = [], []
        with BridgeV2() as b:
            for seed, pick in ((1_000_097, 0), (1_000_402, -1)):
                r = b.call("reset", seed=seed)
                while not r.get("done"):
                    obs.append(r["obs"])
                    masks.append(r["mask"])
                    legal = [a for a, ok in enumerate(r["mask"]) if ok]
                    r = b.call("step", action=legal[pick])
        obs, masks = np.asarray(obs, np.float32), np.asarray(masks, bool)
        self.assertTrue((obs[:, 0] > 0.5).any(), "re-auction states included")
        with tempfile.TemporaryDirectory() as tmp:
            pol = export_policy(Path(tmp) / "policy.json", net, OBS_V2, ACT_V3, trained_steps=0, seed=1, cfg={}, cfg_hash="x",
                                extra_meta={"algorithmName": "openai_es"})
        self.assertEqual(pol["algorithm"], "es", "the loader's identifier (openai_es would be refused and fall back)")
        self.assertEqual(pol["meta"]["algorithmName"], "openai_es")
        self.assertEqual(pol["architecture"], {"input": 80, "hidden": [64, 64], "activation": "tanh", "head": "logits", "actions": 20})
        self.assertEqual(pol["selection"], {"mode": "argmax"})
        par = es.parity(net, pol, obs, masks)
        self.assertLess(par["maxAbsScoreDiff"], 1e-9)
        self.assertEqual(par["byMask"], {"real": 1.0, "passOnly": 1.0, "single": 1.0, "allLegal": 1.0})


if __name__ == "__main__":
    unittest.main()
