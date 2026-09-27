"""Phase 2D.1 — masked A2C: networks, masking, GAE/returns, the one-step A2C
update, config hygiene, act-v3 alignment, export parity and deterministic
evaluation. Randomly initialised networks and hand-made batches only (no
training run here)."""

import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

HAVE_TORCH = all(importlib.util.find_spec(m) for m in ("torch", "numpy", "gymnasium"))

if HAVE_TORCH:
    import numpy as np
    import torch

    from ipl_rl.algos import a2c
    from ipl_rl.algos.a2c import DEFAULTS, A2CAgent, a2c_loss, a2c_update, compute_gae, init_seed, params_digest
    from ipl_rl.bridge import BridgeV2, run_node_script
    from ipl_rl.common.config import load_config
    from ipl_rl.common.masking import MASKED_LOGIT, masked_distribution
    from ipl_rl.common.rollout import policy_step
    from ipl_rl.nets import PolicyNet

ACT_V3 = {"version": "act-v3", "hash": "5f72f510c48b1f46", "count": 20}
OBS_V2_HASH = "629b25783f833af7"
ACTION_NAMES = ["PASS", "BASE", "FV_0.20", "FV_0.30", "FV_0.40", "FV_0.50", "FV_0.60", "FV_0.70", "FV_0.80", "FV_0.90",
                "FV_1.00", "FV_1.10", "FV_1.25", "FV_1.40", "FV_1.60", "FV_1.80", "FV_2.00", "FV_2.50", "FV_3.00", "MAX_SAFE"]
PPO_2C3_INITIAL = {1: "5d3196dbd3f13883", 2: "0f7d17d3a2595f05", 3: "f6c2f9465df72b60"}


def seeded_agent(seed):
    torch.manual_seed(init_seed(seed))
    return A2CAgent()


def cfg(**kw):
    return {**DEFAULTS, **kw}


def real_states(count=60, seed=1_000_071):
    """Real obs-v2 observations and act-v3 masks from one JavaScript episode."""
    with BridgeV2() as b:
        info = b.info
        r = b.call("reset", seed=seed)
        obs, masks, t = [], [], 0
        while len(obs) < count and not r.get("done"):
            obs.append(r["obs"])
            masks.append(r["mask"])
            legal = [a for a, ok in enumerate(r["mask"]) if ok]
            r = b.call("step", action=legal[(3 * t) % len(legal)])
            t += 1
    return info, np.asarray(obs, dtype=np.float32), np.asarray(masks, dtype=bool)


@unittest.skipUnless(HAVE_TORCH, "torch/numpy/gymnasium not installed")
class Networks(unittest.TestCase):
    def test_shapes_heads_and_separate_parameters(self):
        agent = seeded_agent(1)
        self.assertIsInstance(agent.actor, PolicyNet)
        self.assertEqual(agent.actor.algorithm, "a2c")
        obs = torch.randn(7, 80)
        self.assertEqual(tuple(agent.actor(obs).shape), (7, 20))
        self.assertEqual(tuple(agent.value(obs).shape), (7,))
        self.assertEqual(agent.critic[-1].out_features, 1)
        sizes = [(m.in_features, m.out_features) for m in agent.critic if isinstance(m, torch.nn.Linear)]
        self.assertEqual(sizes, [(80, 128), (128, 128), (128, 1)])
        sizes = [(m.in_features, m.out_features) for m in agent.actor.body if isinstance(m, torch.nn.Linear)] + [(128, agent.actor.head.out_features)]
        self.assertEqual(sizes, [(80, 128), (128, 128), (128, 20)])
        self.assertTrue(all(isinstance(m, torch.nn.Tanh) for m in list(agent.actor.body)[1::2]))
        self.assertTrue(all(isinstance(m, torch.nn.Tanh) for m in list(agent.critic)[1:-1:2]))
        self.assertFalse({id(p) for p in agent.actor.parameters()} & {id(p) for p in agent.critic.parameters()})

    def test_orthogonal_init(self):
        agent = seeded_agent(2)
        w = agent.actor.body[2].weight.detach()  # 128 × 128, gain √2
        self.assertTrue(torch.allclose(w @ w.T, 2.0 * torch.eye(128), atol=1e-4))
        h = agent.actor.head.weight.detach()     # 20 × 128, gain 0.01 → orthonormal rows × 0.01
        self.assertTrue(torch.allclose(h @ h.T, 1e-4 * torch.eye(20), atol=1e-8))
        self.assertTrue(torch.all(agent.actor.head.bias == 0))
        v = agent.critic[-1].weight.detach()     # 1 × 128, gain 1
        self.assertAlmostEqual(float(v.norm()), 1.0, places=5)

    def test_init_is_reproducible_independent_per_seed_and_not_ppo(self):
        digests = {s: params_digest(seeded_agent(s)) for s in (1, 2, 3)}
        self.assertEqual(digests, {s: params_digest(seeded_agent(s)) for s in (1, 2, 3)})
        self.assertEqual(len(set(digests.values())), 3)
        for s in (1, 2, 3):
            self.assertNotEqual(digests[s], PPO_2C3_INITIAL[s], "A2C must not draw PPO's initial weights")


@unittest.skipUnless(HAVE_TORCH, "torch/numpy/gymnasium not installed")
class Masking(unittest.TestCase):
    def test_masked_actions_have_zero_probability_and_sampling_is_always_legal(self):
        agent = seeded_agent(1)
        g = torch.Generator().manual_seed(3)
        obs = torch.randn(200, 80, generator=g)
        mask = torch.rand(200, 20, generator=g) < 0.3
        mask[torch.arange(200), torch.randint(0, 20, (200,), generator=g)] = True  # ≥ 1 legal per row
        # Push the illegal actions' raw logits far up: masking must still zero them.
        with torch.no_grad():
            agent.actor.head.bias.copy_(torch.linspace(0, 50, 20))
        dist = masked_distribution(agent.actor(obs), mask)
        self.assertTrue(torch.all(dist.probs[~mask] == 0))
        self.assertTrue(torch.allclose(dist.probs.sum(-1), torch.ones(200)))
        for _ in range(50):
            a = dist.sample()
            self.assertTrue(bool(mask[torch.arange(200), a].all()))
        # The rollout sampler (per-environment generator, one row at a time).
        gen = torch.Generator().manual_seed(9)
        for i in range(200):
            a, logp, _ = policy_step(agent.actor, agent.critic, obs[i].numpy(), mask[i].numpy(), gen)
            self.assertTrue(bool(mask[i, a]))
            self.assertAlmostEqual(logp, float(dist.logits[i, a].detach()), places=4)  # same masked log-softmax as the update
        # Masked actions contribute nothing to the entropy (entropy over legal actions only).
        u = masked_distribution(torch.zeros(200, 20), mask).entropy()
        self.assertTrue(torch.allclose(u, torch.log(mask.sum(-1).float()), atol=1e-5))
        # A masked action's log-probability is effectively −∞ (≈ MASKED_LOGIT).
        bad = (~mask).float().argmax(-1)
        rows = (~mask).any(-1)
        self.assertTrue(torch.all(dist.log_prob(bad)[rows] < MASKED_LOGIT / 2))

    def test_real_act_v3_masks_are_honoured(self):
        agent = seeded_agent(3)
        _, obs, masks = real_states(80)
        o, m = torch.from_numpy(obs), torch.from_numpy(masks)
        dist = masked_distribution(agent.actor(o), m)
        self.assertTrue(torch.all(dist.probs[~m] == 0))
        gen = torch.Generator().manual_seed(1)
        for i in range(len(obs)):
            for _ in range(10):
                a, _, _ = policy_step(agent.actor, agent.critic, obs[i], masks[i], gen)
                self.assertTrue(masks[i][a])


@unittest.skipUnless(HAVE_TORCH, "torch/numpy/gymnasium not installed")
class Gae(unittest.TestCase):
    def test_hand_computed_gamma_1_lambda_095(self):
        r = torch.tensor([[0.1], [0.0], [0.3]])
        v = torch.tensor([[1.0], [0.8], [0.5]])
        d = torch.zeros(3, 1)
        nv = torch.tensor([0.2])
        adv, ret = compute_gae(r, v, d, nv, 1.0, 0.95)
        d2 = 0.3 + 0.2 - 0.5
        d1 = 0.0 + 0.5 - 0.8
        d0 = 0.1 + 0.8 - 1.0
        a2 = d2
        a1 = d1 + 0.95 * a2
        a0 = d0 + 0.95 * a1
        self.assertTrue(torch.allclose(adv.flatten(), torch.tensor([a0, a1, a2]), atol=1e-7))
        self.assertTrue(torch.allclose(ret, adv + v))

    def test_termination_stops_bootstrapping(self):
        r = torch.tensor([[0.0], [0.5], [0.2]])
        v = torch.tensor([[0.4], [0.3], [0.9]])
        d = torch.tensor([[0.0], [1.0], [0.0]])  # step 1 ends an episode; step 2 starts the next
        adv, _ = compute_gae(r, v, d, torch.tensor([0.7]), 1.0, 0.95)
        a2 = 0.2 + 0.7 - 0.9
        a1 = 0.5 - 0.3                    # no bootstrap, no carry from step 2
        a0 = (0.0 + 0.3 - 0.4) + 0.95 * a1
        self.assertTrue(torch.allclose(adv.flatten(), torch.tensor([a0, a1, a2]), atol=1e-7))

    def test_lambda_1_is_monte_carlo_and_lambda_0_is_td(self):
        g = torch.Generator().manual_seed(0)
        T, N = 9, 4
        r = torch.rand(T, N, generator=g)
        v = torch.rand(T, N, generator=g)
        d = (torch.rand(T, N, generator=g) < 0.2).float()
        nv = torch.rand(N, generator=g)
        _, ret = compute_gae(r, v, d, nv, 1.0, 1.0)
        mc = torch.zeros(T, N)
        run = nv.clone()
        for t in reversed(range(T)):
            run = r[t] + (1 - d[t]) * run
            mc[t] = run
        self.assertTrue(torch.allclose(ret, mc, atol=1e-6))
        adv0, _ = compute_gae(r, v, d, nv, 1.0, 0.0)
        nxt = torch.cat([v[1:], nv.unsqueeze(0)])
        self.assertTrue(torch.allclose(adv0, r + (1 - d) * nxt - v, atol=1e-6))

    def test_defaults_are_the_frozen_gamma_and_lambda(self):
        self.assertEqual(DEFAULTS["gamma"], 1.0)
        self.assertEqual(DEFAULTS["gae_lambda"], 0.95)


@unittest.skipUnless(HAVE_TORCH, "torch/numpy/gymnasium not installed")
class Update(unittest.TestCase):
    def batch(self, agent, n=64, seed=5):
        g = torch.Generator().manual_seed(seed)
        obs = torch.randn(n, 80, generator=g)
        mask = torch.rand(n, 20, generator=g) < 0.5
        mask[:, 0] = True
        with torch.no_grad():
            act = masked_distribution(agent.actor(obs), mask).sample()
        adv = torch.randn(n, generator=g)
        ret = torch.randn(n, generator=g)
        return obs, mask, act, adv, ret

    def test_loss_is_the_a2c_objective(self):
        agent = seeded_agent(1)
        obs, mask, act, adv, ret = self.batch(agent)
        c = cfg()
        loss, _ = a2c_loss(agent, obs, mask, act, adv, ret, c)
        dist = masked_distribution(agent.actor(obs), mask)
        a = (adv - adv.mean()) / (adv.std() + 1e-8)
        manual = -(a * dist.log_prob(act)).mean() - c["ent_coef"] * dist.entropy().mean() + c["vf_coef"] * 0.5 * ((agent.value(obs) - ret) ** 2).mean()
        self.assertAlmostEqual(loss.item(), manual.item(), places=6)

    def test_one_step_per_rollout_clipped_gradients_and_on_policy(self):
        agent = seeded_agent(2)
        obs, mask, act, adv, ret = self.batch(agent)
        with torch.no_grad():
            sample_logp = masked_distribution(agent.actor(obs), mask).log_prob(act)
        before = params_digest(agent)
        opt = torch.optim.Adam(agent.parameters(), lr=3e-4, eps=1e-5)
        stats = a2c_update(agent, opt, obs, mask, act, adv * 1e3, ret * 1e3, cfg(), sample_logp=sample_logp)
        self.assertNotEqual(params_digest(agent), before)
        steps = {int(s["step"]) for s in opt.state.values()}
        self.assertEqual(steps, {1}, "exactly one optimiser step per rollout")
        self.assertGreater(stats["grad_norm"], 0.5)  # pre-clip norm is reported …
        clipped = math.sqrt(sum(float(p.grad.norm() ** 2) for p in agent.parameters()))
        self.assertLessEqual(clipped, 0.5 + 1e-5)      # … and the applied gradient was clipped to 0.5
        self.assertLess(stats["logp_drift"], 1e-5)
        self.assertTrue(stats["grads_finite"] and stats["params_finite"])

    def test_policy_moves_towards_positive_advantage_and_value_towards_return(self):
        agent = seeded_agent(3)
        obs = torch.randn(1, 80).repeat(2, 1)
        mask = torch.ones(2, 20, dtype=torch.bool)
        act = torch.tensor([4, 11])
        adv = torch.tensor([1.0, -1.0])
        ret = torch.tensor([2.0, 2.0])
        c = cfg(ent_coef=0.0)
        opt = torch.optim.Adam(agent.parameters(), lr=1e-4, eps=1e-5)  # small first Adam step: direction only
        with torch.no_grad():
            p0 = torch.softmax(agent.actor(obs[:1]), -1)[0]
            v0 = agent.value(obs[:1]).item()
        a2c_update(agent, opt, obs, mask, act, adv, ret, c)
        with torch.no_grad():
            p1 = torch.softmax(agent.actor(obs[:1]), -1)[0]
            v1 = agent.value(obs[:1]).item()
        self.assertGreater(p1[4], p0[4])
        self.assertLess(p1[11], p0[11])
        self.assertLess(abs(v1 - 2.0), abs(v0 - 2.0))

    def test_health_alarms(self):
        from ipl_rl.algos.a2c import health_problems
        ok = {"loss": 1.0, "policy_loss": 0.0, "value_loss": 3.0, "entropy": 2.0, "grad_norm": 5.0, "grads_finite": True, "params_finite": True,
              "entropy_normalised": 0.9, "logp_drift": 0.0}
        acts = {"action_shares": [0.05] * 20}
        self.assertEqual(health_problems(30, ok, acts, 2.9), [])                     # slow critic: not an explosion
        self.assertTrue(health_problems(30, {**ok, "value_loss": 1.5}, acts, 0.01))   # rose far above its best
        self.assertEqual(health_problems(5, {**ok, "value_loss": 1.5}, acts, 0.01), [])  # warm-up
        self.assertTrue(health_problems(1, {**ok, "loss": float("nan")}, acts, 1.0))
        self.assertTrue(health_problems(1, {**ok, "grads_finite": False}, acts, 1.0))
        self.assertTrue(health_problems(1, {**ok, "entropy_normalised": 0.01}, acts, 1.0))
        self.assertTrue(health_problems(1, ok, {"action_shares": [0.96] + [0.04 / 19] * 19}, 1.0))
        self.assertTrue(health_problems(1, {**ok, "logp_drift": 1e-2}, acts, 1.0))

    def test_no_ppo_mechanisms_in_the_config(self):
        for key in ("clip_coef", "clip_vloss", "update_epochs", "minibatch_size", "target_kl", "anneal_lr", "rollout_mode"):
            self.assertNotIn(key, DEFAULTS)
            with self.assertRaises(KeyError):
                load_config(None, DEFAULTS, {key: 1})
        c = load_config(Path(a2c.__file__).parents[1] / "configs" / "a2c_2d1.json", DEFAULTS)
        self.assertEqual((c["num_envs"], c["num_steps"], c["learning_rate"], c["gamma"], c["gae_lambda"], c["ent_coef"], c["vf_coef"],
                          c["max_grad_norm"], c["adam_eps"], c["norm_adv"], c["stage"], c["expected_act_spec"]),
                         (12, 512, 3e-4, 1.0, 0.95, 0.01, 0.5, 0.5, 1e-5, True, "A", ACT_V3["hash"]))


@unittest.skipUnless(HAVE_TORCH, "torch/numpy/gymnasium not installed")
class SpecAlignmentExportAndEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.info, cls.obs, cls.masks = real_states(120, seed=1_000_083)

    def test_act_v3_alignment(self):
        act = self.info["actSpec"]
        self.assertEqual({k: act[k] for k in ("version", "hash", "count")}, ACT_V3)
        self.assertEqual(list(act["actions"]), ACTION_NAMES)
        self.assertEqual(ACTION_NAMES.index("PASS"), 0)
        self.assertEqual(ACTION_NAMES.index("MAX_SAFE"), 19)
        self.assertEqual(self.info["obsSpec"]["hash"], OBS_V2_HASH)
        self.assertEqual(self.info["gamma"], 1)
        self.assertEqual(seeded_agent(1).actor.head.out_features, len(ACTION_NAMES))

    def test_export_parity_and_production_contract(self):
        from ipl_rl.common.evaluation import export_parity
        from ipl_rl.export import to_policy_json
        agent = seeded_agent(1)
        with torch.no_grad():  # a non-trivial head so the argmax is informative
            agent.actor.head.weight.mul_(300)
        obs_spec = {k: self.info["obsSpec"][k] for k in ("version", "hash", "size")}
        policy = to_policy_json(agent.actor, obs_spec, ACT_V3)
        self.assertEqual(policy["algorithm"], "a2c")
        self.assertEqual(policy["selection"], {"mode": "sample", "temperature": 0.3})
        self.assertEqual(policy["architecture"], {"input": 80, "hidden": [128, 128], "activation": "tanh", "head": "logits", "actions": 20})
        par = export_parity(agent.actor, policy, self.obs, self.masks)
        self.assertEqual(par["argmaxAgreement"], 1.0)
        self.assertLess(par["maxAbsScoreDiff"], 1e-4)
        self.assertEqual(par["states"], len(self.obs))

    def test_deterministic_evaluation(self):
        from ipl_rl.common.evaluation import node_evaluate
        from ipl_rl.export import to_policy_json, write_policy
        agent = seeded_agent(2)
        # Python side: masked argmax is a pure function of (obs, mask).
        o, m = torch.from_numpy(self.obs), torch.from_numpy(self.masks)
        with torch.no_grad():
            s = agent.actor(o)
        self.assertTrue(torch.equal(torch.where(m, s, torch.tensor(MASKED_LOGIT)).argmax(-1), torch.where(m, agent.actor(o), torch.tensor(MASKED_LOGIT)).argmax(-1)))
        # Node evaluator (production inference, T = 0.3, learner stream): same policy → same episodes.
        obs_spec = {k: self.info["obsSpec"][k] for k in ("version", "hash", "size")}
        with tempfile.TemporaryDirectory() as tmp:
            p = write_policy(Path(tmp) / "policy.json", to_policy_json(agent.actor, obs_spec, ACT_V3))
            runs = [node_evaluate(p, Path(tmp) / f"eval{k}", limit=3, workers=3)[1]["episodes"]["policy:a2c"] for k in range(2)]
        strip = lambda rows: [{k: v for k, v in r.items() if k not in ("ms", "seconds", "wallMs")} for r in rows]
        self.assertEqual(strip(runs[0]), strip(runs[1]))
        self.assertEqual(len(runs[0]), 3)
        self.assertTrue(all(r["legalXI"] for r in runs[0]))


if __name__ == "__main__":
    unittest.main()
