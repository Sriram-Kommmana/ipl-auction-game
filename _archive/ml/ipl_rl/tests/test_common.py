"""Phase 2C shared training infrastructure (no training here)."""

import unittest

import numpy as np
import torch

from ipl_rl.common.config import config_hash, load_config
from ipl_rl.common.masking import MASKED_LOGIT, assert_legal, masked_argmax, masked_distribution, masked_sample
from ipl_rl.common.seeding import episode_seed
from ipl_rl.common.vec_env import VecIplAuctionEnv
from ipl_rl.env import SPLIT_RANGES, IplAuctionEnv, MaskedActionError


class TestConfigAndSeeds(unittest.TestCase):
    def test_config_rejects_unknown_keys_and_hash_ignores_run_name(self):
        defaults = {"seed": 1, "lr": 0.1, "run_name": None}
        with self.assertRaises(KeyError):
            load_config(None, defaults, {"lr_typo": 1})
        a = load_config(None, defaults, {"run_name": "x"})
        b = load_config(None, defaults, {"run_name": "y"})
        self.assertEqual(config_hash(a), config_hash(b))
        self.assertNotEqual(config_hash(a), config_hash(load_config(None, defaults, {"lr": 0.2})))

    def test_episode_seed_schedule_is_deterministic_and_train_only(self):
        lo, hi = SPLIT_RANGES["train"]
        seeds = [episode_seed(1, i, k) for i in range(12) for k in range(50)]
        self.assertEqual(seeds, [episode_seed(1, i, k) for i in range(12) for k in range(50)])
        self.assertTrue(all(lo <= s <= hi for s in seeds))
        self.assertEqual(len(set(seeds)), len(seeds))
        self.assertNotEqual(episode_seed(1, 0, 0), episode_seed(2, 0, 0))
        with self.assertRaises(ValueError):
            episode_seed(1, 0, 0, split="validation")


class TestMasking(unittest.TestCase):
    def test_masked_actions_have_zero_probability_and_are_never_chosen(self):
        g = torch.Generator().manual_seed(0)
        logits = torch.randn(64, 20, generator=g) * 5
        mask = torch.rand(64, 20, generator=g) < 0.4
        mask[:, 0] = True
        dist = masked_distribution(logits, mask)
        self.assertTrue(torch.all(dist.probs[~mask] == 0))
        for _ in range(20):
            a, logp, ent = masked_sample(logits, mask)
            assert_legal(a, mask)
            self.assertTrue(torch.all(torch.isfinite(logp)) and torch.all(ent >= 0))
        arg = masked_argmax(logits, mask)
        assert_legal(arg, mask)
        expect = torch.where(mask, logits, torch.tensor(MASKED_LOGIT)).argmax(-1)
        self.assertTrue(torch.equal(arg, expect))
        # Entropy over legal actions only: uniform logits → log(#legal).
        u = masked_distribution(torch.zeros(64, 20), mask).entropy()
        self.assertTrue(torch.allclose(u, torch.log(mask.sum(-1).float()), atol=1e-6))
        with self.assertRaises(AssertionError):
            assert_legal(torch.tensor([1]), torch.tensor([[True, False] + [False] * 18]))


class TestVecEnv(unittest.TestCase):
    def test_vector_env_matches_single_env_on_the_same_seeds(self):
        venv = VecIplAuctionEnv(num_envs=3, run_seed=7)
        single = IplAuctionEnv(split="train")
        try:
            obs, masks = venv.reset()
            singles = []
            for i in range(3):
                o, _ = single.reset(options={"episode_seed": episode_seed(7, i, 0)})
                singles.append(o)
            self.assertTrue(np.array_equal(obs, np.stack(singles)))
            # Step env 0 in both with the same (first legal) actions for a while.
            ref_obs, _ = single.reset(options={"episode_seed": episode_seed(7, 0, 0)})
            for _ in range(30):
                acts = masks.argmax(axis=1)  # PASS if legal, else the first legal action
                obs, rewards, dones, masks, infos = venv.step(acts)
                ro, rr, rd, _, _ = single.step(int(acts[0]))
                self.assertEqual(rr, rewards[0])
                if rd:
                    break
                self.assertTrue(np.array_equal(ro, obs[0]))
                self.assertTrue(np.array_equal(single.action_masks(), masks[0]))
            bad = np.flatnonzero(~masks[1])
            if len(bad):
                acts = masks.argmax(axis=1)
                acts[1] = bad[0]
                with self.assertRaises(MaskedActionError):
                    venv.step(acts)
        finally:
            venv.close()
            single.close()

    def test_auto_reset_reports_episode_and_starts_the_next_scheduled_seed(self):
        venv = VecIplAuctionEnv(num_envs=2, run_seed=3)
        try:
            obs, masks = venv.reset()
            finished = []
            steps = 0
            while not finished and steps < 2000:
                # Bid MAX_SAFE whenever possible: squads fill fast, episodes end early.
                acts = np.where(masks[:, 19], 19, masks.argmax(axis=1))
                obs, rewards, dones, masks, infos = venv.step(acts)
                finished = [infos[i]["episode"] for i in np.flatnonzero(dones)]
                steps += 1
            self.assertTrue(finished, "no episode finished")
            ep = finished[0]
            self.assertEqual(ep["invariantViolations"], 0, ep["violations"])
            self.assertEqual(sum(ep["actionCounts"]), ep["decisions"])
            i = ep["envIndex"]
            self.assertEqual(ep["seed"], episode_seed(3, i, 0))
            self.assertEqual(venv.current_seed[i], episode_seed(3, i, 1))
            self.assertTrue(masks[i].any())
        finally:
            venv.close()


if __name__ == "__main__":
    unittest.main()


class TestAsyncRollout(unittest.TestCase):
    def test_async_collection_equals_one_env_at_a_time(self):
        """Arrival order never matters: async == each env driven alone (untrained policy, no updates)."""
        from ipl_rl.algos.ppo import Agent
        from ipl_rl.common.rollout import AsyncRolloutState, collect_async, policy_step

        torch.manual_seed(11)
        agent = Agent()
        n, T = 3, 40

        def fresh_buffers():
            return {"obs": torch.zeros((T, n, 80)), "mask": torch.zeros((T, n, 20), dtype=torch.bool),
                    "act": torch.zeros((T, n), dtype=torch.long), "logp": torch.zeros((T, n)), "val": torch.zeros((T, n)),
                    "rew": torch.zeros((T, n)), "done": torch.zeros((T, n))}

        runs = []
        for _ in range(2):
            venv = VecIplAuctionEnv(num_envs=n, run_seed=5)
            try:
                state = AsyncRolloutState(venv, seed=5)
                bufs, eps = fresh_buffers(), []
                collect_async(state, agent.actor, agent.critic, T, bufs, eps.append)
                collect_async(state, agent.actor, agent.critic, T, bufs, eps.append)  # continues across rollouts
                runs.append((bufs, sorted(e["seed"] for e in eps), state.obs.copy()))
            finally:
                venv.close()

        # Reference: one environment at a time, same generators, same schedule.
        venv = VecIplAuctionEnv(num_envs=n, run_seed=5)
        try:
            state = AsyncRolloutState(venv, seed=5)
            ref = fresh_buffers()
            for i in range(n):
                for _ in range(2):  # two rollouts
                    for k in range(T):
                        a, logp, v = policy_step(agent.actor, agent.critic, state.obs[i], state.mask[i], state.generators[i])
                        ref["obs"][k, i], ref["mask"][k, i] = torch.from_numpy(state.obs[i]), torch.from_numpy(state.mask[i])
                        ref["act"][k, i], ref["logp"][k, i], ref["val"][k, i] = a, logp, v
                        venv.send_step(i, a)
                        j, r = venv.next_reply()
                        self.assertEqual(j, i)
                        ref["rew"][k, i], ref["done"][k, i] = r["reward"], float(r["done"])
                        if r["done"]:
                            venv.send_reset(i)
                            j, r = venv.next_reply()
                        state.obs[i], state.mask[i] = r["obs"], r["mask"]
        finally:
            venv.close()
        for key in ref:
            self.assertTrue(torch.equal(runs[0][0][key], runs[1][0][key]), f"async not reproducible: {key}")
            self.assertTrue(torch.equal(runs[0][0][key], ref[key]), f"async differs from one-at-a-time: {key}")
        self.assertTrue(np.array_equal(runs[0][2], state.obs))
        self.assertEqual(runs[0][1], runs[1][1])
