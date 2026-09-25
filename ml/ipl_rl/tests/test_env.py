"""J / M / N / K — Gymnasium environment, JS↔Python parity, reward parity."""

import importlib.util
import unittest

HAVE_GYM = all(importlib.util.find_spec(m) for m in ("gymnasium", "numpy"))

if HAVE_GYM:
    import gymnasium as gym
    import numpy as np
    from gymnasium.utils.env_checker import check_env

    from ipl_rl.bridge import run_node_script
    from ipl_rl.env import ACTION_COUNT, OBS_SIZE, IplAuctionEnv, MaskedActionError

    class LegalSampling(gym.ActionWrapper):
        """check_env samples arbitrary actions; project them onto the mask so
        the API checks can run. (The environment itself raises on masked actions.)"""

        def action(self, action):
            mask = self.env.unwrapped.action_masks()
            return int(action) if mask[int(action)] else int(np.flatnonzero(mask)[int(action) % int(mask.sum())])


def run_fixed(env, seed):
    """One full episode with a deterministic legal-action rule."""
    obs, info = env.reset(options={"episode_seed": seed})
    entry = info["entry"]
    observations, masks, actions, rewards = [obs], [env.action_masks()], [], []
    t = 0
    while True:
        legal = np.flatnonzero(env.action_masks())
        a = int(legal[(t * 7) % len(legal)])
        t += 1
        obs, reward, terminated, truncated, info = env.step(a)
        actions.append(a)
        rewards.append(reward)
        if terminated:
            return entry, observations, masks, actions, rewards, info
        observations.append(obs)
        masks.append(env.action_masks())


@unittest.skipUnless(HAVE_GYM, "gymnasium/numpy not installed")
class GymEnv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = IplAuctionEnv(split="train")

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_spaces(self):
        self.assertEqual(self.env.observation_space.shape, (OBS_SIZE,))
        self.assertEqual(self.env.observation_space.dtype, np.float32)
        self.assertEqual(float(self.env.observation_space.low[0]), -1.0)
        self.assertEqual(float(self.env.observation_space.high[0]), 5.0)
        self.assertEqual(self.env.action_space.n, ACTION_COUNT)
        self.assertEqual(self.env.gamma, 1.0)
        self.assertEqual(self.env.obs_spec["version"], "obs-v2")
        self.assertEqual(self.env.act_spec["version"], "act-v2")

    def test_check_env(self):
        env = IplAuctionEnv(split="train")
        try:
            check_env(LegalSampling(env), skip_render_check=True)
        finally:
            env.close()

    def test_reset_is_seeded_and_in_split(self):
        o1, i1 = self.env.reset(seed=123)
        o2, i2 = self.env.reset(seed=123)
        self.assertTrue(np.array_equal(o1, o2))
        self.assertEqual(i1["entry"], i2["entry"])
        self.assertGreaterEqual(i1["entry"]["seed"], 1_000_000)
        val = IplAuctionEnv(split="validation")
        try:
            _, info = val.reset(seed=5)
            self.assertTrue(100_000 <= info["entry"]["seed"] <= 100_499)
        finally:
            val.close()

    def test_episode_rewards_masks_and_terminal(self):
        entry, observations, masks, actions, rewards, info = run_fixed(self.env, 1_000_050)
        ep = info["episode"]
        self.assertAlmostEqual(sum(rewards), ep["return"], places=9)
        self.assertAlmostEqual(sum(rewards), ep["xiTotal"] / 110 + (-2 if ep["emptySlots"] else 0), places=9)
        self.assertGreater(ep["reauctionLots"], 0, "main auction + re-auction")
        for o, m in zip(observations, masks):
            self.assertEqual(o.shape, (OBS_SIZE,))
            self.assertTrue(np.all(o >= -1) and np.all(o <= 5))
            self.assertTrue(m[1:].any(), "every step offers a legal bid")
        with self.assertRaises(RuntimeError):
            self.env.step(0)

    def test_masked_action_raises(self):
        self.env.reset(options={"episode_seed": 1_000_051})
        while self.env.action_masks().all():  # find a decision with a masked action
            self.env.step(0 if self.env.action_masks()[0] else 1)
        masked = int(np.flatnonzero(~self.env.action_masks())[0])
        with self.assertRaises(MaskedActionError):
            self.env.step(masked)

    def test_js_python_parity(self):
        """M/N: what Python receives equals an independent JavaScript replay
        (obs to float32 exactly, masks exactly, rewards exactly)."""
        entry, observations, masks, actions, rewards, _ = run_fixed(self.env, 1_000_052)
        js = run_node_script("rl-replay.js", {"entry": entry, "actions": actions, "tremble": 0.01})
        self.assertEqual(len(js["obs"]), len(observations))
        for i, (o, jo) in enumerate(zip(observations, js["obs"])):
            self.assertTrue(np.array_equal(o, np.asarray(jo, dtype=np.float32)), f"obs {i}")
        for i, (m, jm) in enumerate(zip(masks, js["masks"])):
            self.assertTrue(np.array_equal(m, np.asarray(jm, dtype=bool)), f"mask {i}")
        self.assertEqual(rewards, js["rewards"])


if __name__ == "__main__":
    unittest.main()
