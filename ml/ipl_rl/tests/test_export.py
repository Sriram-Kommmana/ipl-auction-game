"""L — rl-policy-v2 export → JavaScript import parity. Randomly initialised
networks, forward passes only: no training happens here."""

import importlib.util
import unittest

HAVE_TORCH = all(importlib.util.find_spec(m) for m in ("torch", "numpy"))

if HAVE_TORCH:
    import numpy as np
    import torch

    from ipl_rl.bridge import BridgeV2, run_node_script
    from ipl_rl.export import to_policy_json
    from ipl_rl.nets import ARCHITECTURES, build


def real_observations(count=40):
    """Real obs-v2 vectors from a real episode (JS side), as float32."""
    with BridgeV2() as b:
        spec = {k: b.info["obsSpec"][k] for k in ("version", "hash", "size")}, {k: b.info["actSpec"][k] for k in ("version", "hash", "count")}
        r = b.call("reset", seed=1_000_060)
        obs = []
        t = 0
        while len(obs) < count and not r.get("done"):
            obs.append(r["obs"])
            legal = [a for a, ok in enumerate(r["mask"]) if ok]
            r = b.call("step", action=legal[t % len(legal)])
            t += 1
    return spec, np.asarray(obs, dtype=np.float32)


@unittest.skipUnless(HAVE_TORCH, "torch/numpy not installed")
class ExportParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (cls.obs_spec, cls.act_spec), cls.obs = real_observations()

    def test_all_five_algorithms_round_trip(self):
        for algo in ARCHITECTURES:
            with self.subTest(algo=algo):
                net = build(algo, seed=7)
                policy = to_policy_json(net, self.obs_spec, self.act_spec, meta={"seed": 7, "config": {"test": "random init"}})
                js = run_node_script("rl-policy-scores.js", {"policy": policy, "observations": self.obs.tolist()})
                self.assertTrue(js["ok"], js.get("error"))
                with torch.no_grad():
                    torch_scores = net.action_scores(torch.from_numpy(self.obs)).double().numpy()
                self.assertTrue(np.allclose(np.asarray(js["scores"]), torch_scores, atol=1e-5), f"{algo}: max diff {np.abs(np.asarray(js['scores']) - torch_scores).max()}")
                self.assertTrue(np.array_equal(np.asarray(js["scores"]).argmax(1), torch_scores.argmax(1)))

    def test_dueling_fold_is_exact(self):
        net = build("d3qn", seed=3)
        (w1, b1), (w2, b2), (wq, bq) = net.export_layers()
        x = torch.from_numpy(self.obs)
        with torch.no_grad():
            z = torch.tanh(torch.tanh(x @ w1.T + b1) @ w2.T + b2)
            self.assertTrue(torch.allclose(z @ wq.T + bq, net(x), atol=1e-5))

    def test_quantile_head_mean(self):
        net = build("qrdqn", seed=4)
        with torch.no_grad():
            q = net(torch.from_numpy(self.obs))
        self.assertEqual(tuple(q.shape[1:]), (20, 32))
        policy = to_policy_json(net, self.obs_spec, self.act_spec)
        js = run_node_script("rl-policy-scores.js", {"policy": policy, "observations": self.obs[:3].tolist()})
        self.assertEqual(len(js["raw"][0]), 20 * 32)

    def test_metadata_and_spec_stamps(self):
        policy = to_policy_json(build("es", seed=1), self.obs_spec, self.act_spec, meta={"seed": 1, "config": {"lr": 0.01}, "gitSha": "abc"})
        self.assertEqual(policy["format"], "rl-policy-v2")
        self.assertEqual(policy["architecture"]["hidden"], [64, 64])
        self.assertEqual(policy["obsSpec"], self.obs_spec)
        self.assertEqual(policy["actSpec"], self.act_spec)
        self.assertEqual(policy["meta"]["trainedSteps"], 0)
        self.assertEqual(policy["meta"]["config"], {"lr": 0.01})

    def test_mismatched_spec_is_rejected_by_the_js_loader(self):
        policy = to_policy_json(build("ppo", seed=2), dict(self.obs_spec, hash="0" * 16), self.act_spec)
        js = run_node_script("rl-policy-scores.js", {"policy": policy, "observations": self.obs[:1].tolist()})
        self.assertFalse(js["ok"])
        self.assertIn("observation spec mismatch", js["error"])


if __name__ == "__main__":
    unittest.main()
