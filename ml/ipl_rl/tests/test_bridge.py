"""I — bridge protocol from Python (standard library only)."""

import hashlib
import json
import unittest

from ipl_rl.bridge import BridgeError, BridgeV2


def legal(mask):
    return [a for a, ok in enumerate(mask) if ok]


def play(bridge, **reset):
    replies = [bridge.call("reset", **reset)]
    t = 0
    while True:
        r = bridge.call("step", action=legal(replies[-1]["mask"])[t % len(legal(replies[-1]["mask"]))])
        t += 1
        replies.append(r)
        if r["done"]:
            return replies


class BridgeProtocol(unittest.TestCase):
    def test_info(self):
        with BridgeV2() as b:
            info = b.info
            self.assertEqual(info["protocol"], "rl-bridge-v2")
            self.assertEqual(info["obsSpec"]["version"], "obs-v2")
            self.assertEqual(info["obsSpec"]["size"], 80)
            self.assertEqual(len(info["obsSpec"]["features"]), 80)
            self.assertEqual(info["actSpec"]["version"], "act-v2")
            self.assertEqual(info["actSpec"]["count"], 20)
            self.assertEqual(info["actSpec"]["actions"][0], "PASS")
            self.assertEqual(info["actSpec"]["actions"][19], "MAX_SAFE")
            self.assertEqual(info["gamma"], 1)
            self.assertEqual(info["lambdaRel"], 0)

    def test_episode_and_reward_sum(self):
        with BridgeV2() as b:
            replies = play(b, seed=1_000_040, split="train")
            ep = replies[-1]["info"]["episode"]
            total = sum(r["reward"] for r in replies[1:])
            self.assertAlmostEqual(total, ep["return"], places=9)
            self.assertAlmostEqual(total, ep["xiTotal"] / 110 + (-2 if ep["emptySlots"] else 0), places=9)
            self.assertGreater(ep["reauctionLots"], 0)

    def test_determinism_across_processes(self):
        digests = []
        for _ in range(2):
            with BridgeV2() as b:
                digests.append(hashlib.sha256(json.dumps(play(b, seed=1_000_041)).encode()).hexdigest())
        self.assertEqual(digests[0], digests[1])

    def test_errors_are_reported(self):
        with BridgeV2() as b:
            with self.assertRaisesRegex(BridgeError, "reset"):
                b.call("step", action=0)
            with self.assertRaisesRegex(BridgeError, "not in the validation split"):
                b.call("reset", seed=1_000_000, split="validation")
            r = b.call("reset", seed=100_000, split="validation")
            while all(r["mask"]):  # find a decision with at least one masked action
                r = b.call("step", action=0 if r["mask"][0] else 1)
            masked = [a for a, ok in enumerate(r["mask"]) if not ok][0]
            with self.assertRaisesRegex(BridgeError, "masked"):
                b.call("step", action=masked)


if __name__ == "__main__":
    unittest.main()
