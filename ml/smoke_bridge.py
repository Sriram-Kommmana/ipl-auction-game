"""Checkpoint 0 — run this before installing anything.

    python ml/smoke_bridge.py

Plays a few auctions with a random agent that only picks LEGAL actions, using
nothing but the Python standard library. If this works, Python can drive the
simulator and the only thing left to install is the ML stack.
"""

import random
import time

from bridge import Bridge


def main():
    bridge = Bridge()
    try:
        info = bridge.call("info")
        print(f"observation: {info['observationSize']} values, actions: {info['actionCount']}")
        print("cap multipliers:", info["capMultipliers"])

        rng = random.Random(0)
        started = time.time()
        decisions = 0
        for episode in range(5):
            step = bridge.call("reset", seed=episode)
            total = 0.0
            while True:
                legal = [a for a, ok in enumerate(step["mask"]) if ok]
                step = bridge.call("step", action=rng.choice(legal))
                decisions += 1
                total += step["reward"]
                if step["done"]:
                    i = step["info"]
                    print(f"episode {episode}: XI strength {i['strength']:5.1f}, "
                          f"rank {i['rank'] + 1}/10, reward {total:+.2f}")
                    break
        elapsed = time.time() - started
        print(f"{decisions} decisions in {elapsed:.1f}s ({decisions / elapsed:.0f}/s)")

        # The simulator must refuse an action outside the mask.
        step = bridge.call("reset", seed=99)
        try:
            bridge.call("step", action=len(step["mask"]))
            print("ERROR: an out-of-range action was accepted")
        except Exception as err:
            print("illegal action correctly refused:", err)
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
