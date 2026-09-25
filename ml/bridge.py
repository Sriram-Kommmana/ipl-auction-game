"""Talks to the JavaScript auction simulator (packages/shared/bin/rl-bridge.js).

Standard library only, so it works before any ML packages are installed —
see smoke_bridge.py.

Why a bridge instead of re-writing the auction in Python? The rules, the
scoring, the rule bots and the observation builder already exist in
JavaScript, and the live server uses exactly that code. Training against the
same code means a policy can never learn rules that differ from the real game.
"""

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BRIDGE_SCRIPT = REPO_ROOT / "packages" / "shared" / "bin" / "rl-bridge.js"


class BridgeError(RuntimeError):
    pass


class Bridge:
    """One Node.js simulator process. Each Gymnasium env owns one."""

    def __init__(self, node="node"):
        self.proc = subprocess.Popen(
            [node, str(BRIDGE_SCRIPT)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # simulator errors go straight to your terminal
            text=True,
            encoding="utf-8",
            bufsize=1,  # line-buffered: one JSON message per line
        )

    def call(self, cmd, **fields):
        if self.proc.poll() is not None:
            raise BridgeError("the simulator process has exited")
        self.proc.stdin.write(json.dumps({"cmd": cmd, **fields}) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise BridgeError("the simulator closed its output")
        reply = json.loads(line)
        if not reply.pop("ok", False):
            raise BridgeError(reply.get("error", "unknown simulator error"))
        return reply

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.terminate()
            self.proc.wait(timeout=5)
