"""Client for packages/shared/bin/rl-bridge-v2.js (standard library only).

Python never implements auction rules. The Node process runs the real
AuctionSim, the frozen opponents, obs-v2, the canonical action mask and the
reward; this class just sends one JSON request per line and reads one JSON
reply per line.
"""

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BRIDGE_SCRIPT = REPO_ROOT / "packages" / "shared" / "bin" / "rl-bridge-v2.js"
NODE_BIN = REPO_ROOT / "packages" / "shared" / "bin"
PROTOCOL = "rl-bridge-v2"
# V8 runtime flag, not game logic: a larger young generation cuts garbage
# collection in the allocation-heavy simulator (~1.7× more episodes/s with
# many simulators running; results are identical).
NODE_FLAGS = ("--max-semi-space-size=32",)


def _unthrottle(proc):
    """Opt a simulator out of Windows EcoQoS throttling (speed only; see common/win_qos.py)."""
    try:
        from .common.win_qos import disable_throttling
        disable_throttling(proc._handle)
    except Exception:
        pass


class BridgeError(RuntimeError):
    pass


class BridgeV2:
    """One Node.js simulator process. Each environment owns one."""

    def __init__(self, node="node", node_flags=NODE_FLAGS):
        self.proc = subprocess.Popen(
            [node, *node_flags, str(BRIDGE_SCRIPT)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        _unthrottle(self.proc)
        info = self.call("info")
        if info.get("protocol") != PROTOCOL:
            self.close()
            raise BridgeError(f"expected {PROTOCOL}, got {info.get('protocol')}")
        self.info = info

    def call(self, cmd, **fields):
        self.send(cmd, **fields)
        return self.receive()

    # send() + receive() split a call in two, so a vector of environments can
    # send to every simulator first and then collect the replies — the Node
    # processes then run in parallel (ipl_rl.common.vec_env).
    def send(self, cmd, **fields):
        if self.proc.poll() is not None:
            raise BridgeError("the simulator process has exited")
        self.proc.stdin.write(json.dumps({"cmd": cmd, **fields}) + "\n")
        self.proc.stdin.flush()

    def receive(self):
        line = self.proc.stdout.readline()
        if not line:
            raise BridgeError("the simulator closed its output")
        reply = json.loads(line)
        if not reply.pop("ok", False):
            raise BridgeError(reply.get("error", "unknown simulator error"))
        return reply

    def kill(self):
        """Hard stop (the deadlock watchdog): unblocks a pending receive()."""
        if self.proc.poll() is None:
            self.proc.kill()

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.terminate()
            self.proc.wait(timeout=5)
        if self.proc.stdout and not self.proc.stdout.closed:
            self.proc.stdout.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def run_node_script(script, payload, node="node"):
    """Run a one-shot helper from packages/shared/bin with JSON on stdin."""
    out = subprocess.run(
        [node, str(NODE_BIN / script)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return json.loads(out.stdout)
