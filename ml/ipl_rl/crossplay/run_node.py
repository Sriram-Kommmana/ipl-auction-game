"""Run a Node script with the Windows EcoQoS opt-out applied (speed only — results unchanged).
    .venv/Scripts/python ipl_rl/crossplay/run_node.py <script.mjs> [args...]"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ipl_rl.bridge import NODE_FLAGS  # noqa: E402
from ipl_rl.common.win_qos import disable_throttling  # noqa: E402

proc = subprocess.Popen(["node", *NODE_FLAGS, *sys.argv[1:]])
disable_throttling(proc._handle)
sys.exit(proc.wait())
