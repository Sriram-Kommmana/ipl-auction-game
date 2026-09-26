"""Reproducibility metadata stamped into every run directory."""

import hashlib
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import gymnasium
import numpy as np
import torch

from ..bridge import REPO_ROOT


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def source_hash(paths):
    """SHA-256 over the given source trees (sorted files, LF line endings)."""
    h = hashlib.sha256()
    files = sorted(p for root in paths for p in Path(root).rglob("*") if p.suffix in (".js", ".py") and "__pycache__" not in p.parts)
    for f in files:
        h.update(str(f.relative_to(REPO_ROOT)).replace("\\", "/").encode())
        h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def run_metadata(cfg, cfg_hash, obs_spec, act_spec, gamma, extra=None):
    node = subprocess.run(["node", "--version"], capture_output=True, text=True).stdout.strip()
    status = _git("status", "--porcelain")
    return {
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": cfg,
        "configHash": cfg_hash,
        "obsSpec": obs_spec,
        "actSpec": act_spec,
        "gamma": gamma,
        "git": {"commit": _git("rev-parse", "HEAD"), "dirty": bool(status), "changed": status.splitlines() if status else []},
        "sourceHash": {
            "sharedSrc": source_hash([REPO_ROOT / "packages" / "shared" / "src"]),
            "iplRl": source_hash([REPO_ROOT / "ml" / "ipl_rl"]),
        },
        "versions": {
            "python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
            "gymnasium": gymnasium.__version__, "node": node,
        },
        "platform": {"system": platform.platform(), "machine": platform.machine(), "cpus": os.cpu_count(), "torchThreads": torch.get_num_threads()},
        "command": sys.argv,
        **(extra or {}),
    }
