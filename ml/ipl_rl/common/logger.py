"""Run logging: TensorBoard scalars, a JSON-lines record per update, and a
final JSON run summary.

    log = RunLogger(run_dir)
    log.scalars("train", {"policy_loss": 0.1, ...}, step=decisions)
    log.record({"update": 3, ...})        # one line in metrics.jsonl
    log.write_json("summary.json", {...})
"""

import json
import math
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def _clean(value):
    """JSON-safe: NaN / inf become null; numpy scalars become Python numbers."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class RunLogger:
    def __init__(self, run_dir, tensorboard=True):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.run_dir / "tb")) if tensorboard else None
        self.jsonl = open(self.run_dir / "metrics.jsonl", "a", encoding="utf-8")

    def scalars(self, prefix, values, step):
        if not self.writer:
            return
        for k, v in values.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                self.writer.add_scalar(f"{prefix}/{k}", v, step)

    def histogram_shares(self, prefix, shares, names, step):
        if self.writer:
            for name, share in zip(names, shares):
                self.writer.add_scalar(f"{prefix}/{name}", share, step)

    def record(self, row):
        self.jsonl.write(json.dumps(_clean(row)) + "\n")
        self.jsonl.flush()

    def write_json(self, name, payload):
        path = self.run_dir / name
        path.write_text(json.dumps(_clean(payload), indent=2), encoding="utf-8")
        return path

    def close(self):
        if self.writer:
            self.writer.flush()
            self.writer.close()
        self.jsonl.close()
