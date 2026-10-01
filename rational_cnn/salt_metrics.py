r"""Lightning callback writing SALT's epoch metrics to ``metrics.csv`` (SALT's
CLI only wires up a Comet logger)."""
import csv
from pathlib import Path

from lightning import Callback


class MetricsCSV(Callback):
    r"""Appends one row of ``trainer.callback_metrics`` per validation epoch to
    ``<default_root_dir>/metrics.csv``."""
    def __init__(self, filename: str = 'metrics.csv'):
        self.filename = filename
        self.rows = []

    def on_validation_epoch_end(self, trainer, module):
        if trainer.sanity_checking or trainer.global_rank != 0:
            return
        row = {'epoch': trainer.current_epoch, 'step': trainer.global_step}
        for k, v in trainer.callback_metrics.items():
            row[k] = float(v)
        self.rows.append(row)
        keys = sorted({k for r in self.rows for k in r},
                      key=lambda k: (k not in ('epoch', 'step'), k))
        path = Path(trainer.default_root_dir) / self.filename
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.rows)
