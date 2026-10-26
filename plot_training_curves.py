"""
Turn the TensorBoard logs of your training run into report-ready figures.

Usage (from the EPDiff project folder, venv active):
    python plot_training_curves.py
    python plot_training_curves.py --logdir output/rtx4050/tb_logs

Creates output/rtx4050/figures/training_curves.png
and prints every scalar tag found in the logs.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

parser = argparse.ArgumentParser()
parser.add_argument("--logdir", default="output/rtx4050/tb_logs")
parser.add_argument("--out", default="output/rtx4050/figures/training_curves.png")
args = parser.parse_args()

ea = EventAccumulator(args.logdir, size_guidance={"scalars": 0})
ea.Reload()
tags = ea.Tags().get("scalars", [])
print("Scalar tags found:")
for t in tags:
    print("  ", t, f"({len(ea.Scalars(t))} points)")


def series(tag):
    if tag not in tags:
        return None
    ev = ea.Scalars(tag)
    return [e.step for e in ev], [e.value for e in ev]


fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

# 1) training loss per epoch
s = series("train/loss")
if s:
    axes[0].plot(*s, marker="o", ms=3)
axes[0].set_title("Training loss (per epoch)")
axes[0].set_xlabel("Epoch")
axes[0].grid(alpha=0.3)

# 2) validation Dice per epoch
for tag, label in [("val/dice/mean", "Mean (3 classes)"), ("val/brats/WT", "WT"),
                   ("val/brats/TC", "TC"), ("val/brats/ET", "ET")]:
    s = series(tag)
    if s:
        axes[1].plot(*s, label=label, marker="o", ms=3)
axes[1].set_title("Validation Dice (per epoch)")
axes[1].set_xlabel("Epoch")
axes[1].set_ylim(0, 1)
axes[1].grid(alpha=0.3)
axes[1].legend()

# 3) learning rate per step
s = series("train/lr")
if s:
    axes[2].plot(*s)
axes[2].set_title("Learning rate (logged every 20 steps)")
axes[2].set_xlabel("Step")
axes[2].grid(alpha=0.3)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
fig.savefig(args.out, dpi=150, bbox_inches="tight")
print("\nSaved", args.out)
