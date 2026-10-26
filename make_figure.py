"""
Make a comparison figure for one test patient:
FLAIR | Ground truth (doctor) | Prediction (model)

Usage (from the EPDiff project folder, venv active):
    python make_figure.py BraTS2021_00001
    python make_figure.py BraTS2021_00001 --split test --pred_dir output/rtx4050/predictions
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap

parser = argparse.ArgumentParser()
parser.add_argument("patient", help="patient folder name, e.g. BraTS2021_00001")
parser.add_argument("--data_root", default="data/BraTS2021")
parser.add_argument("--split", default="test")
parser.add_argument("--pred_dir", default="output/rtx4050/predictions")
parser.add_argument("--out_dir", default="output/rtx4050/figures")
args = parser.parse_args()

pid = args.patient
folder = os.path.join(args.data_root, args.split, pid)
flair = nib.load(os.path.join(folder, f"{pid}_flair.nii.gz")).get_fdata()
gt = np.rint(nib.load(os.path.join(folder, f"{pid}_seg.nii.gz")).get_fdata()).astype(int)
gt[gt == 4] = 3  # original BraTS label 4 (enhancing) -> 3, same as the model's labels
pred = np.rint(
    nib.load(os.path.join(args.pred_dir, f"{pid}_pred_seg.nii.gz")).get_fdata()
).astype(int)

# pick the axial slice with the most true tumour (last axis = slice axis)
z = int(np.argmax((gt > 0).sum(axis=(0, 1))))


def show(ax, mask, title):
    ax.imshow(np.rot90(flair[:, :, z]), cmap="gray")
    if mask is not None:
        m = np.ma.masked_where(mask[:, :, z] == 0, mask[:, :, z])
        cmap = ListedColormap(["#e41a1c", "#4daf4a", "#ffd92f"])  # 1 red, 2 green, 3 yellow
        ax.imshow(np.rot90(m), cmap=cmap, vmin=1, vmax=3, alpha=0.6, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")


fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
show(axes[0], None, f"FLAIR (slice {z})")
show(axes[1], gt, "Ground truth (doctor)")
show(axes[2], pred, "Prediction (model)")
fig.suptitle(f"{pid}    red = necrotic core, green = edema, yellow = enhancing tumour")
os.makedirs(args.out_dir, exist_ok=True)
out = os.path.join(args.out_dir, f"{pid}.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("Saved", out)
