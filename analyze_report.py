"""
Per-patient analysis of inference_report.json (best / worst patients, statistics, plots).

Usage (from the EPDiff project folder, venv active):
    python analyze_report.py
    python analyze_report.py --report output/rtx4050/predictions/inference_report.json

Creates in output/rtx4050/analysis/ :
    per_patient_dice.csv   all patients with WT/TC/ET and their mean
    dice_boxplot.png       box plot of WT/TC/ET over all test patients
    dice_histogram.png     histogram of the per-patient mean Dice
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--report", default="output/rtx4050/predictions/inference_report.json")
parser.add_argument("--out_dir", default="output/rtx4050/analysis")
args = parser.parse_args()

with open(args.report) as f:
    report = json.load(f)

rows = []
for e in report:
    m = e.get("brats_metrics") or {}
    if not all(k in m for k in ("WT", "TC", "ET")):
        continue
    wt, tc, et = float(m["WT"]), float(m["TC"]), float(m["ET"])
    rows.append({
        "subject_id": e["subject_id"],
        "WT": wt, "TC": tc, "ET": et,
        "mean": float(np.nanmean([wt, tc, et])),
        "infer_time_s": e.get("infer_time_s", float("nan")),
    })

if not rows:
    raise SystemExit("No patients with WT/TC/ET metrics found in the report.")

rows.sort(key=lambda r: r["mean"])
os.makedirs(args.out_dir, exist_ok=True)

# ---- CSV ----
csv_path = os.path.join(args.out_dir, "per_patient_dice.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# ---- statistics ----
print(f"\nPatients analysed: {len(rows)}\n")
print(f"{'Region':<8}{'Mean':>8}{'Median':>9}{'Std':>8}{'Min':>8}{'Max':>8}")
for k in ("WT", "TC", "ET"):
    v = np.array([r[k] for r in rows])
    print(f"{k:<8}{np.nanmean(v):8.4f}{np.nanmedian(v):9.4f}{np.nanstd(v):8.4f}"
          f"{np.nanmin(v):8.4f}{np.nanmax(v):8.4f}")
t = np.array([r["infer_time_s"] for r in rows], dtype=float)
print(f"\nMean inference time per patient: {np.nanmean(t):.1f} s")

et = np.array([r["ET"] for r in rows])
print(f"Patients with ET Dice < 0.5: {(et < 0.5).sum()} of {len(rows)}")
print(f"Patients with ET Dice == 0 : {(et == 0).sum()}")


def show(title, items):
    print(f"\n{title}")
    print(f"{'Patient':<20}{'WT':>8}{'TC':>8}{'ET':>8}{'Mean':>8}")
    for r in items:
        print(f"{r['subject_id']:<20}{r['WT']:8.4f}{r['TC']:8.4f}{r['ET']:8.4f}{r['mean']:8.4f}")


show("5 WORST patients (by mean of WT/TC/ET):", rows[:5])
show("5 BEST patients:", rows[-5:][::-1])
show("MEDIAN patient:", [rows[len(rows) // 2]])

# ---- plots ----
fig, ax = plt.subplots(figsize=(6, 5))
ax.boxplot([[r[k] for r in rows] for k in ("WT", "TC", "ET")])
ax.set_xticklabels(["WT", "TC", "ET"])
ax.set_ylabel("Dice")
ax.set_title("Per-patient Dice on the test set")
ax.grid(alpha=0.3)
fig.savefig(os.path.join(args.out_dir, "dice_boxplot.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist([r["mean"] for r in rows], bins=20, color="#4c72b0", edgecolor="white")
ax.set_xlabel("Mean Dice of WT/TC/ET per patient")
ax.set_ylabel("Number of patients")
ax.set_title("Distribution of per-patient Dice")
fig.savefig(os.path.join(args.out_dir, "dice_histogram.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved: {csv_path}")
print(f"Saved: {os.path.join(args.out_dir, 'dice_boxplot.png')}")
print(f"Saved: {os.path.join(args.out_dir, 'dice_histogram.png')}")
