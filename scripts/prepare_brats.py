"""
Turn the Kaggle BraTS 2021 Task 1 download into the layout the code expects:

    <out>/train/<subject>/<subject>_{t1,t1ce,t2,flair,seg}.nii.gz
    <out>/val/...
    <out>/test/...

Usage:
    python scripts/prepare_brats.py --src data/BraTS21 --out data/BraTS2021
"""
import argparse
import random
import shutil
import tarfile
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--src", default="data/BraTS21", help="Kaggle download folder")
p.add_argument("--out", default="data/BraTS2021")
p.add_argument("--val", type=float, default=0.1)
p.add_argument("--test", type=float, default=0.1)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--copy", action="store_true",
               help="copy subject folders instead of moving them")
args = p.parse_args()

src, out = Path(args.src), Path(args.out)

# Extract any .tar archives that have not been extracted yet
extract_dir = src / "extracted"
for tar in sorted(src.rglob("*.tar")):
    marker = extract_dir / f".{tar.name}.done"
    if marker.exists():
        continue
    print(f"Extracting {tar} ...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar) as tf:
        tf.extractall(extract_dir, filter="data")
    marker.touch()

# A subject folder is any folder holding <name>_flair.nii.gz
subjects = {}
for flair in src.rglob("*_flair.nii*"):
    d = flair.parent
    if d.name.startswith("BraTS") and (out not in d.parents):
        subjects[d.name] = d
names = sorted(subjects)
if not names:
    raise SystemExit(f"No subject folders found under {src}")

random.Random(args.seed).shuffle(names)
n_val, n_test = int(len(names) * args.val), int(len(names) * args.test)
splits = {"val": names[:n_val], "test": names[n_val:n_val + n_test],
          "train": names[n_val + n_test:]}

for split, ids in splits.items():
    (out / split).mkdir(parents=True, exist_ok=True)
    for sid in ids:
        dst = out / split / sid
        if dst.exists():
            continue
        if args.copy:
            shutil.copytree(subjects[sid], dst)
        else:
            shutil.move(str(subjects[sid]), dst)
    print(f"{split}: {len(ids)} subjects")
print(f"Done -> {out}")
