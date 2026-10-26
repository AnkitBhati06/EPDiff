"""Create a tiny synthetic BraTS-style dataset for smoke-testing the pipeline."""
import sys
from pathlib import Path
import nibabel as nib
import numpy as np

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/dummy")
shape = (64, 64, 48)
rng = np.random.default_rng(0)
for split, n in [("train", 4), ("val", 2), ("test", 2)]:
    for i in range(n):
        sid = f"BraTS2021_{split}_{i:03d}"
        d = root / split / sid
        d.mkdir(parents=True, exist_ok=True)
        zz, yy, xx = np.mgrid[:shape[0], :shape[1], :shape[2]]
        brain = ((zz - 32) ** 2 + (yy - 32) ** 2 + (xx - 24) ** 2) < 22 ** 2
        c = rng.integers(22, 42, 3)
        tumour = ((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2] // 2 - 12) ** 2) < 6 ** 2
        seg = np.zeros(shape, np.uint8)
        seg[brain & tumour] = 2
        seg[brain & tumour & (rng.random(shape) > 0.5)] = 4
        seg[brain & tumour & (rng.random(shape) > 0.8)] = 1
        for mod in ["t1", "t1ce", "t2", "flair"]:
            vol = brain * (100 + rng.normal(0, 10, shape)) + tumour * 80
            nib.save(nib.Nifti1Image(vol.astype(np.float32), np.eye(4)), d / f"{sid}_{mod}.nii.gz")
        nib.save(nib.Nifti1Image(seg, np.eye(4)), d / f"{sid}_seg.nii.gz")
print("dummy dataset written to", root)
