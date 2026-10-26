# Running EPDiff on a laptop GPU (RTX 4050, 6 GB)

Run each command from the project folder. Where Windows and Linux/macOS differ, both are shown.

## 1. Python environment

Install Python 3.10–3.12, then:

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-modern.txt
```

Check the GPU is visible (should print `True NVIDIA GeForce RTX 4050 ...`):

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If it prints `False`, update your NVIDIA driver and reinstall torch with the command above.

## 2. Quick test with fake data (~1 minute)

```bash
python scripts/make_dummy_data.py
python train_multimodal.py --config multimodal/config/smoke_test.yaml
```

If this finishes with `Training finished`, the setup works.

## 3. Download BraTS 2021 (~13 GB)

1. Create a Kaggle account → Settings → **Create New Token**. This downloads `kaggle.json`.
2. Put it in `C:\Users\<you>\.kaggle\kaggle.json` (Windows) or `~/.kaggle/kaggle.json` (Linux/macOS).
3. Download and arrange the data:

```bash
python data_downloader.py                 # -> data/BraTS21
python scripts/prepare_brats.py           # -> data/BraTS2021/{train,val,test}
```

`prepare_brats.py` unpacks the archives and splits subjects 80/10/10. It needs roughly 30 GB of free disk space while it runs. Afterwards you can delete `data/BraTS21`.

## 4. Train

```bash
python train_multimodal.py --config multimodal/config/rtx4050.yaml
```

- Checkpoints are saved in `output/rtx4050/` (`best.pth`, `last.pth`).
- To watch progress: `tensorboard --logdir output/rtx4050/tb_logs`
- To resume after stopping: `python train_multimodal.py --config multimodal/config/rtx4050.yaml --override resume=output/rtx4050/last.pth`
- If you get `CUDA out of memory`, lower `base_channels` to 8 in `rtx4050.yaml`, or set `patch_size` to `[64, 64, 64]`.
- If Windows shows DataLoader errors, add `--override num_workers=0`.

## 5. Evaluate on the test split

```bash
python infer_multimodal.py --checkpoint output/rtx4050/best.pth \
  --config multimodal/config/rtx4050.yaml --data_root data/BraTS2021 \
  --split test --sliding_window --out_dir output/rtx4050/predictions --save_nifti
```

Dice scores (WT / TC / ET) are printed at the end and saved to `output/rtx4050/predictions/aggregate_metrics.json`.
