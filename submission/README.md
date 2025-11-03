**exp002 Kaggle Submission**

- Script: `submission/exp002_baseline.py`
- Purpose: Minimal, dependency-light baseline that predicts per-`target_name` mean.
- Output: Writes `submission.csv` in the working directory.

How to use on Kaggle (Notebook or Script):
- Ensure the competition dataset is added as an input. The script auto-detects the dataset folder under `/kaggle/input` by finding a directory containing `train.csv` and `test.csv`.
- Run: `python submission/exp002_baseline.py`
- Optionally specify data path: `python submission/exp002_baseline.py --data-dir /kaggle/input/<dataset>`

Local test:
- `python submission/exp002_baseline.py --data-dir data`

---

**exp003 Kaggle Training (single-file)**

- Script: `submission/exp003_train.py`
- Purpose: Train two-stream 3-output model per task.md; saves `best_model_fold{fold}.pth` under `/kaggle/working/exp003` by default.
- Notes:
  - Ignores unknown notebook args (fixes `-f kernel.json` error).
  - Disables Albumentations version check (no internet).
  - Falls back to a small CNN if `timm` is unavailable.

Run on Kaggle:
- `python submission/exp003_train.py`
- Options: `--data-dir /kaggle/input/<dataset> --epochs 3 --train-batch-size 8 --img-size 768 --model-name convnext_tiny`

Local test (requires timm/torch/albumentations/opencv):
- `python submission/exp003_train.py --data-dir data --epochs 1`
