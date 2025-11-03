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

