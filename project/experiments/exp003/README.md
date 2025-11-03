**exp003: Two-stream 3-output Training**

- Model: `convnext_tiny` encoder shared on left/right halves; three regression heads for `['Dry_Total_g','GDM_g','Dry_Green_g']`.
- Transforms: Albumentations with ReplayCompose to synchronize augmentations.
- Splits: 5-fold KFold on unique images.
- Output: Saves `best_model_fold{fold}.pth` to `project/results/exp003/` by default.

Run
- `python -m project.experiments.exp003.train --train-csv data/train.csv --image-dir data/train --output-dir project/results/exp003`

Notes
- Ensure inference uses the same `BiomassModel` architecture (two-stream, three heads) to load checkpoints.
- Uses only image information; no tabular features are consumed.

