# Problem Log

本ドキュメントは提出時の不具合・低スコアの原因と対策を記録します。実験ごとに ID を付与し、再発防止のためのチェックリストも併記します。

---

## PROB-001: Kaggle 提出が「ダメだった」
- 日付: 2025-11-02 (UTC)
- 対象: exp002（`submission/exp002_submission.py`）

### 症状
- Kaggle に提出したが、結果が期待に届かない／もしくは提出が受理されなかった。
  - 想定される失敗パターン:
    1) 予測列に NaN/Inf が混入している（前処理・型変換の不整合）
    2) 物理量として負の予測が出ている（バリデーションで弾かれる可能性）
    3) 行数・順序が `sample_submission.csv` と一致していない
    4) 形式は正しいが、特徴量が貧弱で LB が極端に低い

### 原因（推定）
- Kaggle 側データセットには `test.csv` が存在しない（または列が極端に少ない）ため、`submission` スクリプトが `test.csv` 前提で失敗。
- さらに、本コンペの test 側は `sample_id` から `image_id` と `target_name` が復元可能で、ここから最低限の特徴を作る必要があった。
- 加えて、前処理の端数・欠損により NaN/負値が出ると、提出が拒否される可能性がある。

### 対策（実施済み）
- `submission/exp002_submission.py` と `kaggle/exp002_baseline_submission.py` を修正:
  - `test.csv` が無い場合は `sample_submission.csv` の `sample_id` を分解して `test` DataFrame を生成（`image_path`, `target_name`）。
  - 予測値の非負クリップ: `np.clip(pred, 0.0, None)`
  - NaN/非有限値の検知・明示的エラー: `_validate_submission()`
  - `sample_id` による行順アラインの明示化（既存維持）
- 問題再発防止チェックリスト:
  - [ ] 提出直前に `len(submission) == len(sample_submission)` を assert
  - [ ] `target` 列が全て有限かつ非負
  - [ ] カラム名が `sample_submission.csv` と一致

### 次のアクション（改善案）
- H-02（特徴量拡張）: 画像からの簡易統計量（平均/分散/ヒストグラムベースの特徴など）を抽出し、`target_name` と併用する。
  - 目的: test 側にも生成できる特徴を増やして、LB を底上げ。
  - 実装案: `project/features/image_stats.py` を作成し、`train/` と `test/` の JPG を一括処理 → CSV で保存 → 学習・推論で結合。
- バリデーションの一貫性: Notebook/Script 双方で同一の前処理・検証ロジックを使うユーティリティを整備。

---

## 付録: 参考コマンド
- ローカル検証（exp002 ベースライン）
  - `python -m project.experiments.exp002.baseline --data-dir data`
- Kaggle 提出生成（Notebook 上）
  - `!python submission/exp002_submission.py`
- 出力確認
  - `import pandas as pd; pd.read_csv('/kaggle/working/submission.csv').describe()`
