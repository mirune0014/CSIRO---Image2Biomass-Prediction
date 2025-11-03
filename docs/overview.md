# Image2Biomassコンペ概要

Image2Biomassは、画像・地上計測・公開データセットを用いて牧草バイオマス量を予測するモデルを構築するコンペティションです。  
本モデルは、牧場管理者が家畜の放牧タイミングや方法を判断するために活用されます。

---

## コンペの目的

牧草地のバイオマス（飼料量）は、放牧や休耕、土壌の健康維持などに重要な役割を果たします。  
従来の「刈り取り・計量法」は高精度ですが、手間と時間がかかり大規模運用には不向きです。  
本コンペでは、画像・地上計測・NDVIなど多様なデータを活用し、より効率的かつ高精度なバイオマス推定モデルの開発を目指します。

---

## 評価指標

モデルの性能は、5つの出力次元のスコアの加重平均で評価されます。

- Dry_Green_g: 0.1
- Dry_Dead_g: 0.1
- Dry_Clover_g: 0.1
- GDM_g: 0.2
- Dry_Total_g: 0.5

最終スコアは以下の式で計算されます：
    $$Final Score = \sum_{i=1}^{5} (w_i \times R_i^2)$$

各次元の決定係数 $R^2$ は以下で定義されます：
    $$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$
  
残差平方和 $SS_{res}$（予測誤差の合計）：
    $$SS_{res} = \sum_{j} (y_j - \hat{y}_j)^2$$

全平方和 $SS_{tot}$（データの分散）：
    $$SS_{tot} = \sum_{j} (y_j - \bar{y})^2$$

- $y_j$：実測値
- $\hat{y}_j$：予測値
- $\bar{y}$：実測値の平均

---

## 提出ファイル形式

提出はCSV（long format）で、以下2列を持ちます：

| sample_id | target |
|-----------|--------|
| 画像ID__ターゲット名 | 予測値（g, float） |

例：

```
sample_id,target
ID1001187975__Dry_Green_g,0.0
ID1001187975__Dry_Dead_g,0.0
ID1001187975__Dry_Clover_g,0.0
ID1001187975__GDM_g,0.0
ID1001187975__Dry_Total_g,0.0
```

各画像につき5行（5ターゲット）必要です。

---

## スケジュール

- 2025年10月28日：開始
- 2026年1月21日：エントリー締切・ルール同意必須
- 2026年1月21日：チーム合併締切
- 2026年1月28日：最終提出締切

全てUTC 23:59締切。主催者判断で変更の可能性あり。

---

## コード要件

- Notebook形式で提出
- CPU/GPUとも最大9時間実行
- インターネットアクセス不可
- 公開データ・事前学習モデル利用可
- 提出ファイル名は submission.csv

詳細はFAQ・デバッグドキュメント参照。

---

## データセット概要

予測対象（5成分）：

- 緑色植生（クローバー除く）
- 枯死物質
- クローバー乾燥バイオマス
- 緑色乾燥物質（GDM）
- 総乾燥バイオマス

### ファイル構成

- test.csv：予測対象リスト（画像×ターゲット）
- train/：学習画像（JPEG）
- test/：テスト画像（採点時のみ利用）
- train.csv：学習データ詳細（画像パス、採取日、州、種、NDVI、草丈、ターゲット名、実測値）
- sample_submission.csv：提出サンプル

---

## 引用

本データセットを研究利用する場合は、以下論文を引用してください。

```
@misc{liao2025estimatingpasturebiomasstopview,
  title={Estimating Pasture Biomass from Top-View Images: A Dataset for Precision Agriculture},
  author={Qiyu Liao and Dadong Wang and Rebecca Haling and Jiajun Liu and Xun Li and Martyna Plomecka and Andrew Robson and Matthew Pringle and Rhys Pirie and Megan Walker and Joshua Whelan},
  year={2025},
  eprint={2510.22916},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2510.22916},
}


はい、承知いたしました。ファイル名と対応するMarkdown（LaTeX）形式の数式は以下の通りです。







