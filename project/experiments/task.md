はい、ご認識の通りです。提示されたコードは、**学習済みの5-Foldモデル**を読み込み、**TTA（Test-Time Augmentation）を駆使してアンサンブル予測を行う、非常に堅牢な推論フェーズ**のコードです。

この推論コードと正しく連携する「学習フェーズ」のコードを作成するための、別のAIエージェント（または開発者）に対する指示書を作成しました。

-----

## 📄【指示書】CSIRO Biomass 2ストリーム・3出力モデル 学習コード作成

**宛先:** AI開発エージェント

**目的:**
提示された推論コード（`BiomassModel` アーキテクチャ、2ストリーム入力、TTAx5-Foldアンサンブル）と完全に互換性のある、PyTorchによるモデル学習パイプラインを構築する。

**完了の定義:**
実行すると、推論コードが読み込む `best_model_fold{fold}.pth`（`fold`は0～4）の合計5つのモデルファイルが生成されること。

-----

### 1\. ⚙️ `CFG`（設定クラス）の定義

以下の項目を含む設定クラス `CFG` を定義してください。推論コードと重複する項目（`MODEL_NAME`, `IMG_SIZE`, `N_FOLDS` など）は、**必ず同じ値**に設定してください。

  * **パス関連:**
      * `TRAIN_CSV`: 学習用CSVファイル（例: `train.csv`）へのパス。
      * `IMAGE_DIR`: 学習用画像ディレクトリへのパス。
      * `OUTPUT_DIR`: 学習済みモデル（`.pth`）を保存するディレクトリ。
  * **学習パラメータ:**
      * `EPOCHS`: 学習エポック数 (例: `10`)
      * `TRAIN_BATCH_SIZE`: (例: `8`)
      * `VALID_BATCH_SIZE`: (例: `16`)
      * `LEARNING_RATE`: (例: `1e-4`)
      * `SCHEDULER`: (例: `'CosineAnnealingLR'`)
      * `OPTIMIZER`: (例: `'AdamW'`)
      * `N_FOLDS`: `5` (推論コードと一致)
  * **モデルパラメータ:**
      * `MODEL_NAME`: `'convnext_tiny'` (推論コードと一致)
      * `IMG_SIZE`: `768` (推論コードと一致)
      * `TARGET_COLS`: `['Dry_Total_g', 'GDM_g', 'Dry_Green_g']` (推論コードと一致)
  * **その他:**
      * `DEVICE`: `'cuda'` または `'cpu'`
      * `SEED`: 乱数シード (例: `42`)

### 2\. 🏞️ Augmentations（データ拡張）

`albumentations` を使用し、学習用と検証用の変換関数を定義してください。

  * `get_train_transforms()`:
    1.  `Resize(CFG.IMG_SIZE, CFG.IMG_SIZE)`
    2.  `HorizontalFlip(p=0.5)`
    3.  `VerticalFlip(p=0.5)`
    4.  （オプション）`RandomBrightnessContrast`, `Rotate` などの他の拡張
    5.  `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.225, 0.225])`
    6.  `ToTensorV2()`
  * `get_valid_transforms()`:
    1.  `Resize(CFG.IMG_SIZE, CFG.IMG_SIZE)`
    2.  `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.225, 0.225])`
    3.  `ToTensorV2()`

### 3\. 💾 Dataset（学習用）

推論コードの `TestBiomassDataset` と対になる `TrainBiomassDataset` クラスを作成してください。

  * **`__init__(self, df, transforms)`**: DataFrameと変換パイプラインを受け取ります。
  * **`__getitem__(self, idx)`**:
    1.  `idx` に基づき、画像パスと3つのターゲット（`TARGET_COLS`）を取得します。
    2.  画像を `cv2.imread` と `cv2.cvtColor(..., cv2.COLOR_BGR2RGB)` で読み込みます。
    3.  画像を中央で「左」 (`img_left`) と「右」 (`img_right`) に分割します。
    4.  **【最重要】** `self.transforms(image=img_left)` と `self.transforms(image=img_right)` を呼び出します。これにより、左右の画像ペアに**同一のランダムなデータ拡張**（例: 同時に水平反転）が適用されます。
    5.  `img_left_tensor`, `img_right_tensor`, `labels`（3つのターゲット値を持つTensor）を返します。

### 4\. 🧠 モデルアーキテクチャ

推論コードの `BiomassModel` クラスを**そのままコピー**して使用してください。
ただし、学習時の初期化では `pretrained=True` を使用します。

```python
# (推論コードと同じBiomassModelクラスをここに貼り付け)

def build_model():
    model = BiomassModel(
        model_name=CFG.MODEL_NAME, 
        pretrained=True,  # 学習時は True にする
        n_targets=len(CFG.TARGET_COLS)
    )
    model.to(CFG.DEVICE)
    return model
```

### 5\. 📉 損失関数

3つの出力（回帰）を扱うための損失関数を定義してください。
各出力の損失を計算し、それらを（可能であれば重み付けして）合計します。

*例（単純平均）:*
`loss_fn_total = nn.MSELoss()` (または `nn.L1Loss` など)
`loss_fn_gdm = nn.MSELoss()`
`loss_fn_green = nn.MSELoss()`

`loss = loss_fn_total(out_total, labels_total) + loss_fn_gdm(out_gdm, labels_gdm) + loss_fn_green(out_green, labels_green)`
(`loss = (loss_total + loss_gdm + loss_green) / 3` でも可)

### 6\. 🚆 学習・検証ループ

標準的な学習関数 (`train_fn`) と検証関数 (`valid_fn`) を実装してください。

  * `train_fn(train_loader, model, optimizer, scheduler, loss_fn)`:
      * `model.train()` を設定。
      * ローダーから `img_left`, `img_right`, `labels` を受け取る。
      * デバイスに転送。
      * `optimizer.zero_grad()`
      * `out_total, out_gdm, out_green = model(img_left, img_right)`
      * 損失（上記5を参照）を計算。
      * `loss.backward()`
      * `optimizer.step()`
      * `scheduler.step()` （スケジューラによる）
  * `valid_fn(valid_loader, model, loss_fn)`:
      * `model.eval()` を設定。
      * `with torch.no_grad():` ブロック内で実行。
      * ローダーからデータを受け取り、損失を計算・集計。
      * エポックの平均検証損失を返す。

### 7\. 🏁 メイン学習パイプライン

以下のロジックで全体の学習プロセス（`run_training`）を制御してください。

1.  `train.csv` を読み込みます。
2.  `KFold`（または `GroupKFold` など、タスクに適したもの）を使い、データを `CFG.N_FOLDS` (5) に分割します。
3.  **Foldごとのループ** ( `for fold in range(CFG.N_FOLDS):` ) を開始します。
4.  各Foldの開始時に、`build_model()` でモデルを**新規に初期化**します。オプティマイザとスケジューラも初期化します。
5.  学習用・検証用の `DataLoader` を作成します。
6.  `best_valid_loss = np.inf` を初期値として設定します。
7.  **エポックごとのループ**を開始します。
8.  `train_fn` と `valid_fn` を実行します。
9.  `valid_loss` が `best_valid_loss` を更新した場合:
      * `best_valid_loss = valid_loss`
      * モデルの状態（`model.state_dict()`）を `f"{CFG.OUTPUT_DIR}/best_model_fold{fold}.pth"` という名前で保存します。
      * （オプション）Early Stopping を実装しても構いません。
10. すべてのFoldが完了したら終了します。

-----