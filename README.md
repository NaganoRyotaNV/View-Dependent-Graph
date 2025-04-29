# View-Dependent-Graph

本リポジトリは Pascal3D⁺ と ObjectNet3D を  
「物体 × 視線方向」ラベル付きデータセットへ加工し、  
ViG (Vision GNN) 系バックボーンで学習・評価するまでを  
ワンコマンドで再現できるようにまとめたプロジェクトです。

---

## 1. 必要データの取得

| データセット | 配布ページ | 備考 |
| :--- | :--- | :--- |
| ObjectNet3D | https://cvgl.stanford.edu/projects/objectnet3d/ | からアノテーションデータと画像データを取得してください |
| Pascal3D⁺ | https://paperswithcode.com/dataset/pascal3d-2 | Pascal3D+ Release 1.1 の Images と Annotations を取得してください |

> **注意**  
> 各データセットは研究目的ライセンスです。  
> 利用規約に同意したうえでダウンロード・解凍し、パスをスクリプト内の変数に合わせてください。

---

## 2. 環境構築

```bash
git clone <your-repo-url>  # 例: git@github.com:your-name/View-Dependent-Graph.git
cd View-Dependent-Graph

# Conda 環境
conda create -n vig_env python=3.9
conda activate vig_env
pip install -r requirements.txt  # CUDA 11.8 + PyTorch 2.5.1 前提

# 任意: Apex が必要な場合
# git clone https://github.com/NVIDIA/apex && cd apex && pip install -v . && cd ..
```

---

## 3. データセット生成パイプライン

### 3-1 Pascal3D⁺

```bash
cd Pascal3D_sets
chmod +x run_all.sh
./run_all.sh
```

生成物（例）:

```
Pascal3D_sets/data/
├─ maindata/           # object_viewpoint
├─ cropped_out/        # 正方形クロップ
├─ cropped_split/      # train/val/test（object_viewpoint）
└─ cropped_unified/    # train/val/test（object のみ）
```

---

### 3-2 ObjectNet3D

```bash
cd Object3Dsets
chmod +x make_objectnet3d_dataset.sh
./make_objectnet3d_dataset.sh
```

同様の4段階フォルダと分析レポート（TXT）が生成されます。

---

## 4. 学習

```bash
cd vig_pytorch   # ViG 実装ディレクトリ
python train.py \
  ../Object3Dsets/data/ObjectNet3D_Split \
  --model vig_b_224_gelu \
  --num-classes 436 \
  --opt adamw --sched cosine --epochs 300 \
  --batch-size 128 --lr 1e-3 --weight-decay 0.05 \
  --warmup-epochs 5 --amp --model-ema \
  --scale 0.8 1.0 --ratio 0.75 1.33 --hflip 0.5 \
  --output ../results/train
```

> Pascal3D⁺ で学習する場合は `--num-classes 51` に変更してください。

---

## 5. 評価 & 疑似アスペクトグラフ

```bash
python train_eval1.py \
  ../Object3Dsets/data/ObjectNet3D_Split \
  --evaluate \
  --config ../results/train/XXXX/args.yaml \
  --resume ../results/train/XXXX/last.pth.tar \
  --output ../results/test_data \
  --eval-dir test   # val/train も可
```

---

## 6. よくあるトラブル

| 症状 | チェック項目 |
| :--- | :--- |
| PytorchStreamWriter failed writing… | ディスク容量不足 (`df -h`) |
| DataLoader が遅い | `--workers` 増加 / SSD配置 / `--pin-mem` 有効化 |
| GPU メモリ不足 | `--batch-size` 減少 / `--amp` 有効化 |
| Apex が無い | 上記 Apex 手順を実行 or FP32学習へ切替 |

---