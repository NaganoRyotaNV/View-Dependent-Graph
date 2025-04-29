#!/usr/bin/env bash
# ============================================================
#  ObjectNet3D データセット前処理フルパイプライン
#    1. 画像ソート（方位カテゴリ付与）          sort_objectnet3d.py
#    2. バウンディングボックスでクロップ         crop_ObjectNet.py
#    3. train / val / test へ分割                dataset_split.py
#    4. 方向ラベルを外して物体ラベルだけに統合    flatten_viewpoints.py
#    5. 画像枚数など構造解析                    analyze_structure.py
#
#   使い方:
#       chmod +x make_objectnet3d_dataset.sh
#       ./make_objectnet3d_dataset.sh
#
#   必要に応じてパスや比率を編集してください。
# ============================================================
set -euo pipefail   # どこかで失敗したら即終了&未定義変数はエラー

# ----------[ 0 事前に調整したいパス / パラメータ ]-----------------
# 元データ
OBJ3D_ANN="/mnt/data/ObjectNet3D/Annotations"
OBJ3D_IMG="/mnt/data/ObjectNet3D/Images"

# 各ステップの出力場所
STEP1_OUT="/mnt/data/ObjectNet3D_Sorted_Combined"
STEP2_OUT="/mnt/data/ObjectNet3D_Cropped_Combined"
SPLIT_OUT="/mnt/data/ObjectNet3D_Split"
FLAT_OUT="/mnt/data/ObjectNet3D_Split_Flattened"

# train / val / test 割合
TRAIN_RATIO=0.6   # 60 %
VAL_RATIO=0.3     # 30 %
SEED=42
# --------------------------------------------------------------


echo "========== 1) ソート (方位分類付きコピー) =========="
python sort_objectnet3d.py

echo "========== 2) クロップ (BBox → 正方形) =========="
python crop_ObjectNet.py

echo "========== 3) train / val / test に分割 =========="
python dataset_split.py \
  --input_dir  "$STEP2_OUT" \
  --output_dir "$SPLIT_OUT" \
  --train_ratio "$TRAIN_RATIO" \
  --val_ratio   "$VAL_RATIO" \
  --seed        "$SEED"

echo "========== 4) 方向ラベルを物体ラベルに統合 =========="
python flatten_viewpoints.py \
  --input_dir  "$SPLIT_OUT" \
  --output_dir "$FLAT_OUT"

echo "========== 5-A) フラット構造の枚数確認 =========="
python analyze_structure.py flat \
  --input_dir   "$FLAT_OUT" \
  --output_file "${FLAT_OUT}/flat_analysis_results.txt"

echo "========== 5-B) 階層構造の枚数確認 =========="
python analyze_structure.py hierarchical \
  --input_dir   "$SPLIT_OUT" \
  --output_file "${SPLIT_OUT}/hierarchical_analysis_results.txt"

echo "🎉  すべて完了しました！"
echo "    - フラット構造:           $FLAT_OUT"
echo "      ⇒ 分析結果:            ${FLAT_OUT}/flat_analysis_results.txt"
echo "    - 方向ラベル付き階層構造: $SPLIT_OUT"
echo "      ⇒ 分析結果:            ${SPLIT_OUT}/hierarchical_analysis_results.txt"
