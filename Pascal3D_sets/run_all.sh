#!/usr/bin/env bash
set -e  # 何か失敗したらそこで停止

########################################
# 0) 前処理: 出力用ディレクトリを掃除 (任意)
########################################
# 必要に応じて既存のディレクトリを削除
# rm -rf /mnt/data/maindata
# rm -rf /mnt/data/cropped_out
# rm -rf /mnt/data/cropped_split
# rm -rf /mnt/data/cropped_unified

########################################
# 1) Pascal3D+ から視線方向ごとに分類 ('物体_方向' ラベルで出力)
#    (split_pascal3d_directions.py が '物体_方向' 出力に修正されている前提)
########################################
echo "===== Step 1: Direction classification (output: object_direction) ====="
python split_pascal3d_directions.py \
    --images_dir /mnt/data/PASCAL3D+_release1.1/Images \
    --annotations_dir /mnt/data/PASCAL3D+_release1.1/Annotations \
    --output_dir /mnt/data/maindata # 出力は '物体_方向' ラベルのディレクトリ構造

########################################
# 2) 正方形クロップ (入力・出力ともに '物体_方向' ラベル)
#    (crop_square.py が '物体_方向' 構造に対応している前提)
########################################
echo "===== Step 2: Cropping by bbox (structure: object_direction) ====="
python crop_square.py \
    --src_dir /mnt/data/maindata \
    --ann_dir /mnt/data/PASCAL3D+_release1.1/Annotations \
    --out_dir /mnt/data/cropped_out # 出力も '物体_方向' ラベルのディレクトリ構造

########################################
# 3) train/val/test に分割 (入力・出力ともに '物体_方向' ラベル)
#    (dataset_split.py が '物体_方向' 構造に対応している前提)
########################################
echo "===== Step 3: Dataset split (structure: split/object_direction) ====="
python dataset_split.py \
    --src_dir /mnt/data/cropped_out \
    --out_dir /mnt/data/cropped_split \
    --seed 42 # 出力は split/物体_方向 ラベルのディレクトリ構造

########################################
# 4) '物体_方向' ラベルを '物体' ラベルに統合
#    (unify_direction.py が '物体_方向' -> '物体' 統合に修正されている前提)
########################################
echo "===== Step 4: Merge labels to object only (structure: split/object) ====="
python unify_direction.py \
    --src_dir /mnt/data/cropped_split \
    --out_dir /mnt/data/cropped_unified # 出力は split/物体 ラベルのディレクトリ構造

########################################
# 5) ディレクトリ枚数の集計をテキスト出力 (デバッグ用)
########################################
echo "===== Step 5: Counting images for debug ====="

# A) 分割後の '物体_方向' 構造 (/mnt/data/cropped_split) の枚数をカウント
#    集計は「物体ラベル」ごとに行う
echo "  Counting images in split data (aggregated by object label)..."
# <<< 修正: Pythonコードを '物体_方向' 構造に合わせて修正 >>>
python -c "
import os
from collections import defaultdict

root_dir = '/mnt/data/cropped_split' # 分析対象ディレクトリ
out_txt  = os.path.join(root_dir, 'dataset_counts_object_label_agg.txt') # 出力ファイル名変更

def count_images_in_split(split_dir):
    object_counts = defaultdict(int) # 物体ラベルごとの合計
    label_counts = {} # 物体_方向ラベルごとのカウント（参考用）
    if not os.path.isdir(split_dir): return {}, {}
    try:
        # '物体_方向' ラベルのディレクトリをリストアップ
        label_dirs = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
    except Exception as e:
        print(f'Warning: Error listing labels in {split_dir}: {e}')
        return {}, {}
    for label_name in label_dirs:
        label_path = os.path.join(split_dir, label_name)
        try:
            imgs = [x for x in os.listdir(label_path) if x.lower().endswith(('.jpg','.jpeg','.png'))]
            count = len(imgs)
            label_counts[label_name] = count # 参考用に記録
            # 物体ラベルを抽出して集計
            if '_' in label_name: object_label = label_name.rsplit('_', 1)[0]
            else: object_label = label_name
            object_counts[object_label] += count
        except Exception as e:
            print(f'Warning: Error processing {label_path}: {e}')
    # 物体ラベルごとの集計結果と、参考用のラベル別カウントを返す
    return dict(object_counts), label_counts

splits = ['train','val','test']
grand_total = 0
with open(out_txt,'w', encoding='utf-8') as f:
    f.write(f'# Image Counts in {root_dir} (Aggregated by Object Label)\n')
    f.write('# Source Structure: split/object_direction/image.*\n')
    f.write('='*50 + '\\n')
    for s in splits:
        s_dir = os.path.join(root_dir, s)
        f.write(f'\\n--- {s.upper()} ---\\n')
        # 物体ラベルごとの集計結果を取得
        obj_results, _ = count_images_in_split(s_dir)
        split_total = 0
        if not obj_results: f.write('  No object labels found or error occurred.\\n')
        else:
            # 物体ラベルでソートして出力
            for obj_label, num in sorted(obj_results.items()):
                f.write(f'  {obj_label}: {num}\\n')
                split_total += num
            f.write(f'  --------------------\\n')
            f.write(f'  Subtotal ({s}): {split_total}\\n')
            grand_total += split_total
    f.write('\\n' + '='*50 + '\\n')
    f.write(f'Grand Total (all splits): {grand_total}\\n')
    f.write('='*50 + '\\n')
print(f'Wrote aggregated object counts from {root_dir} to {out_txt}')
"

# B) 物体ラベルのみに統合後のディレクトリ (/mnt/data/cropped_unified) の枚数をカウント
#    このパートは 'split/物体' 構造をカウントするため、修正不要
echo "  Counting images in unified data (object labels only)..."
python -c "
import os

root_dir = '/mnt/data/cropped_unified' # 分析対象ディレクトリ
out_txt  = os.path.join(root_dir, 'dataset_counts_object_only.txt') # 出力ファイル名変更推奨

def count_images(split_dir):
    count_dict = {}
    if not os.path.isdir(split_dir): return {}
    try:
        # 物体ラベルディレクトリをリストアップ
        object_labels = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
    except Exception as e:
        print(f'Warning: Error listing object labels in {split_dir}: {e}')
        return {}
    for obj_label in object_labels:
        obj_path = os.path.join(split_dir, obj_label)
        try:
            imgs = [x for x in os.listdir(obj_path) if x.lower().endswith(('.jpg','.jpeg','.png'))]
            count_dict[obj_label] = len(imgs)
        except Exception as e:
            print(f'Warning: Error processing {obj_path}: {e}')
            count_dict[obj_label] = 'Error'
    return count_dict

splits = ['train','val','test']
grand_total = 0
with open(out_txt,'w', encoding='utf-8') as f:
    f.write(f'# Image Counts in {root_dir} (Object Labels Only)\n')
    f.write('# Structure: split/object/image.*\n')
    f.write('='*50 + '\\n')
    for s in splits:
        s_dir = os.path.join(root_dir, s)
        f.write(f'\\n--- {s.upper()} ---\\n')
        results = count_images(s_dir)
        split_total = 0
        if not results: f.write('  No object labels found or error occurred.\\n')
        else:
            for obj_label, num in sorted(results.items()):
                f.write(f'  {obj_label}: {num}\\n')
                if isinstance(num, int): split_total += num
            f.write(f'  --------------------\\n')
            f.write(f'  Subtotal ({s}): {split_total}\\n')
            grand_total += split_total
    f.write('\\n' + '='*50 + '\\n')
    f.write(f'Grand Total (all splits): {grand_total}\\n')
    f.write('='*50 + '\\n')
print(f'Wrote object-only counts from {root_dir} to {out_txt}')
"

echo "All steps done."