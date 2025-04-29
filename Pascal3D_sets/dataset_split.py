#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import random
import math
import shutil
from tqdm import tqdm # tqdmを追加

def main():
    parser = argparse.ArgumentParser(
        # <<< 修正 >>> 説明文を更新
        description="入力ディレクトリ ('物体_方向' ラベル) の画像を 6:3:1 に分割して train/val/test にコピーし、最終的な物体ラベル別・物体_方向ラベル別の枚数をtxtに出力"
    )
    # <<< 修正 >>> ヘルプメッセージを更新
    parser.add_argument("--src_dir", required=True,
                        help="元画像のディレクトリ (例: /mnt/data/PASCAL3D_Cropped)。直下に '物体_方向' ラベルのディレクトリがある想定。")
    parser.add_argument("--out_dir", required=True,
                        help="分割先のディレクトリ (例: /mnt/data/PASCAL3D_Split)。")
    parser.add_argument("--seed", type=int, default=42,
                        help="ランダムシード (デフォルト: 42)")
    args = parser.parse_args()

    src_dir = args.src_dir
    out_dir = args.out_dir
    seed = args.seed

    if not os.path.isdir(src_dir):
        print(f"Error: Source directory not found: {src_dir}")
        return

    random.seed(seed)

    train_dir = os.path.join(out_dir, "train")
    val_dir   = os.path.join(out_dir, "val")
    test_dir  = os.path.join(out_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # <<< 修正 >>> 集計用の辞書を新しい構造に合わせて変更
    #  1) 物体ラベル別合計 => object_counts[split][object_label]
    #  2) 物体_方向ラベル別 => label_counts[split][label_name]
    # ----------------------------------------------------------------
    object_counts = {"train": {}, "val": {}, "test": {}}
    label_counts = {"train": {}, "val": {}, "test": {}}

    # <<< 修正 >>> src_dir 下の '物体_方向' ラベルディレクトリを走査
    try:
        label_dirs = sorted([d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))])
    except Exception as e:
        print(f"Error listing label directories in '{src_dir}': {e}")
        return

    if not label_dirs:
        print(f"No label directories found in '{src_dir}'.")
        return

    print(f"Found {len(label_dirs)} label directories. Starting split...")

    # <<< 修正 >>> 単一ループで処理
    for label_name in tqdm(label_dirs, desc="Processing labels"):
        source_label_dir = os.path.join(src_dir, label_name)

        # 画像ファイルを列挙
        try:
            image_files = [f for f in os.listdir(source_label_dir)
                           if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        except Exception as e:
            print(f"Warning: Error listing images in '{source_label_dir}': {e}. Skipping.")
            continue

        if not image_files:
            # print(f"Info: No images found in '{label_name}'. Skipping.")
            continue

        random.shuffle(image_files)

        total = len(image_files)
        # 分割数の計算 (train=60%, val=30%, test=残り)
        n_train = math.floor(total * 0.6)
        n_val   = math.floor(total * 0.3)
        # 最小ファイル数チェック（各セットに最低1ファイル割り当てるか）
        if total > 0 and (n_train == 0 or n_val == 0 or (total - n_train - n_val) == 0):
             print(f"Warning: Skipping label '{label_name}' ({total} files). Cannot split into non-empty train/val/test sets.")
             continue # このラベルはスキップ

        n_test = total - n_train - n_val

        train_list = image_files[:n_train]
        val_list   = image_files[n_train : n_train + n_val]
        test_list  = image_files[n_train + n_val :]

        # <<< 修正 >>> 出力先サブフォルダ名を label_name に基づいて生成
        train_subdir = os.path.join(train_dir, label_name)
        val_subdir   = os.path.join(val_dir,   label_name)
        test_subdir  = os.path.join(test_dir,  label_name)
        try:
            os.makedirs(train_subdir, exist_ok=True)
            os.makedirs(val_subdir, exist_ok=True)
            os.makedirs(test_subdir, exist_ok=True)
        except OSError as e:
            print(f"Error creating output subdirectories for '{label_name}': {e}. Skipping.")
            continue

        # コピー処理を関数化してDRYに
        def copy_files(file_list, src_dir, dst_dir):
            copied_count = 0
            for f in file_list:
                src_path = os.path.join(src_dir, f)
                dst_path = os.path.join(dst_dir, f)
                try:
                    shutil.copy2(src_path, dst_path)
                    copied_count += 1
                except Exception as e:
                    print(f"Error copying {src_path} to {dst_path}: {e}")
            return copied_count

        copied_train = copy_files(train_list, source_label_dir, train_subdir)
        copied_val = copy_files(val_list, source_label_dir, val_subdir)
        copied_test = copy_files(test_list, source_label_dir, test_subdir)

        # 簡易ログ出力 (コピー数ベースに変更)
        # print(f"[{label_name}] total={total} => copied: train={copied_train}, val={copied_val}, test={copied_test}")

        # --- 集計 ---
        # 1. 物体_方向ラベル別 (label_counts)
        label_counts["train"][label_name] = copied_train
        label_counts["val"][label_name] = copied_val
        label_counts["test"][label_name] = copied_test

        # 2. 物体ラベル別 (object_counts)
        # ラベル名から物体ラベルを抽出
        if '_' in label_name:
            object_label = label_name.rsplit('_', 1)[0]
        else:
            object_label = label_name # '_' がない場合はそのまま

        object_counts["train"].setdefault(object_label, 0)
        object_counts["val"].setdefault(object_label, 0)
        object_counts["test"].setdefault(object_label, 0)

        object_counts["train"][object_label] += copied_train
        object_counts["val"][object_label]   += copied_val
        object_counts["test"][object_label]  += copied_test


    # <<< 修正 >>> 最後に集計結果を txt に書き出す
    out_txt = os.path.join(out_dir, "dataset_counts.txt")
    print(f"\nWriting summary to {out_txt}...")
    try:
        with open(out_txt, "w", encoding='utf-8') as f:
            grand_total = 0
            # まず物体ラベル別合計
            f.write("===== Object Label Totals =====\n")
            for split_name in ["train", "val", "test"]:
                f.write(f"\n--- {split_name.upper()} ---\n")
                split_total = 0
                # 物体ラベルでソートして出力
                for obj_label, num in sorted(object_counts[split_name].items()):
                    f.write(f"  {obj_label}: {num}\n")
                    split_total += num
                f.write(f"  --------------------\n")
                f.write(f"  Subtotal ({split_name}): {split_total}\n")
                grand_total += split_total

            f.write("\n===============================\n")
            f.write(f"Grand Total (all splits): {grand_total}\n")
            f.write("===============================\n")


            # 続いて物体_方向ラベル別
            f.write("\n===== Object_Direction Label Breakdown =====\n")
            for split_name in ["train", "val", "test"]:
                f.write(f"\n--- {split_name.upper()} ---\n")
                # 物体_方向ラベルでソートして出力
                for label, num in sorted(label_counts[split_name].items()):
                    f.write(f"  {label}: {num}\n")

    except IOError as e:
         print(f"Error writing summary file: {e}")

    print("Split process completed.")
    print(f"Split data saved under: {out_dir}")
    print(f"Dataset counts saved to: {out_txt}")

if __name__ == "__main__":
    main()