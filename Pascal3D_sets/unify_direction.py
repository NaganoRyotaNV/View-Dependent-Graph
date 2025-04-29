#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import shutil
from tqdm import tqdm # tqdmを追加

# <<< 修正 >>> 関数名を変更し、ロジックを新しい構造に合わせる
def merge_labels_to_object(src_dir, out_dir):
    """
    src_dir 下にある train/val/test のサブディレクトリを走査し、
    '物体ラベル_方向ラベル' ディレクトリ内の画像を、
    '物体ラベル' ディレクトリに「コピー」して統合する。
    ファイル名が衝突した場合は警告を出してスキップする。

    Args:
        src_dir (str): 分割済みデータディレクトリ (例: /mnt/data/PASCAL3D_Split)
                       train/val/test 下に '物体_方向' ラベルのディレクトリ構造を持つ想定。
        out_dir (str): 統合後のデータを出力するディレクトリ (例: /mnt/data/PASCAL3D_Unified)
    """
    print(f"Starting merging process from '{src_dir}' to '{out_dir}'...")
    copied_files_count = 0
    skipped_collision_count = 0
    error_count = 0

    try:
        splits = sorted([d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))])
    except FileNotFoundError:
        print(f"Error: Source directory '{src_dir}' not found.")
        return
    except Exception as e:
        print(f"Error listing splits in '{src_dir}': {e}")
        return

    for split_name in tqdm(splits, desc="Processing splits (train/val/test)"):
        split_path = os.path.join(src_dir, split_name)
        out_split_dir = os.path.join(out_dir, split_name)
        os.makedirs(out_split_dir, exist_ok=True)

        # <<< 修正 >>> '物体_方向' ラベルディレクトリを列挙
        try:
            label_dirs = sorted([d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))])
        except Exception as e:
            print(f"Warning: Error listing label directories in '{split_path}': {e}. Skipping this split.")
            continue

        if not label_dirs:
            # print(f"Info: No label directories found in '{split_path}'.")
            continue

        # <<< 修正 >>> 単一ループで処理
        for label_name in tqdm(label_dirs, desc=f"  Merging {split_name}", leave=False):
            source_label_dir = os.path.join(split_path, label_name)

            # <<< 追加 >>> 物体ラベルを抽出
            if '_' in label_name:
                object_label = label_name.rsplit('_', 1)[0]
            else:
                object_label = label_name
                print(f"Warning: Label directory name '{label_name}' does not contain '_'. Using it directly as object label.")

            # <<< 修正 >>> 出力先の物体ラベルディレクトリ
            out_object_dir = os.path.join(out_split_dir, object_label)
            os.makedirs(out_object_dir, exist_ok=True)

            # 画像ファイルをコピー
            try:
                image_files = [f for f in os.listdir(source_label_dir)
                               if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            except Exception as e:
                 print(f"Warning: Error listing images in '{source_label_dir}': {e}. Skipping.")
                 continue

            for img_file in image_files:
                src_img_path = os.path.join(source_label_dir, img_file)
                dst_img_path = os.path.join(out_object_dir, img_file)

                try:
                    # <<< 追加 >>> ファイル名衝突チェック
                    if os.path.exists(dst_img_path):
                        # 既に存在する場合、警告を出してスキップ
                        print(f"Warning: Filename collision: '{img_file}' already exists in '{out_object_dir}'. Skipping copy from '{src_img_path}'.")
                        skipped_collision_count += 1
                        continue
                    else:
                        # copy2 を使う (メタデータ保持)
                        shutil.copy2(src_img_path, dst_img_path)
                        copied_files_count += 1
                except Exception as e:
                     print(f"Error copying {src_img_path} to {dst_img_path}: {e}")
                     error_count += 1

    print("\n--- Merging Summary ---")
    print(f"Successfully copied files: {copied_files_count}")
    print(f"Files skipped due to filename collision: {skipped_collision_count}")
    print(f"Errors during copying: {error_count}")
    print(f"Merged data saved to: {out_dir}")

# --- count_final_images 関数は変更不要 ---
def count_final_images(root_dir, output_txt):
    """
    統合後のディレクトリ構造 (train/val/test下に<物体ラベル>/*.jpg) で、
    各splitの各物体ラベルの画像枚数を数えて output_txt に書き出す。
    """
    print(f"\nCounting final images in '{root_dir}'...")
    try:
        with open(output_txt, "w", encoding='utf-8') as f:
            f.write("Final Image Counts (Object Labels)\n")
            f.write(f"Source Directory: {root_dir}\n")
            f.write("="*40 + "\n")

            grand_total = 0
            splits = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and d in ["train", "val", "test"]]) # train/val/testのみ対象

            for split_name in splits:
                split_path = os.path.join(root_dir, split_name)
                f.write(f"\n--- {split_name.upper()} ---\n")
                split_total = 0

                try:
                    # 物体ラベルディレクトリを列挙
                    object_labels = sorted([d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))])
                except Exception as e:
                    f.write(f"  Error listing object labels: {e}\n")
                    continue

                if not object_labels:
                     f.write("  No object labels found.\n")
                     continue

                for obj_label in object_labels:
                    obj_path = os.path.join(split_path, obj_label)
                    try:
                        # 画像ファイルを数える
                        files = [im for im in os.listdir(obj_path)
                                 if im.lower().endswith((".jpg",".jpeg",".png"))]
                        count = len(files)
                        f.write(f"  {obj_label}: {count}\n")
                        split_total += count
                    except Exception as e:
                         f.write(f"  {obj_label}: Error counting files ({e})\n")

                f.write(f"  --------------------\n")
                f.write(f"  Subtotal ({split_name}): {split_total}\n")
                grand_total += split_total

            f.write("\n" + "="*40 + "\n")
            f.write(f"Grand Total (all splits): {grand_total}\n")
            f.write("="*40 + "\n")
        print(f"Final counts saved to: {output_txt}")
    except IOError as e:
        print(f"Error writing final counts file '{output_txt}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred during final count: {e}")

def main():
    parser = argparse.ArgumentParser(
        # <<< 修正 >>> 説明文を更新
        description="入力ディレクトリ ('物体_方向' ラベル) の画像を、物体ラベルのみのディレクトリにコピーして統合する。最後に画像枚数をtxtに書き出す。"
    )
    # <<< 修正 >>> ヘルプメッセージを更新
    parser.add_argument("--src_dir", required=True,
                        help="分割済みデータディレクトリ (例: /mnt/data/PASCAL3D_Split)。train/val/test 下に '物体_方向' ラベルのディレクトリ構造を持つ想定。")
    parser.add_argument("--out_dir", required=True,
                        help="統合後のデータを出力するディレクトリ (例: /mnt/data/PASCAL3D_Unified)。train/val/test 下に '物体' ラベルのディレクトリ構造が作られる。")
    args = parser.parse_args()

    src_dir = args.src_dir
    out_dir = args.out_dir

    if not os.path.isdir(src_dir):
        print(f"Error: Source directory not found: {src_dir}")
        return

    # <<< 修正 >>> 修正後の関数を呼び出す
    merge_labels_to_object(src_dir, out_dir)

    # 統合後の画像枚数を final_counts.txt に出力
    final_counts_txt = os.path.join(out_dir, "final_counts.txt")
    count_final_images(out_dir, final_counts_txt)

    print("\nProcess finished.")
    print(f"Original split data remains in: {src_dir}")
    print(f"Merged data (object labels only) saved to: {out_dir}")

if __name__ == "__main__":
    main()