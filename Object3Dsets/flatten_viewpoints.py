import os
import shutil
import glob
from tqdm import tqdm
import argparse
import logging

# --- 設定 ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# 処理対象とする画像の拡張子リスト (小文字・大文字両方を考慮)
IMAGE_EXTENSIONS = ['*.jpeg', '*.jpg', '*.JPEG', '*.JPG', '*.png', '*.PNG'] # 必要に応じて追加

# --- 関数 ---
# <<< 修正 >>> 関数名を変更し、説明を更新
def merge_viewpoints_to_object_label(input_dir, output_dir):
    """
    入力ディレクトリ内の train/val/test 構造を維持しつつ、
    "物体ラベル_方向ラベル" ディレクトリを "物体ラベル" ディレクトリに統合し、
    画像を物体ラベルディレクトリ直下にコピーする。

    Args:
        input_dir (str): train/val/test ディレクトリを含む入力ディレクトリ
                         (例: /mnt/data/ObjectNet3D_Split_Combined)。
                         各split内に "物体ラベル_方向ラベル" のディレクトリ構造を持つ想定。
        output_dir (str): 統合された構造を保存する新しい出力ディレクトリ
                          (例: /mnt/data/ObjectNet3D_Split_ObjectOnly)
    """
    logging.info(f"Starting viewpoint merging process...")
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_dir}")

    if not os.path.isdir(input_dir):
        logging.error(f"Input directory '{input_dir}' not found.")
        return

    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        logging.error(f"Failed to create output base directory '{output_dir}': {e}")
        return

    splits = ["train", "val", "test"]
    total_files_copied = 0
    total_files_skipped_collision = 0
    total_errors_copying = 0
    processed_source_dirs = 0 # 処理した入力ディレクトリ数

    for split in splits:
        split_dir_in = os.path.join(input_dir, split)
        split_dir_out = os.path.join(output_dir, split)

        logging.info(f"--- Processing '{split}' set ---")

        if not os.path.isdir(split_dir_in):
            logging.warning(f"Input split directory '{split_dir_in}' not found. Skipping.")
            continue

        try:
            os.makedirs(split_dir_out, exist_ok=True)
        except OSError as e:
            logging.error(f"Failed to create output split directory '{split_dir_out}': {e}")
            continue

        # <<< 修正 >>> 入力splitディレクトリ内の "物体ラベル_方向ラベル" ディレクトリを取得
        try:
            # 'bus_front', 'car_side' などのリスト
            label_dirs_with_viewpoint = sorted([d for d in os.listdir(split_dir_in) if os.path.isdir(os.path.join(split_dir_in, d))])
        except Exception as e:
            # <<< 修正 >>> エラーメッセージを調整
            logging.error(f"Error listing source label directories in '{split_dir_in}': {e}. Skipping this split.")
            continue

        if not label_dirs_with_viewpoint:
            # <<< 修正 >>> 警告メッセージを調整
            logging.warning(f"No source label directories (e.g., 'object_viewpoint') found in '{split_dir_in}'.")
            continue

        # <<< 修正 >>> ログメッセージを調整
        logging.info(f"Found {len(label_dirs_with_viewpoint)} source label directories in '{split}'.")
        processed_source_dirs += len(label_dirs_with_viewpoint)

        # <<< 修正 >>> "物体ラベル_方向ラベル" ごとに処理
        for label_name_with_viewpoint in tqdm(label_dirs_with_viewpoint, desc=f"Merging {split}", unit="label", leave=False):
            source_label_dir = os.path.join(split_dir_in, label_name_with_viewpoint)

            # <<< 追加 >>> 物体ラベルを抽出するロジック
            # ラベル名の最後の'_'より前の部分を物体ラベルとする
            # 例: 'bus_front' -> 'bus', 'dining_table_side' -> 'dining_table'
            # '_' が含まれない場合は、そのままの名前を使用
            if '_' in label_name_with_viewpoint:
                object_label = label_name_with_viewpoint.rsplit('_', 1)[0]
            else:
                # 予期しないケースだが、フォールバックとしてそのまま使用
                object_label = label_name_with_viewpoint
                logging.warning(f"Source directory name '{label_name_with_viewpoint}' does not contain '_'. Using it directly as object label.")

            # <<< 修正 >>> 出力先の物体ラベルディレクトリパス
            object_label_dir_out = os.path.join(split_dir_out, object_label)

            # 出力の物体ラベルディレクトリを作成
            try:
                os.makedirs(object_label_dir_out, exist_ok=True)
            except OSError as e:
                logging.error(f"Failed to create output object label directory '{object_label_dir_out}': {e}")
                continue # このソースラベルディレクトリをスキップ

            files_copied_from_source = 0
            # <<< 修正 >>> 画像ファイルを検索 (再帰は不要)
            all_image_files_in_source = []
            for ext_pattern in IMAGE_EXTENSIONS:
                pattern = os.path.join(source_label_dir, ext_pattern) # 再帰なし
                try:
                    all_image_files_in_source.extend(glob.glob(pattern))
                except Exception as e:
                    logging.warning(f"Error searching for files with pattern '{pattern}': {e}")

            all_image_files_in_source = sorted(list(set(all_image_files_in_source)))

            # 見つかった画像ファイルをコピー
            for src_path in all_image_files_in_source:
                try:
                    filename = os.path.basename(src_path)
                    # <<< 修正 >>> コピー先パス (出力物体ラベルディレクトリ直下)
                    dst_path = os.path.join(object_label_dir_out, filename)

                    # ファイル名衝突チェック
                    if os.path.exists(dst_path):
                        # 異なるビューポイントに同名のファイルが存在した場合
                        logging.warning(f"Filename collision: '{filename}' from '{label_name_with_viewpoint}' already exists in '{object_label_dir_out}'. Skipping copy from '{src_path}'.")
                        total_files_skipped_collision += 1
                        continue

                    shutil.copy2(src_path, dst_path)
                    files_copied_from_source += 1
                except Exception as e:
                    logging.error(f"Error copying file {src_path} to {dst_path}: {e}")
                    total_errors_copying += 1

            total_files_copied += files_copied_from_source

    # --- 最終結果の表示 ---
    print("\n" + "="*30 + " Merging Summary " + "="*30)
    logging.info(f"Viewpoint merging process complete.")
    logging.info(f"Total source label directories processed: {processed_source_dirs}")
    logging.info(f"Total image files copied successfully: {total_files_copied}")
    if total_files_skipped_collision > 0:
        logging.warning(f"Total files skipped due to filename collisions: {total_files_skipped_collision}")
    if total_errors_copying > 0:
        logging.error(f"Total errors encountered during file copying: {total_errors_copying}")
    logging.info(f"Output saved to: {output_dir}")
    print("="*80)


# --- メイン実行部分 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        # <<< 修正 >>> 説明文を更新
        description='Merge viewpoint information into object labels within train/val/test splits. Copies images from "object_viewpoint" directories to "object" directories.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        # <<< 修正 >>> ヘルプメッセージの例を更新
        help="Path to the directory containing train/val/test splits with 'object_viewpoint' structure (e.g., /mnt/data/ObjectNet3D_Split_Combined)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        # <<< 修正 >>> ヘルプメッセージの例を更新
        help="Path to the new directory where the merged structure ('object'/image) will be saved (e.g., /mnt/data/ObjectNet3D_Split_ObjectOnly)."
    )

    args = parser.parse_args()

    # <<< 修正 >>> 実行する関数名を変更
    merge_viewpoints_to_object_label(args.input_dir, args.output_dir)