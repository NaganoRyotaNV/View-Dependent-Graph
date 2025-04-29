import os
import shutil
import random
import glob
from math import floor
from tqdm import tqdm
import argparse
import logging

# --- 設定 ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

DEFAULT_TRAIN_RATIO = 0.6
DEFAULT_VAL_RATIO = 0.3
DEFAULT_SEED = 42

# --- 関数 ---
# <<< 修正 >>> 関数の説明を新しい構造に合わせて更新
def split_data(input_dir, output_dir, train_ratio, val_ratio, seed):
    """
    指定されたディレクトリ内の画像をラベル（物体_方向）ごとに train/val/test に分割する。
    各ラベルディレクトリが train/val/test 全てに最低1ファイル割り当てられない場合はスキップする。

    Args:
        input_dir (str): クロップ済み画像が格納されたディレクトリ (例: /mnt/data/ObjectNet3D_Cropped_Combined)。
                         直下に "物体_方向" ラベルのディレクトリが存在する想定。
        output_dir (str): 分割後のデータを保存するベースディレクトリ。
        train_ratio (float): トレーニングデータの割合 (0.0 ~ 1.0)。
        val_ratio (float): 検証データの割合 (0.0 ~ 1.0)。
        seed (int): 乱数シード。
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    logging.info(f"Starting data splitting process...")
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Target split ratios: Train={train_ratio:.1%}, Val={val_ratio:.1%}, Test={test_ratio:.1%}")
    logging.info(f"Random seed: {seed}")

    if not os.path.isdir(input_dir):
        logging.error(f"Input directory '{input_dir}' not found.")
        return

    if not (train_ratio > 0 and val_ratio > 0 and test_ratio > 0 and abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9):
        logging.error("Invalid train/validation/test ratios. Each ratio must be > 0 and their sum must be 1.0.")
        return

    random.seed(seed)

    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")

    try:
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        logging.info(f"Ensured output directories exist: {train_dir}, {val_dir}, {test_dir}")
    except OSError as e:
        logging.error(f"Failed to create output directories: {e}")
        return

    # <<< 修正 >>> input_dir 直下のディレクトリを "ラベル" (label) として取得
    try:
        # label_dirs は 'bus_front', 'car_side' などのリストになる
        label_dirs = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
    except FileNotFoundError:
        logging.error(f"Cannot list directories in '{input_dir}'. Check permissions or path.")
        return
    except Exception as e:
        logging.error(f"An error occurred while listing label directories: {e}")
        return

    if not label_dirs:
        # <<< 修正 >>> エラーメッセージを調整
        logging.error(f"No label subdirectories (e.g., 'object_viewpoint') found directly under '{input_dir}'. Check the directory structure.")
        return

    # <<< 修正 >>> ログメッセージを調整
    logging.info(f"Found {len(label_dirs)} label directories.")

    # --- 統計情報用変数 ---
    total_files_considered = 0
    total_files_skipped_insufficient = 0
    # <<< 修正 >>> 変数名を変更 (viewpoint -> label)
    skipped_label_dirs_count = 0
    processed_label_dirs_count = 0
    total_train_files = 0
    total_val_files = 0
    total_test_files = 0
    error_copying_files_count = 0
    # ---------------------

    # <<< 修正 >>> カテゴリ (category) ではなくラベル (label_dir) でループ
    for label_name in tqdm(label_dirs, desc="Processing labels"):
        # <<< 修正 >>> ラベルディレクトリのパスを取得
        label_path = os.path.join(input_dir, label_name)

        # <<< 削除 >>> ビューポイントディレクトリの探索は不要になったので削除
        # try:
        #     viewpoints = sorted([vp for vp in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, vp))])
        # ... (中略) ...
        # for viewpoint in viewpoints:
        #     viewpoint_path = os.path.join(category_path, viewpoint)
        #     current_viewpoint_identifier = os.path.join(category, viewpoint)

        # <<< 修正 >>> ラベルディレクトリ内の全画像ファイルを取得
        image_files = []
        try:
            # label_path (例: /mnt/data/ObjectNet3D_Cropped_Combined/bus_front) から直接検索
            image_files.extend(glob.glob(os.path.join(label_path, '*.JPEG')))
            image_files.extend(glob.glob(os.path.join(label_path, '*.jpeg')))
            image_files = sorted(list(set(image_files)))
        except Exception as e:
            # <<< 修正 >>> エラーメッセージ内のパス名を修正
            logging.warning(f"Error finding image files in '{label_path}': {e}. Skipping this label.")
            continue # このラベルディレクトリの処理をスキップ

        num_files = len(image_files)
        total_files_considered += num_files

        if num_files == 0:
            # logging.info(f"No image files found in '{label_name}'. Skipping.")
            continue

        # --- 分割数の計算と最小ファイル数チェック ---
        num_train = floor(num_files * train_ratio)
        num_val = floor(num_files * val_ratio)
        num_test = num_files - num_train - num_val

        if num_train > 0 and num_val > 0 and num_test > 0:
            # <<< 修正 >>> 処理対象のカウント変数名を変更
            processed_label_dirs_count += 1

            random.shuffle(image_files)
            train_files = image_files[:num_train]
            val_files = image_files[num_train : num_train + num_val]
            test_files = image_files[num_train + num_val :]

            total_train_files += len(train_files)
            total_val_files += len(val_files)
            total_test_files += len(test_files)

            # ファイルを対応するディレクトリにコピー
            for split_name, split_dir, file_list in [("train", train_dir, train_files),
                                                     ("val", val_dir, val_files),
                                                     ("test", test_dir, test_files)]:
                if not file_list: continue

                for src_path in file_list:
                    try:
                        # <<< 修正不要 >>> 出力パスの構築ロジックはこのままでOK
                        # relative_path は 'bus_front/image1.JPEG' のようになる
                        relative_path = os.path.relpath(src_path, input_dir)
                        dst_path = os.path.join(split_dir, relative_path)
                        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                        shutil.copy2(src_path, dst_path)
                    except Exception as e:
                        logging.error(f"Error copying file {src_path} to {dst_path}: {e}")
                        error_copying_files_count += 1
        else:
            # <<< 修正 >>> スキップ処理のログメッセージとカウント変数名を変更
            logging.warning(f"Skipping label '{label_name}' ({num_files} files). Cannot split into non-empty train({num_train})/val({num_val})/test({num_test}) sets.")
            skipped_label_dirs_count += 1
            total_files_skipped_insufficient += num_files

    # --- 最終結果の表示 ---
    # <<< 修正 >>> 統計情報やログメッセージの表現を「カテゴリ/ビューポイント」から「ラベル」へ変更
    print("\n" + "="*30 + " Data Splitting Summary " + "="*30)
    logging.info(f"Total label directories found: {len(label_dirs)}")
    total_labels_scanned = processed_label_dirs_count + skipped_label_dirs_count
    logging.info(f"Total label directories scanned: {total_labels_scanned}")
    logging.info(f" - Labels processed and split: {processed_label_dirs_count}")
    logging.info(f" - Labels skipped (insufficient files): {skipped_label_dirs_count}")
    print("-"*80)
    logging.info(f"Total image files considered: {total_files_considered}")
    files_in_processed_labels = total_files_considered - total_files_skipped_insufficient
    logging.info(f" - Files in processed labels: {files_in_processed_labels}")
    logging.info(f" - Files in skipped labels: {total_files_skipped_insufficient}")
    print("-"*80)
    logging.info(f"Files copied to output directories:")
    final_total_copied = total_train_files + total_val_files + total_test_files
    denominator = max(1, files_in_processed_labels) # ゼロ除算回避

    logging.info(f" - Train: {total_train_files} ({total_train_files/denominator:.1%})")
    logging.info(f" - Val:   {total_val_files} ({total_val_files/denominator:.1%})")
    logging.info(f" - Test:  {total_test_files} ({total_test_files/denominator:.1%})")
    logging.info(f" - Total Copied: {final_total_copied} / {files_in_processed_labels}")
    if error_copying_files_count > 0:
        logging.warning(f"Encountered {error_copying_files_count} errors during file copying.")
    print("="*80)
    logging.info(f"Data splitting complete. Output saved to: {output_dir}")


# --- メイン実行部分 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        # <<< 修正 >>> 説明文を更新
        description="Split cropped image data into train/val/test sets per label (object_viewpoint) subdirectory, skipping labels that cannot be split into non-empty sets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        # <<< 修正 >>> ヘルプメッセージの例を更新
        help="Path to the directory containing cropped images (e.g., /mnt/data/ObjectNet3D_Cropped_Combined). Expects subdirectories like 'bus_front', 'car_side'."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Base path to save the split data (train, val, test subdirs will be created)."
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help="Proportion of data for the training set (must be > 0)."
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help="Proportion of data for the validation set (must be > 0)."
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=DEFAULT_SEED,
        help='Random seed for shuffling.'
    )

    args = parser.parse_args()

    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if not (args.train_ratio > 0 and args.val_ratio > 0 and test_ratio > 0):
         parser.error("Train, validation, and the resulting test ratios must all be greater than 0.")
    if abs(args.train_ratio + args.val_ratio + test_ratio - 1.0) > 1e-9:
         parser.error("Train, validation, and test ratios must sum to 1.0.")

    # データ分割関数を実行 (引数は変更なし)
    split_data(args.input_dir, args.output_dir, args.train_ratio, args.val_ratio, args.seed)