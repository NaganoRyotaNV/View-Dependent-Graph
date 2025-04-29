import os
import glob
import argparse
from collections import defaultdict

# 処理対象とする画像の拡張子リスト (小文字・大文字両方を考慮)
IMAGE_EXTENSIONS = ['*.jpeg', '*.jpg', '*.JPEG', '*.JPG', '*.png', '*.PNG']

def count_images_in_dir(directory, extensions):
    """指定されたディレクトリ直下の画像ファイル数をカウントする（再帰しない）"""
    count = 0
    if not os.path.isdir(directory):
        return 0
    for ext_pattern in extensions:
        pattern = os.path.join(directory, ext_pattern)
        try:
            # glob はリストを返すので len() で数を取得
            count += len(glob.glob(pattern))
        except Exception as e:
            print(f"Warning: Error during glob operation for pattern '{pattern}': {e}")
    return count

def analyze_flat_structure(input_dir, output_file, extensions):
    """
    フラット化されたディレクトリ構造 (split/category/image) を分析し、結果をファイルに出力する。
    """
    print(f"Analyzing FLAT structure in '{input_dir}'...")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Directory Structure Analysis (Flat)\n")
            f.write(f"Source Directory: {input_dir}\n")
            f.write("="*50 + "\n")

            splits = ["train", "val", "test"]
            overall_total_images = 0
            overall_total_categories = 0 # 各splitのカテゴリ数の単純合計

            for split in splits:
                split_path = os.path.join(input_dir, split)
                f.write(f"\n--- {split.upper()} ---\n")

                if not os.path.isdir(split_path):
                    f.write("  Directory not found.\n")
                    continue

                try:
                    # split内のカテゴリディレクトリを取得
                    categories = sorted([d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))])
                except Exception as e:
                    f.write(f"  Error listing categories: {e}\n")
                    continue

                if not categories:
                    f.write("  No categories found.\n")
                    f.write("  Total Categories: 0\n")
                    f.write("  Total Images: 0\n")
                    continue

                total_categories_in_split = len(categories)
                total_images_in_split = 0
                category_counts = defaultdict(int)

                print(f"  Processing {split} set ({total_categories_in_split} categories)...")
                for category in categories:
                    category_path = os.path.join(split_path, category)
                    image_count = count_images_in_dir(category_path, extensions)
                    category_counts[category] = image_count
                    total_images_in_split += image_count

                f.write(f"Total Categories: {total_categories_in_split}\n")
                f.write(f"Total Images: {total_images_in_split}\n")
                f.write("Category Counts:\n")
                for category, count in category_counts.items():
                    f.write(f"    {category}: {count}\n")

                overall_total_images += total_images_in_split
                overall_total_categories += total_categories_in_split

            f.write("\n" + "="*50 + "\n")
            f.write(f"Overall Total Images (train+val+test): {overall_total_images}\n")
            # f.write(f"Overall Total Category Entries (train+val+test): {overall_total_categories}\n") # 必要であればコメント解除

        print(f"Flat structure analysis complete. Results saved to '{output_file}'")

    except IOError as e:
        print(f"Error writing to output file '{output_file}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred during flat analysis: {e}")


# ===== ここから修正箇所 =====
def analyze_hierarchical_structure(input_dir, output_file, extensions):
    """
    階層化されたディレクトリ構造 (split/category/viewpoint/image) を分析し、結果をファイルに出力する。
    物体カテゴリ数と、ビューポイント（方向）を考慮したカテゴリ組み合わせ数をカウントし、
    各ビューポイント内の画像数も記録する。
    """
    print(f"Analyzing HIERARCHICAL structure in '{input_dir}'...")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Directory Structure Analysis (Hierarchical)\n")
            f.write(f"Source Directory: {input_dir}\n")
            f.write("="*50 + "\n")

            splits = ["train", "val", "test"]
            overall_total_images = 0
            overall_total_categories = 0 # 各splitの物体カテゴリ数の単純合計
            overall_total_viewpoint_combinations = 0 # 各splitの(カテゴリ,ビューポイント)組み合わせ数の単純合計

            for split in splits:
                split_path = os.path.join(input_dir, split)
                f.write(f"\n--- {split.upper()} ---\n")

                if not os.path.isdir(split_path):
                    f.write("  Directory not found.\n")
                    continue

                try:
                    # split内のカテゴリディレクトリを取得
                    categories = sorted([d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))])
                except Exception as e:
                    f.write(f"  Error listing categories: {e}\n")
                    continue

                if not categories:
                    f.write("  No categories found.\n")
                    f.write("  Total Categories (Object): 0\n") # 修正: ラベルを合わせる
                    f.write("  Total Viewpoint Combinations (Category x Viewpoint): 0\n") # 修正: ラベルを合わせる
                    f.write("  Total Images: 0\n") # 追加: 画像総数も0と明記
                    continue

                total_categories_in_split = len(categories)
                split_total_images = 0
                split_total_viewpoint_combinations = 0 # このsplit内のビューポイント組み合わせ総数

                f.write(f"Total Categories (Object): {total_categories_in_split}\n") # 物体カテゴリ数
                # split_total_images と split_total_viewpoint_combinations は後で集計結果を出力

                print(f"  Processing {split} set ({total_categories_in_split} categories)...")
                for category in categories:
                    category_path = os.path.join(split_path, category)

                    try:
                        # カテゴリ内のビューポイントディレクトリを取得
                        viewpoints = sorted([v for v in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, v))])
                    except Exception as e:
                        print(f"Warning: Error listing viewpoints for '{category_path}': {e}") # エラーはコンソールにも表示
                        # カテゴリごとの詳細出力でエラーを示す
                        f.write(f"\n  {category}:\n")
                        f.write(f"      Error listing viewpoints: {e}\n")
                        continue # このカテゴリの処理をスキップ

                    total_viewpoints_in_category = len(viewpoints)
                    split_total_viewpoint_combinations += total_viewpoints_in_category # ビューポイント数を加算

                    total_images_in_category = 0
                    viewpoint_counts = defaultdict(int) # 各ビューポイントの画像数を格納

                    if not viewpoints:
                        # ビューポイントがない場合の詳細出力
                        f.write(f"\n  {category}:\n")
                        f.write(f"    Total Viewpoints: 0\n")
                        f.write(f"    Total Images: 0\n")
                        # ビューポイントがない場合、カテゴリ直下の画像は仕様上カウントしない
                        continue

                    for viewpoint in viewpoints:
                        viewpoint_path = os.path.join(category_path, viewpoint)
                        image_count = count_images_in_dir(viewpoint_path, extensions)
                        viewpoint_counts[viewpoint] = image_count # ビューポイントごとのカウントを保存
                        total_images_in_category += image_count

                    # --- カテゴリごとの詳細出力 ---
                    f.write(f"\n  {category}:\n") # カテゴリ名をインデントして表示
                    f.write(f"    Total Viewpoints: {total_viewpoints_in_category}\n")
                    f.write(f"    Total Images: {total_images_in_category}\n")
                    if viewpoint_counts: # ビューポイントが存在する場合のみカウントを出力
                        f.write(f"    Viewpoint Counts:\n")
                        for viewpoint, count in viewpoint_counts.items():
                            f.write(f"      {viewpoint}: {count}\n")
                    # --- ここまでカテゴリごとの詳細出力 ---

                    split_total_images += total_images_in_category

                # --- splitごとのサマリー出力 ---
                f.write(f"\n--- {split.upper()} SUMMARY ---\n") # Splitごとのサマリーであることを明記
                f.write(f"Total Categories (Object): {total_categories_in_split}\n") # 再度表示しても良い（カテゴリごとの詳細の上にもあるが）
                f.write(f"Total Viewpoint Combinations (Category x Viewpoint): {split_total_viewpoint_combinations}\n") # ビューポイント組み合わせ数
                f.write(f"Total Images: {split_total_images}\n")

                overall_total_images += split_total_images
                overall_total_categories += total_categories_in_split
                overall_total_viewpoint_combinations += split_total_viewpoint_combinations

            f.write("\n" + "="*50 + "\n")
            f.write(f"Overall Summary (train+val+test):\n")
            f.write(f"  Overall Total Images: {overall_total_images}\n")
            f.write(f"  Overall Total Category Entries (Object): {overall_total_categories}\n") # 各splitの物体カテゴリ数の単純合計
            f.write(f"  Overall Total Viewpoint Combinations (Category x Viewpoint): {overall_total_viewpoint_combinations}\n") # 各splitの組み合わせ数の単純合計

        print(f"Hierarchical structure analysis complete. Results saved to '{output_file}'")

    except IOError as e:
        print(f"Error writing to output file '{output_file}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred during hierarchical analysis: {e}")
# ===== ここまで修正箇所 =====


# --- メイン実行部分 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze directory structures (flat or hierarchical) and count categories/files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "analysis_type",
        choices=['flat', 'hierarchical'],
        help="Type of directory structure to analyze ('flat' for split/category/image, 'hierarchical' for split/category/viewpoint/image)."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the base directory containing train/val/test splits."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output text file to save the analysis results."
    )

    args = parser.parse_args()

    # 分析タイプに応じて適切な関数を呼び出し
    if args.analysis_type == 'flat':
        analyze_flat_structure(args.input_dir, args.output_file, IMAGE_EXTENSIONS)
    elif args.analysis_type == 'hierarchical':
        analyze_hierarchical_structure(args.input_dir, args.output_file, IMAGE_EXTENSIONS)
    else:
        # argparseのchoicesでハンドリングされるはずだが念のため
        print(f"Error: Invalid analysis type '{args.analysis_type}'. Use 'flat' or 'hierarchical'.")