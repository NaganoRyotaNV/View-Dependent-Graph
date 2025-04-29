import os
import shutil
import scipy.io
import numpy as np
from tqdm import tqdm
import math # math.isclose のために追加

# --- 設定 ---
ANNOTATION_DIR = "/mnt/data/ObjectNet3D/Annotations"  # アノテーション(.mat)ファイルがあるディレクトリ
IMAGE_DIR = "/mnt/data/ObjectNet3D/Images"           # 画像(.JPEG)ファイルがあるディレクトリ
OUTPUT_DIR = "/mnt/data/ObjectNet3D_Sorted_Combined" # <<< 修正 >>> 分かりやすいように出力ディレクトリ名も変更 (任意)

# --- 角度正規化関数 ---
def normalize_angle_180(angle_degrees):
    """角度を -180度 < angle <= 180度の範囲に正規化する"""
    angle = angle_degrees % 360
    if angle > 180:
        angle -= 360
    # 0度と-0度を区別しないように念のため
    if math.isclose(angle, -0.0):
        angle = 0.0
    # -180度の場合、180度として扱う (一般的に同じ方向のため)
    if math.isclose(angle, -180.0):
        angle = 180.0
    return angle

# --- 方位角 (azimuth) をユーザー定義のカテゴリにマッピングする関数 ---
def get_viewpoint_category(azimuth_degrees):
    """
    方位角をユーザー定義の範囲に基づいて5つのカテゴリに分類します。
    角度は -180 < angle <= 180 の範囲に正規化して判定します。
    ユーザー定義 (解釈):
    "front":     [-36, 36]
    "frontside": (-72, -36) U (36, 72]
    "side":      (-108, -72] U (72, 108]
    "backside":  (-144, -108] U (108, 144]
    "back":      (-180, -144] U (144, 180]  (正規化により -180 は 180 になる)
    """
    # まず角度を (-180, 180] の範囲に正規化
    angle = normalize_angle_180(azimuth_degrees)

    # ユーザー定義に基づいて分類 (境界値の扱いに注意)
    if -36 <= angle <= 36:
        return "front"
    elif (angle > 36 and angle <= 72) or (angle >= -72 and angle < -36): # frontside
         return "frontside"
    elif (angle > 72 and angle <= 108) or (angle >= -108 and angle < -72): # side
        return "side"
    elif (angle > 108 and angle <= 144) or (angle >= -144 and angle < -108): # backside
        return "backside"
    elif (angle > 144 and angle <= 180) or (angle < -144): # back (-180は正規化で180になるため条件を修正)
         return "back"
    else:
        # 通常ここには来ないはずだが、デバッグ用に表示
        print(f"Warning: Normalized angle {angle} (from {azimuth_degrees}) did not fall into any category.")
        return "unknown"


# --- メイン処理 ---
def sort_images_by_viewpoint(annotation_dir, image_dir, output_dir):
    """
    ObjectNet3Dの画像をアノテーションに基づいてカテゴリと視点方向を組み合わせたディレクトリに分類します。
    """
    print(f"アノテーションディレクトリ: {annotation_dir}")
    print(f"画像ディレクトリ: {image_dir}")
    print(f"出力ディレクトリ: {output_dir}")

    if not os.path.isdir(annotation_dir):
        print(f"エラー: アノテーションディレクトリが見つかりません: {annotation_dir}")
        return
    if not os.path.isdir(image_dir):
        print(f"エラー: 画像ディレクトリが見つかりません: {image_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    print("出力ディレクトリを作成しました (存在しない場合)。")

    mat_files = [f for f in os.listdir(annotation_dir) if f.endswith('.mat')]
    print(f"{len(mat_files)} 個のアノテーションファイルを検出しました。処理を開始します...")

    processed_count = 0
    error_count = 0
    skipped_count = 0

    for mat_file in tqdm(mat_files, desc="Processing annotations"):
        mat_path = os.path.join(annotation_dir, mat_file)

        try:
            data = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

            # --- (データの読み込みと検証部分は変更なし) ---
            if 'record' not in data:
                skipped_count += 1
                continue
            record = data['record']

            if not hasattr(record, 'objects'):
                skipped_count += 1
                continue

            if isinstance(record.objects, np.ndarray):
                 if len(record.objects) == 0:
                     skipped_count += 1
                     continue
                 obj = record.objects[0]
            else:
                 obj = record.objects

            if not hasattr(record, 'filename'):
                 skipped_count += 1
                 continue
            if not hasattr(obj, 'viewpoint') or not hasattr(obj.viewpoint, 'azimuth_coarse'):
                 skipped_count += 1
                 continue
            try:
                object_class = getattr(obj, 'class')
            except AttributeError:
                skipped_count += 1
                continue

            img_filename = record.filename
            azimuth = obj.viewpoint.azimuth_coarse

            if isinstance(azimuth, np.ndarray):
                if azimuth.size == 1:
                    azimuth = azimuth.item()
                elif azimuth.size > 1:
                    azimuth = azimuth[0]
                elif azimuth.size == 0:
                     skipped_count += 1
                     continue

            if not isinstance(azimuth, (int, float, np.number)):
                skipped_count += 1
                continue

            try:
                azimuth_float = float(azimuth)
            except (ValueError, TypeError) as e:
                skipped_count += 1
                continue

            safe_object_class = object_class.replace('/', '_').replace('\\', '_')

            viewpoint_cat = get_viewpoint_category(azimuth_float)

            if viewpoint_cat == "unknown":
                skipped_count += 1
                continue
            # --- (データの読み込みと検証部分は変更なし ここまで) ---


            # --- ファイルパスの構築とコピー ---
            src_image_path = os.path.join(image_dir, img_filename)

            # <<< 修正箇所: 出力先ディレクトリ名を "物体ラベル_方向ラベル" に変更 >>>
            combined_dir_name = f"{safe_object_class}_{viewpoint_cat}"
            dest_dir = os.path.join(output_dir, combined_dir_name)
            # <<< 修正ここまで >>>

            os.makedirs(dest_dir, exist_ok=True) # 目的のディレクトリを作成

            dest_image_path = os.path.join(dest_dir, img_filename)

            if os.path.exists(src_image_path):
                shutil.copy2(src_image_path, dest_image_path)
                processed_count += 1
            else:
                # print(f"Warning: 対応する画像ファイルが見つかりません: {src_image_path} ({mat_file} 用)。スキップします。")
                skipped_count += 1

        except FileNotFoundError:
            print(f"\nエラー: アノテーションファイルが見つかりません: {mat_path}")
            error_count += 1
        except ImportError as e:
             print(f"\nエラー: 必要なライブラリが見つかりません: {e}")
             raise e
        except TypeError as e:
             # print(f"\nエラー: {mat_file} の構造の読み取り中にTypeError: {e}。スキップ。")
             error_count += 1
        except AttributeError as e:
             # print(f"\nエラー: {mat_file} の構造アクセス中にAttributeError: {e}。スキップ。")
             error_count += 1
        except Exception as e:
            print(f"\nエラー: {mat_file} の処理中に予期せぬエラーが発生しました: {e}")
            error_count += 1

    print()
    print("--- 処理完了 ---")
    print(f"処理済み画像数: {processed_count}")
    print(f"スキップ数: {skipped_count} (ファイル欠損、アノテーション不備など)")
    print(f"エラー発生数: {error_count}")

# --- スクリプト実行 ---
if __name__ == "__main__":
    # パスが正しいか再度確認してください
    if not os.path.exists(ANNOTATION_DIR) or not os.path.exists(IMAGE_DIR):
        print("エラー: ANNOTATION_DIR または IMAGE_DIR のパスが存在しません。")
        print(f"ANNOTATION_DIR: {ANNOTATION_DIR}")
        print(f"IMAGE_DIR: {IMAGE_DIR}")
    else:
        sort_images_by_viewpoint(ANNOTATION_DIR, IMAGE_DIR, OUTPUT_DIR)