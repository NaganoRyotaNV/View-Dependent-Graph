#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import re
import cv2
import numpy as np
import scipy.io as sio
# import xml.etree.ElementTree as ET # XMLは使わない想定のためコメントアウト
import pprint
from tqdm import tqdm # tqdmを追加

# --- ヘルパー関数 (変更なし) ---
def matstruct_to_dict(obj):
    """
    scipy.io.loadmat で読み込んだ mat_struct を再帰的に dict に変換
    """
    if isinstance(obj, sio.matlab.mio5_params.mat_struct):
        d = {}
        for f in obj._fieldnames:
            val = getattr(obj, f, None)
            if val is not None:
                d[f] = matstruct_to_dict(val)
            else:
                d[f] = None
        return d
    elif isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            item = obj.item()
            if item is not None:
                return matstruct_to_dict(item)
            else:
                return None
        return [matstruct_to_dict(x) for x in obj if x is not None]
    else:
        return obj

def dump_mat_contents_to_file(mat_path, outfile):
    """
    指定.matファイルの中身を、再帰的に展開してテキスト出力する
    """
    if not os.path.exists(mat_path):
        outfile.write(f"\n--- MAT file not found: {mat_path} ---\n")
        return
    outfile.write(f"\n===== Contents of {mat_path} =====\n")
    try:
        mat_data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        converted = {}
        for k, v in mat_data.items():
             if k.startswith('__'): continue
             converted[k] = matstruct_to_dict(v)
        outfile.write(pprint.pformat(converted, indent=2, width=120))
    except FileNotFoundError:
         outfile.write(f"Error: File not found during loadmat: {mat_path}\n")
    except ValueError as e:
         outfile.write(f"Error loading or parsing (possibly invalid format) {mat_path}: {e}\n")
    except Exception as e:
        outfile.write(f"Unexpected error loading or parsing {mat_path}: {e}\n")
    outfile.write("=" * (len(mat_path) + 20) + "\n\n")

def crop_image_square_by_bbox(image, bbox):
    """
    画像を指定されたBBoxに基づき、中央揃えの正方形にクロップし、不足分を黒でパディングする。
    """
    if bbox is None or len(bbox) != 4:
        print(f"Warning: Invalid bbox received: {bbox}. Skipping crop.")
        return image

    try:
        xmin, ymin, xmax, ymax = map(int, bbox)
        if xmin >= xmax or ymin >= ymax:
            print(f"Warning: Invalid bbox values (min >= max): {bbox}. Skipping crop.")
            return image

        width = xmax - xmin
        height = ymax - ymin
        side = max(width, height)

        # 中心座標
        cx = (xmin + xmax) / 2.0 # 浮動小数点数で計算
        cy = (ymin + ymax) / 2.0

        # 正方形の左上座標 (ターゲット)
        left_f = cx - side / 2.0
        top_f = cy - side / 2.0

        # 整数座標に変換 (切り捨て)
        left = int(np.floor(left_f))
        top = int(np.floor(top_f))
        # right と bottom は side を使って計算
        right = left + side
        bottom = top + side

        h, w = image.shape[:2]

        # 画像内での実際の切り取り領域
        crop_left = max(0, left)
        crop_top = max(0, top)
        crop_right = min(w, right)
        crop_bottom = min(h, bottom)

        # 実際に切り取れるサイズを確認
        actual_crop_width = crop_right - crop_left
        actual_crop_height = crop_bottom - crop_top

        if actual_crop_width <= 0 or actual_crop_height <= 0:
             print(f"Warning: Zero area crop region calculated for bbox {bbox} on image size {w}x{h}. Skipping crop.")
             return image

        # 画像から切り取り
        cropped = image[crop_top:crop_bottom, crop_left:crop_right]

        # パディング量の計算
        pad_left = max(0, -left)
        pad_top = max(0, -top)
        # 右と下のパディングは、ターゲットサイズと実際の切り取りサイズから計算
        pad_right = max(0, side - actual_crop_width - pad_left)
        pad_bottom = max(0, side - actual_crop_height - pad_top)

        # パディングが必要な場合
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            cropped = cv2.copyMakeBorder(
                cropped,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0) # 黒でパディング
            )

        # 最終サイズチェック (デバッグ用)
        # final_h, final_w = cropped.shape[:2]
        # if final_h != side or final_w != side:
        #     print(f"Warning: Final size {final_w}x{final_h} != target {side}x{side} for bbox {bbox}")

        return cropped
    except Exception as e:
        print(f"Error during cropping image with bbox {bbox}: {e}")
        return image # エラー時は元の画像を返す

def get_bbox_from_mat(mat_path, target_class):
    """指定されたクラスに一致する最初のオブジェクトのBboxを取得"""
    try:
        mat_data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    except FileNotFoundError:
        raise ValueError(f"MAT file not found at {mat_path}")
    except Exception as e:
        raise ValueError(f"Failed to load MAT file {mat_path}: {e}")

    if "record" not in mat_data: raise ValueError(f"No 'record' in {mat_path}")
    record = mat_data["record"]
    if not hasattr(record, "objects"): raise ValueError(f"No 'objects' in record of {mat_path}")
    objs = record.objects
    if objs is None: raise ValueError(f"'record.objects' is None in {mat_path}")

    if isinstance(objs, np.ndarray):
         if objs.size == 0: raise ValueError("Empty 'record.objects' array in {mat_path}")
         obj_list = objs
    elif hasattr(objs, '_fieldnames'): obj_list = [objs]
    else: raise ValueError(f"Unexpected type for record.objects: {type(objs)}")

    for obj in obj_list:
         if obj is None: continue
         obj_class = getattr(obj, "class", None)
         if isinstance(obj_class, str) and obj_class == target_class:
             if hasattr(obj, "bbox"):
                 bbox = obj.bbox
                 if bbox is not None and isinstance(bbox, np.ndarray) and bbox.shape == (4,):
                     try:
                         bbox_int = tuple(map(int, np.floor(bbox))) # floorで切り捨ててからintへ
                         if bbox_int[0] >= bbox_int[2] or bbox_int[1] >= bbox_int[3]: continue
                         return bbox_int
                     except (ValueError, TypeError): continue
    raise ValueError(f"No valid bbox found for class '{target_class}' in {mat_path}")

# <<< === 修正箇所 === >>>
def find_bbox(annotation_base_dir, object_label, image_filename):
    """
    アノテーションディレクトリから指定された物体ラベルに対応するBboxを探す。
    Pascal3D+のアノテーションディレクトリ構造
    (Annotations/<クラス名_imagenet または _pascal>/) を想定。
    """
    base_name, _ = os.path.splitext(image_filename)

    # --- 画像ファイル名からデータセットタイプ (_imagenet or _pascal) を推測 ---
    if base_name.startswith("n0"):
        dataset_suffix = "_imagenet"
    else:
        dataset_suffix = "_pascal"
    # -----------------------------------------------------------------

    # --- 正しいアノテーションサブディレクトリ名を構築 ---
    annotation_subdir_name = f"{object_label}{dataset_suffix}" # 例: "bus_imagenet"
    annotation_subdir_path = os.path.join(annotation_base_dir, annotation_subdir_name)
    # ---------------------------------------------------

    # --- .mat ファイルのフルパスを構築 ---
    mat_path = os.path.join(annotation_subdir_path, base_name + ".mat")
    # -----------------------------------

    # アノテーションファイルが存在するかチェック
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Annotation .mat file not found at '{mat_path}' (derived from image '{image_filename}')")

    # .mat ファイルから Bbox を取得
    try:
        bbox = get_bbox_from_mat(mat_path, object_label)
        return bbox, mat_path
    except ValueError as e:
        # エラーメッセージにファイルパスが含まれるようにする
        raise ValueError(f"Error getting bbox from '{mat_path}': {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error processing '{mat_path}': {e}")
# <<< === 修正ここまで === >>>

def main():
    parser = argparse.ArgumentParser(
        description="入力ディレクトリ ('物体_方向' ラベル) の画像をクロップし、同じ構造で出力する。"
    )
    parser.add_argument("--src_dir", required=True,
                        help="仕分け済み画像の親ディレクトリ (例: /path/to/PASCAL3D_Sorted)。直下に '物体_方向' ラベルのディレクトリがある想定。")
    parser.add_argument("--ann_dir", required=True,
                        help="PASCAL3D+ アノテーションディレクトリのルート (例: /path/to/PASCAL3D+_release1.1/Annotations)。")
    parser.add_argument("--out_dir", required=True,
                        help="正方形クロップ後の出力先 (例: /path/to/PASCAL3D_Cropped)。")
    args = parser.parse_args()

    src_dir = args.src_dir
    ann_dir = args.ann_dir # これは Annotations のルートパス
    out_dir = args.out_dir

    # ディレクトリ存在チェック
    if not os.path.isdir(src_dir):
        print(f"Error: Source directory not found: {src_dir}")
        return
    if not os.path.isdir(ann_dir):
        print(f"Error: Annotation directory not found: {ann_dir}")
        return

    # 出力ディレクトリ作成
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directory '{out_dir}': {e}")
        return

    # ログファイル準備
    skip_file = os.path.join(out_dir, "skip_list.txt")
    try:
        with open(skip_file, "w", encoding='utf-8') as sf:
            sf.write("# List of images skipped during cropping process\n")
    except IOError as e:
        print(f"Error: Could not open skip list file '{skip_file}': {e}")
        return

    debug_file = os.path.join(out_dir, "debug_info.txt")
    try:
        with open(debug_file, "w", encoding='utf-8') as df:
            df.write("# Image counts per source label directory\n")
    except IOError as e:
        print(f"Error: Could not open debug info file '{debug_file}': {e}")
        return

    # 入力ディレクトリ内のラベルディレクトリを取得
    try:
        label_dirs = sorted([d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))])
    except Exception as e:
        print(f"Error listing label directories in '{src_dir}': {e}")
        return

    if not label_dirs:
        print(f"Warning: No label directories found in '{src_dir}'. Nothing to crop.")
        return

    print(f"Found {len(label_dirs)} label directories in '{src_dir}'. Starting cropping...")

    processed_count = 0
    skipped_count = 0
    error_count = 0

    # ラベルディレクトリごとにループ
    for label_name in tqdm(label_dirs, desc="Processing labels"):
        source_label_dir = os.path.join(src_dir, label_name)

        # ラベル名から物体ラベルを抽出
        if '_' in label_name:
            object_label = label_name.rsplit('_', 1)[0]
        else:
            object_label = label_name
            # print(f"Warning: Label '{label_name}' used directly as object label.") # 必要ならコメント解除

        # 画像ファイル一覧を取得
        try:
            image_files = [f for f in os.listdir(source_label_dir)
                           if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        except Exception as e:
            print(f"Warning: Error listing images in '{source_label_dir}': {e}. Skipping.")
            continue

        num_images_in_label = len(image_files)
        # デバッグ情報書き込み
        try:
            with open(debug_file, "a", encoding='utf-8') as df:
                df.write(f"\n--- Label: {label_name} (Object: {object_label}) ---\n")
                df.write(f"  Images Found: {num_images_in_label}\n")
        except IOError: pass

        if num_images_in_label == 0: continue

        # 出力先ラベルディレクトリ作成
        out_label_dir = os.path.join(out_dir, label_name)
        try:
            os.makedirs(out_label_dir, exist_ok=True)
        except OSError as e:
            print(f"Error creating output directory '{out_label_dir}': {e}. Skipping label '{label_name}'.")
            continue

        # ラベル内の画像を処理
        for img_file in image_files:
            src_img_path = os.path.join(source_label_dir, img_file)
            dst_img_path = os.path.join(out_label_dir, img_file)
            mat_path_used_for_error = None # エラー時にダンプするためのパス

            try:
                # Bbox をアノテーションから取得
                bbox, mat_path_used_for_error = find_bbox(ann_dir, object_label, img_file)

                # 画像読み込み
                img = cv2.imread(src_img_path)
                if img is None: raise ValueError(f"Failed to load image file: {src_img_path}")

                # クロップ処理
                cropped = crop_image_square_by_bbox(img, bbox)

                # クロップ結果を保存
                success = cv2.imwrite(dst_img_path, cropped, [cv2.IMWRITE_JPEG_QUALITY, 95]) # 品質指定(任意)
                if not success: raise IOError(f"Failed to save cropped image to: {dst_img_path}")
                processed_count += 1

            except (FileNotFoundError, ValueError, IOError, OSError, Exception) as e:
                # エラー発生時の処理
                with open(skip_file, "a", encoding='utf-8') as sf:
                    sf.write(f"{src_img_path} | Error: {e}\n")
                    # find_bbox が成功していた場合（例：bbox抽出失敗）、使用したmatファイルをダンプ
                    if isinstance(e, ValueError) and mat_path_used_for_error:
                        dump_mat_contents_to_file(mat_path_used_for_error, sf)
                    # find_bbox が失敗した場合（FileNotFound）、エラーメッセージにパスが含まれるはず
                    # その他のエラーの場合も、推測パスをダンプしてみる
                    elif not mat_path_used_for_error:
                         try:
                             base_name, _ = os.path.splitext(img_file)
                             if base_name.startswith("n0"): dataset_suffix = "_imagenet"
                             else: dataset_suffix = "_pascal"
                             annotation_subdir_name = f"{object_label}{dataset_suffix}"
                             mat_path_guess = os.path.join(ann_dir, annotation_subdir_name, base_name + ".mat")
                             if os.path.exists(mat_path_guess):
                                  dump_mat_contents_to_file(mat_path_guess, sf)
                         except Exception: pass # ダンプ失敗は無視

                skipped_count += 1
                error_count += 1 # エラーとしてカウント

    # --- 完了メッセージ ---
    print("\n" + "="*30 + " Cropping Summary " + "="*30)
    print(f"Cropping process finished.")
    print(f"Images successfully processed and saved: {processed_count}")
    print(f"Images skipped due to errors: {skipped_count}")
    print(f"  (Including {error_count} files with specific errors during processing)")
    print(f"Details for skipped files saved to: {skip_file}")
    print(f"Source image counts per label directory saved to: {debug_file}")
    print(f"Cropped images saved under: {out_dir}")
    print("="*80)

if __name__ == "__main__":
    main()