#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import shutil
import numpy as np
import scipy.io as sio
import xml.etree.ElementTree as ET # XML読み込みに必要
import pprint
import re
from tqdm import tqdm # tqdmを追加して進捗を表示

# 5方向の角度レンジ定義（azimuth は -180〜180 を想定）
angle_ranges = {
    "front": {"left": (-36, 0), "right": (0, 36)},
    "frontside": {"left": (-72, -36), "right": (36, 72)},
    "side": {"left": (-108, -72), "right": (72, 108)},
    "backside": {"left": (-144, -108), "right": (108, 144)},
    "back": {"left": (-180, -144), "right": (144, 180)},
}

def map_azimuth_to_direction(azimuth):
    """
    azimuth を -180 <= azimuth <= 180 の範囲に正規化し、
    angle_ranges で定義した5方向×左右に当てはめて
    'front','frontside','side','backside','back' のいずれかを返す。
    境界値の扱いを <= に変更 (例: 36度はfront, 72度はfrontside)
    -180度は180度として back に含める。
    """
    # azimuthが数値でない場合やNoneの場合を考慮
    if not isinstance(azimuth, (int, float, np.number)):
        print(f"Warning: Invalid azimuth type received: {type(azimuth)}. Cannot map to direction.")
        return None
    if azimuth is None:
        print(f"Warning: Received None azimuth. Cannot map to direction.")
        return None

    try:
        azimuth = float(azimuth) # 念のためfloatに変換
        normalized = ((azimuth + 180) % 360) - 180
        # -180度の場合、180度として扱う（rangeチェックを簡単にするため）
        if np.isclose(normalized, -180.0):
            normalized = 180.0

        for direction, rng in angle_ranges.items():
            left_min, left_max = rng["left"]
            right_min, right_max = rng["right"]
            # 境界を含むように修正 (min <= normalized <= max)
            # 浮動小数点誤差を考慮して isclose を使う方がより安全
            is_in_left = (left_min <= normalized or np.isclose(left_min, normalized)) and \
                         (normalized <= left_max or np.isclose(normalized, left_max))
            is_in_right = (right_min <= normalized or np.isclose(right_min, normalized)) and \
                          (normalized <= right_max or np.isclose(normalized, right_max))

            # 0.0 付近の扱い: front に厳密に割り当てる
            if np.isclose(normalized, 0.0) and direction == "front":
                 return "front"
            # ±36.0 付近の扱い: front に含める
            if np.isclose(normalized, 36.0) and direction == "front":
                 return "front"
            if np.isclose(normalized, -36.0) and direction == "front":
                 return "front"
            # ±180.0 (正規化後) 付近の扱い: back に含める
            if np.isclose(normalized, 180.0) and direction == "back":
                 return "back"
            # -144.0 以下 (正規化後) の扱い: back に含める
            if normalized <= -144.0 and direction == "back":
                 return "back"

            # 通常の範囲チェック (境界値は既に上で処理されているはずだが念のため)
            if is_in_left or is_in_right:
                 return direction

        # どの範囲にも当てはまらない場合はNoneを返す (デバッグ用)
        print(f"Warning: Azimuth {azimuth} (normalized to {normalized:.2f}) did not fall into any defined range.")
        return None
    except Exception as e:
        print(f"Error mapping azimuth {azimuth} to direction: {e}")
        return None

##############################################################################
# mat_struct などをPythonのdictに再帰的に変換するヘルパー
##############################################################################
def matstruct_to_dict(obj):
    """
    scipy.io.loadmat で読み込んだ mat_struct を再帰的に dict に変換
    """
    if isinstance(obj, sio.matlab.mio5_params.mat_struct):
        d = {}
        for f in obj._fieldnames:
            val = getattr(obj, f, None) # 属性が存在しない場合None
            if val is not None:
                d[f] = matstruct_to_dict(val)
            else:
                d[f] = None # 属性がない場合
        return d
    elif isinstance(obj, np.ndarray):
        if obj.ndim == 0:
             # 配列が空でないか、または要素がNoneでないかチェック
             item = obj.item()
             if item is not None:
                 return matstruct_to_dict(item)
             else:
                 return None
        # 要素がNoneでないことを確認しながらリストを構築
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
        # struct_as_record=False だと属性アクセス可能になる
        mat_data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        converted = {}
        for k, v in mat_data.items():
            # __header__, __version__, __globals__ は除外
            if k.startswith('__'):
                continue
            converted[k] = matstruct_to_dict(v)
        # Pretty Printで出力
        outfile.write(pprint.pformat(converted, indent=2, width=120)) # 幅調整
    except FileNotFoundError:
         outfile.write(f"Error: File not found during loadmat: {mat_path}\n") # loadmat中にFileNotFoundになることも
    except ValueError as e:
         outfile.write(f"Error loading or parsing (possibly invalid format) {mat_path}: {e}\n")
    except Exception as e:
        outfile.write(f"Unexpected error loading or parsing {mat_path}: {e}\n")
    outfile.write("=" * (len(mat_path) + 20) + "\n\n")

##############################################################################
# ここからメインロジック
##############################################################################

def get_azimuth_from_mat(mat_path, target_class):
    """
    .matファイルから azimuth を抽出する（Pascal3D+想定）。
    'record.objects' の中から:
      - 'class' が target_class と一致
      - 'viewpoint' があり viewpoint.azimuth が存在
    の最初のオブジェクトを使う。
    見つからなければ ValueError。
    """
    try:
        # struct_as_record=False にすることで . アクセスが可能に
        mat_data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    except FileNotFoundError:
        raise ValueError(f"MAT file not found at {mat_path}")
    except Exception as e:
        raise ValueError(f"Failed to load MAT file {mat_path}: {e}")

    # record キーの存在確認
    if "record" not in mat_data:
        raise ValueError(f"No 'record' key found in {mat_path}")
    record = mat_data["record"]
    # record が None や他の型でないか、objects属性を持つか確認
    if not hasattr(record, "objects"):
        raise ValueError(f"No 'objects' attribute found in record of {mat_path}")

    # record.objects が None でないか確認
    objs = record.objects
    if objs is None:
         raise ValueError(f"'record.objects' is None in {mat_path}")

    # オブジェクトリストの準備 (単一 or 配列)
    if isinstance(objs, np.ndarray):
        if objs.size == 0: raise ValueError(f"Empty 'record.objects' array in {mat_path}")
        obj_list = objs
    elif hasattr(objs, '_fieldnames'):
        obj_list = [objs]
    else:
        raise ValueError(f"Unexpected type for 'record.objects' in {mat_path}: {type(objs)}")

    # オブジェクトを順番にチェック
    for obj in obj_list:
        if obj is None: continue # オブジェクトがNoneならスキップ

        # クラス名の一致を確認
        obj_class = getattr(obj, "class", None)
        if isinstance(obj_class, str) and obj_class == target_class:
            # viewpoint 属性と中身を確認
            if hasattr(obj, "viewpoint"):
                vp = obj.viewpoint
                if vp is not None and hasattr(vp, "azimuth"):
                    azimuth_raw = getattr(vp, "azimuth", None)
                    # azimuth の値が存在し、数値に変換可能か確認
                    if azimuth_raw is not None:
                        try:
                            # azimuthが配列の場合、最初の要素を使う (例: [[120]])
                            if isinstance(azimuth_raw, np.ndarray) and azimuth_raw.size > 0:
                                azimuth_value = float(azimuth_raw.item(0))
                            else:
                                azimuth_value = float(azimuth_raw)
                            return azimuth_value # 正常に取得できたら返す
                        except (TypeError, ValueError):
                             continue # 数値変換失敗、次のオブジェクトへ
    # ループで見つからなかった場合
    raise ValueError(f"No object of class '{target_class}' with a valid numeric viewpoint.azimuth found in {mat_path}")


def get_azimuth_from_xml(xml_path, target_class):
    """
    .xmlファイルから azimuth を抽出する(PascalVOC形式)。
    target_class と一致する最初の <object> 内の viewpoint.azimuth を返す。
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            name_elem = obj.find('name')
            # クラス名が一致するか確認
            if name_elem is not None and name_elem.text == target_class:
                vp = obj.find('viewpoint')
                if vp is not None:
                    az_elem = vp.find('azimuth')
                    if az_elem is not None and az_elem.text:
                        try:
                            # 空白文字などを除去してからfloatに変換
                            return float(az_elem.text.strip())
                        except (TypeError, ValueError):
                            continue # 数値変換できない場合は次へ
        # 見つからなかった
        raise ValueError(f"No object of class '{target_class}' with viewpoint/azimuth found in {xml_path}")
    except FileNotFoundError:
        raise ValueError(f"XML file not found at {xml_path}")
    except ET.ParseError as e:
        raise ValueError(f"Failed to parse XML file {xml_path}: {e}")
    except Exception as e:
        raise ValueError(f"Error processing XML file {xml_path}: {e}")


def get_azimuth(annotation_dir, base_name, target_class):
    """
    annotation_dir 内の base_name.mat or base_name.xml を探し、
    target_class にあったオブジェクトの viewpoint.azimuth を取得
    """
    mat_path = os.path.join(annotation_dir, base_name + ".mat")
    xml_path = os.path.join(annotation_dir, base_name + ".xml")

    last_error = None # エラーメッセージ保持用
    mat_exists = os.path.exists(mat_path)
    xml_exists = os.path.exists(xml_path)

    # .mat ファイルを優先的に試す
    if mat_exists:
        try:
            az = get_azimuth_from_mat(mat_path, target_class)
            return az, mat_path # 成功したら返す
        except ValueError as e:
            last_error = f"MAT Error ({mat_path}): {e}" # エラー詳細を記録

    # .mat が失敗または存在しない場合に .xml を試す
    if xml_exists:
        try:
            az = get_azimuth_from_xml(xml_path, target_class)
            # .mat が存在しなかった、または .mat でエラーが出ていた場合に XML の結果を返す
            if not mat_exists or last_error:
                return az, None # XMLで成功したら返す（mat_pathはNone）
        except ValueError as e:
            # .mat も失敗していたら、そのエラーを優先するかもしれない
            if last_error is None: # .matが存在せず、XMLでエラーの場合
                last_error = f"XML Error ({xml_path}): {e}"
            # else: .mat のエラーが既に記録されているので、そちらを優先

    # 両方失敗した場合、またはどちらかしか存在せず失敗した場合
    if last_error:
        raise ValueError(last_error) # 記録されたエラーを発生させる
    elif not mat_exists and not xml_exists:
        # どちらのファイルも存在しなかった場合
        raise ValueError(f"No annotation file (.mat or .xml) found for '{base_name}' in '{annotation_dir}'")
    else:
        # ファイルは存在するが、他の理由で azimuth が見つからなかった場合
        # (通常は上のlast_errorで捕捉されるはずだが、念のため)
        raise ValueError(f"Could not find valid azimuth for '{target_class}' in existing annotation(s) for '{base_name}' in '{annotation_dir}'")


def main():
    parser = argparse.ArgumentParser(
        description="指定クラス名＋viewpoint.azimuthを用いた5方向分類を行い、'物体_方向'ディレクトリに画像を振り分ける"
    )
    parser.add_argument("--images_dir", required=True,
                        help="PASCAL3D+ の Images ディレクトリ (例: /path/to/PASCAL3D+_release1.1/Images)")
    parser.add_argument("--annotations_dir", required=True,
                        help="PASCAL3D+ の Annotations ディレクトリ (例: /path/to/PASCAL3D+_release1.1/Annotations)")
    parser.add_argument("--output_dir", required=True,
                        help="振り分け先の出力ディレクトリ (例: /path/to/PASCAL3D_Sorted)")
    args = parser.parse_args()

    images_dir = args.images_dir
    annotations_dir = args.annotations_dir
    output_dir = args.output_dir # これは分類済み画像全体のルートディレクトリ

    # 出力ディレクトリ作成（既にあってもOK）
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error: Could not create output directory '{output_dir}': {e}")
        return

    no_view_file = os.path.join(output_dir, "no_viewpoint_list.txt")
    try:
        with open(no_view_file, "w", encoding='utf-8') as f: # encoding指定
            f.write("# List of images for which viewpoint could not be determined or errors occurred\n")
    except IOError as e:
        print(f"Error: Could not open log file '{no_view_file}': {e}")
        return

    debug_file = os.path.join(output_dir, "debug_info.txt")
    try:
        with open(debug_file, "w", encoding='utf-8') as f: # encoding指定
            f.write("# Image counts per source subdirectory\n")
    except IOError as e:
         print(f"Error: Could not open debug info file '{debug_file}': {e}")
         return

    processed_count = 0
    skipped_count = 0
    error_count = 0

    # 画像ディレクトリ直下のサブディレクトリ（クラス_データセット名）を取得
    try:
        subdirs = sorted([d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))])
    except FileNotFoundError:
        print(f"Error: Images directory not found at '{images_dir}'")
        return
    except Exception as e:
        print(f"Error listing subdirectories in '{images_dir}': {e}")
        return

    if not subdirs:
        print(f"Warning: No subdirectories found in '{images_dir}'. Nothing to process.")
        return

    print(f"Found {len(subdirs)} source subdirectories in '{images_dir}'. Processing...")

    # tqdmでサブディレクトリごとの進捗を表示
    for sd in tqdm(subdirs, desc="Processing source dirs"):
        subdir_path = os.path.join(images_dir, sd)
        # _imagenet や _pascal をクラス名から除去
        target_class = re.sub(r"_(imagenet|pascal)$", "", sd)

        # --- アノテーションディレクトリのパス構築（修正済み） ---
        # sd ('bus_imagenet' など) をそのまま使う
        ann_dir = os.path.join(annotations_dir, sd)
        # ----------------------------------------------------

        # アノテーションディレクトリが存在するか確認
        if not os.path.exists(ann_dir):
            # この警告はデバッグ時に有用なので残す
            # print(f"Warning: Annotation directory '{ann_dir}' not found (for source: {sd}). Skipping.")
            continue # 存在しない場合はスキップ

        # 画像ファイル一覧取得
        try:
            image_files = [f for f in os.listdir(subdir_path)
                           if f.lower().endswith((".jpg", ".jpeg", ".png"))] # pngも考慮
        except FileNotFoundError:
             # print(f"Warning: Image subdirectory '{subdir_path}' missing? Skipping.")
             continue
        except Exception as e:
             print(f"Warning: Error listing images in '{subdir_path}': {e}. Skipping.")
             continue

        total_images = len(image_files)
        # デバッグ情報書き込み
        try:
            with open(debug_file, "a", encoding='utf-8') as df:
                df.write(f"\n--- Source: {sd} (Target Class: {target_class}) ---\n")
                df.write(f"  Annotation Dir Used: {ann_dir}\n")
                df.write(f"  Images Found: {total_images}\n")
        except IOError:
            pass # 書き込めなくても処理は続行

        if total_images == 0:
            continue # 画像がなければ次へ

        # tqdmでサブディレクトリ内のファイル処理進捗を表示 (ネストさせる)
        for file in tqdm(image_files, desc=f"  Processing {sd}", leave=False):
            img_path = os.path.join(subdir_path, file)
            base_name, _ = os.path.splitext(file)

            try:
                # --- アノテーションから方位角を取得（修正済み） ---
                az, mat_path_used = get_azimuth(ann_dir, base_name, target_class)
                # ---------------------------------------------

                # 方位角を方向カテゴリにマッピング
                direction = map_azimuth_to_direction(az)

                if direction is None:
                    # どの方向にも分類できなかった場合
                    with open(no_view_file, "a", encoding='utf-8') as outf:
                        outf.write(f"{img_path} | Azimuth: {az if az is not None else 'N/A'} | Could not map to direction\n")
                        if mat_path_used: # 原因調査用に.matファイルの内容をダンプ
                             dump_mat_contents_to_file(mat_path_used, outf)
                    skipped_count += 1
                    continue

                # --- 出力先ディレクトリのパス構築（修正済み） ---
                combined_label = f"{target_class}_{direction}"
                # out_dir はスクリプト全体の出力ルートなので、その下にサブディレクトリを作る
                out_subdir = os.path.join(output_dir, combined_label)
                # --------------------------------------------

                # 出力ディレクトリを作成 (存在してもエラーにしない)
                os.makedirs(out_subdir, exist_ok=True)

                # --- 画像ファイルを出力ディレクトリにコピー（修正済み） ---
                dst_path = os.path.join(out_subdir, file)
                # ----------------------------------------------------

                # ファイル名衝突チェック (必要な場合コメント解除)
                # if os.path.exists(dst_path):
                #     print(f"Warning: Destination file already exists, skipping copy: {dst_path}")
                #     skipped_count += 1
                #     continue

                shutil.copy2(img_path, dst_path) # メタデータもコピー
                processed_count += 1

            except ValueError as e:
                # get_azimuth でエラー (アノテーションが見つからない、azimuthがない等)
                with open(no_view_file, "a", encoding='utf-8') as outf:
                    outf.write(f"{img_path} | Error getting azimuth: {e}\n")
                    # エラーの原因究明のため、対応する可能性のある .mat ファイルの内容を出力
                    mat_candidate = os.path.join(ann_dir, base_name + ".mat")
                    if os.path.exists(mat_candidate):
                         dump_mat_contents_to_file(mat_candidate, outf)
                    else:
                         # xmlも試してみる
                         xml_candidate = os.path.join(ann_dir, base_name + ".xml")
                         if os.path.exists(xml_candidate):
                              try:
                                   with open(xml_candidate, 'r', encoding='utf-8') as xmlf:
                                        outfile.write(f"\n--- Contents of {xml_candidate} ---\n")
                                        outfile.write(xmlf.read())
                                        outfile.write("\n" + "=" * (len(xml_candidate) + 20) + "\n\n")
                              except Exception as xml_e:
                                   outfile.write(f"Error reading XML {xml_candidate}: {xml_e}\n")

                skipped_count += 1
                error_count += 1 # エラーとしてカウント
            except OSError as e:
                 # ディレクトリ作成やファイルコピーでのOSエラー
                 print(f"\nOS Error processing file {img_path} or writing to {out_subdir}: {e}")
                 with open(no_view_file, "a", encoding='utf-8') as outf:
                      outf.write(f"{img_path} | OS Error: {e}\n")
                 skipped_count += 1
                 error_count += 1
            except Exception as e:
                # その他の予期せぬエラー
                print(f"\nUnexpected error processing file {img_path}: {e}")
                # import traceback; traceback.print_exc() # 詳細なトレースバックが必要な場合
                with open(no_view_file, "a", encoding='utf-8') as outf:
                     outf.write(f"{img_path} | Unexpected Error: {e}\n")
                skipped_count += 1
                error_count += 1


    # 完了メッセージ
    print("\n" + "="*30 + " Processing Summary " + "="*30)
    print(f"Processing finished.")
    print(f"Images successfully processed and copied: {processed_count}")
    print(f"Images skipped (viewpoint error, mapping error, file error, etc.): {skipped_count}")
    print(f"  (Including {error_count} files with specific errors during processing)")
    print(f"Details for skipped/error files saved to: {no_view_file}")
    print(f"Source image counts per subdirectory saved to: {debug_file}")
    print(f"Sorted images saved under: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()