import os
from PIL import Image
import glob
from tqdm import tqdm
import scipy.io # .matファイル読み込み用
import numpy as np # NumPyをインポート

# --- 設定 ---
# !!! 実際の環境に合わせてパスを修正してください !!!
# <<< 修正 >>> INPUT_BASE_DIRを新しい構造のディレクトリに変更
INPUT_BASE_DIR = "/mnt/data/ObjectNet3D_Sorted_Combined"  # 画像が格納されているディレクトリ (例: bus_front/image.JPEG)
# <<< 修正 >>> OUTPUT_BASE_DIRも新しい名前に変更 (任意)
OUTPUT_BASE_DIR = "/mnt/data/ObjectNet3D_Cropped_Combined" # クロップ画像の保存先ディレクトリ
ANNOTATION_DIR = "/mnt/data/ObjectNet3D/Annotations" # アノテーションファイル(.mat)があるディレクトリ (変更なし)
IMAGE_EXTENSION = ".JPEG" # 画像ファイルの拡張子 (大文字に注意)
ANNOTATION_EXTENSION = ".mat" # アノテーションファイルの拡張子 (変更なし)

# --- Helper Function ---
def get_bbox_from_annotation(annotation_path):
    """
    指定されたアノテーションファイルから最初のオブジェクトのBboxを取得します。
    ObjectNet3Dの .mat ファイル構造に合わせて修正済み。

    Args:
        annotation_path (str): .matアノテーションファイルへのパス。

    Returns:
        tuple: (xmin, ymin, xmax, ymax) の形式のBbox (int型)、または見つからない場合はNone。
    """
    # (この関数の内容は変更なし)
    try:
        mat = scipy.io.loadmat(annotation_path, squeeze_me=True, struct_as_record=False)
        if 'record' not in mat:
            # print(f"Warning: 'record' key not found in {annotation_path}")
            return None
        record = mat['record']
        if not hasattr(record, 'objects') or record.objects is None:
            return None
        objects = record.objects
        if isinstance(objects, np.ndarray) and objects.size == 0:
            return None

        first_object = None
        if isinstance(objects, np.ndarray) and objects.ndim > 0 :
             if objects.size > 0:
                 first_object = objects.item(0)
             else:
                 return None
        elif hasattr(objects, 'bbox'):
             first_object = objects
        else:
             return None

        if first_object is None:
            return None
        if not hasattr(first_object, 'bbox') or first_object.bbox is None:
            return None
        bbox = first_object.bbox
        if not isinstance(bbox, (list, np.ndarray)) or len(np.shape(bbox)) == 0 or np.shape(bbox)[-1] != 4:
             print(f"Warning: Unexpected format or length for 'bbox' in {annotation_path}: {bbox}")
             return None
        if isinstance(bbox, np.ndarray):
            bbox = bbox.tolist()
        try:
             if np.any(np.isinf(bbox)) or np.any(np.isnan(bbox)):
                 print(f"Warning: NaN or Inf value found in bbox for {annotation_path}: {bbox}. Skipping.")
                 return None
             bbox_int = tuple(map(int, bbox))
             if bbox_int[0] >= bbox_int[2] or bbox_int[1] >= bbox_int[3]:
                  print(f"Warning: Invalid bbox values (min >= max) for {annotation_path}: {bbox_int}. Original: {bbox}. Skipping.")
                  return None
             return bbox_int
        except (ValueError, TypeError) as e:
             print(f"Error converting bbox values to int for {annotation_path}: {bbox}. Error: {e}")
             return None
    except FileNotFoundError:
        return None
    except ImportError:
        print(f"Error: Failed to import module in annotation file {annotation_path}. Check file integrity.")
        return None
    except Exception as e:
        print(f"Error reading or parsing annotation file {annotation_path}: {e}")
        return None


def crop_and_pad_image(image_path, bbox, output_path):
    """
    指定されたロジックに基づいて画像をクロップし、パディングします。

    Args:
        image_path (str): 入力画像へのパス。
        bbox (tuple): (xmin, ymin, xmax, ymax) のバウンディングボックス (int型)。
        output_path (str): クロップされた画像を保存するパス。
    """
    # (この関数の内容は変更なし)
    try:
        img = Image.open(image_path).convert('RGB')
        img_width, img_height = img.size
        xmin, ymin, xmax, ymax = bbox
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(img_width, xmax)
        ymax = min(img_height, ymax)
        if xmin >= xmax or ymin >= ymax:
            # print(f"Warning: Invalid or zero-area bbox {bbox} after clipping...")
            return

        bbox_width = xmax - xmin
        bbox_height = ymax - ymin
        target_size = max(bbox_width, bbox_height)
        if target_size <= 0:
             # print(f"Warning: Calculated target size is non-positive...")
             return

        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        crop_left = int(center_x - target_size / 2)
        crop_top = int(center_y - target_size / 2)
        crop_right = crop_left + target_size
        crop_bottom = crop_top + target_size

        actual_crop_left = max(0, crop_left)
        actual_crop_top = max(0, crop_top)
        actual_crop_right = min(img_width, crop_right)
        actual_crop_bottom = min(img_height, crop_bottom)
        if actual_crop_left >= actual_crop_right or actual_crop_top >= actual_crop_bottom:
            # print(f"Warning: Calculated actual crop area has zero size...")
            return

        cropped_img = img.crop((actual_crop_left, actual_crop_top, actual_crop_right, actual_crop_bottom))
        cropped_width, cropped_height = cropped_img.size

        final_image = Image.new('RGB', (target_size, target_size), (0, 0, 0))
        paste_x = actual_crop_left - crop_left
        paste_y = actual_crop_top - crop_top
        paste_x = max(0, paste_x)
        paste_y = max(0, paste_y)

        # Ensure the cropped image fits within the target size considering the paste coordinates
        paste_width = min(cropped_width, target_size - paste_x)
        paste_height = min(cropped_height, target_size - paste_y)
        if paste_width != cropped_width or paste_height != cropped_height:
             # print(f"Debug: Cropping pasted image for {output_path}")
             cropped_img = cropped_img.crop((0, 0, paste_width, paste_height))


        if paste_width > 0 and paste_height > 0: # Only paste if there is a valid area
             final_image.paste(cropped_img, (paste_x, paste_y))
        # else:
        #      print(f"Debug: Skipping paste due to zero size for {output_path}")


        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_image.save(output_path)

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
    except Exception as e:
        print(f"Error processing image {image_path} with bbox {bbox}: {e}")


# --- メイン処理 ---
def main():
    print(f"Starting image cropping process...")
    # <<< 修正 >>> プリントするパスも更新
    print(f"Input image directory: {INPUT_BASE_DIR}")
    print(f"Annotation directory: {ANNOTATION_DIR}")
    print(f"Output directory: {OUTPUT_BASE_DIR}")

    if not os.path.isdir(INPUT_BASE_DIR):
        print(f"Error: Input base directory '{INPUT_BASE_DIR}' not found.")
        return
    if not os.path.isdir(ANNOTATION_DIR):
        print(f"Error: Annotation directory '{ANNOTATION_DIR}' not found.")
        return

    # <<< 修正 >>> 新しい構造でもこの検索パターンで動作します
    # INPUT_BASE_DIR の直下にある "物体_方向" ディレクトリ内の画像を検索します
    search_pattern = os.path.join(INPUT_BASE_DIR, '*', '*' + IMAGE_EXTENSION) # 1階層下のディレクトリを検索
    # もし、さらに深い階層も検索対象にする場合は recursive=True を使います
    # search_pattern = os.path.join(INPUT_BASE_DIR, '**', '*' + IMAGE_EXTENSION)
    # image_files = glob.glob(search_pattern, recursive=True)
    image_files = glob.glob(search_pattern) # recursive=False (デフォルト) で1階層のみ検索

    # 小文字の .jpeg も考慮する場合
    search_pattern_lower = os.path.join(INPUT_BASE_DIR, '*', '*.jpeg')
    image_files.extend(glob.glob(search_pattern_lower))
    image_files = list(set(image_files)) # 重複削除

    if not image_files:
        print(f"No images with extension '{IMAGE_EXTENSION}' or '.jpeg' found directly under subdirectories in '{INPUT_BASE_DIR}'.")
        # もし recursive=True を使う場合はメッセージを調整:
        # print(f"No images with extension '{IMAGE_EXTENSION}' or '.jpeg' found recursively in '{INPUT_BASE_DIR}'.")
        return

    print(f"Found {len(image_files)} candidate image files.")

    processed_count = 0
    skipped_count = 0
    annotation_not_found_count = 0
    error_count = 0

    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            annotation_filename = base_filename + ANNOTATION_EXTENSION
            annotation_path = os.path.join(ANNOTATION_DIR, annotation_filename)

            if not os.path.exists(annotation_path):
                annotation_not_found_count += 1
                skipped_count += 1
                continue

            bbox = get_bbox_from_annotation(annotation_path)

            if bbox:
                # <<< 修正不要 >>> このパス構築ロジックは新しい構造でも正しく動作します
                # 例: image_path = /mnt/data/ObjectNet3D_Sorted_Combined/bus_front/image1.JPEG
                #     relative_path = bus_front/image1.JPEG
                #     output_path = /mnt/data/ObjectNet3D_Cropped_Combined/bus_front/image1.JPEG
                relative_path = os.path.relpath(image_path, INPUT_BASE_DIR)
                output_path = os.path.join(OUTPUT_BASE_DIR, relative_path)

                crop_and_pad_image(image_path, bbox, output_path)
                processed_count += 1
            else:
                skipped_count += 1

        except Exception as e:
            print(f"Error during main loop for image {image_path}: {e}")
            error_count += 1
            skipped_count += 1

    print("\n--- Processing Summary ---")
    print(f"Total candidate image files found: {len(image_files)}")
    print(f"Successfully processed and saved: {processed_count} images.")
    print(f"Skipped (Annotation not found): {annotation_not_found_count} images.")
    # <<< 修正 >>> スキップ理由の計算を微調整 (エラーもスキップに含まれるため)
    other_skipped = skipped_count - annotation_not_found_count
    print(f"Skipped (Invalid/missing bbox, errors, etc.): {other_skipped} images.")
    # print(f"Skipped (Errors during processing): {error_count} images.") # エラー数は↑に含まれるため、冗長ならコメントアウト
    print(f"Total skipped: {skipped_count} images.")
    print(f"Cropped images saved to: {OUTPUT_BASE_DIR}")

if __name__ == "__main__":
    main()