#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import scipy.io
import pprint
from scipy.io.matlab import _mio5_params, mat_struct # mat_struct もインポート

def mat_struct_to_dict(matobj):
    """
    MATLAB の mat_struct オブジェクトを再帰的に辞書に変換する関数
    """
    d = {}
    for field in matobj._fieldnames:
        value = getattr(matobj, field)
        d[field] = _convert_value(value)
    return d

def _convert_value(value):
    """
    MATLAB の構造体（mat_struct）を辞書に変換、また ndarray ならリストに変換
    """
    if isinstance(value, _mio5_params.mat_struct) or isinstance(value, mat_struct): # mat_struct もチェック
        return mat_struct_to_dict(value)
    elif isinstance(value, np.ndarray):
        return _tolist(value)
    else:
        return value

def _tolist(ndarray):
    """
    ndarray 内の要素を再帰的に変換してリストにする関数
    """
    # ndarray の dtype が object なら各要素に対して変換を実施
    if ndarray.dtype.kind == 'O':
        return [_convert_value(item) for item in ndarray]
    else:
        return ndarray.tolist()

def print_mat_structure(file_path):
    try:
        # MATファイルを読み込み (詳細表示用に追加)
        mat_data_full = scipy.io.loadmat(file_path, squeeze_me=True, struct_as_record=False)

        print(f"MATファイルの構造 (簡易表示): {file_path}")
        print("-" * 50)

        # トップレベルのキーを表示 (元の check_matfile.py の機能)
        print("トップレベルキー:")
        for key in mat_data_full.keys(): # mat_data_full を使用
            if not key.startswith('__'):
                print(f" - {key}: {type(mat_data_full[key])}")

                value = mat_data_full[key]
                if isinstance(value, dict):
                    print_nested_dict_simple(value, indent=4)
                elif isinstance(value, mat_struct):
                    print_nested_struct_simple(value, indent=4)
                elif isinstance(value, np.ndarray):
                    print_numpy_array_info_simple(value, indent=4)

        print("\n=== MATファイルの内容 (詳細表示) ===")
        pp = pprint.PrettyPrinter(indent=2)
        pp.pprint(mat_data_full)

        if 'record' in mat_data_full:
            print("\n=== 'record' の詳細内容 (変換後) ===")
            record = mat_data_full['record']
            if isinstance(record, (list, tuple, np.ndarray)):
                for i, rec in enumerate(record):
                    if isinstance(rec, (mat_struct, _mio5_params.mat_struct)):
                        record_dict = mat_struct_to_dict(rec)
                        print(f"\n--- record[{i}] ---")
                        pp.pprint(record_dict)
                    else:
                        print(f"\n--- record[{i}] (変換なし) ---")
                        pp.pprint(rec)
            elif isinstance(record, (mat_struct, _mio5_params.mat_struct)):
                record_dict = mat_struct_to_dict(record)
                pp.pprint(record_dict)
            else:
                pp.pprint(record)
        else:
            print("No 'record' field found in the .mat file.")

    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")

def print_nested_dict_simple(dictionary, indent=0):
    space = " " * indent
    for key, value in dictionary.items():
        print(f"{space}- {key}: {type(value)}")
        if isinstance(value, dict):
            print_nested_dict_simple(value, indent + 4)
        elif isinstance(value, mat_struct):
            print_nested_struct_simple(value, indent + 4)
        elif isinstance(value, np.ndarray):
            print_numpy_array_info_simple(value, indent + 4)

def print_nested_struct_simple(struct, indent=0):
    space = " " * indent
    if isinstance(struct, mat_struct):
        for field in struct._fieldnames:
            value = getattr(struct, field)
            print(f"{space}- {field}: {type(value)}")
            if isinstance(value, mat_struct):
                print_nested_struct_simple(value, indent + 4)
            elif isinstance(value, np.ndarray):
                print_numpy_array_info_simple(value, indent + 4)
    else:
        print(f"{space}  内容: {struct}")

def print_numpy_array_info_simple(array, indent=0):
    space = " " * indent
    print(f"{space}- shape: {array.shape}")
    print(f"{space}- dtype: {array.dtype}")

def main():
    if len(sys.argv) != 2:
        print("使用方法: python check_matfile.py <matファイルのパス>")
        sys.exit(1)

    print_mat_structure(sys.argv[1])

if __name__ == "__main__":
    main()