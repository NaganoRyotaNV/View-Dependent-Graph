ソート
python sort_objectnet3d.py
クロップ
python crop_ObjectNet.py

train test val分割
python dataset_split.py     --input_dir /mnt/data/ObjectNet3D_Cropped     --output_dir /mnt/data/ObjectNet3D_Split     --train_ratio 0.6     --val_ratio 0.3     --seed 42
統合
python flatten_viewpoints.py     --input_dir /mnt/data/ObjectNet3D_Split     --output_dir /mnt/data/ObjectNet3D_Split_Flattened


フラット構造 (物体ラベル用) の分析:
python analyze_structure.py flat \
    --input_dir /mnt/data/ObjectNet3D_Split_Flattened \
    --output_file flat_analysis_results.txt
階層構造 (方向分類用) の分析:
    python analyze_structure.py hierarchical \
    --input_dir /mnt/data/ObjectNet3D_Split \
    --output_file hierarchical_analysis_results.txt