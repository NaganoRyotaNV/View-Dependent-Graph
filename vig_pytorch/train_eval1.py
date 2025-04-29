#!/usr/bin/env python
"""
評価用スクリプト（修正版）
"""

import sys
sys.path.insert(0, "/mnt/Efficient-AI-Backbones/vig_pytorch")

import warnings
warnings.filterwarnings('ignore')

import argparse, time, yaml, os, logging, json
from collections import defaultdict, OrderedDict
from datetime import datetime
from contextlib import suppress
import vig

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from timm.data import resolve_data_config
from timm.models import create_model
from timm.utils import setup_default_logging, get_outdir

_logger = logging.getLogger('eval')

def propagate(scores):
    updated = {}
    updated["front"]     = scores["front"] + scores["frontside"]
    updated["frontside"] = scores["frontside"] + (scores["front"] + scores["side"]) / 2
    updated["side"]      = scores["side"] + (scores["frontside"] + scores["backside"]) / 2
    updated["backside"]  = scores["backside"] + (scores["side"] + scores["back"]) / 2
    updated["back"]      = scores["back"] + scores["backside"]
    return updated

def dict_softmax(scores_dict):
    valid_keys = [k for k, v in scores_dict.items() if not np.isnan(v)]
    if not valid_keys:
        return {k: float('nan') for k in scores_dict}
    vals = np.array([scores_dict[k] for k in valid_keys])
    max_val = np.max(vals)
    exp_vals = np.exp(vals - max_val)
    sum_exp = np.sum(exp_vals)
    result = {}
    for k in scores_dict:
        if k in valid_keys:
            result[k] = np.exp(scores_dict[k] - max_val) / sum_exp
        else:
            result[k] = float('nan')
    return result

class MyImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, labeled=True):
        self.root = root
        self.transform = transform
        self.samples = []
        self.labeled = labeled

        if self.labeled:
            subdirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
            self.classes = sorted(subdirs)
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
            self.idx_to_class = {i: cls_name for cls_name, i in self.class_to_idx.items()}
            for cls_name in self.classes:
                full_dir = os.path.join(root, cls_name)
                label = self.class_to_idx[cls_name]
                for fname in os.listdir(full_dir):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        fpath = os.path.join(full_dir, fname)
                        self.samples.append((fpath, label))
        else:
            for fname in os.listdir(root):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    fpath = os.path.join(root, fname)
                    self.samples.append((fpath, -1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label, path

def parse_class_name(class_name):
    parts = class_name.rsplit('_', 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return class_name, None

def extract_logits(model, loader, output_file, idx_to_class, labeled=True, amp_autocast=suppress):
    model.eval()
    desired_order = ["front", "frontside", "side", "backside", "back"]

    # object_to_direction[obj][direction] = class_idx
    object_to_direction = defaultdict(dict)
    for idx, cls_name in idx_to_class.items():
        obj, direction = parse_class_name(cls_name)
        if direction is not None:
            object_to_direction[obj][direction] = idx
    object_list = sorted(object_to_direction.keys())

    # 方向別の TOP1 / TOP5 集計用
    stats = {
        obj: {d: {"total": 0, "top1": 0, "top5": 0} for d in desired_order}
        for obj in object_list
    }

    # オブジェクト単位で Top1/Top5 を集計
    object_acc_stats = defaultdict(lambda: {"total": 0, "top1": 0, "top5": 0})

    with torch.no_grad(), open(output_file, "w", encoding="utf-8") as f:
        f.write("Test Results\n\n")
        overall_obj_total = 0
        overall_obj_top1 = 0
        overall_obj_top5 = 0

        for batch_idx, (inputs, targets, paths) in enumerate(loader):
            inputs = inputs.cuda()
            with amp_autocast():
                outputs = model(inputs)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            logits = outputs.cpu()
            softmax_all = F.softmax(logits, dim=1).cpu().numpy()
            logits_np = logits.numpy()

            for i, path in enumerate(paths):
                f.write(f"[Image: {path}]\n")
                parts = path.split(os.sep)
                folder = parts[-2] if len(parts) >= 2 else "unknown"
                true_class = folder
                true_object, true_direction = parse_class_name(true_class)
                f.write(f"  True label: {true_class}\n")

                topk = 5
                _, pred_topk = torch.tensor(softmax_all[i]).topk(topk, largest=True, sorted=True)
                pred_topk = pred_topk.numpy().tolist()
                pred_top1 = pred_topk[0]

                # 各物体ごとに伝播前の softmax スコアを取得 → propagate
                propagated_dict = {}
                for obj in object_list:
                    raw_scores = {}
                    softmax_scores = {}
                    for d in desired_order:
                        if d in object_to_direction[obj]:
                            idx_val = object_to_direction[obj][d]
                            raw_scores[d] = logits_np[i][idx_val]
                            softmax_scores[d] = softmax_all[i][idx_val]
                        else:
                            raw_scores[d] = float('nan')
                            softmax_scores[d] = float('nan')
                    propagated = propagate(softmax_scores)
                    propagated_dict[obj] = propagated
                    f.write(f"  {obj} classification:\n")
                    f.write("    Raw Logits:\n")
                    for d in desired_order:
                        f.write(f"      ({d}): {raw_scores[d]:.4f}\n")
                    f.write("    Softmax Scores:\n")
                    for d in desired_order:
                        f.write(f"      ({d}): {softmax_scores[d]:.4f}\n")
                    f.write("    After Propagation (raw):\n")
                    for d in desired_order:
                        f.write(f"      ({d}): {propagated[d]:.4f}\n")
                    f.write("\n")

                # 51クラス全体で最終的に re-softmax
                final_scores = np.empty(len(idx_to_class))
                for idx_, cls_name in idx_to_class.items():
                    obj_, direction_ = parse_class_name(cls_name)
                    if obj_ in propagated_dict and direction_ in propagated_dict[obj_]:
                        final_scores[idx_] = propagated_dict[obj_][direction_]
                    else:
                        final_scores[idx_] = softmax_all[i][idx_]
                final_scores_clean = np.where(np.isnan(final_scores), -np.inf, final_scores)
                exp_final = np.exp(final_scores_clean - np.max(final_scores_clean))
                final_prob = exp_final / np.sum(exp_final)
                final_pred = np.argmax(final_prob)
                predicted_class_name = idx_to_class.get(final_pred, "unknown")
                final_pred_object, _ = parse_class_name(predicted_class_name)
                f.write("Final re_softmax (51-class softmax over propagated scores):\n")
                for idx_, cls_name in idx_to_class.items():
                    f.write(f"  {cls_name}: {final_prob[idx_]:.4f}\n")
                f.write(f"Final predicted class: {predicted_class_name}\n")
                f.write(f"Final predicted object: {final_pred_object}\n\n")

                # オブジェクト単位でスコア最大値を用いて TOP1/TOP5 を算出
                object_scores = {}
                for obj in object_list:
                    if obj in propagated_dict:
                        scores = propagated_dict[obj]
                        object_scores[obj] = max(scores.values())
                    else:
                        object_scores[obj] = float('nan')
                sorted_objects = sorted(
                    object_scores.items(),
                    key=lambda x: x[1] if not np.isnan(x[1]) else -1,
                    reverse=True
                )
                top1_object = sorted_objects[0][0] if sorted_objects else "unknown"
                top5_objects = [obj for obj, score in sorted_objects[:5]]
                f.write(f"  Predicted label (full): {idx_to_class.get(pred_top1, 'unknown')}\n")
                f.write(f"  Predicted object (TOP1): {top1_object}\n")
                f.write(f"  Top 5 objects: {', '.join(top5_objects)}\n\n")

                overall_obj_total += 1
                if true_object == top1_object:
                    overall_obj_top1 += 1
                if true_object in top5_objects:
                    overall_obj_top5 += 1

                # オブジェクト単位の集計
                object_acc_stats[true_object]["total"] += 1
                if true_object == top1_object:
                    object_acc_stats[true_object]["top1"] += 1
                if true_object in top5_objects:
                    object_acc_stats[true_object]["top5"] += 1

                # 方向別の TOP1/TOP5 集計
                if true_direction in desired_order:
                    stats[true_object][true_direction]["total"] += 1
                    # direction-level の Top1
                    valid_dirs = [d for d in desired_order if not np.isnan(propagated_dict[true_object][d])]
                    best_dir = max(valid_dirs, key=lambda d: propagated_dict[true_object][d]) if valid_dirs else None
                    if best_dir == true_direction:
                        stats[true_object][true_direction]["top1"] += 1
                    # direction-level の Top5
                    sorted_dirs = sorted(
                        [(d, propagated_dict[true_object][d]) for d in valid_dirs],
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]
                    top5_dirs = [d for d, s in sorted_dirs]
                    if true_direction in top5_dirs:
                        stats[true_object][true_direction]["top5"] += 1

        # ============================================
        # 結果を per_object_direction_evaluation.txt に出力
        # ============================================
        with open(os.path.join(os.path.dirname(output_file), "per_object_direction_evaluation.txt"), "w", encoding="utf-8") as outf:
            outf.write("Object\tDirection\t#Samples\tTOP1 (%)\tTOP5 (%)\n")
            for obj in object_list:
                for d in desired_order:
                    total = stats[obj][d]["total"]
                    if total == 0:
                        top1_acc = "nan"
                        top5_acc = "nan"
                    else:
                        # ★ 小数点第2位まで表示
                        top1_acc_val = stats[obj][d]['top1'] / total * 100
                        top1_acc = f"{top1_acc_val:.2f}"
                        top5_acc_val = stats[obj][d]['top5'] / total * 100
                        top5_acc = f"{top5_acc_val:.2f}"
                    outf.write(f"{obj}\t{d}\t{total}\t{top1_acc}\t{top5_acc}\n")

            outf.write("\nOverall Object Evaluation:\n")
            overall_top1_acc = overall_obj_top1 / overall_obj_total * 100 if overall_obj_total > 0 else 0.0
            overall_top5_acc = overall_obj_top5 / overall_obj_total * 100 if overall_obj_total > 0 else 0.0

            # こちらも小数点第2位まで
            outf.write(f"TOP1: {overall_obj_top1}/{overall_obj_total} ({overall_top1_acc:.2f}%)\n")
            outf.write(f"TOP5: {overall_obj_top5}/{overall_obj_total} ({overall_top5_acc:.2f}%)\n")

            outf.write("\nPer Object TOP1/TOP5 Accuracy:\n")
            for obj in object_list:
                total_obj = object_acc_stats[obj]["total"]
                top1_corr = object_acc_stats[obj]["top1"]
                top5_corr = object_acc_stats[obj]["top5"]
                if total_obj == 0:
                    top1_acc = "nan"
                    top5_acc = "nan"
                else:
                    # ★ ここも小数点第2位
                    top1_acc_val = (top1_corr / total_obj) * 100
                    top1_acc = f"{top1_acc_val:.2f}"
                    top5_acc_val = (top5_corr / total_obj) * 100
                    top5_acc = f"{top5_acc_val:.2f}"
                outf.write(
                    f"{obj}: TOP1 = {top1_corr}/{total_obj} ({top1_acc}%), "
                    f"TOP5 = {top5_corr}/{total_obj} ({top5_acc}%)\n"
                )

        _logger.info(
            f"Evaluation results written to: {os.path.abspath(os.path.join(os.path.dirname(output_file), 'per_object_direction_evaluation.txt'))}"
        )

    return

def _parse_args():
    config_parser = argparse.ArgumentParser(description='Evaluation Config', add_help=False)
    config_parser.add_argument('-c', '--config',
                               default='/mnt/Efficient-AI-Backbones/result/train/dir/args.yaml',
                               type=str, metavar='FILE',
                               help='YAML config file specifying default arguments')
    parser = argparse.ArgumentParser(description='Vision GNN Evaluation')
    parser.add_argument('data', metavar='DIR', default='/mnt/ViHGNN_pme/data/main_data', nargs='?',
                        help='データセットのルートディレクトリ（例: /mnt/ViHGNN_pme/data/main_data）')
    parser.add_argument('--model', default='vig_b_224_gelu', type=str, metavar='MODEL',
                        help='モデル名（例: vig_b_224_gelu）')
    parser.add_argument('--num-classes', type=int, default=51, metavar='N',
                        help='クラス数（今回は 51）')
    parser.add_argument('--resume', default='/mnt/Efficient-AI-Backbones/result/train/dir/checkpoint-100.pth.tar',
                        type=str, metavar='PATH', help='モデルチェックポイントのパス')
    parser.add_argument('-b', '--batch-size', type=int, default=64, metavar='N')
    parser.add_argument('--workers', type=int, default=4, metavar='N')
    parser.add_argument('--pin-mem', action='store_true', default=False)
    parser.add_argument('--output', default='/mnt/Efficient-AI-Backbones/result', type=str, metavar='PATH',
                        help='出力ディレクトリ')
    parser.add_argument('--evaluate', action='store_true', default=True,
                        help='評価のみ実行する')
    parser.add_argument('--amp', action='store_true', default=True)
    parser.add_argument('--img-size', type=int, default=224, metavar='N')
    parser.add_argument('--seed', type=int, default=42, metavar='S')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--eval-dir', default='val', type=str,
                        help='評価用ディレクトリ名（例: "val" or "test"）')
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    args = parser.parse_args(remaining)
    if args.img_size is None:
        args.img_size = 224
    if not args.resume:
        args.resume = '/mnt/Efficient-AI-Backbones/result/train/dir/checkpoint-100.pth.tar'
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

def main():
    setup_default_logging()
    args, args_text = _parse_args()
    output_dir = get_outdir(args.output, 'eval', datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'args.yaml'), 'w', encoding='utf-8') as f:
        f.write(args_text)
    _logger.info(f"Output directory: {output_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    amp_autocast = suppress
    if args.amp and hasattr(torch.cuda.amp, 'autocast'):
        amp_autocast = torch.cuda.amp.autocast

    # 評価用ディレクトリ
    eval_root = os.path.join(args.data, args.eval_dir)
    if not os.path.isdir(eval_root):
        _logger.error(f"Evaluation folder not found: {eval_root}")
        exit(1)

    data_config = resolve_data_config(vars(args), model=create_model(args.model, num_classes=args.num_classes), verbose=False)
    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_config['mean'], std=data_config['std'])
    ])
    dataset_val = MyImageDataset(eval_root, transform=val_transform, labeled=True)
    loader_val = DataLoader(dataset_val, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers,
                            pin_memory=args.pin_mem)

    model = create_model(args.model, num_classes=args.num_classes,
                         pretrained=False, img_size=args.img_size)
    model = model.to(device)
    _logger.info(f"Loading checkpoint from {args.resume}")
    state_dict = torch.load(args.resume, map_location=device)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    _logger.info(f"Checkpoint loaded (missing: {missing}, unexpected: {unexpected})")
    model.eval()

    output_file = os.path.join(output_dir, "logits_val.txt")
    _logger.info(f"Extracting logits for validation data to {output_file}")
    extract_logits(model, loader_val, output_file, dataset_val.idx_to_class,
                   labeled=True, amp_autocast=amp_autocast)

    _logger.info("Evaluation finished.")

if __name__ == "__main__":
    main()
