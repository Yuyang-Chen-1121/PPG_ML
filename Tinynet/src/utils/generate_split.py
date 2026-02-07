# Author: Yuyang Chen
# School: Zhejiang University
# Last Modified: 2026.2.7
# Description: Split generation and dataset balancing utility script.

"""Generate subject-isolated balanced splits with hard-negative accounting."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Tuple

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(SRC_ROOT)

from utils.config import get_bpm_bins, load_config

BUCKETS = ("hr_low", "hr_mid", "hr_high", "af_pos", "nsr", "hard_neg")


# Purpose: Implement `load_simband_source_map` for the TinyNet workflow.
# Inputs: Parameters defined in `load_simband_source_map` signature.
# Outputs: Return value produced by `load_simband_source_map`.
# Assumptions: Caller provides valid types/shapes for this operation.
def load_simband_source_map(processed_dir: str) -> Dict[str, str]:
    meta_path = os.path.join(processed_dir, "simband_source_map.json")
    if not os.path.exists(meta_path):
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return {str(k): str(v) for k, v in payload.items()}
    except Exception:
        return {}


# Purpose: Implement `is_simband_clinical` for the TinyNet workflow.
# Inputs: Parameters defined in `is_simband_clinical` signature.
# Outputs: Return value produced by `is_simband_clinical`.
# Assumptions: Caller provides valid types/shapes for this operation.
def is_simband_clinical(stem: str, source_map: Dict[str, str]) -> bool:
    if stem in source_map:
        return source_map[stem].strip().lower() == "clinical"
    return "clinical" in stem.lower()


# Purpose: Implement `count_buckets_for_file` for the TinyNet workflow.
# Inputs: Parameters defined in `count_buckets_for_file` signature.
# Outputs: Return value produced by `count_buckets_for_file`.
# Assumptions: Caller provides valid types/shapes for this operation.
def count_buckets_for_file(
    y_path: str,
    bpm_bins: np.ndarray,
    source: str,
    stem: str,
    simband_map: Dict[str, str],
    hr_low_th: float,
    hr_high_th: float,
) -> Dict[str, int]:
    counts = {bucket: 0 for bucket in BUCKETS}
    try:
        y_arr = np.load(y_path, mmap_mode="r")
    except Exception:
        return counts

    if y_arr.ndim == 2 and y_arr.shape[1] == bpm_bins.shape[0]:
        bpm = np.sum(y_arr * bpm_bins, axis=1)
        counts["hr_low"] = int(np.sum(bpm < hr_low_th))
        counts["hr_mid"] = int(np.sum((bpm >= hr_low_th) & (bpm <= hr_high_th)))
        counts["hr_high"] = int(np.sum(bpm > hr_high_th))
        return counts

    if y_arr.ndim == 2 and y_arr.shape[1] == 2:
        labels = np.argmax(y_arr, axis=1)
    elif y_arr.ndim == 1:
        labels = (y_arr > 0.5).astype(np.int32)
    else:
        return counts

    af_pos = int(np.sum(labels == 1))
    af_neg = int(np.sum(labels == 0))
    counts["af_pos"] = af_pos

    is_clinical = source == "UMMCSIMBAND" and is_simband_clinical(stem, simband_map)
    if is_clinical:
        counts["hard_neg"] = af_neg
    else:
        counts["nsr"] = af_neg
    return counts


# Purpose: Implement `scan_processed_dir` for the TinyNet workflow.
# Inputs: Parameters defined in `scan_processed_dir` signature.
# Outputs: Return value produced by `scan_processed_dir`.
# Assumptions: Caller provides valid types/shapes for this operation.
def scan_processed_dir(
    processed_dir: str,
    source_prefix: str,
    bpm_bins: np.ndarray,
    simband_map: Dict[str, str],
    hr_low_th: float,
    hr_high_th: float,
) -> Tuple[Dict[str, List[str]], Dict[str, np.ndarray]]:
    subjects: Dict[str, List[str]] = {}
    subject_counts: Dict[str, np.ndarray] = {}

    if not os.path.exists(processed_dir):
        return subjects, subject_counts

    for root, _, files in os.walk(processed_dir):
        for file_name in sorted(files):
            if not file_name.endswith("_y.npy"):
                continue
            stem = file_name.replace("_y.npy", "")
            entry = f"{source_prefix}_{stem}"
            subject_id = entry

            y_path = os.path.join(root, file_name)
            counts = count_buckets_for_file(
                y_path=y_path,
                bpm_bins=bpm_bins,
                source=source_prefix,
                stem=stem,
                simband_map=simband_map,
                hr_low_th=hr_low_th,
                hr_high_th=hr_high_th,
            )

            if subject_id not in subjects:
                subjects[subject_id] = []
                subject_counts[subject_id] = np.zeros(len(BUCKETS), dtype=np.int64)

            subjects[subject_id].append(entry)
            subject_counts[subject_id] += np.array([counts[b] for b in BUCKETS], dtype=np.int64)

    return subjects, subject_counts


# Purpose: Implement `aggregate_counts` for the TinyNet workflow.
# Inputs: Parameters defined in `aggregate_counts` signature.
# Outputs: Return value produced by `aggregate_counts`.
# Assumptions: Caller provides valid types/shapes for this operation.
def aggregate_counts(subject_ids: List[str], subject_counts: Dict[str, np.ndarray]) -> np.ndarray:
    if not subject_ids:
        return np.zeros(len(BUCKETS), dtype=np.int64)
    return np.sum([subject_counts[sid] for sid in subject_ids], axis=0)


# Purpose: Implement `balance_score` for the TinyNet workflow.
# Inputs: Parameters defined in `balance_score` signature.
# Outputs: Return value produced by `balance_score`.
# Assumptions: Caller provides valid types/shapes for this operation.
def balance_score(split_counts: List[np.ndarray], global_counts: np.ndarray) -> float:
    total_global = float(np.sum(global_counts))
    if total_global <= 0:
        return float("inf")
    global_ratio = global_counts / total_global

    score = 0.0
    for counts in split_counts:
        split_total = float(np.sum(counts))
        if split_total <= 0:
            score += 1e6
            continue
        split_ratio = counts / split_total
        for idx, g_ratio in enumerate(global_ratio):
            if global_counts[idx] <= 0:
                continue
            score += float(abs(split_ratio[idx] - g_ratio))
    return score


# Purpose: Compute the requested metric or transformed value.
# Inputs: Parameters defined in `compute_weights` signature.
# Outputs: Return value produced by `compute_weights`.
# Assumptions: Caller provides valid types/shapes for this operation.
def compute_weights(global_counts: np.ndarray, eps: float = 1e-6) -> Dict[str, Dict[str, float]]:
    bucket = {name: int(global_counts[i]) for i, name in enumerate(BUCKETS)}
    af_pos = bucket["af_pos"]
    af_neg = bucket["nsr"] + bucket["hard_neg"]
    hr_low = bucket["hr_low"]
    hr_mid = bucket["hr_mid"]
    hr_high = bucket["hr_high"]

    af_total = af_pos + af_neg
    hr_total = hr_low + hr_mid + hr_high

    task_target = max(af_total, hr_total, 1)
    af_task_w = task_target / (af_total + eps)
    hr_task_w = task_target / (hr_total + eps)

    af_target = max(af_pos, af_neg, 1)
    af_pos_w = af_target / (af_pos + eps)
    af_neg_w = af_target / (af_neg + eps)

    hr_target = max(hr_low, hr_mid, hr_high, 1)
    hr_low_w = hr_target / (hr_low + eps)
    hr_mid_w = hr_target / (hr_mid + eps)
    hr_high_w = hr_target / (hr_high + eps)

    return {
        "task": {"af": float(af_task_w), "hr": float(hr_task_w)},
        "af": {"pos": float(af_pos_w), "neg": float(af_neg_w)},
        "hr": {"low": float(hr_low_w), "mid": float(hr_mid_w), "high": float(hr_high_w)},
    }


# Purpose: Implement `format_table` for the TinyNet workflow.
# Inputs: Parameters defined in `format_table` signature.
# Outputs: Return value produced by `format_table`.
# Assumptions: Caller provides valid types/shapes for this operation.
def format_table(rows: List[List[str]]) -> str:
    col_widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = []
    for idx, row in enumerate(rows):
        line = " | ".join(value.ljust(col_widths[i]) for i, value in enumerate(row))
        lines.append(line)
        if idx == 0:
            lines.append("-+-".join("-" * w for w in col_widths))
    return "\n".join(lines)


# Purpose: Implement `generate_split` for the TinyNet workflow.
# Inputs: Parameters defined in `generate_split` signature.
# Outputs: Return value produced by `generate_split`.
# Assumptions: Caller provides valid types/shapes for this operation.
def generate_split(config_path: str, iterations: int, seed: int) -> None:
    cfg = load_config(config_path)
    bpm_bins = get_bpm_bins(cfg)

    processed_root = cfg.get("paths.processed_root", "./data/processed")
    dalia_dir = cfg.get("paths.processed.dalia", os.path.join(processed_root, "dalia"))
    bami_dir = cfg.get("paths.processed.bami", os.path.join(processed_root, "BAMI"))
    simband_dir = cfg.get("paths.processed.simband", os.path.join(processed_root, "UMMCSIMBAND"))

    hr_low_th = float(cfg.get("training.sampler.hr_low_threshold", 75.0))
    hr_high_th = float(cfg.get("training.sampler.hr_high_threshold", 120.0))

    simband_map = load_simband_source_map(simband_dir)

    subjects: Dict[str, List[str]] = {}
    subject_counts: Dict[str, np.ndarray] = {}

    for processed_dir, prefix in [(dalia_dir, "dalia"), (bami_dir, "BAMI"), (simband_dir, "UMMCSIMBAND")]:
        sub_subjects, sub_counts = scan_processed_dir(
            processed_dir=processed_dir,
            source_prefix=prefix,
            bpm_bins=bpm_bins,
            simband_map=simband_map,
            hr_low_th=hr_low_th,
            hr_high_th=hr_high_th,
        )
        for sid, entries in sub_subjects.items():
            subjects[sid] = entries
        for sid, counts in sub_counts.items():
            subject_counts[sid] = counts

    if not subjects:
        print("No processed data found. Run preprocessing first.")
        return

    subject_ids = list(subjects.keys())
    n_subjects = len(subject_ids)
    n_train = int(n_subjects * 0.75)
    n_val = int(n_subjects * 0.15)
    n_test = n_subjects - n_train - n_val

    rng = random.Random(seed)

    global_counts = aggregate_counts(subject_ids, subject_counts)
    best_score = float("inf")
    best_split = ([], [], [])
    best_counts = (np.zeros(len(BUCKETS), dtype=np.int64),) * 3

    for idx in range(iterations):
        shuffled = list(subject_ids)
        rng.shuffle(shuffled)
        train_ids = shuffled[:n_train]
        val_ids = shuffled[n_train : n_train + n_val]
        test_ids = shuffled[n_train + n_val :]

        train_counts = aggregate_counts(train_ids, subject_counts)
        val_counts = aggregate_counts(val_ids, subject_counts)
        test_counts = aggregate_counts(test_ids, subject_counts)
        score = balance_score([train_counts, val_counts, test_counts], global_counts)

        if score < best_score:
            best_score = score
            best_split = (train_ids, val_ids, test_ids)
            best_counts = (train_counts, val_counts, test_counts)

        if (idx + 1) % max(1, iterations // 10) == 0:
            print(f"Iteration {idx + 1}/{iterations} best_score={best_score:.6f}")

    train_entries = [entry for sid in best_split[0] for entry in subjects[sid]]
    val_entries = [entry for sid in best_split[1] for entry in subjects[sid]]
    test_entries = [entry for sid in best_split[2] for entry in subjects[sid]]

    sampler_weights = compute_weights(global_counts)
    sampler_weights["hr"]["low_threshold"] = hr_low_th
    sampler_weights["hr"]["high_threshold"] = hr_high_th

    output_path = cfg.get("paths.split_json", "./split_optimized.json")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "train": sorted(train_entries),
                "val": sorted(val_entries),
                "test": sorted(test_entries),
                "sampler_config": sampler_weights,
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    rows = [
        ["Split", "Subjects", "AF_Pos", "NSR", "Hard_Neg", "HR_Low", "HR_Mid", "HR_High", "Total"],
    ]
    for name, ids, counts in [
        ("Train", best_split[0], best_counts[0]),
        ("Val", best_split[1], best_counts[1]),
        ("Test", best_split[2], best_counts[2]),
        ("Global", subject_ids, global_counts),
    ]:
        total = int(np.sum(counts))
        rows.append(
            [
                name,
                str(len(ids)),
                str(int(counts[BUCKETS.index("af_pos")])),
                str(int(counts[BUCKETS.index("nsr")])),
                str(int(counts[BUCKETS.index("hard_neg")])),
                str(int(counts[BUCKETS.index("hr_low")])),
                str(int(counts[BUCKETS.index("hr_mid")])),
                str(int(counts[BUCKETS.index("hr_high")])),
                str(total),
            ]
        )

    print("\nSplit distribution (window counts):")
    print(format_table(rows))
    print(f"\nBest balance score: {best_score:.6f}")
    print(f"Subjects: total={n_subjects}, train={n_train}, val={n_val}, test={n_test}")

    print("\nSampler weights (stage 1):")
    print(json.dumps(sampler_weights, indent=2, sort_keys=True))
    print(f"\nSaved split file: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/config.yaml")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_split(args.config, iterations=args.iterations, seed=args.seed)
