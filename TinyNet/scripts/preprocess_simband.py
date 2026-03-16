# Author: Yuyang Chen
# School: Zhejiang University
# Last Modified: 2026.2.7
# Description: Preprocessing entrypoint for UMMCSIMBAND dataset signals.

"""Preprocess UMMCSIMBAND AF data into TinyNet windows."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile

import numpy as np
import pandas as pd
from scipy import signal

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from data.preprocessing import pack_multimodal_channels, preprocess_ppg_signal, z_score_normalization
from utils.config import load_config

VALID_LABELS = {
    0.0: 0,
    1.0: 1,
    2.0: 0,
    0: 0,
    1: 1,
    2: 0,
    "0": 0,
    "1": 1,
    "2": 0,
    "0.0": 0,
    "1.0": 1,
    "2.0": 0,
}


# Purpose: Convert one synchronized AF segment to fixed-length windows.
# Inputs: raw ppg, raw acc magnitude, binary AF label, config accessor.
# Outputs: tuple(list_of_X_windows, list_of_AF_labels).
# Assumptions: PPG/ACC vectors are aligned and sampled at config source_fs.simband.
def process_simband_segment(
    ppg_raw: np.ndarray,
    acc_mag_raw: np.ndarray,
    label_val: int,
    cfg,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    target_fs = cfg.get_int("data.sampling_rate_hz", 32)
    window_len = cfg.get_int("data.window_seconds", 10) * target_fs

    low = cfg.get_float("preprocessing.bandpass_low_hz", 0.5)
    high = cfg.get_float("preprocessing.bandpass_high_hz", 4.0)
    order = cfg.get_int("preprocessing.bandpass_order", 2)
    source_fs = cfg.get_float("preprocessing.source_fs.simband", 50.0)
    hampel_window = cfg.get_int("preprocessing.hampel_window_size", 7)
    hampel_th = cfg.get_float("preprocessing.hampel_threshold_std", 3.0)
    enable_zscore = cfg.get_bool("preprocessing.enable_zscore", True)
    z_eps = cfg.get_float("preprocessing.zscore_epsilon", 1e-6)

    ppg_1d = ppg_raw[:, 0] if ppg_raw.ndim > 1 else ppg_raw
    acc_1d = acc_mag_raw[:, 0] if acc_mag_raw.ndim > 1 else acc_mag_raw

    ppg_norm = preprocess_ppg_signal(
        raw_ppg=ppg_1d,
        source_fs=source_fs,
        target_fs=target_fs,
        lowcut=low,
        highcut=high,
        order=order,
        hampel_window_size=hampel_window,
        hampel_threshold_std=hampel_th,
        enable_zscore=enable_zscore,
        zscore_epsilon=z_eps,
    )
    acc_resampled = signal.resample(acc_1d, ppg_norm.shape[0])
    acc_mag_norm = z_score_normalization(acc_resampled, epsilon=z_eps)
    zeros = np.zeros_like(acc_mag_norm)

    af_label = np.array([1.0, 0.0], dtype=np.float32) if label_val == 0 else np.array([0.0, 1.0], dtype=np.float32)

    x_segments: list[np.ndarray] = []
    y_labels: list[np.ndarray] = []
    for start in range(0, len(ppg_norm) - window_len + 1, window_len):
        end = start + window_len
        x_segments.append(
            pack_multimodal_channels(
                ppg=ppg_norm[start:end],
                acc_mag=acc_mag_norm[start:end],
                acc_x=zeros[start:end],
                acc_y=zeros[start:end],
                acc_z=zeros[start:end],
                target_len=window_len,
            )
        )
        y_labels.append(af_label)

    return x_segments, y_labels


# Purpose: Stream one tar archive and produce AF windows matched by label table.
# Inputs: tar path, label DataFrame, config accessor.
# Outputs: tuple(list_of_X_windows, list_of_AF_labels).
# Assumptions: tar includes paired *_ppg_*.txt and *_accel_*.txt files.
def process_tar_file(tar_path: str, label_df: pd.DataFrame, cfg) -> tuple[list[np.ndarray], list[np.ndarray]]:
    ppg_buffers = {}
    acc_buffers = {}

    with tarfile.open(tar_path, "r:*") as archive:
        for member in archive:
            if not member.isfile() or member.size == 0:
                continue
            name = os.path.basename(member.name)
            if "_ppg_" in name and name.endswith(".txt"):
                key = name.replace("_ppg_", "_").replace(".txt", "")
                handle = archive.extractfile(member)
                if handle is not None:
                    ppg_buffers[key] = np.loadtxt(handle)
            elif "_accel_" in name and name.endswith(".txt"):
                key = name.replace("_accel_", "_").replace(".txt", "")
                handle = archive.extractfile(member)
                if handle is not None:
                    acc_buffers[key] = np.loadtxt(handle)

    batch_x: list[np.ndarray] = []
    batch_y: list[np.ndarray] = []
    for key, ppg_data in ppg_buffers.items():
        if key not in acc_buffers:
            continue
        last_us = key.rfind("_")
        if last_us < 0:
            continue

        label_key = f"{key[:last_us]}_ppg{key[last_us:]}"
        if label_key not in label_df.index:
            continue

        raw_label = label_df.loc[label_key]["label"]
        if raw_label not in VALID_LABELS:
            continue

        acc_data = acc_buffers[key]
        min_len = min(len(ppg_data), len(acc_data))
        if min_len <= 0:
            continue

        x_seg, y_seg = process_simband_segment(
            ppg_raw=ppg_data[:min_len],
            acc_mag_raw=acc_data[:min_len],
            label_val=VALID_LABELS[raw_label],
            cfg=cfg,
        )
        batch_x.extend(x_seg)
        batch_y.extend(y_seg)

    return batch_x, batch_y


# Purpose: Load and normalize all label CSV files into a lookup table.
# Inputs: label directory path.
# Outputs: DataFrame indexed by filename with numeric AF labels.
# Assumptions: CSV columns are [fname, label].
def load_labels(labels_dir: str) -> pd.DataFrame:
    all_frames = []
    if not os.path.exists(labels_dir):
        return pd.DataFrame()

    for file_name in sorted(os.listdir(labels_dir)):
        if not file_name.endswith(".csv"):
            continue
        frame = pd.read_csv(
            os.path.join(labels_dir, file_name),
            header=None,
            names=["fname", "label"],
            dtype=str,
        )
        frame = frame[frame["fname"] != "table_file_name"]
        frame["fname"] = frame["fname"].astype(str).str.strip().str.replace(".txt", "", regex=False).apply(os.path.basename)
        frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
        all_frames.append(frame)

    if not all_frames:
        return pd.DataFrame()

    merged = pd.concat(all_frames, axis=0)
    return merged.drop_duplicates(subset=["fname"]).set_index("fname")


# Purpose: Run SIMBAND preprocessing and store AF windows.
# Inputs: config path.
# Outputs: saved .npy feature and label files.
# Side effects: file I/O for processed dataset folders.
def main(config_path: str) -> None:
    cfg = load_config(config_path)
    raw_root = cfg.get("paths.raw.simband", "./data/raw/UMMCSIMBAND")
    processed_dir = cfg.get("paths.processed.simband", "./data/processed/UMMCSIMBAND")
    os.makedirs(processed_dir, exist_ok=True)

    labels_df = load_labels(os.path.join(raw_root, "Labels"))
    if labels_df.empty:
        print("No SIMBAND labels found.")
        return

    total = 0
    source_map = {}
    for subset in ["AF", "Clinical"]:
        subset_dir = os.path.join(raw_root, subset)
        if not os.path.exists(subset_dir):
            continue

        for tar_name in sorted(f for f in os.listdir(subset_dir) if f.endswith(".tar")):
            tar_path = os.path.join(subset_dir, tar_name)
            try:
                x_list, y_list = process_tar_file(tar_path, labels_df, cfg)
                if not x_list:
                    continue
                stem = os.path.splitext(tar_name)[0]
                x_arr = np.asarray(x_list, dtype=np.float32)
                y_arr = np.asarray(y_list, dtype=np.float32)
                np.save(os.path.join(processed_dir, f"{stem}_X.npy"), x_arr)
                np.save(os.path.join(processed_dir, f"{stem}_y.npy"), y_arr)
                source_map[stem] = subset
                total += x_arr.shape[0]
                print(f"Saved {stem}: X{x_arr.shape}, y{y_arr.shape}")
            except Exception as exc:
                print(f"Failed {tar_name}: {exc}")

    if source_map:
        meta_path = os.path.join(processed_dir, "simband_source_map.json")
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(source_map, handle, indent=2, sort_keys=True)
        print(f"Saved SIMBAND source map: {meta_path}")

    print(f"SIMBAND preprocessing done. Total windows: {total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/config.yaml")
    args = parser.parse_args()
    main(args.config)
