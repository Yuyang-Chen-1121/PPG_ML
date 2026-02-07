# Author: Yuyang Chen
# School: Zhejiang University
# Last Modified: 2026.2.7
# Description: Preprocessing entrypoint for DALIA dataset signals.

"""Preprocess DaLiA into TinyNet hardware-compliant windows."""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
from scipy import signal

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from data.preprocessing import preprocess_ppg_signal, sliding_window_multimodal, z_score_normalization
from utils.config import get_bpm_bins, load_config


# Purpose: Process one DaLiA subject file into model-ready windows and labels.
# Inputs: file name, loaded config accessor, BPM bins.
# Outputs: tuple(X_windows, y_distributions).
# Assumptions: input pickle contains wrist BVP/ACC and scalar HR labels.
def process_subject(file_name: str, cfg, bpm_bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw_dir = cfg.get("paths.raw.dalia", "./data/raw/dalia")
    target_fs = cfg.get_int("data.sampling_rate_hz", 32)
    window_len = cfg.get_int("data.window_seconds", 10) * target_fs
    step_len = cfg.get_int("data.window_step_seconds", 2) * target_fs

    low = cfg.get_float("preprocessing.bandpass_low_hz", 0.5)
    high = cfg.get_float("preprocessing.bandpass_high_hz", 4.0)
    order = cfg.get_int("preprocessing.bandpass_order", 2)
    source_fs = cfg.get_float("preprocessing.source_fs.dalia", 64.0)
    hampel_window = cfg.get_int("preprocessing.hampel_window_size", 7)
    hampel_th = cfg.get_float("preprocessing.hampel_threshold_std", 3.0)
    enable_zscore = cfg.get_bool("preprocessing.enable_zscore", True)
    z_eps = cfg.get_float("preprocessing.zscore_epsilon", 1e-6)
    label_sigma = cfg.get_float("labels.gaussian_sigma", 2.0)

    with open(os.path.join(raw_dir, file_name), "rb") as handle:
        data = pickle.load(handle, encoding="latin1")

    raw_ppg = data["signal"]["wrist"]["BVP"].flatten()
    raw_acc = data["signal"]["wrist"]["ACC"]
    hr_labels = data["label"]

    ppg_norm = preprocess_ppg_signal(
        raw_ppg=raw_ppg,
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

    num_samples = ppg_norm.shape[0]
    acc_mag = np.sqrt(np.sum(raw_acc ** 2, axis=1))

    acc_mag_norm = z_score_normalization(signal.resample(acc_mag, num_samples), epsilon=z_eps)
    acc_x_norm = z_score_normalization(signal.resample(raw_acc[:, 0], num_samples), epsilon=z_eps)
    acc_y_norm = z_score_normalization(signal.resample(raw_acc[:, 1], num_samples), epsilon=z_eps)
    acc_z_norm = z_score_normalization(signal.resample(raw_acc[:, 2], num_samples), epsilon=z_eps)
    acc_combined = np.stack([acc_mag_norm, acc_x_norm, acc_y_norm, acc_z_norm], axis=1)

    label_fs_ratio = len(ppg_norm) / max(len(hr_labels), 1)
    return sliding_window_multimodal(
        ppg=ppg_norm,
        acc_data=acc_combined,
        labels=np.asarray(hr_labels),
        window_size=window_len,
        step_size=step_len,
        label_fs_ratio=label_fs_ratio,
        bpm_bins=bpm_bins,
        label_sigma=label_sigma,
    )


# Purpose: Run DaLiA preprocessing for all subjects in raw folder.
# Inputs: config path.
# Outputs: saved NumPy windows under processed folder.
# Side effects: creates directories and writes .npy files.
def main(config_path: str) -> None:
    cfg = load_config(config_path)
    raw_dir = cfg.get("paths.raw.dalia", "./data/raw/dalia")
    processed_dir = cfg.get("paths.processed.dalia", "./data/processed/dalia")
    os.makedirs(processed_dir, exist_ok=True)

    bpm_bins = get_bpm_bins(cfg)
    for file_name in sorted(os.listdir(raw_dir)):
        if not file_name.endswith(".pkl"):
            continue
        try:
            x_arr, y_arr = process_subject(file_name, cfg, bpm_bins)
            stem = os.path.splitext(file_name)[0]
            np.save(os.path.join(processed_dir, f"{stem}_X.npy"), x_arr.astype(np.float32))
            np.save(os.path.join(processed_dir, f"{stem}_y.npy"), y_arr.astype(np.float32))
            print(f"Saved {file_name}: X{x_arr.shape}, y{y_arr.shape}")
        except Exception as exc:
            print(f"Failed {file_name}: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/config.yaml")
    args = parser.parse_args()
    main(args.config)
