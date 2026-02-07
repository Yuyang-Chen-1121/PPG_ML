# Author: Yuyang Chen
# School: Zhejiang University
# Last Modified: 2026.2.7
# Description: Signal preprocessing and label generation routines.

"""Signal preprocessing utilities for TinyNet datasets."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import scipy.signal
from scipy.ndimage import gaussian_filter1d


# Purpose: Apply Butterworth band-pass filtering on 1D or columnar arrays.
# Inputs: data array, low/high cut frequencies, sampling rate, filter order.
# Outputs: filtered array with same shape as input.
# Assumptions: 0 < lowcut < highcut < fs/2.
def butter_bandpass_filter(
    data: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 2,
) -> np.ndarray:
    nyquist = 0.5 * fs
    b, a = scipy.signal.butter(order, [lowcut / nyquist, highcut / nyquist], btype="band")
    return scipy.signal.filtfilt(b, a, data, axis=0)


# Purpose: Remove impulsive outliers using a Hampel filter.
# Inputs: signal array (1D), odd/even window size, sigma threshold in std units.
# Outputs: filtered 1D array with outliers replaced by local median.
# Assumptions: window_size >= 1; threshold_std > 0.
def hampel_filter(signal: np.ndarray, window_size: int = 7, threshold_std: float = 3.0) -> np.ndarray:
    if signal.ndim != 1:
        raise ValueError("hampel_filter expects a 1D array")

    half = max(1, int(window_size) // 2)
    x = signal.astype(np.float32, copy=True)
    length = x.shape[0]

    for idx in range(length):
        start = max(0, idx - half)
        end = min(length, idx + half + 1)
        window = x[start:end]
        local_median = np.median(window)
        local_std = np.std(window)
        if local_std < 1e-8:
            continue
        if abs(x[idx] - local_median) > threshold_std * local_std:
            x[idx] = local_median

    return x


# Purpose: Standardize a signal to zero-mean and unit-variance.
# Inputs: data array, epsilon for numerical stability.
# Outputs: normalized array.
# Assumptions: normalization is done over all values in input array.
def z_score_normalization(data: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    mean = float(np.mean(data))
    std = float(np.std(data))
    if std < epsilon:
        return data - mean
    return (data - mean) / std


# Purpose: End-to-end PPG preprocessing with filter -> resample -> Hampel -> z-score.
# Inputs: raw_ppg vector, source fs, target fs, band-pass settings, Hampel settings, z-score flag.
# Outputs: processed PPG vector sampled at target_fs.
# Assumptions: raw_ppg is 1D and finite.
def preprocess_ppg_signal(
    raw_ppg: np.ndarray,
    source_fs: float,
    target_fs: float,
    lowcut: float,
    highcut: float,
    order: int,
    hampel_window_size: int,
    hampel_threshold_std: float,
    enable_zscore: bool,
    zscore_epsilon: float,
) -> np.ndarray:
    filtered = butter_bandpass_filter(raw_ppg, lowcut, highcut, source_fs, order=order)
    num_samples = int(len(filtered) * float(target_fs) / float(source_fs))
    resampled = scipy.signal.resample(filtered, num_samples)
    denoised = hampel_filter(resampled, window_size=hampel_window_size, threshold_std=hampel_threshold_std)
    if enable_zscore:
        return z_score_normalization(denoised, epsilon=zscore_epsilon)
    return denoised


# Purpose: Pack PPG and acceleration channels into 16-channel tensor expected by TinyNet.
# Inputs: 1D arrays for ppg, acc magnitude, acc xyz; target output length.
# Outputs: feature array with shape (target_len, 16).
# Assumptions: all provided channels already aligned to target_len.
def pack_multimodal_channels(
    ppg: np.ndarray,
    acc_mag: np.ndarray,
    acc_x: np.ndarray,
    acc_y: np.ndarray,
    acc_z: np.ndarray,
    target_len: int,
) -> np.ndarray:
    packed = np.zeros((target_len, 16), dtype=np.float32)
    packed[:, 0] = ppg
    packed[:, 1] = acc_mag
    packed[:, 2] = acc_x
    packed[:, 3] = acc_y
    packed[:, 4] = acc_z
    return packed


# Purpose: Build Gaussian-smoothed HR distribution on hardware-compliant BPM bins.
# Inputs: bpm value, bpm bin centers, Gaussian sigma.
# Outputs: normalized label distribution with shape (num_bins,).
# Assumptions: bpm_bins length is <= 128 and sigma > 0.
def generate_hr_distribution_label(bpm_value: float, bpm_bins: np.ndarray, sigma: float) -> np.ndarray:
    label = np.zeros_like(bpm_bins, dtype=np.float32)
    if np.isnan(bpm_value) or bpm_value <= 0:
        return label

    target_idx = int(np.argmin(np.abs(bpm_bins - bpm_value)))
    label[target_idx] = 1.0
    label = gaussian_filter1d(label, sigma=sigma)
    norm = float(np.sum(label))
    if norm > 0.0:
        label = label / norm
    return label.astype(np.float32)


# Purpose: Slice synchronized multimodal streams into windows with Gaussian HR labels.
# Inputs: normalized ppg, normalized acc [N,4], scalar labels, window and step sizes, label ratio.
# Outputs: tuple(X_windows, y_distributions).
# Assumptions: labels are scalar BPM values sampled slower than the signal stream.
def sliding_window_multimodal(
    ppg: np.ndarray,
    acc_data: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    step_size: int,
    label_fs_ratio: float,
    bpm_bins: np.ndarray,
    label_sigma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    n_samples = len(ppg)
    x_list = []
    y_list = []

    for start in range(0, n_samples - window_size + 1, step_size):
        end = start + window_size
        win_ppg = ppg[start:end]
        win_acc = acc_data[start:end, :]

        center_idx = start + window_size // 2
        label_idx = int(center_idx / max(label_fs_ratio, 1e-8))
        if label_idx >= len(labels):
            continue

        x_win = pack_multimodal_channels(
            ppg=win_ppg,
            acc_mag=win_acc[:, 0],
            acc_x=win_acc[:, 1],
            acc_y=win_acc[:, 2],
            acc_z=win_acc[:, 3],
            target_len=window_size,
        )
        y_win = generate_hr_distribution_label(float(labels[label_idx]), bpm_bins=bpm_bins, sigma=label_sigma)

        x_list.append(x_win)
        y_list.append(y_win)

    if not x_list:
        return np.zeros((0, window_size, 16), dtype=np.float32), np.zeros((0, bpm_bins.shape[0]), dtype=np.float32)

    return np.asarray(x_list, dtype=np.float32), np.asarray(y_list, dtype=np.float32)
