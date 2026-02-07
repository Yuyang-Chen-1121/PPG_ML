# Author: Yuyang Chen
# School: Zhejiang University
# Last Modified: 2026.2.7
# Description: Evaluation helper functions for metrics, plots, and post-processing.

"""Evaluation plotting and temporal decoding helpers."""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import scipy.ndimage
from scipy.stats import gaussian_kde, pearsonr

from train.helper import decode_hr_smart
from utils.config import ConfigAccessor


class TemporalHeartRateDecoder:
    # Purpose: Smooth frame-level HR predictions using prior-state Bayesian weighting.
    # Inputs: search_window, init_frames, debounce_threshold.
    # Outputs: stateful decoder object.
    # Assumptions: decode() is called on temporally ordered frames.

    # Purpose: Initialize temporal decoder state.
    # Inputs: search window, number of startup frames, debounce threshold.
    # Outputs: initialized decoder.
    # Assumptions: bins passed to decode are fixed across sequence.
    def __init__(self, search_window: int = 20, init_frames: int = 8, debounce_threshold: int = 5) -> None:
        self.search_window = search_window
        self.sigma = search_window / 2.0
        self.init_frames = init_frames
        self.debounce_threshold = debounce_threshold
        self.reset()

    # Purpose: Reset decoder state for a new subject/recording.
    # Inputs: none.
    # Outputs: none.
    # Side effects: clears internal buffers.
    def reset(self) -> None:
        self.prev_bpm = None
        self.init_buffer = []
        self.divergence_count = 0
        self.divergence_bpm_sum = 0.0

    # Purpose: Decode one HR distribution with temporal smoothing.
    # Inputs: probability/logit vector, BPM bins, optional config.
    # Outputs: scalar BPM.
    # Assumptions: consecutive calls correspond to same recording unless reset.
    def decode(self, logits: np.ndarray, bins: np.ndarray, config=None) -> float:
        values = np.asarray(logits, dtype=np.float64)
        if np.max(values) > 1.0 or np.sum(values) > 1.0 + 1e-4:
            exps = np.exp(values - np.max(values))
            probs = exps / (np.sum(exps) + 1e-8)
        else:
            probs = values / (np.sum(values) + 1e-8)

        raw_bpm = decode_hr_smart(probs, bins, config=config)

        if self.prev_bpm is None:
            self.init_buffer.append(raw_bpm)
            if len(self.init_buffer) < self.init_frames:
                return raw_bpm
            self.prev_bpm = float(np.median(np.array(self.init_buffer)))
            return self.prev_bpm

        prior = np.exp(-0.5 * ((bins - self.prev_bpm) / self.sigma) ** 2)
        posterior = probs * prior
        posterior = posterior / (np.sum(posterior) + 1e-8)
        fused_bpm = float(np.sum(posterior * bins))

        diff = abs(raw_bpm - self.prev_bpm)
        if diff < self.search_window:
            self.prev_bpm = 0.7 * self.prev_bpm + 0.3 * fused_bpm
            self.divergence_count = 0
            self.divergence_bpm_sum = 0.0
            return self.prev_bpm

        self.divergence_count += 1
        self.divergence_bpm_sum += raw_bpm
        if self.divergence_count > self.debounce_threshold:
            self.prev_bpm = self.divergence_bpm_sum / self.divergence_count
            self.divergence_count = 0
            self.divergence_bpm_sum = 0.0
        return self.prev_bpm


# Purpose: Validate signal quality based on z-score amplitude, flatline ratio, and NaN/Inf checks.
# Inputs: waveform (1D numpy array), config accessor or dict.
# Outputs: (is_valid, reason) tuple.
# Assumptions: waveform is preprocessed and z-score normalized.
def validate_signal_quality(waveform: np.ndarray, config=None) -> tuple[bool, str]:
    cfg = config if isinstance(config, ConfigAccessor) else ConfigAccessor(config or {})
    if isinstance(config, dict) and "signal_quality" not in config:
        cfg = ConfigAccessor({"signal_quality": config})

    check_nan = cfg.get_bool("signal_quality.check_nan", True)
    if check_nan and np.any(~np.isfinite(waveform)):
        return False, "NaN/Inf"

    max_z = cfg.get_float("signal_quality.max_z_score", 20.0)
    max_val = float(np.max(np.abs(waveform)))
    if max_val > max_z:
        return False, "Extreme Artifact"

    flatline_ratio = cfg.get_float("signal_quality.flatline_ratio", 0.3)
    diff = np.diff(waveform)
    if diff.size == 0:
        return False, "Flatline"
    flat_count = int(np.sum(np.abs(diff) < 1e-6))
    if (flat_count / diff.size) > flatline_ratio:
        return False, "Flatline"

    return True, "OK"


# Purpose: Apply median smoothing, thresholding, and short-episode pruning to AF probabilities.
# Inputs: 1D probability array, config accessor or dict.
# Outputs: binary AF prediction array.
# Assumptions: median_window is odd; min_duration_windows >= 1.
def apply_post_processing_pipeline(probs: np.ndarray, config=None) -> np.ndarray:
    cfg = config if isinstance(config, ConfigAccessor) else ConfigAccessor(config or {})
    arr = np.asarray(probs, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"apply_post_processing_pipeline expects 1D array, got shape {arr.shape}")

    window = int(cfg.get_int("postprocessing.median_window", 7))
    if window < 1:
        raise ValueError(f"median_window must be >= 1, got {window}")
    if window % 2 == 0:
        raise ValueError(f"median_window must be odd, got {window}")
    smoothed = scipy.signal.medfilt(arr, kernel_size=window)

    threshold = float(cfg.get_float("postprocessing.fixed_threshold", 0.5))
    binary = (smoothed >= threshold).astype(np.int32)

    min_len = int(cfg.get_int("postprocessing.min_duration_windows", 1))
    if min_len <= 1:
        return binary

    labeled, num = scipy.ndimage.label(binary)
    if num == 0:
        return binary

    sizes = np.bincount(labeled.ravel())
    for label_id in range(1, num + 1):
        if sizes[label_id] < min_len:
            binary[labeled == label_id] = 0
    return binary


# Purpose: Save density scatter plot for true vs predicted BPM.
# Inputs: true BPM array, predicted BPM array, output path.
# Outputs: none.
# Side effects: writes image file.
def plot_regression_scatter(y_true: np.ndarray, y_pred: np.ndarray, save_path: str) -> None:
    plt.figure(figsize=(8, 8), dpi=120)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    x, y = y_true[mask], y_pred[mask]
    if len(x) < 2:
        plt.close()
        return

    r, _ = pearsonr(x, y)
    try:
        idx_sample = np.random.choice(len(x), min(len(x), 5000), replace=False)
        xy_sample = np.vstack([x[idx_sample], y[idx_sample]])
        kde = gaussian_kde(xy_sample)
        z = kde(np.vstack([x, y]))
        order = z.argsort()
        x, y, z = x[order], y[order], z[order]
        plt.scatter(x, y, c=z, s=5, cmap="jet", alpha=0.6, edgecolor="none")
    except Exception:
        plt.scatter(x, y, c="blue", s=5, alpha=0.3)

    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=2)
    plt.title(f"Density Scatter (r={r:.3f})")
    plt.xlabel("True BPM")
    plt.ylabel("Predicted BPM")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# Purpose: Save Bland-Altman consistency plot.
# Inputs: true BPM array, predicted BPM array, output path.
# Outputs: none.
# Side effects: writes image file.
def plot_bland_altman(y_true: np.ndarray, y_pred: np.ndarray, save_path: str) -> None:
    plt.figure(figsize=(10, 6), dpi=120)
    mean = (y_true + y_pred) / 2
    diff = y_true - y_pred
    md = float(np.mean(diff))
    sd = float(np.std(diff))
    plt.scatter(mean, diff, c="purple", alpha=0.2, s=5, edgecolors="none")
    plt.axhline(md, color="red", linestyle="-", lw=2)
    plt.axhline(md + 1.96 * sd, color="blue", linestyle="--")
    plt.axhline(md - 1.96 * sd, color="blue", linestyle="--")
    plt.xlabel("Mean BPM")
    plt.ylabel("Difference BPM")
    plt.title(f"Bland-Altman (Mean Error: {md:.2f})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# Purpose: Save correlation scatter with identity reference line.
# Inputs: true BPM array, predicted BPM array, output path.
# Outputs: none.
# Side effects: writes image file.
def plot_correlation(y_true: np.ndarray, y_pred: np.ndarray, save_path: str) -> None:
    plt.figure(figsize=(8, 8), dpi=120)
    plt.scatter(y_true, y_pred, alpha=0.3, s=3, c="blue")
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
    r, _ = pearsonr(y_true, y_pred)
    plt.title(f"Correlation (r={r:.3f})")
    plt.xlabel("True BPM")
    plt.ylabel("Predicted BPM")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# Purpose: Save AF probability distribution histograms by class.
# Inputs: binary AF truths, AF probabilities, threshold, output path.
# Outputs: none.
# Side effects: writes image file.
def plot_af_prob_distribution(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, save_path: str) -> None:
    plt.figure(figsize=(10, 6), dpi=120)
    if np.any(y_true == 0):
        plt.hist(y_prob[y_true == 0], bins=50, alpha=0.5, color="green", label="NSR", density=True, range=(0, 1))
    if np.any(y_true == 1):
        plt.hist(y_prob[y_true == 1], bins=50, alpha=0.5, color="red", label="AF", density=True, range=(0, 1))
    plt.axvline(threshold, color="black", linestyle="--", label=f"Th={threshold:.2f}")
    plt.title("AF Probability Distribution")
    plt.xlabel("Probability")
    plt.ylabel("Density")
    plt.xlim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# Purpose: Save error distribution histogram (legacy poincare alias).
# Inputs: true BPM array, predicted BPM array, output path.
# Outputs: none.
# Side effects: writes image file.
def plot_poincare(y_true: np.ndarray, y_pred: np.ndarray, save_path: str) -> None:
    error = y_true - y_pred
    plt.figure(figsize=(10, 6), dpi=120)
    plt.hist(error, bins=50, color="orange", alpha=0.7, edgecolor="black", density=True)
    plt.xlabel("Error (BPM)")
    plt.ylabel("Density")
    plt.title(f"Error Distribution (MAE={np.mean(np.abs(error)):.2f})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# Purpose: Save confusion matrix plot.
# Inputs: y_true (0/1), y_pred (0/1), class labels, output path, title.
# Outputs: none.
# Side effects: writes image file.
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    save_path: str,
    title: str = "Confusion Matrix",
) -> None:
    if y_true.size == 0 or y_pred.size == 0:
        return
    cm = np.zeros((2, 2), dtype=np.int64)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        if 0 <= t <= 1 and 0 <= p <= 1:
            cm[t, p] += 1

    plt.figure(figsize=(6, 5), dpi=120)
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
