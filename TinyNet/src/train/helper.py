# Author: Yuyang Chen
# School: Zhejiang University
# Last Modified: 2026.2.7
# Description: Training helper utilities for optimization and checkpointing.

"""Training helper utilities for TinyNet."""

from __future__ import annotations

import glob
import os
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import median_filter as nd_median_filter
from sklearn.metrics import confusion_matrix

from models.tinynet import ResBlock


# Purpose: Search AF threshold maximizing G-Mean.
# Inputs: y_true binary array, y_pred_prob array.
# Outputs: best threshold scalar.
# Assumptions: y_true contains both classes for meaningful optimization.
def find_best_threshold(y_true: np.ndarray, y_pred_prob: np.ndarray) -> float:
    best_th = 0.5
    best_gmean = -1.0
    for th in np.linspace(0.1, 0.9, 33):
        y_bin = (y_pred_prob >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_bin, labels=[0, 1]).ravel()
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
        gmean = float(np.sqrt(sens * spec))
        if gmean > best_gmean:
            best_gmean = gmean
            best_th = float(th)
    return best_th


# Purpose: Apply centered median filter to smooth AF probabilities.
# Inputs: 1D probability array, odd window size.
# Outputs: filtered probability array with same length.
# Assumptions: window is odd and >= 1.
def apply_median_filter(probs: np.ndarray, window: int = 7) -> np.ndarray:
    if probs is None:
        return probs
    arr = np.asarray(probs, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"apply_median_filter expects 1D array, got shape {arr.shape}")
    window = int(window)
    if window < 1:
        raise ValueError(f"median_window must be >= 1, got {window}")
    if window % 2 == 0:
        raise ValueError(f"median_window must be odd, got {window}")
    if window == 1:
        return arr.copy()
    return nd_median_filter(arr, size=window, mode="reflect")


# Purpose: Decode HR distribution into scalar BPM via weighted expectation.
# Inputs: probability vector, BPM bin vector, optional config.
# Outputs: scalar BPM prediction.
# Assumptions: prob_dist and bins have same length.
def decode_hr_smart(prob_dist: np.ndarray, bins: np.ndarray, config=None) -> float:
    probs = np.asarray(prob_dist, dtype=np.float64)
    probs = probs / (np.sum(probs) + 1e-8)
    return float(np.sum(probs * bins))


# Purpose: Recursively collect *_X.npy feature files under a folder.
# Inputs: directory path.
# Outputs: sorted file path list.
# Assumptions: feature files follow *_X.npy naming.
def get_all_files(dir_path: str) -> List[str]:
    if not os.path.exists(dir_path):
        return []
    return sorted(glob.glob(os.path.join(dir_path, "**", "*_X.npy"), recursive=True))


# Purpose: Split file list into train and test subsets.
# Inputs: file list, test ratio, random seed.
# Outputs: tuple(train_files, test_files).
# Assumptions: list length can be zero.
def strict_train_test_split(file_list: List[str], test_ratio: float = 0.15, seed: int = 42) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    shuffled = list(file_list)
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * test_ratio)
    if cut <= 0:
        return shuffled, []
    return shuffled[:-cut], shuffled[-cut:]


# Purpose: Split file list into train and validation subsets.
# Inputs: file list, validation ratio, random seed.
# Outputs: tuple(train_files, val_files).
# Assumptions: list length can be zero.
def internal_train_val_split(file_list: List[str], val_ratio: float = 0.2, seed: int = 100) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    shuffled = list(file_list)
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * val_ratio)
    if cut <= 0:
        return shuffled, []
    return shuffled[cut:], shuffled[:cut]


# Purpose: Set deterministic seeds for Python, NumPy, and PyTorch.
# Inputs: integer seed.
# Outputs: none.
# Side effects: sets global RNG states.
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Purpose: Initialize model weights for stable optimization and AF gate behavior.
# Inputs: model instance.
# Outputs: none.
# Side effects: mutates parameters in-place.
def initialize_weights(model) -> None:
    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    if hasattr(model, "af_temporal_gate_conv"):
        nn.init.xavier_uniform_(model.af_temporal_gate_conv.weight)
        if model.af_temporal_gate_conv.bias is not None:
            nn.init.zeros_(model.af_temporal_gate_conv.bias)

    for module in model.modules():
        if isinstance(module, ResBlock) and hasattr(module, "bn2") and module.bn2.weight is not None:
            nn.init.zeros_(module.bn2.weight)


# Purpose: Serialize checkpoint with model state and evaluation metadata.
# Inputs: model, AF threshold, epoch number, score, destination path.
# Outputs: none.
# Side effects: writes checkpoint file to disk.
def save_checkpoint(model, threshold: float, epoch: int, score: float, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "threshold": float(threshold),
        "epoch": int(epoch),
        "score": float(score),
    }
    torch.save(payload, path)


class EarlyStopping:
    # Purpose: Stop training when monitored metric stalls for configured patience.
    # Inputs: patience, mode, delta, optional initial score.
    # Outputs: stateful stopper instance.
    # Assumptions: mode is "max" or "min".

    # Purpose: Initialize early stopping state.
    # Inputs: patience, mode, delta, initial score.
    # Outputs: initialized object.
    # Assumptions: delta >= 0.
    def __init__(self, patience: int = 10, mode: str = "max", delta: float = 1e-3, initial_score=None) -> None:
        self.patience = int(patience)
        self.mode = mode
        self.delta = float(delta)
        self.best_score = initial_score
        self.counter = 0
        self.early_stop = False
        if self.best_score is None:
            self.best_score = -np.inf if mode == "max" else np.inf

    # Purpose: Update stopper with new score and report if checkpoint should be saved.
    # Inputs: current metric score.
    # Outputs: bool indicating improvement.
    # Side effects: updates internal counters and early_stop flag.
    def __call__(self, score: float) -> bool:
        improved = (score > self.best_score + self.delta) if self.mode == "max" else (score < self.best_score - self.delta)
        if improved:
            self.best_score = score
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
        return False


# Purpose: Enable gradients only for modules relevant to current sequential stage.
# Inputs: model instance, stage index.
# Outputs: none.
# Side effects: mutates requires_grad flags.
def set_trainable_and_mode(model, stage: int) -> None:
    if stage == 1:
        for param in model.parameters():
            param.requires_grad = True
        return

    for name, param in model.named_parameters():
        if stage == 2:
            param.requires_grad = "af_" in name
        elif stage == 3:
            param.requires_grad = "hr_" in name
        else:
            param.requires_grad = True


# Purpose: Freeze non-target branch BatchNorm statistics during stage-specific training.
# Inputs: model instance, stage index.
# Outputs: none.
# Side effects: switches selected modules between train/eval modes.
def enforce_bn_mode(model, stage: int) -> None:
    model.train()
    if stage == 2:
        if hasattr(model, "stem"):
            model.stem.eval()
        if hasattr(model, "hr_block1"):
            model.hr_block1.eval()
        if hasattr(model, "hr_block2"):
            model.hr_block2.eval()
        if hasattr(model, "hr_block3"):
            model.hr_block3.eval()
    elif stage == 3:
        if hasattr(model, "stem"):
            model.stem.eval()
        if hasattr(model, "af_spatial_block1"):
            model.af_spatial_block1.eval()
        if hasattr(model, "af_spatial_block2"):
            model.af_spatial_block2.eval()
        if hasattr(model, "af_spatial_block3"):
            model.af_spatial_block3.eval()
        if hasattr(model, "af_temporal_block1"):
            model.af_temporal_block1.eval()
        if hasattr(model, "af_temporal_block2"):
            model.af_temporal_block2.eval()
        if hasattr(model, "af_block1"):
            model.af_block1.eval()
        if hasattr(model, "af_block2"):
            model.af_block2.eval()
        if hasattr(model, "af_block3"):
            model.af_block3.eval()
        if hasattr(model, "af_block4"):
            model.af_block4.eval()
