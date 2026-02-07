# Author: Yuyang Chen
# School: Zhejiang University
# Last Modified: 2026.2.7
# Description: Dataset, augmentation, and sampling utilities for TinyNet training.

"""Dataset and sampler utilities for TinyNet training/evaluation."""

from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from utils.config import ConfigAccessor, get_bpm_bins


# Purpose: Map a file path to dataset source id used in metrics.
# Inputs: feature file path.
# Outputs: int source id (0=simband, 1=dalia, 2=bami, 3=unknown).
# Assumptions: source keyword appears in path.
def source_id_from_path(path: str) -> int:
    lower = path.lower()
    if "ummcsimband" in lower:
        return 0
    if "dalia" in lower:
        return 1
    if "bami" in lower:
        return 2
    return 3


# Purpose: Ensure per-window feature tensor is channel-first and fixed temporal length.
# Inputs: raw feature window, target length.
# Outputs: float32 array with shape (C, target_len).
# Assumptions: input is either (L, C) or (C, L).
def ensure_channel_first_length(
    x_window: np.ndarray,
    target_len: int,
    random_crop: bool = False,
    shift_offset: Optional[int] = None,
    shift_pad: int = 0,
    pad_mode: str = "constant",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    x = x_window.astype(np.float32, copy=False)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D window, got shape {x.shape}")

    if x.shape[0] > x.shape[1]:
        x = x.transpose(1, 0)

    current_len = x.shape[1]
    if current_len < target_len:
        pad = target_len - current_len
        x = np.pad(x, ((0, 0), (0, pad)), mode=pad_mode)
        current_len = target_len

    if shift_pad > 0:
        x = np.pad(x, ((0, 0), (shift_pad, shift_pad)), mode=pad_mode)
        current_len = x.shape[1]

    if current_len > target_len:
        if shift_offset is not None:
            base_start = (current_len - target_len) // 2
            start = base_start + int(shift_offset)
        elif random_crop:
            rng = rng or np.random.default_rng()
            start = int(rng.integers(0, current_len - target_len + 1))
        else:
            start = (current_len - target_len) // 2
        start = max(0, min(start, current_len - target_len))
        x = x[:, start : start + target_len]

    return x.astype(np.float32, copy=False)


class SignalAugmenter:
    """Lightweight signal augmentation utilities for numpy/torch inputs."""

    # Purpose: Initialize class state and runtime configuration.
    # Inputs: Parameters defined in `__init__` signature.
    # Outputs: Return value produced by `__init__`.
    # Assumptions: Caller provides valid types/shapes for this operation.
    def __init__(self, cfg: ConfigAccessor) -> None:
        self.enable = cfg.get_bool("aug.enable", False)
        self.gaussian_prob = self._clamp01(cfg.get_float("aug.gaussian_prob", 0.5))
        self.gaussian_std = float(cfg.get_float("aug.gaussian_std", 0.01))
        self.time_mask_prob = self._clamp01(cfg.get_float("aug.time_mask_prob", 0.5))
        self.time_mask_ratio = self._clamp01(cfg.get_float("aug.time_mask_ratio", 0.1))
        self.freq_mask_prob = self._clamp01(cfg.get_float("aug.freq_mask_prob", 0.5))
        self.freq_mask_ratio = self._clamp01(cfg.get_float("aug.freq_mask_ratio", 0.1))
        self.shift_prob = self._clamp01(cfg.get_float("aug.shift_prob", 0.5))
        self.shift_limit = int(cfg.get_int("aug.shift_limit", 20))
        self.rng = np.random.default_rng()

    @staticmethod
    # Purpose: Implement `_clamp01` for the TinyNet workflow.
    # Inputs: Parameters defined in `_clamp01` signature.
    # Outputs: Return value produced by `_clamp01`.
    # Assumptions: Caller provides valid types/shapes for this operation.
    def _clamp01(value: float) -> float:
        return float(max(0.0, min(1.0, value)))

    @staticmethod
    def _is_torch(x) -> bool:
        return torch.is_tensor(x)

    # Purpose: Implement `_should_apply` for the TinyNet workflow.
    # Inputs: Parameters defined in `_should_apply` signature.
    # Outputs: Return value produced by `_should_apply`.
    # Assumptions: Caller provides valid types/shapes for this operation.
    def _should_apply(self, prob: float) -> bool:
        return self.enable and prob > 0.0 and random.random() < prob

    def sample_shift_offset(self) -> int:
        if self.shift_limit <= 0:
            return 0
        return random.randint(-self.shift_limit, self.shift_limit)

    # Purpose: Implement `should_shift` for the TinyNet workflow.
    # Inputs: Parameters defined in `should_shift` signature.
    # Outputs: Return value produced by `should_shift`.
    # Assumptions: Caller provides valid types/shapes for this operation.
    def should_shift(self) -> bool:
        return self._should_apply(self.shift_prob)

    def apply_gaussian(self, x):
        if not self._should_apply(self.gaussian_prob) or self.gaussian_std <= 0:
            return x
        if self._is_torch(x):
            noise = torch.randn_like(x) * self.gaussian_std
            return x + noise
        x += self.rng.normal(0.0, self.gaussian_std, size=x.shape).astype(x.dtype, copy=False)
        return x

    # Purpose: Implement `apply_time_mask` for the TinyNet workflow.
    # Inputs: Parameters defined in `apply_time_mask` signature.
    # Outputs: Return value produced by `apply_time_mask`.
    # Assumptions: Caller provides valid types/shapes for this operation.
    def apply_time_mask(self, x):
        if not self._should_apply(self.time_mask_prob) or self.time_mask_ratio <= 0:
            return x
        length = int(x.shape[-1])
        mask_len = max(1, int(round(self.time_mask_ratio * length)))
        mask_len = min(mask_len, length)
        if self._is_torch(x):
            start = int(torch.randint(0, length - mask_len + 1, (1,), device=x.device).item())
            x[..., start : start + mask_len] = 0.0
            return x
        start = int(self.rng.integers(0, length - mask_len + 1))
        x[..., start : start + mask_len] = 0.0
        return x

    # Purpose: Implement `apply_freq_mask` for the TinyNet workflow.
    # Inputs: Parameters defined in `apply_freq_mask` signature.
    # Outputs: Return value produced by `apply_freq_mask`.
    # Assumptions: Caller provides valid types/shapes for this operation.
    def apply_freq_mask(self, x):
        if not self._should_apply(self.freq_mask_prob) or self.freq_mask_ratio <= 0:
            return x
        length = int(x.shape[-1])
        if self._is_torch(x):
            spectrum = torch.fft.rfft(x, dim=-1)
            n_freq = int(spectrum.shape[-1])
            mask_len = max(1, int(round(self.freq_mask_ratio * n_freq)))
            mask_len = min(mask_len, n_freq)
            start = int(torch.randint(0, n_freq - mask_len + 1, (1,), device=x.device).item())
            spectrum[..., start : start + mask_len] = 0
            return torch.fft.irfft(spectrum, n=length, dim=-1).to(x.dtype)

        spectrum = np.fft.rfft(x, axis=-1)
        n_freq = int(spectrum.shape[-1])
        mask_len = max(1, int(round(self.freq_mask_ratio * n_freq)))
        mask_len = min(mask_len, n_freq)
        start = int(self.rng.integers(0, n_freq - mask_len + 1))
        spectrum[..., start : start + mask_len] = 0
        return np.fft.irfft(spectrum, n=length, axis=-1).astype(np.float32, copy=False)

    # Purpose: Implement `apply_general` for the TinyNet workflow.
    # Inputs: Parameters defined in `apply_general` signature.
    # Outputs: Return value produced by `apply_general`.
    # Assumptions: Caller provides valid types/shapes for this operation.
    def apply_general(self, x):
        x = self.apply_gaussian(x)
        x = self.apply_time_mask(x)
        x = self.apply_freq_mask(x)
        return x


class ChunkMultiModalDataset(Dataset):
    # Purpose: Load preprocessed windows from npy files for AF/HR multi-task training.
    # Inputs: list of *_X.npy file paths, mode string, config dictionary/accessor.
    # Outputs: samples (x, y_af, y_hr, mask_af, mask_hr, source_id).
    # Assumptions: paired label files exist as *_y.npy.

    # Purpose: Build index map and per-window metadata for fast access and sampling.
    # Inputs: file paths, mode, config.
    # Outputs: initialized dataset instance.
    # Side effects: scans file headers and label arrays.
    def __init__(self, file_paths: List[str], mode: str = "train", config=None) -> None:
        self.files = list(file_paths)
        self.mode = mode
        self.cfg = config if isinstance(config, ConfigAccessor) else ConfigAccessor(config or {})
        self.target_len = self.cfg.get_int("data.input_length", 320)
        self.simband_ppg_only = self.cfg.get_bool("data.simband_ppg_only", True)
        self.augmenter = SignalAugmenter(self.cfg)
        self.bpm_bins = get_bpm_bins(self.cfg)
        self.hr_bins = int(self.bpm_bins.shape[0])
        self.af_classes = self.cfg.get_int("labels.af_num_classes", 2)

        self.indices: List[Tuple[int, int]] = []
        self.meta: List[Dict[str, float]] = []

        for file_idx, x_path in enumerate(self.files):
            y_path = x_path.replace("_X.npy", "_y.npy")
            if not os.path.exists(y_path):
                continue

            try:
                x_shape = np.load(x_path, mmap_mode="r").shape
                y_data = np.load(y_path, mmap_mode="r")
            except Exception:
                continue

            n_windows = int(x_shape[0]) if len(x_shape) > 0 else 0
            if n_windows <= 0:
                continue

            source_id = source_id_from_path(x_path)
            is_hr_file = (y_data.ndim == 2 and y_data.shape[1] == self.hr_bins)
            is_af_file = (y_data.ndim == 2 and y_data.shape[1] == self.af_classes)

            for win_idx in range(n_windows):
                self.indices.append((file_idx, win_idx))
                meta = {
                    "is_hr": float(is_hr_file),
                    "is_af": float(is_af_file),
                    "is_af_pos": 0.0,
                    "hr_bpm": 0.0,
                    "source_id": float(source_id),
                }

                if is_af_file:
                    af_label = y_data[win_idx]
                    if af_label.shape[0] >= 2 and af_label[1] > af_label[0]:
                        meta["is_af_pos"] = 1.0
                elif is_hr_file:
                    hr_dist = y_data[win_idx]
                    meta["hr_bpm"] = float(np.sum(hr_dist * self.bpm_bins))

                self.meta.append(meta)

        print(f"Dataset[{mode}] windows: {len(self.indices)}")

    # Purpose: Report dataset sample count.
    # Inputs: none.
    # Outputs: integer number of windows.
    # Assumptions: index map is initialized.
    def __len__(self) -> int:
        return len(self.indices)

    # Purpose: Return metadata dictionary for one indexed sample.
    # Inputs: sample index.
    # Outputs: dict with task/source flags and scalar bpm.
    # Assumptions: index is in-range.
    def get_sample_meta(self, idx: int) -> Dict[str, float]:
        return self.meta[idx]

    # Purpose: Load one training/eval sample from disk.
    # Inputs: sample index.
    # Outputs: tuple(tensors) -> x, y_af, y_hr, m_af, m_hr, source_id.
    # Assumptions: paired X/y arrays are consistent by first dimension.
    def __getitem__(self, idx: int):
        file_idx, window_idx = self.indices[idx]
        x_path = self.files[file_idx]
        y_path = x_path.replace("_X.npy", "_y.npy")

        x_data = np.load(x_path, mmap_mode="r")
        y_data = np.load(y_path, mmap_mode="r")

        meta = self.meta[idx]
        x_raw = x_data[window_idx]

        is_train_aug = self.mode == "train" and self.augmenter.enable
        apply_shift = (
            is_train_aug
            and int(meta["source_id"]) == 0
            and meta["is_af_pos"] > 0.5
            and self.augmenter.should_shift()
        )

        if apply_shift:
            if x_raw.ndim != 2:
                raise ValueError(f"Expected 2D window, got shape {x_raw.shape}")
            raw_len = int(x_raw.shape[0] if x_raw.shape[0] > x_raw.shape[1] else x_raw.shape[1])
            if raw_len > self.target_len:
                extra = raw_len - self.target_len
                base_start = extra // 2
                max_left = min(self.augmenter.shift_limit, base_start)
                max_right = min(self.augmenter.shift_limit, extra - base_start)
                shift_offset = random.randint(-max_left, max_right) if self.augmenter.shift_limit > 0 else 0
                shift_pad = 0
            else:
                shift_pad = self.augmenter.shift_limit
                shift_offset = self.augmenter.sample_shift_offset()
            x_win = ensure_channel_first_length(
                x_raw,
                self.target_len,
                shift_offset=shift_offset,
                shift_pad=shift_pad,
                pad_mode="constant",
            )
        else:
            x_win = ensure_channel_first_length(x_raw, self.target_len)

        x_win = np.array(x_win, copy=True)
        if is_train_aug:
            x_win = self.augmenter.apply_general(x_win)

        if self.simband_ppg_only and int(meta["source_id"]) == 0 and x_win.shape[0] > 1:
            x_win[1:, :] = 0.0

        y_af = np.zeros(self.af_classes, dtype=np.float32)
        y_hr = np.zeros(self.hr_bins, dtype=np.float32)
        m_af = np.zeros(1, dtype=np.float32)
        m_hr = np.zeros(1, dtype=np.float32)

        if y_data.ndim == 2 and y_data.shape[1] == self.hr_bins:
            y_hr = y_data[window_idx].astype(np.float32)
            m_hr[0] = 1.0
        elif y_data.ndim == 2 and y_data.shape[1] == self.af_classes:
            y_af = y_data[window_idx].astype(np.float32)
            m_af[0] = 1.0
        elif y_data.ndim == 1:
            value = int(y_data[window_idx] > 0.5)
            y_af = np.array([1.0 - value, float(value)], dtype=np.float32)
            m_af[0] = 1.0

        source_id = np.array(meta["source_id"], dtype=np.float32)
        return (
            torch.from_numpy(x_win),
            torch.from_numpy(y_af),
            torch.from_numpy(y_hr),
            torch.from_numpy(m_af),
            torch.from_numpy(m_hr),
            torch.tensor(source_id, dtype=torch.float32),
        )


# Purpose: Build stage-specific weighted sampler for imbalance control.
# Inputs: dataset, optional weights dict, training stage index, verbose flag.
# Outputs: WeightedRandomSampler.
# Assumptions: dataset exposes per-sample metadata via get_sample_meta.
def create_weighted_sampler(
    dataset: ChunkMultiModalDataset,
    weights: Dict[str, float] | None = None,
    stage_idx: int = 1,
    verbose: bool = True,
) -> WeightedRandomSampler:
    weights = dict(weights or {})

    task_af_w = float(weights.get("task_af", 1.0))
    task_hr_w = float(weights.get("task_hr", 1.0))
    af_pos_w = float(weights.get("af_positive", 1.0))
    af_neg_w = float(weights.get("af_negative", 1.0))
    hr_low_w = float(weights.get("hr_low", 1.0))
    hr_mid_w = float(weights.get("hr_mid", 1.0))
    hr_high_w = float(weights.get("hr_high", 1.0))
    hr_low_th = float(weights.get("hr_low_threshold", 75.0))
    hr_high_th = float(weights.get("hr_high_threshold", 120.0))

    sample_weights = []
    kept = 0
    for idx in range(len(dataset)):
        meta = dataset.get_sample_meta(idx)

        is_af = meta["is_af"] > 0.5
        is_hr = meta["is_hr"] > 0.5

        if stage_idx == 2 and not is_af:
            sample_weights.append(0.0)
            continue
        if stage_idx == 3 and not is_hr:
            sample_weights.append(0.0)
            continue

        weight = 1.0
        if is_af:
            weight *= task_af_w
            if meta["is_af_pos"] > 0.5:
                weight *= af_pos_w
            else:
                weight *= af_neg_w
        if is_hr:
            weight *= task_hr_w
            bpm = meta["hr_bpm"]
            if bpm < hr_low_th:
                weight *= hr_low_w
            elif bpm > hr_high_th:
                weight *= hr_high_w
            else:
                weight *= hr_mid_w

        kept += 1
        sample_weights.append(weight)

    tensor_weights = torch.tensor(sample_weights, dtype=torch.double)
    if torch.sum(tensor_weights) <= 0:
        tensor_weights = torch.ones_like(tensor_weights)
    if verbose:
        print(f"Sampler stage {stage_idx}: kept={kept}/{len(dataset)}")

    return WeightedRandomSampler(tensor_weights, len(tensor_weights), replacement=True)


# Purpose: Collate dataset samples into batch tensors.
# Inputs: list of sample tuples.
# Outputs: batched tensors with stacked dimensions.
# Assumptions: all samples share consistent shapes.
def batch_collate_fn(batch):
    if not batch:
        return None
    bx, by_af, by_hr, bm_af, bm_hr, b_source = zip(*batch)
    return (
        torch.stack(bx),
        torch.stack(by_af),
        torch.stack(by_hr),
        torch.stack(bm_af).squeeze(-1),
        torch.stack(bm_hr).squeeze(-1),
        torch.stack(b_source),
    )


HybridStrategyDataset = ChunkMultiModalDataset
