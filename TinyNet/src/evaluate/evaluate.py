# Author: Yuyang Chen
# School: Zhejiang University
# Last Modified: 2026.2.7
# Description: Evaluation pipeline and report generation for TinyNet.

"""TinyNet evaluation pipeline."""

from __future__ import annotations

import glob
import json
import os
from collections import Counter
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from evaluate.helper import (
    TemporalHeartRateDecoder,
    apply_post_processing_pipeline,
    plot_af_prob_distribution,
    plot_bland_altman,
    plot_correlation,
    plot_confusion_matrix,
    plot_poincare,
    plot_regression_scatter,
    validate_signal_quality,
)
from models.tinynet import build_model_from_config
from train.helper import decode_hr_smart
from utils.config import get_bpm_bins, load_config, get_device


# Purpose: Render a simple fixed-width table.
# Inputs: list of rows (list of strings).
# Outputs: formatted string for printing.
def format_table(rows: List[List[str]]) -> str:
    col_widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = []
    for idx, row in enumerate(rows):
        line = " | ".join(value.ljust(col_widths[i]) for i, value in enumerate(row))
        lines.append(line)
        if idx == 0:
            lines.append("-+-".join("-" * w for w in col_widths))
    return "\n".join(lines)


# Purpose: Implement `_safe_metric` for the TinyNet workflow.
# Inputs: Parameters defined in `_safe_metric` signature.
# Outputs: Return value produced by `_safe_metric`.
# Assumptions: Caller provides valid types/shapes for this operation.
def _safe_metric(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.2f}"


# Purpose: Implement `_compute_af_metrics_binary` for the TinyNet workflow.
# Inputs: Parameters defined in `_compute_af_metrics_binary` signature.
# Outputs: Return value produced by `_compute_af_metrics_binary`.
# Assumptions: Caller provides valid types/shapes for this operation.
def _compute_af_metrics_binary(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_bin = y_pred.astype(int)
    tn = int(np.sum((y_true == 0) & (y_bin == 0)))
    fp = int(np.sum((y_true == 0) & (y_bin == 1)))
    fn = int(np.sum((y_true == 1) & (y_bin == 0)))
    tp = int(np.sum((y_true == 1) & (y_bin == 1)))
    total = tn + fp + fn + tp
    if total == 0:
        return {}

    recall = (tp / (tp + fn + 1e-8)) * 100.0
    specificity = (tn / (tn + fp + 1e-8)) * 100.0
    accuracy = ((tp + tn) / (total + 1e-8)) * 100.0
    f1 = (2 * tp / (2 * tp + fp + fn + 1e-8)) * 100.0
    return {
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "f1": f1,
        "total": total,
    }


# Purpose: Resolve split entry id/path into concrete feature file path.
# Inputs: split entry string, processed root.
# Outputs: full path to *_X.npy.
# Assumptions: split IDs use {source}_{stem} format.
def resolve_split_entry(entry: str, processed_root: str) -> str:
    if entry.endswith("_X.npy"):
        return entry if os.path.isabs(entry) else os.path.join(processed_root, entry)
    source, stem = entry.split("_", 1)
    if source == "UMMCSIMBAND":
        return os.path.join(processed_root, "UMMCSIMBAND", f"{stem}_X.npy")
    if source == "dalia":
        return os.path.join(processed_root, "dalia", f"{stem}_X.npy")
    if source == "BAMI":
        return os.path.join(processed_root, "BAMI", f"{stem}_X.npy")
    return os.path.join(processed_root, f"{entry}_X.npy")


# Purpose: Build evaluation file list according to mode and split config.
# Inputs: config accessor.
# Outputs: list of feature file paths.
# Assumptions: processed root contains preprocessed files.
def get_eval_files(cfg) -> List[str]:
    mode = cfg.get("evaluation.mode", "all_data")
    processed_root = cfg.get("paths.processed_root", "./data/processed")

    if mode == "test_set":
        split_json = cfg.get("paths.split_json", "./split_optimized.json")
        if not os.path.exists(split_json):
            raise FileNotFoundError(f"Missing split file: {split_json}")
        with open(split_json, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        test_entries = payload.get("test", [])
        if not test_entries and isinstance(payload.get("splits"), dict):
            test_entries = payload["splits"].get("test", [])
        files = [resolve_split_entry(entry, processed_root) for entry in test_entries]
        return [path for path in files if os.path.exists(path)]

    return sorted(glob.glob(os.path.join(processed_root, "**", "*_X.npy"), recursive=True))


# Purpose: Run evaluation and plot/report AF/HR performance.
# Inputs: config path.
# Outputs: none.
# Side effects: writes plot files and prints metrics.
def evaluate_model(config_path: str) -> None:
    cfg = load_config(config_path)
    config_dir = os.path.dirname(os.path.abspath(config_path))
    project_root = os.path.abspath(os.path.join(config_dir, ".."))
    if os.path.basename(config_dir) != "config":
        project_root = config_dir

    # Purpose: Resolve and normalize input path values for consistent file access.
    # Inputs: Parameters defined in `_resolve_path` signature.
    # Outputs: Return value produced by `_resolve_path`.
    # Assumptions: Caller provides valid types/shapes for this operation.
    def _resolve_path(path: str) -> str:
        if not path:
            return path
        return path if os.path.isabs(path) else os.path.abspath(os.path.join(project_root, path))

    # Resolve key paths relative to the project root so CWD doesn't matter.
    paths_cfg = cfg.data.setdefault("paths", {})
    paths_cfg["processed_root"] = _resolve_path(paths_cfg.get("processed_root", "./data/processed"))
    paths_cfg["split_json"] = _resolve_path(paths_cfg.get("split_json", "./split_optimized.json"))
    paths_cfg["plots_dir"] = _resolve_path(paths_cfg.get("plots_dir", "./plots"))
    bpm_bins = get_bpm_bins(cfg)

    device = get_device(cfg)
    print(f"🚀 Runtime Device: {device}")
    model = build_model_from_config(cfg)

    ckpt_path = _resolve_path(cfg.get("evaluation.checkpoint_path", "./checkpoints/final_model.pth"))
    af_threshold = cfg.get_float("evaluation.default_af_threshold", 0.5)

    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
            af_threshold = float(checkpoint.get("threshold", af_threshold))
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print(f"Warning: checkpoint not found at {ckpt_path}. Using random weights.")

    model.to(device).eval()
    files = get_eval_files(cfg)
    print(f"Evaluation files: {len(files)}")

    infer_bs = cfg.get_int("evaluation.infer_batch_size", 2048)
    input_len = cfg.get_int("data.input_length", 320)
    input_channels = cfg.get_int("data.input_channels", 16)
    simband_ppg_only = cfg.get_bool("data.simband_ppg_only", True)
    include_hr_as_af_negative = cfg.get_bool("evaluation.include_hr_as_af_negative", False)
    use_temporal_decoder = cfg.get_bool("evaluation.use_temporal_decoder", True)
    sqa_enable = cfg.get_bool("signal_quality.enable", False)
    sqa_action = cfg.get("signal_quality.action", "ignore")

    pred_af_all = []
    true_af_all = []
    mask_af_all = []
    mask_af_neg_all = []
    pred_hr_all = []
    true_hr_all = []
    mask_hr_all = []
    source_all = []

    decoder = None
    if use_temporal_decoder:
        decoder_cfg = cfg.get("evaluation.temporal_decoder", {}) or {}
        decoder = TemporalHeartRateDecoder(
            search_window=int(decoder_cfg.get("search_window", 10)),
            init_frames=int(decoder_cfg.get("init_frames", 8)),
            debounce_threshold=int(decoder_cfg.get("debounce_threshold", 5)),
        )
    last_sequence_key = None
    warned_channel_shape = False

    with torch.no_grad():
        for x_path in tqdm(files, desc="Eval"):
            y_path = x_path.replace("_X.npy", "_y.npy")
            if not os.path.exists(y_path):
                continue

            x_arr = np.load(x_path)
            y_arr = np.load(y_path)
            if x_arr.size == 0:
                continue

            if x_arr.ndim != 3:
                print(f"Skip {x_path}: expected 3D array, got {x_arr.shape}")
                continue

            if x_arr.shape[1] == input_channels:
                pass
            elif x_arr.shape[2] == input_channels:
                x_arr = x_arr.transpose(0, 2, 1)
            else:
                if x_arr.shape[1] > x_arr.shape[2]:
                    x_arr = x_arr.transpose(0, 2, 1)
                if not warned_channel_shape and x_arr.shape[1] != input_channels:
                    print(
                        f"Warning: {x_path} channel dim {x_arr.shape[1]} "
                        f"!= expected {input_channels}"
                    )
                    warned_channel_shape = True

            cur_len = x_arr.shape[2]
            if cur_len > input_len:
                start = (cur_len - input_len) // 2
                x_arr = x_arr[:, :, start : start + input_len]
            elif cur_len < input_len:
                x_arr = np.pad(x_arr, ((0, 0), (0, 0), (0, input_len - cur_len)), mode="constant")
            if simband_ppg_only and "UMMCSIMBAND" in x_path and x_arr.shape[1] > 1:
                x_arr[:, 1:, :] = 0.0

            is_hr = (y_arr.ndim == 2 and y_arr.shape[1] == bpm_bins.shape[0])
            n = x_arr.shape[0]

            sequence_key = x_path
            if use_temporal_decoder and decoder is not None and sequence_key != last_sequence_key:
                decoder.reset()
                last_sequence_key = sequence_key

            p_af_chunks = []
            p_hr_chunks = []
            tensor_x = torch.from_numpy(x_arr.astype(np.float32))
            for start in range(0, n, infer_bs):
                end = min(start + infer_bs, n)
                out_af, out_hr = model(tensor_x[start:end].to(device))
                p_af_chunks.append(torch.sigmoid(out_af).view(-1).cpu().numpy())
                p_hr_chunks.append(torch.softmax(out_hr, dim=1).cpu().numpy())

            p_af = np.concatenate(p_af_chunks)
            p_hr_dist = np.concatenate(p_hr_chunks)

            if sqa_enable:
                invalid_mask = np.zeros(n, dtype=bool)
                reason_counts = Counter()
                for idx in range(n):
                    waveform = x_arr[idx, 0, :]
                    is_valid, reason = validate_signal_quality(waveform, cfg)
                    if not is_valid:
                        invalid_mask[idx] = True
                        reason_counts[reason] += 1

                if np.any(invalid_mask):
                    if sqa_action == "force_nsr":
                        p_af[invalid_mask] = 0.0
                    if cfg.get_bool("evaluation.debug_stats", False):
                        summary = ", ".join(f"{k}={v}" for k, v in reason_counts.items())
                        print(
                            f"SQA rejected {int(np.sum(invalid_mask))}/{n} in {os.path.basename(x_path)}"
                            f"{(' (' + summary + ')') if summary else ''}"
                        )

            if is_hr:
                if use_temporal_decoder and decoder is not None:
                    p_hr_scalar = np.array(
                        [decoder.decode(row, bpm_bins, config=cfg) for row in p_hr_dist],
                        dtype=np.float32,
                    )
                else:
                    p_hr_scalar = np.array(
                        [decode_hr_smart(row, bpm_bins, config=cfg) for row in p_hr_dist],
                        dtype=np.float32,
                    )
                t_hr = np.sum(y_arr * bpm_bins, axis=1).astype(np.float32)
                m_hr = np.ones(n, dtype=np.float32)
                m_af = np.zeros(n, dtype=np.float32)
                m_af_neg = np.zeros(n, dtype=np.float32)
                t_af = np.zeros(n, dtype=np.float32)
            else:
                p_hr_scalar = np.zeros(n, dtype=np.float32)
                t_hr = np.zeros(n, dtype=np.float32)
                m_hr = np.zeros(n, dtype=np.float32)
                m_af = np.ones(n, dtype=np.float32)
                m_af_neg = np.zeros(n, dtype=np.float32)
                if y_arr.ndim == 2 and y_arr.shape[1] == 2:
                    t_af = np.argmax(y_arr, axis=1).astype(np.float32)
                else:
                    t_af = y_arr.astype(np.float32).reshape(-1)

            source_id = 0 if "UMMCSIMBAND" in x_path else (1 if "dalia" in x_path.lower() else (2 if "bami" in x_path.lower() else 3))
            if include_hr_as_af_negative and is_hr and source_id in (1, 2):
                m_af_neg = np.ones(n, dtype=np.float32)

            pred_af_all.append(p_af)
            true_af_all.append(t_af)
            mask_af_all.append(m_af)
            mask_af_neg_all.append(m_af_neg)
            pred_hr_all.append(p_hr_scalar)
            true_hr_all.append(t_hr)
            mask_hr_all.append(m_hr)
            source_all.append(np.full(n, source_id, dtype=np.int32))

    if not pred_af_all:
        print("No evaluation samples found.")
        return

    pred_af = np.concatenate(pred_af_all)
    true_af = np.concatenate(true_af_all)
    mask_af = np.concatenate(mask_af_all)
    mask_af_neg = np.concatenate(mask_af_neg_all)
    pred_hr = np.concatenate(pred_hr_all)
    true_hr = np.concatenate(true_hr_all)
    mask_hr = np.concatenate(mask_hr_all)
    sources = np.concatenate(source_all)

    af_prob = pred_af
    post_enable = cfg.get_bool("postprocessing.enable", False)
    if post_enable:
        pred_af_binary = apply_post_processing_pipeline(af_prob, cfg)
        af_threshold = cfg.get_float("postprocessing.fixed_threshold", cfg.get_float("evaluation.default_af_threshold", 0.5))
    else:
        af_threshold = cfg.get_float("evaluation.default_af_threshold", 0.5)
        pred_af_binary = (af_prob >= af_threshold).astype(int)

    valid_af = mask_af > 0.5
    valid_af_neg = mask_af_neg > 0.5
    print(f"AF threshold: {af_threshold:.3f}")
    if np.sum(valid_af) > 0:
        af_true_valid = true_af[valid_af].astype(int)
        af_bin_valid = pred_af_binary[valid_af]
        tn = np.sum((af_true_valid == 0) & (af_bin_valid == 0))
        fp = np.sum((af_true_valid == 0) & (af_bin_valid == 1))
        fn = np.sum((af_true_valid == 1) & (af_bin_valid == 0))
        tp = np.sum((af_true_valid == 1) & (af_bin_valid == 1))
        rec = tp / (tp + fn + 1e-8) * 100.0
        spec = tn / (tn + fp + 1e-8) * 100.0
        print(f"AF: threshold={af_threshold:.3f} recall={rec:.2f}% spec={spec:.2f}% TP/FN/TN/FP={tp}/{fn}/{tn}/{fp}")

        dalia_mask = (valid_af_neg if include_hr_as_af_negative else valid_af) & (sources == 1)
        if np.sum(dalia_mask) > 0:
            dalia_fp = np.sum((true_af[dalia_mask] == 0) & (pred_af_binary[dalia_mask] == 1))
            dalia_neg = np.sum(true_af[dalia_mask] == 0)
            fpr = (dalia_fp / (dalia_neg + 1e-8)) * 100.0
            print(f"AF Dalia false-positive rate: {fpr:.2f}%")

    valid_hr = mask_hr > 0.5
    if np.sum(valid_hr) > 0:
        err = np.abs(true_hr[valid_hr] - pred_hr[valid_hr])
        mae = np.mean(err)
        rmse = np.sqrt(np.mean(err ** 2))
        print(f"HR: MAE={mae:.2f}, RMSE={rmse:.2f}")

    if cfg.get_bool("evaluation.debug_stats", False):
        print("\nDebug Stats")
        print(f"AF valid={int(np.sum(valid_af))} HR valid={int(np.sum(valid_hr))}")
        for sid, name in [(0, "simband"), (1, "dalia"), (2, "bami"), (3, "unknown")]:
            mask_af_src = valid_af & (sources == sid)
            mask_hr_src = valid_hr & (sources == sid)
            if np.any(mask_af_src):
                af_true_src = true_af[mask_af_src].astype(int)
                pos = int(np.sum(af_true_src == 1))
                neg = int(np.sum(af_true_src == 0))
                print(f"{name} AF: pos={pos} neg={neg} prob_mean={float(np.mean(af_prob[mask_af_src])):.4f}")
            if np.any(mask_hr_src):
                hr_true_src = true_hr[mask_hr_src]
                hr_pred_src = pred_hr[mask_hr_src]
                print(
                    f"{name} HR: n={int(np.sum(mask_hr_src))} "
                    f"true_mean={float(np.mean(hr_true_src)):.2f} "
                    f"pred_mean={float(np.mean(hr_pred_src)):.2f}"
                )

    low_th = cfg.get_float("training.sampler.hr_low_threshold", 75.0)
    high_th = cfg.get_float("training.sampler.hr_high_threshold", 120.0)

    # Detailed test result matrix.
    print("\nTest Result Matrix")

    hr_rows = [["HR MAE/RMSE/(n)", "Low", "Mid", "High", "Overall"]]

    # Purpose: Implement `_hr_cell` for the TinyNet workflow.
    # Inputs: Parameters defined in `_hr_cell` signature.
    # Outputs: Return value produced by `_hr_cell`.
    # Assumptions: Caller provides valid types/shapes for this operation.
    def _hr_cell(mask: np.ndarray) -> str:
        if not np.any(mask):
            return "n/a"
        err = true_hr[mask] - pred_hr[mask]
        mae = np.mean(np.abs(err))
        rmse = np.sqrt(np.mean(err ** 2))
        return f"{mae:.2f}/{rmse:.2f}/({int(np.sum(mask))})"

    # Purpose: Implement `hr_row` for the TinyNet workflow.
    # Inputs: Parameters defined in `hr_row` signature.
    # Outputs: Return value produced by `hr_row`.
    # Assumptions: Caller provides valid types/shapes for this operation.
    def hr_row(name: str, mask: np.ndarray) -> List[str]:
        low_mask = mask & (true_hr < low_th)
        mid_mask = mask & (true_hr >= low_th) & (true_hr <= high_th)
        high_mask = mask & (true_hr > high_th)
        overall_mask = mask
        return [
            name,
            _hr_cell(low_mask),
            _hr_cell(mid_mask),
            _hr_cell(high_mask),
            _hr_cell(overall_mask),
        ]

    hr_rows.append(hr_row("dalia", valid_hr & (sources == 1)))
    hr_rows.append(hr_row("bami", valid_hr & (sources == 2)))
    hr_rows.append(hr_row("overall", valid_hr))
    print(format_table(hr_rows))

    af_rows = [["AF Metrics", "Recall", "Specificity", "Accuracy", "F1-Score"]]
    simband_mask = valid_af & (sources == 0)
    simband_metrics = _compute_af_metrics_binary(true_af[simband_mask], pred_af_binary[simband_mask]) if np.any(simband_mask) else {}
    af_rows.append(
        [
            "simband",
            _safe_metric(simband_metrics.get("recall") if simband_metrics else np.nan),
            _safe_metric(simband_metrics.get("specificity") if simband_metrics else np.nan),
            _safe_metric(simband_metrics.get("accuracy") if simband_metrics else np.nan),
            _safe_metric(simband_metrics.get("f1") if simband_metrics else np.nan),
        ]
    )

    # Purpose: Implement `af_specificity_row` for the TinyNet workflow.
    # Inputs: Parameters defined in `af_specificity_row` signature.
    # Outputs: Return value produced by `af_specificity_row`.
    # Assumptions: Caller provides valid types/shapes for this operation.
    def af_specificity_row(name: str, mask: np.ndarray) -> List[str]:
        metrics = _compute_af_metrics_binary(true_af[mask], pred_af_binary[mask]) if np.any(mask) else {}
        spec = metrics.get("specificity") if metrics else np.nan
        return [name, "n/a", _safe_metric(spec), "n/a", "n/a"]

    if include_hr_as_af_negative:
        af_rows.append(af_specificity_row("dalia", valid_af_neg & (sources == 1)))
        af_rows.append(af_specificity_row("bami", valid_af_neg & (sources == 2)))
    else:
        af_rows.append(af_specificity_row("dalia", valid_af & (sources == 1)))
        af_rows.append(af_specificity_row("bami", valid_af & (sources == 2)))
    print(format_table(af_rows))

    plots_root = cfg.get("paths.plots_dir", "./plots")
    result_dir = _resolve_path(cfg.get("evaluation.result_dir", os.path.join(plots_root, "evaluation_results")))
    os.makedirs(result_dir, exist_ok=True)

    if np.sum(valid_hr) > 1:
        plot_regression_scatter(true_hr[valid_hr], pred_hr[valid_hr], os.path.join(result_dir, "Scatter_Density.png"))
        plot_correlation(true_hr[valid_hr], pred_hr[valid_hr], os.path.join(result_dir, "Correlation_Global.png"))
        plot_bland_altman(true_hr[valid_hr], pred_hr[valid_hr], os.path.join(result_dir, "BlandAltman_Global.png"))
        plot_poincare(true_hr[valid_hr], pred_hr[valid_hr], os.path.join(result_dir, "ErrorDist_Global.png"))

    if np.sum(valid_af) > 1:
        plot_af_prob_distribution(true_af[valid_af], af_prob[valid_af], af_threshold, os.path.join(result_dir, "AF_Prob_Dist.png"))
        simband_mask = valid_af & (sources == 0)
        if np.any(simband_mask):
            y_true_sim = true_af[simband_mask].astype(int)
            y_pred_sim = pred_af_binary[simband_mask]
            plot_confusion_matrix(
                y_true_sim,
                y_pred_sim,
                labels=["NSR", "AF"],
                save_path=os.path.join(result_dir, "AF_Confusion_Simband.png"),
                title="Simband AF Confusion Matrix",
            )

    print(f"Evaluation complete. Results saved to {result_dir}")
