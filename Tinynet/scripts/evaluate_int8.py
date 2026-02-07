# Author: Yuyang Chen
# School: Zhejiang University
# Last Modified: 2026.2.7
# Description: Entrypoint for evaluating TinyNet INT8 models.

"""Evaluate TinyNet INT8 model with FP32-equivalent report formatting."""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if SRC_ROOT not in sys.path:
    sys.path.append(SRC_ROOT)

from evaluate.evaluate import format_table, get_eval_files, _compute_af_metrics_binary, _safe_metric
from evaluate.helper import (
    TemporalHeartRateDecoder,
    apply_post_processing_pipeline,
    plot_af_prob_distribution,
    plot_bland_altman,
    plot_confusion_matrix,
    plot_correlation,
    plot_poincare,
    plot_regression_scatter,
    validate_signal_quality,
)
from train.helper import decode_hr_smart
from utils.config import get_bpm_bins, load_config


# Purpose: Resolve and normalize input path values for consistent file access.
# Inputs: Parameters defined in `_resolve_path` signature.
# Outputs: Return value produced by `_resolve_path`.
# Assumptions: Caller provides valid types/shapes for this operation.
def _resolve_path(project_root: str, path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    cwd_path = os.path.abspath(path)
    if os.path.exists(cwd_path):
        return cwd_path
    return os.path.abspath(os.path.join(project_root, path))


# Purpose: Load required resources and return parsed content.
# Inputs: Parameters defined in `_load_int8_model` signature.
# Outputs: Return value produced by `_load_int8_model`.
# Assumptions: Caller provides valid types/shapes for this operation.
def _load_int8_model(model_path: str) -> torch.nn.Module:
    torch.backends.quantized.engine = "qnnpack"
    try:
        model = torch.jit.load(model_path, map_location="cpu")
        model.eval()
        return model
    except Exception:
        payload = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(payload, torch.nn.Module):
            payload.eval()
            return payload
        if isinstance(payload, dict):
            for key in ("model", "quantized_model", "module"):
                candidate = payload.get(key)
                if isinstance(candidate, torch.nn.Module):
                    candidate.eval()
                    return candidate
        raise TypeError(f"Unable to load INT8 model from: {model_path}")


# Purpose: Implement `main` for the TinyNet workflow.
# Inputs: Parameters defined in `main` signature.
# Outputs: Return value produced by `main`.
# Assumptions: Caller provides valid types/shapes for this operation.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=os.path.join(PROJECT_ROOT, "config", "config.yaml"))
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    config_path = _resolve_path(PROJECT_ROOT, args.config)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = load_config(config_path)
    config_dir = os.path.dirname(os.path.abspath(config_path))
    project_root = os.path.abspath(os.path.join(config_dir, ".."))
    if os.path.basename(config_dir) != "config":
        project_root = config_dir

    paths_cfg = cfg.data.setdefault("paths", {})
    paths_cfg["processed_root"] = _resolve_path(project_root, paths_cfg.get("processed_root", "./data/processed"))
    paths_cfg["split_json"] = _resolve_path(project_root, paths_cfg.get("split_json", "./split_optimized.json"))
    paths_cfg["plots_dir"] = _resolve_path(project_root, paths_cfg.get("plots_dir", "./plots"))

    bpm_bins = get_bpm_bins(cfg)

    device = torch.device("cpu")
    print(f"🚀 Runtime Device: {device}")

    output_root = _resolve_path(project_root, cfg.get("paths.output_root", "./output"))
    model_path_cfg = cfg.get("evaluation.int8_model_path", os.path.join(output_root, "int8_model", "model_quantized.pth"))
    model_path = _resolve_path(project_root, args.model if args.model else model_path_cfg)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"INT8 model not found: {model_path}")
    model = _load_int8_model(model_path)
    print(f"Loaded INT8 model: {model_path}")

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
    def hr_row(name: str, mask: np.ndarray) -> list[str]:
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
    def af_specificity_row(name: str, mask: np.ndarray) -> list[str]:
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

    result_dir_cfg = cfg.get("evaluation.int8_result_dir", os.path.join(output_root, "visualizations"))
    result_dir = _resolve_path(project_root, result_dir_cfg)
    os.makedirs(result_dir, exist_ok=True)

    if np.sum(valid_hr) > 1:
        plot_regression_scatter(true_hr[valid_hr], pred_hr[valid_hr], os.path.join(result_dir, "Scatter_Density_Int8.png"))
        plot_correlation(true_hr[valid_hr], pred_hr[valid_hr], os.path.join(result_dir, "Correlation_Global_Int8.png"))
        plot_bland_altman(true_hr[valid_hr], pred_hr[valid_hr], os.path.join(result_dir, "BlandAltman_Global_Int8.png"))
        plot_poincare(true_hr[valid_hr], pred_hr[valid_hr], os.path.join(result_dir, "ErrorDist_Global_Int8.png"))

    if np.sum(valid_af) > 1:
        plot_af_prob_distribution(true_af[valid_af], af_prob[valid_af], af_threshold, os.path.join(result_dir, "AF_Prob_Dist_Int8.png"))
        simband_mask = valid_af & (sources == 0)
        if np.any(simband_mask):
            y_true_sim = true_af[simband_mask].astype(int)
            y_pred_sim = pred_af_binary[simband_mask]
            plot_confusion_matrix(
                y_true_sim,
                y_pred_sim,
                labels=["NSR", "AF"],
                save_path=os.path.join(result_dir, "AF_Confusion_Simband_Int8.png"),
                title="Simband AF Confusion Matrix (Int8)",
            )

    print(f"Evaluation complete. Results saved to {result_dir}")


if __name__ == "__main__":
    main()
