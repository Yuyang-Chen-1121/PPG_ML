# Author: Yuyang Chen
# School: Zhejiang University
# Last Modified: 2026.2.7
# Description: Core training loops, validation, and stage scheduling logic.

"""Training loops and dataloader preparation for TinyNet."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataloader import ChunkMultiModalDataset, batch_collate_fn, create_weighted_sampler
from train.helper import (
    EarlyStopping,
    apply_median_filter,
    decode_hr_smart,
    enforce_bn_mode,
    find_best_threshold,
    get_all_files,
    internal_train_val_split,
    save_checkpoint,
    set_trainable_and_mode,
)
from utils.config import ConfigAccessor, get_bpm_bins


# Purpose: Implement `_tensor_stats` for the TinyNet workflow.
# Inputs: Parameters defined in `_tensor_stats` signature.
# Outputs: Return value produced by `_tensor_stats`.
# Assumptions: Caller provides valid types/shapes for this operation.
def _tensor_stats(x: torch.Tensor) -> Dict[str, object]:
    x = x.detach()
    stats = {"shape": list(x.shape), "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    if x.numel() == 0:
        return stats
    x_f = x.float()
    stats["mean"] = float(x_f.mean().item())
    stats["std"] = float(x_f.std(unbiased=False).item())
    stats["min"] = float(x_f.min().item())
    stats["max"] = float(x_f.max().item())
    return stats


# Purpose: Implement `_diagnose_report_path` for the TinyNet workflow.
# Inputs: Parameters defined in `_diagnose_report_path` signature.
# Outputs: Return value produced by `_diagnose_report_path`.
# Assumptions: Caller provides valid types/shapes for this operation.
def _diagnose_report_path() -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(project_root, "diagnose_report.txt")


# Purpose: Implement `_record_af_branch_trace` for the TinyNet workflow.
# Inputs: Parameters defined in `_record_af_branch_trace` signature.
# Outputs: Return value produced by `_record_af_branch_trace`.
# Assumptions: Caller provides valid types/shapes for this operation.
def _record_af_branch_trace(model, batch, device, report_path: str) -> None:
    if batch is None:
        return
    bx, _, _, _, _, _ = batch
    x = bx.to(device)

    was_training = model.training
    model.eval()
    lines = []
    try:
        with torch.no_grad():
            x = model.quant(x)
            x = model.stem(x)
            lines.append(("stem_out", _tensor_stats(x)))

            x_spatial = model.af_spatial_block1(x)
            lines.append(("af_spatial_block1", _tensor_stats(x_spatial)))
            x_spatial = model.af_spatial_block2(x_spatial)
            lines.append(("af_spatial_block2", _tensor_stats(x_spatial)))
            x_spatial = model.af_spatial_block3(x_spatial)
            lines.append(("af_spatial_block3", _tensor_stats(x_spatial)))

            x_temporal = model.af_temporal_pool1(x)
            lines.append(("af_temporal_pool1", _tensor_stats(x_temporal)))
            x_temporal = model.af_temporal_pool2(x_temporal)
            lines.append(("af_temporal_pool2", _tensor_stats(x_temporal)))
            x_temporal = model.af_temporal_block1(x_temporal)
            lines.append(("af_temporal_block1", _tensor_stats(x_temporal)))
            x_temporal = model.af_temporal_block2(x_temporal)
            lines.append(("af_temporal_block2", _tensor_stats(x_temporal)))

            x_data = model.af_temporal_data_conv(x_temporal)
            x_data = model.af_temporal_data_bn(x_data)
            x_data = model.af_temporal_data_relu(x_data)
            lines.append(("af_temporal_data", _tensor_stats(x_data)))

            x_gate = model.af_temporal_gate_conv(x_temporal)
            x_gate = model.af_temporal_gate_bn(x_gate)
            x_gate = model.af_temporal_gate_sigmoid(x_gate)
            lines.append(("af_temporal_gate", _tensor_stats(x_gate)))

            x_temporal = model.af_temporal_gate_mul.mul(x_data, x_gate)
            lines.append(("af_temporal_gated", _tensor_stats(x_temporal)))
            x_temporal = model.af_temporal_global_pool(x_temporal)
            lines.append(("af_temporal_global_pool", _tensor_stats(x_temporal)))

            x_af = model.af_fusion_add.add(x_spatial, x_temporal)
            lines.append(("af_fusion_add", _tensor_stats(x_af)))
            x_af = model.af_fusion_se(x_af)
            lines.append(("af_fusion_se", _tensor_stats(x_af)))

            x_af = model.af_head_conv(x_af)
            x_af = model.af_head_bn(x_af)
            x_af = model.af_head_relu(x_af)
            lines.append(("af_head_conv", _tensor_stats(x_af)))

            x_af = model.af_gap(x_af)
            lines.append(("af_gap", _tensor_stats(x_af)))
            x_af = model.af_dropout(x_af)
            lines.append(("af_dropout", _tensor_stats(x_af)))
    finally:
        model.train(was_training)

    if not lines:
        return

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\nAF Branch Trace (single batch)\n")
        f.write(f"batch_size: {int(x.size(0))}\n")
        for name, stats in lines:
            f.write(
                f"{name}: shape={stats['shape']} mean={stats['mean']:.6f} std={stats['std']:.6f} "
                f"min={stats['min']:.6f} max={stats['max']:.6f}\n"
            )


# Purpose: Convert split json entry to concrete feature file path.
# Inputs: split entry string/path, processed root.
# Outputs: resolved *_X.npy path.
# Assumptions: split entries may be dataset IDs like "dalia_S10".
def resolve_split_entry(entry: str, processed_root: str) -> str:
    if entry.endswith("_X.npy"):
        return entry if os.path.isabs(entry) else os.path.join(processed_root, entry)

    parts = entry.split("_", 1)
    if len(parts) != 2:
        return os.path.join(processed_root, entry)

    source, stem = parts
    if source == "UMMCSIMBAND":
        return os.path.join(processed_root, "UMMCSIMBAND", f"{stem}_X.npy")
    if source == "dalia":
        return os.path.join(processed_root, "dalia", f"{stem}_X.npy")
    if source == "BAMI":
        return os.path.join(processed_root, "BAMI", f"{stem}_X.npy")
    return os.path.join(processed_root, f"{entry}_X.npy")


# Purpose: Normalize sampler config from split json or config into flat dict.
# Inputs: raw sampler config (possibly nested).
# Outputs: flat dict with sampler keys.
# Assumptions: numeric values are castable to float.
def normalize_sampler_config(raw_cfg: Any) -> Dict[str, float]:
    if not isinstance(raw_cfg, dict):
        return {}

    weights: Dict[str, float] = {}

    # Flat keys (legacy or already normalized).
    flat_keys = [
        "task_af",
        "task_hr",
        "af_positive",
        "af_negative",
        "hr_low",
        "hr_mid",
        "hr_high",
        "hr_low_threshold",
        "hr_high_threshold",
    ]
    for key in flat_keys:
        if key in raw_cfg:
            weights[key] = float(raw_cfg[key])

    # Nested keys from split_optimized.json.
    task_cfg = raw_cfg.get("task", {}) if isinstance(raw_cfg.get("task", {}), dict) else {}
    af_cfg = raw_cfg.get("af", {}) if isinstance(raw_cfg.get("af", {}), dict) else {}
    hr_cfg = raw_cfg.get("hr", {}) if isinstance(raw_cfg.get("hr", {}), dict) else {}

    if "af" in task_cfg:
        weights["task_af"] = float(task_cfg["af"])
    if "hr" in task_cfg:
        weights["task_hr"] = float(task_cfg["hr"])

    if "pos" in af_cfg:
        weights["af_positive"] = float(af_cfg["pos"])
    if "neg" in af_cfg:
        weights["af_negative"] = float(af_cfg["neg"])

    if "low" in hr_cfg:
        weights["hr_low"] = float(hr_cfg["low"])
    if "mid" in hr_cfg:
        weights["hr_mid"] = float(hr_cfg["mid"])
    if "high" in hr_cfg:
        weights["hr_high"] = float(hr_cfg["high"])
    if "low_threshold" in hr_cfg:
        weights["hr_low_threshold"] = float(hr_cfg["low_threshold"])
    if "high_threshold" in hr_cfg:
        weights["hr_high_threshold"] = float(hr_cfg["high_threshold"])

    return weights


# Purpose: Build stage-specific sampler weights from base config.
# Inputs: base weight dict, stage index.
# Outputs: weight dict tailored for training stage.
# Assumptions: stage 1 uses full weights, stage 2 AF-focused, stage 3 HR-focused.
def build_stage_sampler_weights(base_weights: Dict[str, float], stage_idx: int) -> Dict[str, float]:
    weights = dict(base_weights or {})
    if stage_idx == 2:
        weights["task_hr"] = 0.0
        weights["hr_low"] = 0.0
        weights["hr_mid"] = 0.0
        weights["hr_high"] = 0.0
    elif stage_idx == 3:
        weights["task_af"] = 0.0
        weights["af_positive"] = 0.0
        weights["af_negative"] = 0.0
    return weights


# Purpose: Build train/val file lists from split json or fallback random split.
# Inputs: config accessor.
# Outputs: tuple(train_files, val_files).
# Assumptions: processed_root contains preprocessed *_X.npy files.
def load_train_val_files(cfg: ConfigAccessor) -> Tuple[List[str], List[str], Dict[str, float]]:
    processed_root = cfg.get("paths.processed_root", "./data/processed")
    split_json = cfg.get("paths.split_json", "./split_optimized.json")
    sampler_cfg: Dict[str, float] = {}

    if split_json and os.path.exists(split_json):
        with open(split_json, "r", encoding="utf-8") as handle:
            split_payload = json.load(handle)
        train_entries = split_payload.get("train", [])
        val_entries = split_payload.get("val", [])
        sampler_cfg = normalize_sampler_config(split_payload.get("sampler_config", {}))
        if train_entries and val_entries:
            train_files = [resolve_split_entry(entry, processed_root) for entry in train_entries]
            val_files = [resolve_split_entry(entry, processed_root) for entry in val_entries]
            train_files = [path for path in train_files if os.path.exists(path)]
            val_files = [path for path in val_files if os.path.exists(path)]
            return sorted(train_files), sorted(val_files), sampler_cfg

    all_files = get_all_files(processed_root)
    seed = cfg.get_int("training.seed", 42)
    val_ratio = cfg.get_float("training.val_ratio", 0.2)
    train_files, val_files = internal_train_val_split(all_files, val_ratio=val_ratio, seed=seed)
    return train_files, val_files, sampler_cfg


# Purpose: Build train/val dataloaders with stage-1 weighted sampler.
# Inputs: config dictionary/accessor.
# Outputs: tuple(train_loader, val_loader).
# Assumptions: preprocessed data is present under configured paths.
def prepare_data_loaders(config: Any):
    cfg = config if isinstance(config, ConfigAccessor) else ConfigAccessor(config or {})

    train_files, val_files, split_sampler_cfg = load_train_val_files(cfg)
    if not train_files:
        raise RuntimeError("No training files found. Run preprocessing first.")

    train_ds = ChunkMultiModalDataset(train_files, mode="train", config=cfg)
    val_ds = ChunkMultiModalDataset(val_files, mode="val", config=cfg)

    base_sampler_cfg = split_sampler_cfg or normalize_sampler_config(cfg.get("training.sampler", {}) or {})
    stage1_weights = build_stage_sampler_weights(base_sampler_cfg, stage_idx=1)
    stage1_sampler = create_weighted_sampler(train_ds, weights=stage1_weights, stage_idx=1, verbose=True)

    batch_size = cfg.get_int("training.batch_size", 512)
    num_workers = cfg.get_int("training.num_workers", 4)
    pin_memory = cfg.get_bool("training.pin_memory", True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=stage1_sampler,
        shuffle=False,
        collate_fn=batch_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=batch_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader, base_sampler_cfg


# Purpose: Run validation forward pass and collect AF/HR predictions with masks.
# Inputs: model, val loader, device, inner batch size.
# Outputs: tuple of concatenated numpy arrays for metric computation plus avg AF/HR losses.
# Assumptions: model forward returns (af_logits, hr_logits).
def run_val_loop(model, val_loader, device, real_bs: int, criterion=None):
    model.eval()

    preds_af, trues_af, masks_af = [], [], []
    preds_hr, trues_hr, masks_hr = [], [], []
    sources = []
    loss_af_sum = 0.0
    loss_hr_sum = 0.0
    weight_af = 0.0
    weight_hr = 0.0

    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            bx, by_af, by_hr, bm_af, bm_hr, b_source = batch

            n = bx.size(0)
            for start in range(0, n, real_bs):
                end = min(start + real_bs, n)
                inputs = bx[start:end].to(device)
                out_af, out_hr = model(inputs)

                preds_af.append(torch.sigmoid(out_af).view(-1).cpu().numpy())
                preds_hr.append(torch.softmax(out_hr, dim=1).cpu().numpy())

                af_slice = by_af[start:end]
                if af_slice.dim() > 1 and af_slice.size(1) > 1:
                    trues_af.append(torch.argmax(af_slice, dim=1).cpu().numpy())
                else:
                    trues_af.append(af_slice.view(-1).cpu().numpy())

                trues_hr.append(by_hr[start:end].cpu().numpy())
                masks_af.append(bm_af[start:end].cpu().numpy())
                masks_hr.append(bm_hr[start:end].cpu().numpy())
                sources.append(b_source[start:end].cpu().numpy())

                if criterion is not None:
                    af_slice_dev = af_slice.to(device)
                    hr_slice_dev = by_hr[start:end].to(device)
                    m_af_slice = bm_af[start:end].to(device)
                    m_hr_slice = bm_hr[start:end].to(device)

                    loss_af = criterion.compute_af_loss(out_af, af_slice_dev, m_af_slice)
                    loss_hr, _ = criterion.compute_hr_loss(out_hr, hr_slice_dev, m_hr_slice)

                    af_weight = float(torch.sum(m_af_slice).item())
                    hr_weight = float(torch.sum(m_hr_slice).item())
                    if af_weight > 0:
                        loss_af_sum += float(loss_af.detach().cpu()) * af_weight
                        weight_af += af_weight
                    if hr_weight > 0:
                        loss_hr_sum += float(loss_hr.detach().cpu()) * hr_weight
                        weight_hr += hr_weight

    if not preds_af:
        return None

    return (
        np.concatenate(trues_af),
        np.concatenate(preds_af),
        np.concatenate(trues_hr),
        np.concatenate(preds_hr),
        np.concatenate(masks_af),
        np.concatenate(masks_hr),
        np.concatenate(sources),
        (loss_af_sum / (weight_af + 1e-8)) if criterion is not None else 0.0,
        (loss_hr_sum / (weight_hr + 1e-8)) if criterion is not None else 0.0,
    )


# Purpose: Compute AF and HR metrics plus composite score from validation outputs.
# Inputs: AF/HR truths and predictions, task masks, source ids, BPM bins, config.
# Outputs: dictionary of metrics.
# Assumptions: HR distributions have same length as BPM bins.
def calculate_metrics_balanced(
    y_t_af,
    y_p_af,
    y_t_hr,
    y_p_hr,
    m_af,
    m_hr,
    sources,
    bpm_bins,
    config=None,
):
    cfg = config if isinstance(config, ConfigAccessor) else ConfigAccessor(config or {})
    metrics = {
        "af_gmean": 0.0,
        "af_th": 0.5,
        "af_rec": 0.0,
        "af_spec": 0.0,
        "hr_mae": 99.9,
        "hr_rmse": 99.9,
        "hr_outliers": 0,
        "hr_outlier_rate": 0.0,
        "score": 0.0,
    }

    valid_af = m_af > 0.5
    if np.sum(valid_af) > 0:
        af_true = y_t_af[valid_af].astype(int)
        af_prob = y_p_af[valid_af]
        if cfg.get_bool("postprocessing.enable_median_filter", False):
            window = cfg.get_int("postprocessing.median_window", 7)
            af_prob = apply_median_filter(af_prob, window=window)

        search_best = cfg.get_bool("evaluation.search_best_threshold", False)
        if search_best and np.any(af_true == 0) and np.any(af_true == 1):
            best_th = find_best_threshold(af_true, af_prob)
        else:
            best_th = cfg.get_float("evaluation.default_af_threshold", 0.5)

        af_bin = (af_prob >= best_th).astype(int)
        tn, fp, fn, tp = confusion_matrix(af_true, af_bin, labels=[0, 1]).ravel()
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)

        metrics["af_gmean"] = float(np.sqrt(sens * spec))
        metrics["af_th"] = float(best_th)
        metrics["af_rec"] = float(sens * 100.0)
        metrics["af_spec"] = float(spec * 100.0)

    valid_hr = m_hr > 0.5
    if np.sum(valid_hr) > 0:
        true_dist = y_t_hr[valid_hr]
        pred_dist = y_p_hr[valid_hr]
        true_bpm = np.sum(true_dist * bpm_bins, axis=1)
        pred_bpm = np.array([decode_hr_smart(prob, bpm_bins, config=cfg) for prob in pred_dist])
        err = np.abs(true_bpm - pred_bpm)
        metrics["hr_mae"] = float(np.mean(err))
        metrics["hr_rmse"] = float(np.sqrt(np.mean(err ** 2)))
        outlier_th = cfg.get_float("evaluation.hr_outlier_threshold", 10.0)
        outliers = int(np.sum(err > outlier_th))
        metrics["hr_outliers"] = outliers
        metrics["hr_outlier_rate"] = float(outliers / (len(err) + 1e-8) * 100.0)

    score_cfg = cfg.get("training.score_weights", {}) or {}
    w_af = float(score_cfg.get("w_af", 0.4))
    w_mae = float(score_cfg.get("w_mae", 0.3))
    w_rmse = float(score_cfg.get("w_rmse", 0.3))
    mae_norm = float(score_cfg.get("mae_norm", 15.0))
    rmse_norm = float(score_cfg.get("rmse_norm", 25.0))

    af_component = metrics["af_gmean"]
    mae_component = max(0.0, 1.0 - metrics["hr_mae"] / mae_norm)
    rmse_component = max(0.0, 1.0 - metrics["hr_rmse"] / rmse_norm)
    metrics["score"] = float(w_af * af_component + w_mae * mae_component + w_rmse * rmse_component)

    return metrics


# Purpose: Train one sequential stage with stage-specific sampling/optimization policy.
# Inputs: stage settings, model/loaders/config/loss/device/checkpoint arguments.
# Outputs: none.
# Side effects: updates model weights and writes checkpoint files.
def run_training_stage(
    stage_idx,
    epochs,
    lr,
    patience,
    monitor_metric,
    model,
    train_loader,
    val_loader,
    train_cfg,
    config,
    criterion,
    device,
    ckpt_dir,
    sampler_cfg=None,
    mode="max",
    save_name="best.pth",
    prev_best_th=None,
):
    del monitor_metric  # stage index defines monitored metric below.
    cfg = config if isinstance(config, ConfigAccessor) else ConfigAccessor(config or {})

    real_bs = int(train_cfg.get("batch_size", cfg.get_int("training.batch_size", 512)))
    bpm_bins_np = get_bpm_bins(cfg)

    set_trainable_and_mode(model, stage_idx)

    base_sampler_cfg = sampler_cfg or normalize_sampler_config(cfg.get("training.sampler", {}) or {})
    stage_weights = build_stage_sampler_weights(base_sampler_cfg, stage_idx=stage_idx)
    new_sampler = create_weighted_sampler(train_loader.dataset, weights=stage_weights, stage_idx=stage_idx, verbose=True)
    train_loader = DataLoader(
        train_loader.dataset,
        batch_size=real_bs,
        sampler=new_sampler,
        num_workers=train_loader.num_workers,
        pin_memory=train_loader.pin_memory,
        collate_fn=train_loader.collate_fn,
        drop_last=True,
    )

    model_params = [param for param in model.parameters() if param.requires_grad]
    loss_params = [param for param in criterion.parameters() if param.requires_grad]
    optimizer = optim.AdamW(
        model_params + loss_params,
        lr=lr,
        weight_decay=cfg.get_float("training.weight_decay", 0.01),
    )

    sched_cfg = cfg.get("training.scheduler", {}) or {}
    if sched_cfg.get("type", "plateau") == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(sched_cfg.get("step_size", 10)),
            gamma=float(sched_cfg.get("gamma", 0.8)),
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=float(sched_cfg.get("factor", 0.5)),
            patience=int(sched_cfg.get("patience", 5)),
        )

    print(f"Stage {stage_idx} start: lr={lr:.3e}, epochs={epochs}, patience={patience}")
    early_stopper = EarlyStopping(patience=patience, mode=mode)
    best_th = float(prev_best_th if prev_best_th is not None else cfg.get_float("evaluation.default_af_threshold", 0.5))

    if not getattr(model, "_af_trace_logged", False):
        trace_batch = next(iter(train_loader), None)
        if trace_batch is not None:
            _record_af_branch_trace(model, trace_batch, device, _diagnose_report_path())
            model._af_trace_logged = True

    for epoch in range(int(epochs)):
        model.train()
        enforce_bn_mode(model, stage_idx)

        train_af_loss_sum = 0.0
        train_hr_loss_sum = 0.0
        train_af_weight = 0.0
        train_hr_weight = 0.0
        pbar = tqdm(train_loader, leave=False, desc=f"S{stage_idx} E{epoch+1}/{epochs}")
        for batch in pbar:
            if batch is None:
                continue
            bx, by_af, by_hr, bm_af, bm_hr, _ = batch
            x = bx.to(device)
            y_af = by_af.to(device)
            y_hr = by_hr.to(device)
            m_af = bm_af.to(device)
            m_hr = bm_hr.to(device)

            optimizer.zero_grad()
            out_af, out_hr = model(x)

            loss = torch.tensor(0.0, device=device)
            if stage_idx in (1, 2):
                loss_af = criterion.compute_af_loss(out_af, y_af, m_af)
                af_precision = torch.exp(-criterion.log_vars[0])
                loss = loss + af_precision * loss_af + 0.5 * criterion.log_vars[0]
            else:
                with torch.no_grad():
                    loss_af = criterion.compute_af_loss(out_af, y_af, m_af)
            if stage_idx in (1, 3):
                loss_hr, _ = criterion.compute_hr_loss(out_hr, y_hr, m_hr)
                hr_precision = torch.exp(-criterion.log_vars[1])
                loss = loss + hr_precision * loss_hr + 0.5 * criterion.log_vars[1]
            else:
                with torch.no_grad():
                    loss_hr, _ = criterion.compute_hr_loss(out_hr, y_hr, m_hr)

            loss.backward()
            grad_clip = cfg.get_float("training.grad_clip_norm", 2.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            af_weight = float(torch.sum(m_af).item())
            hr_weight = float(torch.sum(m_hr).item())
            if af_weight > 0:
                train_af_loss_sum += float(loss_af.detach().cpu()) * af_weight
                train_af_weight += af_weight
            if hr_weight > 0:
                train_hr_loss_sum += float(loss_hr.detach().cpu()) * hr_weight
                train_hr_weight += hr_weight

        val_data = run_val_loop(model, val_loader, device, real_bs, criterion=criterion)
        if val_data is None:
            continue

        (
            y_t_af,
            y_p_af,
            y_t_hr,
            y_p_hr,
            m_af,
            m_hr,
            sources,
            val_loss_af,
            val_loss_hr,
        ) = val_data
        metrics = calculate_metrics_balanced(
            y_t_af,
            y_p_af,
            y_t_hr,
            y_p_hr,
            m_af,
            m_hr,
            sources,
            bpm_bins=bpm_bins_np,
            config=cfg,
        )
        epoch_th = float(metrics.get("af_th", best_th))
        if stage_idx == 1:
            score = metrics["score"]
        elif stage_idx == 2:
            score = metrics["af_gmean"]
            best_th = metrics["af_th"]
        else:
            score = metrics["hr_mae"]

        train_af_loss = train_af_loss_sum / (train_af_weight + 1e-8)
        train_hr_loss = train_hr_loss_sum / (train_hr_weight + 1e-8)
        log_vars = criterion.log_vars.detach().cpu().tolist()
        print(
            f"Stage {stage_idx} Ep {epoch+1:03d} | Train AF Loss {train_af_loss:.4f} "
            f"| Train HR Loss {train_hr_loss:.4f} | Val AF Loss {val_loss_af:.4f} | Val HR Loss {val_loss_hr:.4f} "
            f"| log_vars [{log_vars[0]:.3f}, {log_vars[1]:.3f}] "
            f"| AF GMean {metrics['af_gmean']:.3f} | AF Th {epoch_th:.3f} "
            f"| AF Rec {metrics['af_rec']:.2f}% "
            f"| AF Spec {metrics['af_spec']:.2f}% | HR MAE {metrics['hr_mae']:.2f} "
            f"| HR RMSE {metrics['hr_rmse']:.2f} | HR Out {metrics['hr_outliers']} "
            f"({metrics['hr_outlier_rate']:.2f}%)"
        )

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(score)
        else:
            scheduler.step()

        if early_stopper(score):
            best_th = epoch_th
            save_checkpoint(model, best_th, epoch + 1, score, os.path.join(ckpt_dir, save_name))
            print(f"Saved checkpoint: {save_name} (score={score:.4f})")

        if early_stopper.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"Stage {stage_idx} complete")
