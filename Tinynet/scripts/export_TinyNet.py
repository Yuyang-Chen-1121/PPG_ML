# Author: Yuyang Chen
# School: Zhejiang University
# Last Modified: 2026.2.7
# Description: PTQ export pipeline for INT8 model, hex data, and graph artifacts.

"""Export TinyNet to INT8 artifacts and hardware hex dumps."""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import shutil
import sys
import types
from collections import Counter, defaultdict
from typing import Any, Dict

import numpy as np
import torch
import yaml
from torch import nn
from torch.fx import symbolic_trace

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if SRC_ROOT not in sys.path:
    sys.path.append(SRC_ROOT)

from src.models.tinynet import CascadedAvgPoolToOne, SqueezeExcitationBlock, TinyNet
from src.utils.quant_export_utils import CustomFxGraphDrawer, HexExporter, patch_module_file


class FloatBnBridge(nn.Module):
    """Run BatchNorm in float path between quantized ops."""

    # Purpose: Initialize class state and runtime configuration.
    # Inputs: Parameters defined in `__init__` signature.
    # Outputs: Return value produced by `__init__`.
    # Assumptions: Caller provides valid types/shapes for this operation.
    def __init__(self, bn: nn.BatchNorm1d) -> None:
        super().__init__()
        self.dequant = torch.quantization.DeQuantStub()
        self.bn = bn
        self.quant = torch.quantization.QuantStub()

    # Purpose: Execute forward computation for this module.
    # Inputs: Parameters defined in `forward` signature.
    # Outputs: Return value produced by `forward`.
    # Assumptions: Caller provides valid types/shapes for this operation.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dequant(x)
        x = self.bn(x)
        return self.quant(x)


# Purpose: Implement `_se_forward_quantized` for the TinyNet workflow.
# Inputs: Parameters defined in `_se_forward_quantized` signature.
# Outputs: Return value produced by `_se_forward_quantized`.
# Assumptions: Caller provides valid types/shapes for this operation.
def _se_forward_quantized(self, x: torch.Tensor) -> torch.Tensor:
    z = self.squeeze(x).squeeze(-1)
    z = self.fc1(z)
    z = self.relu(z)
    z = self.fc2(z)
    z = self.sigmoid(z).unsqueeze(-1)
    return self.q_mul.mul(x, z)


# Purpose: Resolve and normalize input path values for consistent file access.
# Inputs: Parameters defined in `_resolve_project_path` signature.
# Outputs: Return value produced by `_resolve_project_path`.
# Assumptions: Caller provides valid types/shapes for this operation.
def _resolve_project_path(path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    cwd_path = os.path.abspath(path)
    if os.path.exists(cwd_path):
        return cwd_path
    return os.path.abspath(os.path.join(PROJECT_ROOT, path))


# Purpose: Implement `_xml_escape` for the TinyNet workflow.
# Inputs: Parameters defined in `_xml_escape` signature.
# Outputs: Return value produced by `_xml_escape`.
# Assumptions: Caller provides valid types/shapes for this operation.
def _xml_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


# Purpose: Load required resources and return parsed content.
# Inputs: Parameters defined in `_load_config` signature.
# Outputs: Return value produced by `_load_config`.
# Assumptions: Caller provides valid types/shapes for this operation.
def _load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# Purpose: Implement `_extract_state_dict` for the TinyNet workflow.
# Inputs: Parameters defined in `_extract_state_dict` signature.
# Outputs: Return value produced by `_extract_state_dict`.
# Assumptions: Caller provides valid types/shapes for this operation.
def _extract_state_dict(checkpoint: Any) -> Dict[str, Any]:
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            return checkpoint["model"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    return checkpoint


# Purpose: Resolve and normalize input path values for consistent file access.
# Inputs: Parameters defined in `_resolve_checkpoint_path` signature.
# Outputs: Return value produced by `_resolve_checkpoint_path`.
# Assumptions: Caller provides valid types/shapes for this operation.
def _resolve_checkpoint_path(config: Dict[str, Any]) -> str:
    train_cfg = config.get("train", {}) or {}
    training_cfg = config.get("training", {}) or {}
    eval_cfg = config.get("evaluation", {}) or {}

    candidates = [
        train_cfg.get("checkpoint_path"),
        training_cfg.get("checkpoint_path"),
        eval_cfg.get("checkpoint_path"),
        "./checkpoints/stage1_best.pth",
    ]
    for ckpt in candidates:
        if not ckpt:
            continue
        ckpt_path = _resolve_project_path(ckpt)
        if os.path.exists(ckpt_path):
            return ckpt_path

    requested = train_cfg.get("checkpoint_path")
    if requested:
        return _resolve_project_path(requested)
    return _resolve_project_path("./checkpoints/stage1_best.pth")


# Purpose: Implement `_safe_fuse_modules` for the TinyNet workflow.
# Inputs: Parameters defined in `_safe_fuse_modules` signature.
# Outputs: Return value produced by `_safe_fuse_modules`.
# Assumptions: Caller provides valid types/shapes for this operation.
def _safe_fuse_modules(model: torch.nn.Module, modules_to_fuse: list[str]) -> None:
    try:
        torch.quantization.fuse_modules(model, modules_to_fuse, inplace=True)
    except Exception:
        pass


# Purpose: Implement `_fuse_tinynet` for the TinyNet workflow.
# Inputs: Parameters defined in `_fuse_tinynet` signature.
# Outputs: Return value produced by `_fuse_tinynet`.
# Assumptions: Caller provides valid types/shapes for this operation.
def _fuse_tinynet(model: TinyNet) -> None:
    if hasattr(model, "stem"):
        _safe_fuse_modules(model.stem, ["0", "1", "2"])

    res_blocks = [
        "hr_block1",
        "hr_block2",
        "hr_block3",
        "af_spatial_block1",
        "af_spatial_block2",
        "af_spatial_block3",
        "af_temporal_block1",
        "af_temporal_block2",
    ]
    for name in res_blocks:
        if not hasattr(model, name):
            continue
        block = getattr(model, name)
        # ResBlock reuses `relu` both after conv1 and after skip-add.
        # Fusing conv1+bn1+relu would replace the shared relu with Identity,
        # silently removing the final activation path. Fuse only conv+bn here.
        _safe_fuse_modules(block, ["conv1", "bn1"])
        _safe_fuse_modules(block, ["conv2", "bn2"])
        if getattr(block, "downsample", None) is not None:
            _safe_fuse_modules(block.downsample, ["0", "1"])

    _safe_fuse_modules(model, ["af_temporal_data_conv", "af_temporal_data_bn", "af_temporal_data_relu"])
    _safe_fuse_modules(model, ["af_temporal_gate_conv", "af_temporal_gate_bn"])
    _safe_fuse_modules(model, ["af_head_conv", "af_head_bn", "af_head_relu"])

    # Keep AF spatial BN numerically active by running it in a float island.
    if isinstance(getattr(model, "af_spatial_bn", None), torch.nn.BatchNorm1d):
        model.af_spatial_bn = FloatBnBridge(model.af_spatial_bn)


# Purpose: Implement `_patch_se_mul_for_quant` for the TinyNet workflow.
# Inputs: Parameters defined in `_patch_se_mul_for_quant` signature.
# Outputs: Return value produced by `_patch_se_mul_for_quant`.
# Assumptions: Caller provides valid types/shapes for this operation.
def _patch_se_mul_for_quant(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, SqueezeExcitationBlock):
            if not hasattr(module, "q_mul"):
                module.q_mul = torch.nn.quantized.FloatFunctional()
            module.forward = types.MethodType(_se_forward_quantized, module)


# Purpose: Build and return a configured object for this pipeline step.
# Inputs: Parameters defined in `_build_int8_model` signature.
# Outputs: Return value produced by `_build_int8_model`.
# Assumptions: Caller provides valid types/shapes for this operation.
def _build_int8_model(model: TinyNet, calibration_input: torch.Tensor, backend: str = "qnnpack") -> torch.nn.Module:
    torch.backends.quantized.engine = backend
    model = model.cpu().eval()
    _patch_se_mul_for_quant(model)
    _fuse_tinynet(model)

    model.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.quantization.prepare(model, inplace=True)
    with torch.no_grad():
        model(calibration_input)
    return torch.quantization.convert(model, inplace=False)


# Purpose: Implement `_patch_fx_control_flow_for_export` for the TinyNet workflow.
# Inputs: Parameters defined in `_patch_fx_control_flow_for_export` signature.
# Outputs: Return value produced by `_patch_fx_control_flow_for_export`.
# Assumptions: Caller provides valid types/shapes for this operation.
def _patch_fx_control_flow_for_export(model: torch.nn.Module) -> None:
    def _forward_no_assert(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pool1(x)
        out = self.pool2(out)
        out = self.pool3(out)
        return out

    for module in model.modules():
        if isinstance(module, CascadedAvgPoolToOne):
            module.forward = types.MethodType(_forward_no_assert, module)


# Purpose: Implement `_restore_fx_control_flow_after_export` for the TinyNet workflow.
# Inputs: Parameters defined in `_restore_fx_control_flow_after_export` signature.
# Outputs: Return value produced by `_restore_fx_control_flow_after_export`.
# Assumptions: Caller provides valid types/shapes for this operation.
def _restore_fx_control_flow_after_export(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, CascadedAvgPoolToOne):
            module.forward = CascadedAvgPoolToOne.forward.__get__(module, CascadedAvgPoolToOne)


# Purpose: Implement `_prepare_output_dirs` for the TinyNet workflow.
# Inputs: Parameters defined in `_prepare_output_dirs` signature.
# Outputs: Return value produced by `_prepare_output_dirs`.
# Assumptions: Caller provides valid types/shapes for this operation.
def _prepare_output_dirs(config: Dict[str, Any]) -> tuple[str, str, str]:
    paths_cfg = config.get("paths", {}) or {}
    export_cfg = config.get("export", {}) or {}
    output_root = _resolve_project_path(paths_cfg.get("output_root", "./output"))

    int8_model_dir = _resolve_project_path(export_cfg.get("int8_model_dir", os.path.join(output_root, "int8_model")))
    hex_data_dir = _resolve_project_path(export_cfg.get("hex_data_dir", os.path.join(output_root, "hex_data")))
    viz_dir = _resolve_project_path(export_cfg.get("visualizations_dir", os.path.join(output_root, "visualizations")))

    for path in (int8_model_dir, hex_data_dir, viz_dir):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    return int8_model_dir, hex_data_dir, viz_dir


# Purpose: Implement `_write_graph_summary_svg` for the TinyNet workflow.
# Inputs: Parameters defined in `_write_graph_summary_svg` signature.
# Outputs: Return value produced by `_write_graph_summary_svg`.
# Assumptions: Caller provides valid types/shapes for this operation.
def _write_graph_summary_svg(graph_svg_path: str, traced_model: torch.nn.Module, reason: str) -> None:
    node_lines: list[str] = []
    for idx, node in enumerate(traced_model.graph.nodes):
        node_lines.append(f"{idx:03d} | {node.op} | {node.target}")
        if len(node_lines) >= 35:
            node_lines.append("... (truncated)")
            break

    line_h = 18
    width = 1280
    height = max(220, 140 + line_h * (len(node_lines) + 2))
    y = 28
    svg_lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        "<rect x='0' y='0' width='100%' height='100%' fill='#ffffff'/>",
        "<text x='16' y='28' font-family='monospace' font-size='14' fill='#111'>TinyNet FX Graph Summary (fallback)</text>",
        f"<text x='16' y='50' font-family='monospace' font-size='12' fill='#aa0000'>Reason: {_xml_escape(reason)}</text>",
    ]
    y = 74
    for line in node_lines:
        svg_lines.append(
            f"<text x='16' y='{y}' font-family='monospace' font-size='12' fill='#222'>{_xml_escape(line)}</text>"
        )
        y += line_h
    svg_lines.append("</svg>")
    with open(graph_svg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_lines))


# Purpose: Resolve and normalize input path values for consistent file access.
# Inputs: Parameters defined in `_resolve_split_entry` signature.
# Outputs: Return value produced by `_resolve_split_entry`.
# Assumptions: Caller provides valid types/shapes for this operation.
def _resolve_split_entry(entry: str, processed_root: str) -> str:
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


# Purpose: Implement `_source_from_path` for the TinyNet workflow.
# Inputs: Parameters defined in `_source_from_path` signature.
# Outputs: Return value produced by `_source_from_path`.
# Assumptions: Caller provides valid types/shapes for this operation.
def _source_from_path(path: str) -> str:
    lower = path.lower()
    if "ummcsimband" in lower:
        return "simband"
    if "/dalia/" in lower:
        return "dalia"
    if "/bami/" in lower:
        return "bami"
    return "unknown"


# Purpose: Return derived values required by downstream steps.
# Inputs: Parameters defined in `_get_calibration_files` signature.
# Outputs: Return value produced by `_get_calibration_files`.
# Assumptions: Caller provides valid types/shapes for this operation.
def _get_calibration_files(config: Dict[str, Any]) -> list[str]:
    paths_cfg = config.get("paths", {}) or {}
    processed_root = _resolve_project_path(paths_cfg.get("processed_root", "./data/processed"))
    quant_cfg = config.get("quantization", {}) or {}
    eval_cfg = config.get("evaluation", {}) or {}
    mode = str(quant_cfg.get("calibration_mode", eval_cfg.get("mode", "all_data")))
    if mode in {"train_set", "val_set", "test_set", "split"}:
        split_json = _resolve_project_path(paths_cfg.get("split_json", "./split_optimized.json"))
        if os.path.exists(split_json):
            with open(split_json, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if mode == "split":
                split_name = str(quant_cfg.get("calibration_split", "train")).strip().lower()
            else:
                split_name = mode.replace("_set", "")
            split_entries = payload.get(split_name, [])
            if not split_entries and isinstance(payload.get("splits"), dict):
                split_entries = payload["splits"].get(split_name, [])
            files = [_resolve_split_entry(entry, processed_root) for entry in split_entries]
            return sorted([p for p in files if os.path.exists(p)])
    return sorted(glob.glob(os.path.join(processed_root, "**", "*_X.npy"), recursive=True))


# Purpose: Implement `_normalize_calibration_window` for the TinyNet workflow.
# Inputs: Parameters defined in `_normalize_calibration_window` signature.
# Outputs: Return value produced by `_normalize_calibration_window`.
# Assumptions: Caller provides valid types/shapes for this operation.
def _normalize_calibration_window(
    x_win: np.ndarray,
    input_channels: int,
    input_length: int,
    simband_ppg_only: bool,
    source_key: str,
) -> np.ndarray | None:
    if x_win.ndim != 2:
        return None

    if x_win.shape[0] == input_channels:
        x_cf = x_win
    elif x_win.shape[1] == input_channels:
        x_cf = x_win.T
    else:
        x_cf = x_win.T if x_win.shape[0] > x_win.shape[1] else x_win
        if x_cf.shape[0] != input_channels:
            return None

    cur_len = x_cf.shape[1]
    if cur_len > input_length:
        start = (cur_len - input_length) // 2
        x_cf = x_cf[:, start : start + input_length]
    elif cur_len < input_length:
        x_cf = np.pad(x_cf, ((0, 0), (0, input_length - cur_len)), mode="constant")

    if simband_ppg_only and source_key == "simband" and x_cf.shape[0] > 1:
        x_cf = x_cf.copy()
        x_cf[1:, :] = 0.0

    return x_cf.astype(np.float32, copy=False)


# Purpose: Implement `_collect_calibration_windows` for the TinyNet workflow.
# Inputs: Parameters defined in `_collect_calibration_windows` signature.
# Outputs: Return value produced by `_collect_calibration_windows`.
# Assumptions: Caller provides valid types/shapes for this operation.
def _collect_calibration_windows(
    file_list: list[str],
    target: int,
    rng: np.random.Generator,
    input_channels: int,
    input_length: int,
    simband_ppg_only: bool,
) -> tuple[list[np.ndarray], list[str]]:
    selected: list[np.ndarray] = []
    selected_sources: list[str] = []
    if target <= 0:
        return selected, selected_sources

    for x_path in file_list:
        if len(selected) >= target:
            break
        try:
            x_arr = np.load(x_path, mmap_mode="r")
        except Exception:
            continue
        if x_arr.ndim != 3 or x_arr.shape[0] == 0:
            continue

        source_key = _source_from_path(x_path)
        n = int(x_arr.shape[0])
        order = rng.permutation(n)
        need = target - len(selected)
        for idx in order[:need]:
            x_win = np.asarray(x_arr[int(idx)], dtype=np.float32)
            x_cf = _normalize_calibration_window(
                x_win,
                input_channels=input_channels,
                input_length=input_length,
                simband_ppg_only=simband_ppg_only,
                source_key=source_key,
            )
            if x_cf is None:
                continue
            selected.append(x_cf)
            selected_sources.append(source_key)
            if len(selected) >= target:
                break

    return selected, selected_sources


# Purpose: Build and return a configured object for this pipeline step.
# Inputs: Parameters defined in `_build_calibration_input` signature.
# Outputs: Return value produced by `_build_calibration_input`.
# Assumptions: Caller provides valid types/shapes for this operation.
def _build_calibration_input(
    config: Dict[str, Any],
    input_channels: int,
    input_length: int,
    max_samples: int = 512,
) -> tuple[torch.Tensor | None, Dict[str, int]]:
    files = _get_calibration_files(config)
    if not files:
        return None, {}

    simband_ppg_only = bool((config.get("data", {}) or {}).get("simband_ppg_only", True))
    quant_cfg = config.get("quantization", {}) or {}
    seed = int((config.get("training", {}) or {}).get("seed", 42))
    strategy = str(quant_cfg.get("calibration_strategy", "stratified_source")).strip().lower()
    rng = np.random.default_rng(seed)

    if strategy == "sequential":
        shuffled_files = list(files)
        rng.shuffle(shuffled_files)
        samples, sample_sources = _collect_calibration_windows(
            shuffled_files,
            target=max_samples,
            rng=rng,
            input_channels=input_channels,
            input_length=input_length,
            simband_ppg_only=simband_ppg_only,
        )
    else:
        per_source_files: dict[str, list[str]] = defaultdict(list)
        for path in files:
            per_source_files[_source_from_path(path)].append(path)

        source_order = ["simband", "dalia", "bami", "unknown"]
        active_sources = [src for src in source_order if per_source_files.get(src)]

        samples = []
        sample_sources: list[str] = []
        if active_sources:
            base = max_samples // len(active_sources)
            extra = max_samples % len(active_sources)
            for idx, src in enumerate(active_sources):
                target = base + (1 if idx < extra else 0)
                src_files = list(per_source_files[src])
                rng.shuffle(src_files)
                src_samples, src_sources = _collect_calibration_windows(
                    src_files,
                    target=target,
                    rng=rng,
                    input_channels=input_channels,
                    input_length=input_length,
                    simband_ppg_only=simband_ppg_only,
                )
                samples.extend(src_samples)
                sample_sources.extend(src_sources)

        if len(samples) < max_samples:
            shuffled_files = list(files)
            rng.shuffle(shuffled_files)
            rem_samples, rem_sources = _collect_calibration_windows(
                shuffled_files,
                target=max_samples - len(samples),
                rng=rng,
                input_channels=input_channels,
                input_length=input_length,
                simband_ppg_only=simband_ppg_only,
            )
            samples.extend(rem_samples)
            sample_sources.extend(rem_sources)

    if not samples:
        return None, {}
    counts = dict(Counter(sample_sources))
    return torch.from_numpy(np.stack(samples, axis=0)), counts


# Purpose: Implement `main` for the TinyNet workflow.
# Inputs: Parameters defined in `main` signature.
# Outputs: Return value produced by `main`.
# Assumptions: Caller provides valid types/shapes for this operation.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=os.path.join(PROJECT_ROOT, "config", "config.yaml"))
    args = parser.parse_args()

    config_path = _resolve_project_path(args.config)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = _load_config(config_path)
    data_cfg = config.get("data", {}) or {}
    training_cfg = config.get("training", {}) or {}
    quant_cfg = config.get("quantization", {}) or {}

    seed = int(training_cfg.get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)

    input_channels = int(data_cfg.get("input_channels", 16))
    window_size = int(data_cfg.get("window_size", data_cfg.get("input_length", 320)))

    int8_model_dir, hex_data_dir, viz_dir = _prepare_output_dirs(config)
    calib_samples = int(quant_cfg.get("calibration_samples", 512))
    calib_mode = str(quant_cfg.get("calibration_mode", (config.get("evaluation", {}) or {}).get("mode", "all_data")))
    calibration_input, calibration_mix = _build_calibration_input(
        config,
        input_channels=input_channels,
        input_length=window_size,
        max_samples=max(1, calib_samples),
    )
    if calibration_input is None:
        calibration_input = torch.randn(1, input_channels, window_size)
        print("Warning: calibration data not found; falling back to random calibration input.")
    else:
        print(f"Calibration samples: {int(calibration_input.shape[0])}")
    print(f"Calibration mode: {calib_mode}")
    if calibration_mix:
        mix_text = ", ".join(f"{k}={v}" for k, v in sorted(calibration_mix.items()))
        print(f"Calibration source mix: {mix_text}")
    dummy_input = calibration_input[:1].clone()

    ckpt_path = _resolve_checkpoint_path(config)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = TinyNet(config).cpu()
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint load mismatch. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}"
        )
    print(f"Loaded checkpoint tensors: {len(state_dict)}")

    fp32_ref_model = copy.deepcopy(model).eval()
    int8_model = _build_int8_model(model, calibration_input=calibration_input, backend="qnnpack")

    with torch.no_grad():
        af_fp, hr_fp = fp32_ref_model(dummy_input)
        af_int8, hr_int8 = int8_model(dummy_input)
        af_diff = float(torch.mean(torch.abs(af_fp - af_int8)).cpu())
        hr_diff = float(torch.mean(torch.abs(hr_fp - hr_int8)).cpu())
    print(f"Sanity diff |AF logits| mean: {af_diff:.6f}, |HR logits| mean: {hr_diff:.6f}")

    _patch_fx_control_flow_for_export(int8_model)

    traced_model = symbolic_trace(int8_model)
    _restore_fx_control_flow_after_export(int8_model)

    exporter = HexExporter(hex_data_dir, traced_model)
    with torch.no_grad():
        exporter.export_(dummy_input)
    exporter.remove_hooks()

    traced_model.to_folder(int8_model_dir, "TinyNetQuant")
    module_path = os.path.join(int8_model_dir, "module.py")
    if os.path.exists(module_path):
        patch_module_file(module_path)

    model_out_path = os.path.join(int8_model_dir, "model_quantized.pth")
    try:
        scripted_model = torch.jit.script(int8_model)
    except Exception:
        scripted_model = torch.jit.trace(int8_model, dummy_input)
    scripted_model.save(model_out_path)

    graph_base = os.path.join(viz_dir, "TinyNet_Graph")
    graph_svg_path = f"{graph_base}.svg"
    try:
        drawer = CustomFxGraphDrawer(traced_model, graph_base)
        with open(graph_svg_path, "wb") as f:
            f.write(drawer.get_dot_graph().create_svg())
    except Exception as exc:
        _write_graph_summary_svg(graph_svg_path, traced_model, str(exc))
        print(f"Warning: graph generation fallback used ({exc})")

    print(f"Config: {config_path}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"INT8 model: {model_out_path}")
    print(f"module.py: {module_path}")
    print(f"Hex dir: {hex_data_dir}")
    print(f"Graph: {graph_svg_path}")


if __name__ == "__main__":
    main()
