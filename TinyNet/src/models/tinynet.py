# Author: Yuyang Chen
# School: Zhejiang University
# Last Modified: 2026.2.7
# Description: TinyNet model architecture definition and builders.

"""TinyNet model definition constrained for CNN_Base V1.0."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.quantization import DeQuantStub, QuantStub

from utils.config import ConfigAccessor, get_bpm_bins, load_config


# Purpose: Round a positive integer up to a target multiple.
# Inputs: value integer, multiple integer.
# Outputs: rounded integer >= value.
# Assumptions: multiple > 0.
def _round_up_to_multiple(value: int, multiple: int) -> int:
    return int(math.ceil(value / float(multiple)) * multiple)


class CascadedAvgPoolToOne(nn.Module):
    # Purpose: Compress temporal length 160 -> 1 using fixed three-stage average pooling.
    # Inputs: Tensor (N, C, 160).
    # Outputs: Tensor (N, C, 1).
    # Assumptions: input temporal length is exactly 160.

    # Purpose: Initialize fixed hardware-safe pooling layers.
    # Inputs: none.
    # Outputs: initialized module.
    # Assumptions: pool kernels/strides follow hardware limits.
    def __init__(self) -> None:
        super().__init__()
        self.pool1 = nn.AvgPool1d(kernel_size=8, stride=7, padding=0)
        self.pool2 = nn.AvgPool1d(kernel_size=8, stride=7, padding=0)
        self.pool3 = nn.AvgPool1d(kernel_size=3, stride=1, padding=0)

    # Purpose: Apply the fixed pooling cascade and validate final length.
    # Inputs: x tensor with shape (N, C, 160).
    # Outputs: pooled tensor with shape (N, C, 1).
    # Assumptions: caller passes features with length 160.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pool1(x)
        out = self.pool2(out)
        out = self.pool3(out)
        if out.size(-1) != 1:
            raise RuntimeError(f"Cascaded pooling expected length 1, got {out.size(-1)}")
        return out


class SqueezeExcitationBlock(nn.Module):
    # Purpose: Channel reweighting block using hardware-safe squeeze/excitation operators.
    # Inputs: Tensor (N, C, 160).
    # Outputs: Tensor (N, C, 160).
    # Assumptions: C is divisible by 16 and input length is 160.

    # Purpose: Initialize squeeze pooling and excitation MLP.
    # Inputs: channels integer, reduction_divisor integer.
    # Outputs: initialized SE block.
    # Assumptions: reduced channel count remains divisible by 16.
    def __init__(self, channels: int, reduction_divisor: int = 4) -> None:
        super().__init__()
        reduced = _round_up_to_multiple(max(1, channels // reduction_divisor), 16)
        reduced = min(reduced, channels)
        self.squeeze = CascadedAvgPoolToOne()
        self.fc1 = nn.Linear(channels, reduced)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(reduced, channels)
        self.sigmoid = nn.Sigmoid()

    # Purpose: Compute SE gates and apply broadcast multiplication to feature map.
    # Inputs: x tensor (N, C, 160).
    # Outputs: scaled tensor (N, C, 160).
    # Assumptions: temporal squeeze returns (N, C, 1).
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.squeeze(x).squeeze(-1)
        z = self.fc1(z)
        z = self.relu(z)
        z = self.fc2(z)
        z = self.sigmoid(z).unsqueeze(-1)
        return x * z


class ResBlock(nn.Module):
    # Purpose: Standard residual block with hardware-safe Conv1d kernels and dropout regularization.
    # Inputs: Tensor (N, C_in, L).
    # Outputs: Tensor (N, C_out, L_out).
    # Assumptions: kernel size in {3, 5, 7}; stride in {1, 2}.

    # Purpose: Build residual branch, projection branch, and quantization-safe add op.
    # Inputs: channel sizes, kernel size, stride, dropout probability.
    # Outputs: initialized block.
    # Assumptions: channel counts are hardware-compliant multiples of 16.
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dropout_p: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_p)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        self.skip_add = nn.quantized.FloatFunctional()

    # Purpose: Execute residual forward path with dropout after first ReLU.
    # Inputs: x tensor (N, C_in, L).
    # Outputs: residual output tensor (N, C_out, L_out).
    # Assumptions: downsample branch is configured when dimensions differ.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.skip_add.add(out, identity)
        out = self.relu(out)
        return out


class TinyNet(nn.Module):
    # Purpose: Multi-task TinyNet with HR energy-integration head and AF dual-stream gated branch.
    # Inputs: config dictionary or ConfigAccessor.
    # Outputs: model instance that returns AF and HR logits.
    # Assumptions: configuration follows CNN_Base V1.0 channel and length limits.

    # Purpose: Initialize all backbone blocks, heads, and quantization stubs.
    # Inputs: config object.
    # Outputs: initialized model.
    # Assumptions: input length is 320 so branch output length is 160 before GAP heads.
    def __init__(self, config: Optional[Any] = None) -> None:
        super().__init__()
        cfg = config if isinstance(config, ConfigAccessor) else ConfigAccessor(config or {})

        input_channels = cfg.get_int("data.input_channels", 16)
        stem_channels = cfg.get_int("model.stem_channels", 32)
        af_stream_channels = cfg.get_int("model.af_stream_channels", 64)
        af_head_channels = cfg.get_int("model.af_head_conv_channels", 128)
        hr_channels = cfg.get_int("model.hr_channels", 48)
        hr_kernels = cfg.get_list("model.hr_kernels", [7, 7, 7])
        hr_strides = cfg.get_list("model.hr_strides", [2, 1, 1])
        af_spatial_kernels = cfg.get_list("model.af_spatial_kernels", [3, 3, 3])
        af_spatial_strides = cfg.get_list("model.af_spatial_strides", [2, 1, 1])
        af_temporal_pool_kernels = cfg.get_list("model.af_temporal_pool_kernels", [5, 4])
        af_temporal_pool_strides = cfg.get_list("model.af_temporal_pool_strides", [5, 4])
        af_temporal_kernels = cfg.get_list("model.af_temporal_kernels", [7, 7])
        af_temporal_gate_kernel = cfg.get_int("model.af_temporal_gate_kernel", 3)
        af_temporal_global_pool_kernel = cfg.get_int("model.af_temporal_global_pool_kernel", 16)
        af_temporal_global_pool_stride = cfg.get_int("model.af_temporal_global_pool_stride", 1)
        reduction_divisor = cfg.get_int("model.se_reduction_divisor", 4)
        resblock_dropout = cfg.get_float("model.resblock_dropout", 0.5)
        af_head_dropout = cfg.get_float("model.af_head_dropout", 0.5)
        hr_head_dropout = cfg.get_float("model.hr_head_dropout", 0.5)
        hr_head_channels = cfg.get_int("model.hr_head_conv_channels", 128)
        af_output_dim = cfg.get_int("model.af_output_dim", 1)

        bpm_bins = get_bpm_bins(cfg)
        self.hr_bins = int(bpm_bins.shape[0])

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, stem_channels, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(stem_channels),
            nn.ReLU(inplace=True),
        )

        self.hr_block1 = ResBlock(stem_channels, hr_channels, int(hr_kernels[0]), int(hr_strides[0]), resblock_dropout)
        self.hr_block2 = ResBlock(hr_channels, hr_channels, int(hr_kernels[1]), int(hr_strides[1]), resblock_dropout)
        self.hr_block3 = ResBlock(hr_channels, hr_channels, int(hr_kernels[2]), int(hr_strides[2]), resblock_dropout)
        self.hr_head_conv = nn.Conv1d(hr_channels, hr_head_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.hr_gap = CascadedAvgPoolToOne()
        self.hr_dropout = nn.Dropout(p=hr_head_dropout)
        self.hr_fc = nn.Linear(hr_head_channels, self.hr_bins)

        self.af_spatial_block1 = ResBlock(
            stem_channels,
            af_stream_channels,
            int(af_spatial_kernels[0]),
            int(af_spatial_strides[0]),
            resblock_dropout,
        )
        self.af_spatial_block2 = ResBlock(
            af_stream_channels,
            af_stream_channels,
            int(af_spatial_kernels[1]),
            int(af_spatial_strides[1]),
            resblock_dropout,
        )
        self.af_spatial_block3 = ResBlock(
            af_stream_channels,
            af_stream_channels,
            int(af_spatial_kernels[2]),
            int(af_spatial_strides[2]),
            resblock_dropout,
        )
        self.af_spatial_bn = nn.BatchNorm1d(af_stream_channels)

        self.af_temporal_pool1 = nn.AvgPool1d(
            kernel_size=int(af_temporal_pool_kernels[0]),
            stride=int(af_temporal_pool_strides[0]),
            padding=0,
        )
        self.af_temporal_pool2 = nn.AvgPool1d(
            kernel_size=int(af_temporal_pool_kernels[1]),
            stride=int(af_temporal_pool_strides[1]),
            padding=0,
        )
        self.af_temporal_block1 = ResBlock(
            stem_channels,
            af_stream_channels,
            int(af_temporal_kernels[0]),
            1,
            resblock_dropout,
        )
        self.af_temporal_block2 = ResBlock(
            af_stream_channels,
            af_stream_channels,
            int(af_temporal_kernels[1]),
            1,
            resblock_dropout,
        )
        self.af_temporal_data_conv = nn.Conv1d(
            af_stream_channels,
            af_stream_channels,
            kernel_size=int(af_temporal_gate_kernel),
            stride=1,
            padding=int((af_temporal_gate_kernel - 1) // 2),
            bias=False,
        )
        self.af_temporal_data_bn = nn.BatchNorm1d(af_stream_channels)
        self.af_temporal_data_relu = nn.ReLU(inplace=True)
        self.af_temporal_gate_conv = nn.Conv1d(
            af_stream_channels,
            af_stream_channels,
            kernel_size=int(af_temporal_gate_kernel),
            stride=1,
            padding=int((af_temporal_gate_kernel - 1) // 2),
            bias=False,
        )
        self.af_temporal_gate_bn = nn.BatchNorm1d(af_stream_channels)
        self.af_temporal_gate_sigmoid = nn.Sigmoid()
        self.af_temporal_gate_mul = nn.quantized.FloatFunctional()
        self.af_temporal_global_pool = nn.AvgPool1d(
            kernel_size=int(af_temporal_global_pool_kernel),
            stride=int(af_temporal_global_pool_stride),
            padding=0,
        )

        self.af_fusion_add = nn.quantized.FloatFunctional()
        self.af_fusion_se = SqueezeExcitationBlock(af_stream_channels, reduction_divisor=reduction_divisor)
        self.af_head_conv = nn.Conv1d(af_stream_channels, af_head_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.af_head_bn = nn.BatchNorm1d(af_head_channels)
        self.af_head_relu = nn.ReLU(inplace=True)
        self.af_gap = CascadedAvgPoolToOne()
        self.af_dropout = nn.Dropout(p=af_head_dropout)
        self.af_fc = nn.Linear(af_head_channels, af_output_dim)

        self._validate_hardware_constraints(cfg)

    # Purpose: Validate channel and class dimensions against hardware limits.
    # Inputs: cfg accessor.
    # Outputs: none.
    # Side effects: raises ValueError when configuration is invalid.
    def _validate_hardware_constraints(self, cfg: ConfigAccessor) -> None:
        max_channels = cfg.get_int("hardware.max_channels", 8192)
        channel_multiple = cfg.get_int("hardware.channel_multiple", 16)
        softmax_limit = cfg.get_int("hardware.softmax_max_features", 128)

        channel_values = [
            cfg.get_int("data.input_channels", 16),
            cfg.get_int("model.stem_channels", 32),
            cfg.get_int("model.hr_channels", 48),
            cfg.get_int("model.af_stream_channels", 64),
            cfg.get_int("model.hr_head_conv_channels", 128),
            cfg.get_int("model.af_head_conv_channels", 128),
        ]
        for c in channel_values:
            if c > max_channels or (c % channel_multiple) != 0:
                raise ValueError(f"Channel count {c} violates hardware constraints")

        if self.hr_bins > softmax_limit:
            raise ValueError(f"HR bins {self.hr_bins} exceed hardware softmax limit {softmax_limit}")

    # Purpose: Execute forward pass for AF and HR heads.
    # Inputs: x tensor (N, C=16, L=320).
    # Outputs: tuple(af_logits(N,1), hr_logits(N,106)).
    # Assumptions: stem output and branch lengths remain hardware-safe.
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.quant(x)
        x = self.stem(x)

        x_hr = self.hr_block1(x)
        x_hr = self.hr_block2(x_hr)
        x_hr = self.hr_block3(x_hr)
        x_hr = self.hr_head_conv(x_hr)
        x_hr = self.hr_gap(x_hr).squeeze(-1)
        x_hr = self.hr_dropout(x_hr)
        out_hr = self.hr_fc(x_hr)

        x_spatial = self.af_spatial_block1(x)
        x_spatial = self.af_spatial_block2(x_spatial)
        x_spatial = self.af_spatial_block3(x_spatial)
        x_spatial = self.af_spatial_bn(x_spatial)

        x_temporal = self.af_temporal_pool1(x)
        x_temporal = self.af_temporal_pool2(x_temporal)
        x_temporal = self.af_temporal_block1(x_temporal)
        x_temporal = self.af_temporal_block2(x_temporal)
        x_data = self.af_temporal_data_conv(x_temporal)
        x_data = self.af_temporal_data_bn(x_data)
        x_data = self.af_temporal_data_relu(x_data)
        x_gate = self.af_temporal_gate_conv(x_temporal)
        x_gate = self.af_temporal_gate_bn(x_gate)
        x_gate = self.af_temporal_gate_sigmoid(x_gate)
        x_temporal = self.af_temporal_gate_mul.mul(x_data, x_gate)
        x_temporal = self.af_temporal_global_pool(x_temporal)

        x_af = self.af_fusion_add.add(x_spatial, x_temporal)
        x_af = self.af_fusion_se(x_af)
        x_af = self.af_head_conv(x_af)
        x_af = self.af_head_bn(x_af)
        x_af = self.af_head_relu(x_af)
        x_af = self.af_gap(x_af).squeeze(-1)
        x_af = self.af_dropout(x_af)
        out_af = self.af_fc(x_af)

        return self.dequant(out_af), self.dequant(out_hr)


# Purpose: Build TinyNet from config dict/accessor/path-loaded payload.
# Inputs: config dictionary or ConfigAccessor.
# Outputs: TinyNet instance.
# Assumptions: caller has already loaded YAML when passing dictionary.
def build_model_from_config(config_input: Any) -> TinyNet:
    if isinstance(config_input, str):
        cfg = load_config(config_input)
    else:
        cfg = config_input if isinstance(config_input, ConfigAccessor) else ConfigAccessor(config_input or {})
    return TinyNet(cfg)
