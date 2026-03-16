# Author: Yuyang Chen
# School: Zhejiang University
# Last Modified: 2026.2.7
# Description: Integrated AF and HR multi-task loss definitions.

"""Integrated multi-task losses for TinyNet."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import ConfigAccessor, get_bpm_bins


class IntegratedLoss(nn.Module):
    # Purpose: Compute AF BCE loss and HR distribution loss with stage-compatible masks.
    # Inputs: config dictionary/accessor, device string.
    # Outputs: loss module instance.
    # Assumptions: HR labels are 106-dim distributions and AF labels are binary.

    # Purpose: Initialize loss weights and class-specific criteria.
    # Inputs: config object, device identifier.
    # Outputs: initialized loss module.
    # Assumptions: af_pos_weight is non-negative.
    def __init__(self, config: Any, device: str = "cuda") -> None:
        super().__init__()
        self.cfg = config if isinstance(config, ConfigAccessor) else ConfigAccessor(config or {})
        self.device = device

        self.hr_smoothing = self.cfg.get_float("loss.hr_label_smoothing", 0.1)

        pos_weight = torch.tensor(self.cfg.get_float("loss.af_pos_weight", 1.0), device=device)
        self.af_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

        self.log_vars = nn.Parameter(torch.zeros(2))

        self.bpm_bins = torch.tensor(get_bpm_bins(self.cfg), dtype=torch.float32, device=device)

    # Purpose: Compute masked AF BCE loss from logits and one-hot/binary labels.
    # Inputs: af_logits (N,1), target_af (N,2 or N), mask_af (N).
    # Outputs: scalar AF loss tensor.
    # Assumptions: AF positive class is index 1 for one-hot labels.
    def compute_af_loss(self, af_logits: torch.Tensor, target_af: torch.Tensor, mask_af: torch.Tensor) -> torch.Tensor:
        if torch.sum(mask_af) <= 0:
            return torch.tensor(0.0, device=af_logits.device)

        if target_af.dim() > 1 and target_af.size(1) >= 2:
            y_target = target_af[:, 1].float().view(-1, 1)
        else:
            y_target = target_af.float().view(-1, 1)

        raw_loss = self.af_criterion(af_logits, y_target).view(-1)
        mask = mask_af.view(-1)
        return torch.sum(raw_loss * mask) / (torch.sum(mask) + 1e-8)

    # Purpose: Compute masked HR distribution cross-entropy with optional label smoothing.
    # Inputs: hr_logits (N,106), target_hr (N,106), mask_hr (N).
    # Outputs: tuple(loss_tensor, metric_dict).
    # Assumptions: target_hr rows are valid probability distributions.
    def compute_hr_loss(
        self,
        hr_logits: torch.Tensor,
        target_hr: torch.Tensor,
        mask_hr: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if torch.sum(mask_hr) <= 0:
            zero = torch.tensor(0.0, device=hr_logits.device)
            return zero, {"hr_ce": 0.0}

        targets = target_hr.float()
        if self.hr_smoothing > 0.0:
            num_classes = targets.size(1)
            uniform = 1.0 / float(num_classes)
            targets = targets * (1.0 - self.hr_smoothing) + uniform * self.hr_smoothing

        log_probs = F.log_softmax(hr_logits, dim=1)
        ce = -torch.sum(targets * log_probs, dim=1)
        mask = mask_hr.view(-1)
        final = torch.sum(ce * mask) / (torch.sum(mask) + 1e-8)
        return final, {"hr_ce": float(final.detach().cpu())}

    # Purpose: Compute total weighted loss for joint training.
    # Inputs: AF/HR predictions, labels, and task masks.
    # Outputs: tuple(total_loss, metric_dict).
    # Assumptions: masks identify task-valid samples.
    def forward(
        self,
        pred_af: torch.Tensor,
        pred_hr: torch.Tensor,
        target_af: torch.Tensor,
        target_hr: torch.Tensor,
        mask_af: torch.Tensor,
        mask_hr: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_loss = torch.tensor(0.0, device=pred_af.device)
        metrics: Dict[str, float] = {}

        if torch.sum(mask_af) > 0:
            af_loss = self.compute_af_loss(pred_af, target_af, mask_af)
            af_precision = torch.exp(-self.log_vars[0])
            af_weighted = af_precision * af_loss + 0.5 * self.log_vars[0]
            total_loss = total_loss + af_weighted
            metrics["loss_af"] = float(af_loss.detach().cpu())

        if torch.sum(mask_hr) > 0:
            hr_loss, hr_stats = self.compute_hr_loss(pred_hr, target_hr, mask_hr)
            hr_precision = torch.exp(-self.log_vars[1])
            hr_weighted = hr_precision * hr_loss + 0.5 * self.log_vars[1]
            total_loss = total_loss + hr_weighted
            metrics.update(hr_stats)
            metrics["loss_hr"] = float(hr_loss.detach().cpu())

        metrics["loss_total"] = float(total_loss.detach().cpu())
        return total_loss, metrics
