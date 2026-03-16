# Author: Yuyang Chen
# School: Zhejiang University
# Last Modified: 2026.2.7
# Description: Sequential multi-stage FP32 training entrypoint for TinyNet.

"""Sequential TinyNet training entrypoint."""

from __future__ import annotations

import argparse
import os
import sys

import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from loss.loss import IntegratedLoss
from models.tinynet import build_model_from_config
from train.helper import initialize_weights, set_seed
from train.train import prepare_data_loaders, run_training_stage
from utils.config import load_config, get_device


# Purpose: Run three-stage decoupled training using centralized config.
# Inputs: config path.
# Outputs: none.
# Side effects: creates checkpoints and updates model weights.
def train_decoupled(config_path: str) -> None:
    cfg = load_config(config_path)

    set_seed(cfg.get_int("training.seed", 42))
    device = get_device(cfg)
    print(f"🚀 Runtime Device: {device}")
    train_loader, val_loader, sampler_cfg = prepare_data_loaders(cfg)
    model = build_model_from_config(cfg).to(device)
    initialize_weights(model)
    criterion = IntegratedLoss(cfg, device=str(device))

    ckpt_dir = cfg.get("paths.checkpoints_dir", "./checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    seq_cfg = cfg.get("training.sequential", {}) or {}
    start_stage = int(seq_cfg.get("start_stage", 1))
    base_lr = cfg.get_float("training.learning_rate", 1e-3)
    default_patience = cfg.get_int("training.early_stop_patience", 20)

    best_threshold = cfg.get_float("evaluation.default_af_threshold", 0.5)

    if start_stage <= 1:
        run_training_stage(
            stage_idx=1,
            epochs=int(seq_cfg.get("stage1_epochs", cfg.get_int("training.epochs", 100))),
            lr=base_lr,
            patience=default_patience,
            monitor_metric="score",
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            train_cfg=cfg.get("training", {}),
            config=cfg,
            sampler_cfg=sampler_cfg,
            criterion=criterion,
            device=device,
            ckpt_dir=ckpt_dir,
            mode="max",
            save_name="stage1_best.pth",
            prev_best_th=best_threshold,
        )

    stage1_path = os.path.join(ckpt_dir, "stage1_best.pth")
    if os.path.exists(stage1_path):
        checkpoint = torch.load(stage1_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        best_threshold = float(checkpoint.get("threshold", best_threshold))

    if start_stage <= 2:
        run_training_stage(
            stage_idx=2,
            epochs=int(seq_cfg.get("stage2_epochs", 80)),
            lr=base_lr * float(seq_cfg.get("stage2_lr_scale", 0.5)),
            patience=int(seq_cfg.get("stage2_patience", default_patience)),
            monitor_metric="af_gmean",
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            train_cfg=cfg.get("training", {}),
            config=cfg,
            sampler_cfg=sampler_cfg,
            criterion=criterion,
            device=device,
            ckpt_dir=ckpt_dir,
            mode="max",
            save_name="stage2_af_best.pth",
            prev_best_th=best_threshold,
        )

    stage2_path = os.path.join(ckpt_dir, "stage2_af_best.pth")
    if os.path.exists(stage2_path):
        checkpoint = torch.load(stage2_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        best_threshold = float(checkpoint.get("threshold", best_threshold))

    if start_stage <= 3:
        run_training_stage(
            stage_idx=3,
            epochs=int(seq_cfg.get("stage3_epochs", 80)),
            lr=base_lr * float(seq_cfg.get("stage3_lr_scale", 0.5)),
            patience=int(seq_cfg.get("stage3_patience", default_patience)),
            monitor_metric="hr_mae",
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            train_cfg=cfg.get("training", {}),
            config=cfg,
            sampler_cfg=sampler_cfg,
            criterion=criterion,
            device=device,
            ckpt_dir=ckpt_dir,
            mode="min",
            save_name="final_model.pth",
            prev_best_th=best_threshold,
        )

    print("Training complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/config.yaml")
    args = parser.parse_args()

    if os.path.exists(args.config):
        train_decoupled(args.config)
    else:
        print(f"Config not found: {args.config}")
