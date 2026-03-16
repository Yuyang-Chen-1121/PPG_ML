# Author: Yuyang Chen
# School: Zhejiang University
# Last Modified: 2026.2.7
# Description: Entrypoint for evaluating TinyNet FP32 checkpoints.

# scripts/evaluate_decoupled.py
# scripts/evaluate_decoupled.py
import os
import sys
import argparse

# --- Path Setup ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) 
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from evaluate.evaluate import evaluate_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./config/config.yaml")
    args = parser.parse_args()
    
    if os.path.exists(args.config):
        evaluate_model(args.config)
    else:
        print(f"❌ Config not found: {args.config}")