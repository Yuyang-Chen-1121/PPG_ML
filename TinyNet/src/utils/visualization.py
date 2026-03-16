# Author: Yuyang Chen
# School: Zhejiang University
# Last Modified: 2026.2.7
# Description: Training metric visualization helper classes and plotting utilities.

# src/utils/visualization.py
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class TrainingVisualizer:
    # Purpose: Initialize class state and runtime configuration.
    # Inputs: Parameters defined in `__init__` signature.
    # Outputs: Return value produced by `__init__`.
    # Assumptions: Caller provides valid types/shapes for this operation.
    def __init__(self, save_dir):
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        self.log_csv_path = os.path.join(self.save_dir, "training_log.csv")
        self.data = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'val_rmse': [],
            'val_af_gmean': [],
            'learning_rate': []
        }

    # Purpose: Implement `log` for the TinyNet workflow.
    # Inputs: Parameters defined in `log` signature.
    # Outputs: Return value produced by `log`.
    # Assumptions: Caller provides valid types/shapes for this operation.
    def log(self, epoch, train_loss, val_loss, val_mae, val_rmse, val_af_gmean, lr):
        """记录一个 Epoch 的数据"""
        self.data['epoch'].append(epoch)
        self.data['train_loss'].append(train_loss)
        self.data['val_loss'].append(val_loss)
        self.data['val_mae'].append(val_mae)
        self.data['val_rmse'].append(val_rmse)
        self.data['val_af_gmean'].append(val_af_gmean)
        self.data['learning_rate'].append(lr)
        
        # 实时保存到 CSV 防止中断丢失
        df = pd.DataFrame(self.data)
        df.to_csv(self.log_csv_path, index=False)

    # Purpose: Render and save the requested visualization artifact.
    # Inputs: Parameters defined in `plot_curves` signature.
    # Outputs: Return value produced by `plot_curves`.
    # Assumptions: Caller provides valid types/shapes for this operation.
    def plot_curves(self):
        """绘制并保存曲线图"""
        epochs = self.data['epoch']
        
        # 创建一个 2x2 的画布
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Metrics Visualization', fontsize=16)

        # 1. Loss Curve
        axs[0, 0].plot(epochs, self.data['train_loss'], label='Train Loss', color='blue')
        axs[0, 0].plot(epochs, self.data['val_loss'], label='Val Loss', color='orange', linestyle='--')
        axs[0, 0].set_title('Loss Curve')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].legend()
        axs[0, 0].grid(True, alpha=0.3)

        # 2. HR Error (MAE vs RMSE)
        axs[0, 1].plot(epochs, self.data['val_mae'], label='HR MAE', color='green')
        axs[0, 1].plot(epochs, self.data['val_rmse'], label='HR RMSE (Outliers)', color='red', linestyle=':')
        axs[0, 1].set_title('Heart Rate Estimation Error')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('BPM Error')
        axs[0, 1].legend()
        axs[0, 1].grid(True, alpha=0.3)

        # 3. AF Performance
        axs[1, 0].plot(epochs, self.data['val_af_gmean'], label='AF G-Mean', color='purple')
        axs[1, 0].set_title('AF Detection Performance (G-Mean)')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('Score (0-1)')
        axs[1, 0].legend()
        axs[1, 0].grid(True, alpha=0.3)

        # 4. Learning Rate
        axs[1, 1].plot(epochs, self.data['learning_rate'], label='Learning Rate', color='brown')
        axs[1, 1].set_title('Learning Rate Decay')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].set_ylabel('LR')
        axs[1, 1].set_yscale('log') # LR 通常用对数坐标看
        axs[1, 1].legend()
        axs[1, 1].grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(self.save_dir, "metrics_plot.png"), dpi=300)
        plt.close()