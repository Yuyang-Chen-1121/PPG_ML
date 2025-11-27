import sys
import os
import torch
import hydra
import numpy as np
from omegaconf import DictConfig
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
from src.models import ArtifactAwareBeliefNetwork
from src.data.dataset import PPGDataset
from src.utils.dsp import get_heart_rate_from_distribution

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def evaluate(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载模型
    model = ArtifactAwareBeliefNetwork().to(device)
    # 假设加载最新的 checkpoint
    ckpt_path = f"{cfg.training.save_dir}/model_epoch_{cfg.training.epochs}.pth"
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path))
        print(f"Loaded model from {ckpt_path}")
    else:
        print("Warning: No checkpoint found, evaluating initialized model.")
    
    model.eval()
    
    # 2. 数据加载 (Mock or Real)
    test_dataset = PPGDataset(data_path=None, config=cfg, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    mae_list =
    
    # 3. 推理循环
    print("Starting evaluation...")
    bins = np.linspace(30, 210, 180) # 心率刻度
    
    with torch.no_grad():
        for batch in test_loader:
            raw = batch['raw_ppg'].to(device)
            spec = batch['spectrogram'].to(device)
            
            # 获取真实 HR (从分布标签中还原，或者 Dataset 应该直接返回标量 label)
            # 这里简化处理：假设 label_dist 的最大值对应真实 HR
            target_dist = batch['label_dist'].cpu().numpy()
            true_hr = get_heart_rate_from_distribution(target_dist, bins, method='argmax')
            
            # 模型预测
            out = model(raw, spec)
            pred_dist = torch.exp(out['hr_refined']).cpu().numpy().squeeze(1) # (Batch, Classes)
            
            # 计算 HR
            pred_hr = get_heart_rate_from_distribution(pred_dist, bins, method='expectation')
            
            # 记录误差
            error = np.abs(pred_hr - true_hr)
            mae_list.append(error)
            
    # 4. 统计结果
    mae_mean = np.mean(mae_list)
    print(f"Evaluation Complete.")
    print(f"Mean Absolute Error (MAE): {mae_mean:.4f} BPM")

if __name__ == "__main__":
    evaluate()