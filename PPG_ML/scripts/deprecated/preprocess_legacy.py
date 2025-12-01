import sys
import os
import glob
import numpy as np
import pickle
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

# 确保能导入 src
sys.path.append(os.getcwd())
from src.data.preprocessing import filter_ppg, generate_spectrogram

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def preprocess(cfg: DictConfig):
    """
    读取 Raw 数据 -> 滤波/STFT -> 保存为 Processed (.pt 或.npy)
    """
    raw_dir = cfg.dataset.data_root
    save_dir = os.path.join(os.path.dirname(raw_dir), "processed")
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Processing data from {raw_dir} to {save_dir}...")
    
    # 假设数据是.pkl 格式 (PPG-DaLiA 常见格式)
    files = glob.glob(os.path.join(raw_dir, "*.pkl"))
    
    if not files:
        print("No.pkl files found. Creating mock processed data for demonstration.")
        # 创建假数据用于测试
        for i in range(5):
            mock_data = {
                'ppg': np.random.randn(256 * 100),
                'acc': np.random.randn(256 * 100, 3),
                'hr': np.random.randint(60, 100, size=(100,))
            }
            with open(os.path.join(save_dir, f"subject_{i}.pkl"), 'wb') as f:
                pickle.dump(mock_data, f)
        return

    for fpath in tqdm(files):
        with open(fpath, 'rb') as f:
            data = pickle.load(f)
            
        # 1. 滤波
        ppg_clean = filter_ppg(data['ppg'], fs=cfg.dataset.fs)
        
        # 2. 生成频谱 (预计算以节省训练时间)
        spec = generate_spectrogram(ppg_clean, fs=cfg.dataset.fs)
        
        # 3. 保存
        processed_data = {
            'ppg': ppg_clean,
            'spectrogram': spec,
            'label': data['label']
        }
        
        fname = os.path.basename(fpath)
        with open(os.path.join(save_dir, fname), 'wb') as out_f:
            pickle.dump(processed_data, out_f)

if __name__ == "__main__":
    preprocess()