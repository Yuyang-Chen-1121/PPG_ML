import torch
from torch.utils.data import Dataset
import numpy as np
from.preprocessing import filter_ppg, generate_spectrogram, generate_gaussian_label

class PPGDataset(Dataset):
    def __init__(self, data_path, config, mode='train'):
        self.config = config
        self.mode = mode
        
        # 实际工程中，这里会加载 pickle/h5/mat 文件
        # 为了演示，我们初始化一个标志位，如果没有数据就生成随机数据
        self.use_mock = True if data_path is None else False
        
        if self.use_mock:
            print(f" Using Mock Data for {mode} mode. 训练结果将无意义，仅用于调试流程。")
            self.length = 1000
        else:
            # TODO: 实现真实数据的加载逻辑 (例如读取 self.data = np.load(data_path))
            pass

    def __len__(self):
        return self.length if self.use_mock else len(self.data)

    def __getitem__(self, idx):
        # 1. 获取原始信号片段
        if self.use_mock:
            # 模拟 30-210 BPM 的正弦波作为PPG信号，叠加噪声
            fs = self.config.dataset.fs
            win_len = self.config.dataset.window_size
            t = np.linspace(0, win_len/fs, win_len)
            
            true_hr = np.random.uniform(50, 150)
            clean_sig = np.sin(2 * np.pi * (true_hr/60) * t)
            noise = np.random.normal(0, 0.5, size=win_len)
            raw_ppg = clean_sig + noise
            
            # 模拟伪影 Mask (假设噪声大于某个阈值的地方是伪影)
            mask = (np.abs(noise) > 0.8).astype(np.float32)
        else:
            # TODO: 从 self.data 中切片
            pass

        # 2. 预处理
        # 滤波 (Belief分支需要干净一点的信号，Tiny分支通常鲁棒性强可以直接吃Raw)
        filtered_ppg = filter_ppg(raw_ppg, fs=self.config.dataset.fs)
        
        # 生成STFT频谱 (Belief分支输入)
        # 增加一个维度作为 Channel (1, F, T)
        spectrogram = generate_spectrogram(filtered_ppg, 
                                           fs=self.config.dataset.fs,
                                           n_fft=self.config.preprocessing.stft.n_fft,
                                           hop_length=self.config.preprocessing.stft.hop_length)
        spectrogram = spectrogram[np.newaxis,...] 

        # 生成概率分布标签 (Label)
        hr_dist = generate_gaussian_label(true_hr)

        # 3. 转为 Tensor
        return {
            "raw_ppg": torch.FloatTensor(raw_ppg).unsqueeze(0), # (1, Time)
            "spectrogram": torch.FloatTensor(spectrogram),      # (1, Freq, Time)
            "mask": torch.FloatTensor(mask).unsqueeze(0),       # (1, Time) - 伪影标签
            "label_dist": torch.FloatTensor(hr_dist)            # (180,) - 心率分布标签
        }