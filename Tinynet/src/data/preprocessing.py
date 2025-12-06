import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import gaussian_filter1d

#基础信号处理
def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.filtfilt(b, a, data, axis=0)
    return y

def z_score_normalization(data):
    mean = np.mean(data)
    std = np.std(data)
    if std < 1e-6: return data - mean
    return (data - mean) / std

#特征工程
def compute_envelope_diff(signal_data, window_size=15):
    s = pd.Series(signal_data)
    upper = s.rolling(window=window_size, center=True).max().bfill().ffill()
    lower = s.rolling(window=window_size, center=True).min().bfill().ffill()
    env_diff = (upper - lower).values
    return z_score_normalization(env_diff)

def compute_derivative(signal_data):
    diff = np.diff(signal_data)
    diff = np.insert(diff, 0, 0)
    return z_score_normalization(diff)

def pack_multimodal_channels(ppg, acc_main, acc_aux1, acc_aux2, target_len=320):
    ppg_delta = compute_derivative(ppg)
    ppg_env = compute_envelope_diff(ppg)
    
    X = np.zeros((target_len, 16), dtype=np.float32)
    
    X[:, 0] = ppg
    X[:, 1] = acc_main  # Ch1: ACC Magnitude
    X[:, 2] = acc_aux1  # Ch2: Zero
    X[:, 3] = acc_aux2  # Ch3: Zero
    X[:, 4] = ppg_delta # Ch4: Delta
    X[:, 5] = ppg_env   # Ch5: Envelope
    # Ch 6-15: Zero
    
    return X

def generate_distribution_label(bpm_val, num_classes=128, min_bpm=30, max_bpm=210, sigma=2.0):
    if np.isnan(bpm_val) or bpm_val <= 0:
        return np.zeros(num_classes)
    bins = np.linspace(min_bpm, max_bpm, num_classes)
    target_idx = np.argmin(np.abs(bins - bpm_val))
    label = np.zeros(num_classes)
    label[target_idx] = 1.0
    label = gaussian_filter1d(label, sigma=sigma)
    if np.sum(label) > 0: label = label / np.sum(label)
    return label

def sliding_window_multimodal(ppg, acc_3ch, labels, window_size, step_size, label_fs_ratio):
    n_samples = len(ppg)
    X_list = []
    y_list = []
    
    for i in range(0, n_samples - window_size, step_size):
        win_ppg = ppg[i : i + window_size]
        win_acc = acc_3ch[i : i + window_size, :] 
        
        # 传入 (Mag, 0, 0)
        X_win = pack_multimodal_channels(
            win_ppg, 
            win_acc[:, 0], # Mag
            win_acc[:, 1], # 0
            win_acc[:, 2], # 0
            target_len=window_size
        )
        
        center_idx = i + window_size // 2
        label_idx = int(center_idx / label_fs_ratio)
        
        if label_idx < len(labels):
            win_label = labels[label_idx]
            X_list.append(X_win)
            y_list.append(win_label)
            
    return np.array(X_list), np.array(y_list)