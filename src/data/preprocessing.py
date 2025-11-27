import numpy as np
import scipy.signal as signal
import torch

def butter_bandpass(lowcut, highcut, fs, order=4):
    """设计巴特沃斯带通滤波器"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def filter_ppg(data, fs=32, low=0.5, high=4.0):
    """应用带通滤波 (双向滤波以消除相位偏移)"""
    b, a = butter_bandpass(low, high, fs)
    # axis=-1 保证对最后一个维度(时间)进行滤波
    y = signal.filtfilt(b, a, data, axis=-1)
    return y

def generate_spectrogram(sig, fs=32, n_fft=128, hop_length=32):
    """
    生成STFT频谱图，用于Belief-PPG分支。
    Input: (Time,)
    Output: (Frequency, Time)
    """
    f, t, Zxx = signal.stft(sig, fs=fs, nperseg=n_fft, noverlap=n_fft-hop_length)
    # 取幅值谱
    spec = np.abs(Zxx)
    # 标准化: 这一步对神经网络训练非常重要
    spec = (spec - spec.mean()) / (spec.std() + 1e-8)
    return spec

def generate_gaussian_label(hr_val, num_classes=180, min_hr=30, max_hr=210, sigma=1.5):
    """
    将标量心率转换为概率分布 (Label Smoothing)。
    这是Belief-PPG训练的关键：我们不教网络“是70”，而是教它“大概率在70附近”。
    """
    bins = np.linspace(min_hr, max_hr, num_classes)
    # 高斯分布公式
    dist = np.exp(-0.5 * ((bins - hr_val) / sigma) ** 2)
    # 归一化，使概率和为1
    dist = dist / dist.sum()
    return dist