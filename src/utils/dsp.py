import numpy as np
import scipy.signal as signal

def get_heart_rate_from_distribution(probs, bins, method='expectation'):
    """
    从概率分布中解码心率值。
    Args:
        probs: (B, C) 概率分布
        bins: (C,) 心率刻度 (30, 31,..., 210)
        method: 'argmax' 或 'expectation' (加权平均，精度更高)
    """
    if method == 'argmax':
        idx = np.argmax(probs, axis=1)
        return bins[idx]
    elif method == 'expectation':
        # E[x] = sum(x * p(x))
        return np.sum(probs * bins, axis=1)
    else:
        raise ValueError("Unknown method")

def calculate_snr(clean, noise):
    """计算信噪比 (dB)"""
    p_signal = np.sum(clean ** 2)
    p_noise = np.sum(noise ** 2)
    return 10 * np.log10(p_signal / (p_noise + 1e-10))

def peak_detection(signal_1d, fs):
    """简单的峰值检测封装"""
    distance = int(0.5 * fs) # 假设最大心率不超过 120bpm -> 0.5s 间隔
    peaks, _ = signal.find_peaks(signal_1d, distance=distance)
    return peaks