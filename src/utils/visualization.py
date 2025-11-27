import matplotlib.pyplot as plt
import numpy as np

def plot_spectrogram(spec, title="Spectrogram"):
    """绘制 STFT 频谱图"""
    plt.figure(figsize=(10, 4))
    plt.imshow(spec, aspect='auto', origin='lower', cmap='inferno')
    plt.title(title)
    plt.colorbar(format='%+2.0f dB')
    plt.ylabel('Frequency Bin')
    plt.xlabel('Time Frame')
    plt.tight_layout()
    plt.close()

def plot_uncertainty_heatmap(probs, true_hr, bins, title="HR Probability Distribution"):
    """
    绘制随时间变化的心率概率热图 (BeliefPPG 的标志性可视化)
    probs: (Time, Classes)
    true_hr: (Time,) 真实心率曲线
    """
    plt.figure(figsize=(12, 6))
    
    # 绘制热图
    plt.imshow(probs.T, aspect='auto', origin='lower', 
               extent=[0, probs.shape, bins, bins[-1]], 
               cmap='viridis', interpolation='nearest')
    
    # 叠加真实心率曲线
    plt.plot(true_hr, color='red', linewidth=2, label='True HR', linestyle='--')
    
    plt.colorbar(label='Probability')
    plt.title(title)
    plt.xlabel('Time Window')
    plt.ylabel('Heart Rate (BPM)')
    plt.legend()
    plt.tight_layout()
    # 返回 figure 对象以便保存
    return plt.gcf()