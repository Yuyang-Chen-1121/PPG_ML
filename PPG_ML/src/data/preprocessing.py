import numpy as np
from scipy import signal
import librosa

# ============================================================
# 1. PPG 滤波（安全带保护）
# ============================================================

def filter_ppg(data, fs=64):
    """
    对 PPG 做 0.9–5 Hz 带通滤波，内部做短序列保护。
    data: 1D array, 原始 PPG
    """
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 1:
        data = data.reshape(-1)

    # 3 阶 Butterworth
    b, a = signal.butter(
        N=3,
        Wn=[0.9 / (fs / 2.0), 5.0 / (fs / 2.0)],
        btype="band"
    )

    # 最小长度保护（经验值 + padlen 预估）
    MIN_LEN = 3 * max(len(a), len(b)) + 1
    if len(data) < MIN_LEN:
        return data

    try:
        y = signal.filtfilt(b, a, data, axis=-1)
    except ValueError:
        # 再保险：scipy 仍然嫌短就原样返回
        y = data

    return np.asarray(y, dtype=np.float32)


# ============================================================
# 2. 滑窗切片（PPG / z-score 共用）
# ============================================================

def slice_windows(x, win, stride):
    """
    将 1D 序列切成滑动窗口。
    返回:
      windows: (N, win)
      centers: (N,) 每个窗口中心样本索引
    """
    x = np.asarray(x)
    if x.ndim != 1:
        x = x.reshape(-1)

    total = len(x)
    if total < win:
        return np.zeros((0, win), dtype=x.dtype), np.zeros((0,), dtype=int)

    windows = []
    centers = []

    for start in range(0, total - win + 1, stride):
        end = start + win
        windows.append(x[start:end])
        centers.append(start + win // 2)

    return np.stack(windows, axis=0), np.asarray(centers, dtype=int)


# ============================================================
# 3. ACC → motion mask（在 PPG 时间轴上对齐）
# ============================================================

def generate_acc_mask(
    ppg_len,
    acc_raw,
    fs_ppg=64,
    fs_acc=32,
    win=256,
    stride=64,
    z_thresh=2.5,
    ma_window_sec=5.0,
):
    """
    基于 ACC 生成 motion mask，并映射到 PPG 采样域。
    步骤：
      1. ACC 取模并做移动平均
      2. 对 ACC 序列做 z-score
      3. 上采样到 PPG 采样率（简单重复）
      4. 在 PPG 采样域上滑窗，得到 (N,1,win) 的二值 mask
    """
    acc_raw = np.asarray(acc_raw, dtype=np.float32)
    assert acc_raw.ndim == 2 and acc_raw.shape[1] == 3, f"ACC shape should be (T,3), got {acc_raw.shape}"

    # 1. ACC magnitude
    acc_mag = np.sqrt(np.sum(acc_raw**2, axis=-1))  # (T_acc,)

    # 2. 移动平均平滑
    ma_window = int(ma_window_sec * fs_acc)
    if ma_window < 1:
        ma_window = 1
    kernel = np.ones(ma_window, dtype=np.float32) / ma_window
    smooth = np.convolve(acc_mag, kernel, mode="same")

    # 3. z-score (ACC 采样域)
    z_acc = (smooth - smooth.mean()) / (smooth.std() + 1e-8)  # (T_acc,)

    # 4. 上采样到 PPG 采样域
    factor = fs_ppg / float(fs_acc)  # 128 / 64 = 2
    idx = (np.arange(ppg_len) / factor).astype(int)
    idx = np.clip(idx, 0, len(z_acc) - 1)
    z_ppg = z_acc[idx]  # (T_ppg,)

    # 5. 在 PPG 采样域滑窗
    z_win, _ = slice_windows(z_ppg, win, stride)  # (N, win)
    if z_win.shape[0] == 0:
        return np.zeros((0, 1, win), dtype=np.float32)

    mask = (np.abs(z_win) < z_thresh).astype(np.float32)  # (N, win)
    return mask[:, None, :]  # (N,1,win)


# ============================================================
# 4. STFT 频谱（Belief-PPG 风格）
# ============================================================

def generate_stft(sig, fs=64, n_fft=128, hop_length=32):
    """
    生成 STFT 线性频谱，输出形状：(1, F, T)
    """
    sig = np.asarray(sig, dtype=np.float32)
    if len(sig) < n_fft:
        pad_len = n_fft - len(sig)
        sig = np.pad(sig, (0, pad_len), mode="constant")

    f, t, Zxx = signal.stft(
        sig,
        fs=fs,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
    )

    spec = np.abs(Zxx)
    spec = (spec - spec.mean()) / (spec.std() + 1e-8 + 1e-12)
    return spec[np.newaxis, :, :]  # (1,F,T)


# ============================================================
# 5. Mel 频谱（Tiny-PPG 风格）
# ============================================================

def generate_mel(sig, fs=64, n_fft=128, hop_length=32, n_mels=32):
    """
    生成 Mel 频谱，输出形状：(1, n_mels, T)
    """
    sig = np.asarray(sig, dtype=np.float32)
    if len(sig) < n_fft:
        pad_len = n_fft - len(sig)
        sig = np.pad(sig, (0, pad_len), mode="constant")

    S = librosa.feature.melspectrogram(
        y=sig,
        sr=fs,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )

    S_db = librosa.power_to_db(S, ref=np.max)
    S_db = (S_db - S_db.mean()) / (S_db.std() + 1e-8 + 1e-12)

    return S_db[np.newaxis, :, :]  # (1,M,T)