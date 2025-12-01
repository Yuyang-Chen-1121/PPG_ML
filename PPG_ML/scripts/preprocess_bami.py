"""
BAMI-1 & BAMI-2 preprocessing
与当前 DaLiA 预处理 pipeline 对齐：
- 统一采样率：64 Hz
- 窗口长度：256 samples (4 秒)
- 滑动步长：64 samples (1 秒)
- 滤波：使用 src.data.preprocessing.filter_ppg（你当前版本已改为 0.9–5 Hz, fs=64）
- STFT / Mel：与 DaLiA 一致，通过 generate_stft / generate_mel 显式传参
- ACC：用于生成 motion mask（generate_acc_mask）
"""

import os
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly

# 允许 import src.*
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.preprocessing import (
    filter_ppg,
    slice_windows,
    generate_acc_mask,
    generate_stft,
    generate_mel,
)

# ============================================================
# 和 “当前 DaLiA 配置” 完全一致的参数
# ============================================================
FS_TARGET = 64          # 你现在认定的 PPG 采样率
WIN = 256               # 4 秒窗口（64 * 4）
STRIDE = 64             # 1 秒步长（64 * 1）

# STFT / Mel 参数：与你现在准备使用的一致
STFT_NFFT = 128
STFT_HOP = 32

MEL_NFFT = 128
MEL_HOP = 32
MEL_BINS = 64  # 如果你之后在 config 里改成 32，这里也可以一起改

# ACC 伪影检测参数：和 config 保持相同数值
ACC_MA_WINDOW_SEC = 5.0
ACC_Z_THRESH = 2.5


# ============================================================
# 处理单个 BAMI .mat 文件
# ============================================================
def process_one_file(mat_path: Path):
    d = loadmat(mat_path)

    bpm_ecg = d["bpm_ecg"].astype(np.float32).reshape(-1)   # (N_hr,)
    rawPPG = d["rawPPG"].astype(np.float32)                 # (3, T_src)
    rawAcc = d["rawAcc"].astype(np.float32)                 # (3, T_src)

    T_src = rawPPG.shape[1]
    N_hr = bpm_ecg.shape[0]

    # 估计原始采样率（大约 100 Hz 左右）
    fs_est = T_src / N_hr
    fs_src = int(round(fs_est))
    print(f"[{mat_path.name}] T={T_src}, N_hr={N_hr}, fs_est≈{fs_est:.2f}, fs_src={fs_src}")

    # 使用通道 0 作为主 PPG，并去均值
    ppg_raw = rawPPG[0] - rawPPG[0].mean()

    # ACC 去均值
    acc_raw = rawAcc - rawAcc.mean(axis=1, keepdims=True)

    # --------------------------------------------------------
    # 1) 重采样到 64 Hz（PPG & ACC 都拉到 64Hz）
    # --------------------------------------------------------
    ppg_64 = resample_poly(ppg_raw, up=FS_TARGET, down=fs_src)  # (T_new,)
    acc_64 = np.vstack([
        resample_poly(acc_raw[i], up=FS_TARGET, down=fs_src)
        for i in range(acc_raw.shape[0])
    ])  # (3, T_new)

    T_new = ppg_64.shape[0]

    # --------------------------------------------------------
    # 2) HR 对齐：bpm_ecg 是每秒一个值，我们按时间映射
    #    time_sec = sample_idx / FS_TARGET → floor 到整数秒
    # --------------------------------------------------------
    duration = T_src / fs_src  # 原始总时长（秒）
    # HR 序列的时间刻度：0,1,2,...,N_hr-1（秒）
    # 这里不用插值，直接用整数秒索引
    # 后面在窗口中心做一次映射即可

    # --------------------------------------------------------
    # 3) 带通滤波：使用你当前版本的 filter_ppg（已经改成 0.9–5 Hz, fs=64）
    # --------------------------------------------------------
    ppg_filt = filter_ppg(ppg_64, fs=FS_TARGET)

    # --------------------------------------------------------
    # 4) 滑窗切片（与 DaLiA 完全一致）
    # --------------------------------------------------------
    ppg_win, centers = slice_windows(ppg_filt, WIN, STRIDE)  # (N_win, 256)
    if ppg_win.shape[0] == 0:
        print(f"[WARN] {mat_path.name}: no valid windows after slicing.")
        return (
            np.zeros((0, 1, WIN), dtype=np.float32),
            np.zeros((0, 1, 1, 1), dtype=np.float32),
            np.zeros((0, 1, 1, 1), dtype=np.float32),
            np.zeros((0, 1, WIN), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    # 窗口中心对应的时间（秒）
    time_sec = centers / FS_TARGET  # (N_win,)
    # HR index：按秒取整
    hr_idx = np.floor(time_sec).astype(int)
    hr_idx = np.clip(hr_idx, 0, N_hr - 1)
    hr_win = bpm_ecg[hr_idx]  # (N_win,)

    # --------------------------------------------------------
    # 5) ACC 生成 mask：和 DaLiA 一样用 generate_acc_mask
    #    注意 BAMI 的 ACC 是 (3, T)，generate_acc_mask 期望 (T,3)，要转置
    # --------------------------------------------------------
    acc_64_T = acc_64.T  # (T_new, 3)

    mask = generate_acc_mask(
        ppg_len=len(ppg_filt),
        acc_raw=acc_64_T,
        fs_ppg=FS_TARGET,
        fs_acc=FS_TARGET,  # 因为已经重采样到 64 Hz
        win=WIN,
        stride=STRIDE,
        z_thresh=ACC_Z_THRESH,
        ma_window_sec=ACC_MA_WINDOW_SEC,
    )  # (N_win,1,WIN)

    # 为安全起见，对齐窗口数量
    N = min(ppg_win.shape[0], hr_win.shape[0], mask.shape[0])
    ppg_win = ppg_win[:N]
    hr_win = hr_win[:N]
    mask = mask[:N]

    # --------------------------------------------------------
    # 6) 生成 STFT / Mel（与 DaLiA current config 一致）
    # --------------------------------------------------------
    stft_list = []
    mel_list = []
    for w in ppg_win:
        stft = generate_stft(
            w,
            fs=FS_TARGET,
            n_fft=STFT_NFFT,
            hop_length=STFT_HOP,
        )  # 期望 (1,F,T)
        mel = generate_mel(
            w,
            fs=FS_TARGET,
            n_fft=MEL_NFFT,
            hop_length=MEL_HOP,
            n_mels=MEL_BINS,
        )  # 期望 (1,M,T)

        stft_list.append(stft.astype(np.float32))
        mel_list.append(mel.astype(np.float32))

    STFT = np.stack(stft_list, axis=0)  # (N,1,F,T)
    MEL = np.stack(mel_list, axis=0)    # (N,1,M,T)

    # PPG / MASK 按 (N,1,256) 组织
    PPG = ppg_win[:, None, :]          # (N,1,256)
    MASK = mask                        # 已经是 (N,1,256)
    HR = hr_win.astype(np.float32)     # (N,)

    return PPG, STFT, MEL, MASK, HR


# ============================================================
# 处理整个 BAMI-1 或 BAMI-2 目录
# ============================================================
def process_group(group_name: str, raw_subdir: str, out_subdir: str):
    base_raw = ROOT / "data" / "raw" / "BAMI" / raw_subdir
    base_out = ROOT / "data" / "processed" / out_subdir
    base_out.mkdir(parents=True, exist_ok=True)

    print(f"\n==== Processing {group_name} from {base_raw} ====")

    all_ppg = []
    all_stft = []
    all_mel = []
    all_mask = []
    all_hr = []

    mat_files = sorted([f for f in base_raw.iterdir() if f.suffix == ".mat"])

    for mat_path in mat_files:
        PPG, STFT, MEL, MASK, HR = process_one_file(mat_path)
        if PPG.shape[0] == 0:
            continue

        all_ppg.append(PPG)
        all_stft.append(STFT)
        all_mel.append(MEL)
        all_mask.append(MASK)
        all_hr.append(HR)

    if not all_ppg:
        print(f"[ERROR] No valid windows for {group_name}")
        return

    PPG = np.concatenate(all_ppg, axis=0)
    STFT = np.concatenate(all_stft, axis=0)
    MEL = np.concatenate(all_mel, axis=0)
    MASK = np.concatenate(all_mask, axis=0)
    HR = np.concatenate(all_hr, axis=0)

    print(f"\n[{group_name}] FINAL SHAPES:")
    print("  PPG :", PPG.shape)
    print("  STFT:", STFT.shape)
    print("  MEL :", MEL.shape)
    print("  MASK:", MASK.shape)
    print("  HR  :", HR.shape)

    np.save(base_out / "ppg.npy", PPG)
    np.save(base_out / "stft.npy", STFT)
    np.save(base_out / "mel.npy", MEL)
    np.save(base_out / "mask.npy", MASK)
    np.save(base_out / "hr.npy", HR)

    print(f"[{group_name}] Saved to {base_out}")


if __name__ == "__main__":
    process_group("bami1", "BAMI-1", "bami1")
    process_group("bami2", "BAMI-2", "bami2")
    print("\n[Done] BAMI-1 & BAMI-2 preprocessing finished.")