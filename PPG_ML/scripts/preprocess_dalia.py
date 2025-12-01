import os
import sys
import pickle
import numpy as np
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from src.data.preprocessing import (
    filter_ppg,
    slice_windows,
    generate_acc_mask,
    generate_stft,
    generate_mel,
)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def preprocess_dalia(cfg: DictConfig):

    # 使用绝对路径，避免 Hydra 改工作目录导致相对路径失效
    raw_dir = os.path.abspath(cfg.data.dalia_raw_dir)
    out_dir = os.path.abspath(cfg.data.dalia_out_dir)

    fs_ppg = cfg.preprocessing.fs_ppg
    fs_acc = cfg.preprocessing.fs_acc

    win = cfg.preprocessing.window_size
    stride = cfg.preprocessing.window_stride

    print(f"\n [PPG-DaLiA] 读取目录: {raw_dir}")

    pkl_files = [f for f in os.listdir(raw_dir) if f.endswith(".pkl")]
    pkl_files.sort()
    print(f" [PPG-DaLiA] 找到 {len(pkl_files)} 个受试者文件")

    all_ppg = []
    all_stft = []
    all_mel = []
    all_mask = []
    all_hr = []

    for fname in tqdm(pkl_files, desc="Processing subjects"):
        sid = os.path.splitext(fname)[0]
        fpath = os.path.join(raw_dir, fname)

        with open(fpath, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        # ---- 取 PPG / ACC / HR ----
        ppg_raw = np.asarray(data["signal"]["wrist"]["BVP"], dtype=np.float32)
        acc_raw = np.asarray(data["signal"]["wrist"]["ACC"], dtype=np.float32)
        hr_seq  = np.asarray(data["label"], dtype=np.float32)  # (seconds,)

        if len(ppg_raw) < win:
            print(f"[WARN] {sid}: PPG 长度 {len(ppg_raw)} < 窗口长度 {win}，跳过")
            continue

        # ---- PPG 滤波 ----
        ppg_clean = filter_ppg(ppg_raw, fs=fs_ppg)

        # ---- 滑窗切片（PPG 域）----
        ppg_win, centers = slice_windows(ppg_clean, win, stride)  # (N, win)
        if ppg_win.shape[0] == 0:
            print(f"[WARN] {sid}: 滑窗后无有效窗口，跳过")
            continue

        # ---- HR 对齐（窗口中心时间 → HR 序列）----
        time_sec = centers / fs_ppg                    # (N,)
        time_sec = np.clip(time_sec, 0, len(hr_seq)-1)
        hr_win = hr_seq[time_sec.astype(int)]          # (N,)

        # ---- ACC 伪影 mask（映射到 PPG 域）----
        mask = generate_acc_mask(
            ppg_len=len(ppg_clean),
            acc_raw=acc_raw,
            fs_ppg=fs_ppg,
            fs_acc=fs_acc,
            win=win,
            stride=stride,
            z_thresh=cfg.preprocessing.acc_z_thresh,
            ma_window_sec=cfg.preprocessing.acc_ma_window,
        )  # (N,1,win)

        # 确保 mask 与 ppg_win 数量一致（可能最后一两个窗口略有差异）
        if mask.shape[0] != ppg_win.shape[0]:
            min_n = min(mask.shape[0], ppg_win.shape[0], hr_win.shape[0])
            ppg_win = ppg_win[:min_n]
            hr_win  = hr_win[:min_n]
            mask    = mask[:min_n]

        # ---- STFT 频谱 (Belief-PPG) ----
        if cfg.preprocessing.stft.enabled:
            stft_list = []
            for w in ppg_win:
                stft_list.append(
                    generate_stft(
                        w,
                        fs=fs_ppg,
                        n_fft=cfg.preprocessing.stft.n_fft,
                        hop_length=cfg.preprocessing.stft.hop_length,
                    )
                )
            stft_array = np.stack(stft_list, axis=0)  # (N,1,F,T)
        else:
            stft_array = np.zeros((ppg_win.shape[0], 1, 1, 1), dtype=np.float32)

        # ---- Mel 频谱 (Tiny-PPG) ----
        if cfg.preprocessing.mel.enabled:
            mel_list = []
            for w in ppg_win:
                mel_list.append(
                    generate_mel(
                        w,
                        fs=fs_ppg,
                        n_fft=cfg.preprocessing.mel.n_fft,
                        hop_length=cfg.preprocessing.mel.hop_length,
                        n_mels=cfg.preprocessing.mel.n_mels,
                    )
                )
            mel_array = np.stack(mel_list, axis=0)  # (N,1,M,T)
        else:
            mel_array = np.zeros((ppg_win.shape[0], 1, 1, 1), dtype=np.float32)

        # ---- 累积 ----
        all_ppg.append(ppg_win[:, None, :])  # (N,1,win)
        all_stft.append(stft_array)          # (N,1,F,T)
        all_mel.append(mel_array)            # (N,1,M,T)
        all_mask.append(mask)                # (N,1,win)
        all_hr.append(hr_win)                # (N,)

    # ============================================================
    # 拼接 & 保存
    # ============================================================
    if not all_ppg:
        raise RuntimeError("没有生成任何窗口，请检查数据与配置")

    PPG  = np.concatenate(all_ppg, axis=0)
    STFT = np.concatenate(all_stft, axis=0)
    MEL  = np.concatenate(all_mel, axis=0)
    MASK = np.concatenate(all_mask, axis=0)
    HR   = np.concatenate(all_hr, axis=0)

    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "ppg.npy"),  PPG)
    np.save(os.path.join(out_dir, "stft.npy"), STFT)
    np.save(os.path.join(out_dir, "mel.npy"),  MEL)
    np.save(os.path.join(out_dir, "mask.npy"), MASK)
    np.save(os.path.join(out_dir, "hr.npy"),   HR)

    print("\n ======= FINAL SHAPES ========")
    print("PPG :", PPG.shape)
    print("STFT:", STFT.shape)
    print("MEL :", MEL.shape)
    print("MASK:", MASK.shape)
    print("HR  :", HR.shape)
    print(f"\n[FINISHED] 已保存到: {out_dir}")


if __name__ == "__main__":
    preprocess_dalia()