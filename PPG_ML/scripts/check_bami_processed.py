import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def check_one_dataset(folder_name):
    print("\n================ CHECKING:", folder_name, "================")

    base = ROOT / "data" / "processed" / folder_name

    # 1. 检查文件是否存在
    req_files = ["ppg.npy", "stft.npy", "mel.npy", "mask.npy", "hr.npy"]
    print("== Checking files ==")
    for f in req_files:
        path = base / f
        if not path.exists():
            print(f"  ✗ Missing: {f}")
            return
        else:
            print(f"  ✓ {f} found")

    # 2. 载入 numpy
    print("\n== Loading arrays ==")
    ppg = np.load(base / "ppg.npy")
    stft = np.load(base / "stft.npy")
    mel = np.load(base / "mel.npy")
    mask = np.load(base / "mask.npy")
    hr = np.load(base / "hr.npy")

    print("PPG shape :", ppg.shape)
    print("STFT shape:", stft.shape)
    print("MEL shape :", mel.shape)
    print("MASK shape:", mask.shape)
    print("HR shape  :", hr.shape)

    N = ppg.shape[0]

    # 3. 基本维度检查
    print("\n== Dimension checks ==")
    assert stft.shape[0] == N == mel.shape[0] == mask.shape[0] == hr.shape[0], "Batch size mismatch!"
    assert ppg.shape[1] == 1, "PPG should have channel dim = 1"
    assert mask.shape == ppg.shape, "Mask must match PPG shape"

    print("✓ All dimensions OK")

    # 4. 检查 NaN / Inf
    print("\n== NaN / Inf checks ==")

    def check_nan_inf(name, arr):
        if np.isnan(arr).any():
            print(f"  ✗ {name} has NaN")
        elif np.isinf(arr).any():
            print(f"  ✗ {name} has Inf")
        else:
            print(f"  ✓ {name} clean")

    check_nan_inf("PPG", ppg)
    check_nan_inf("STFT", stft)
    check_nan_inf("MEL", mel)
    check_nan_inf("MASK", mask)
    check_nan_inf("HR", hr)

    # 5. HR 范围
    print("\n== HR Stats ==")
    print(f"HR range: {hr.min():.2f} ~ {hr.max():.2f}")

    # 6. 可视化（保存图像）
    print("\n== Visualizing sample ==")

    idx = random.randint(0, N - 1)
    print(f"Plot sample index: {idx}")

    sig = ppg[idx, 0]

    # FIX: 自动 squeeze STFT/MEL，统一为 (freq, time)
    spec_stft = np.squeeze(stft[idx])
    spec_mel = np.squeeze(mel[idx])
    mask_i = mask[idx, 0]

    print("  STFT squeezed shape:", spec_stft.shape)
    print("  MEL squeezed shape :", spec_mel.shape)

    # 创建保存目录
    plot_dir = os.path.join(ROOT, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # 输出文件名
    out_path = os.path.join(plot_dir, f"bami_sample_{idx}.png")

    # 画图但不显示
    fig = plt.figure(figsize=(14, 10))

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(sig)
    ax1.set_title(f"PPG (idx={idx})")

    ax2 = fig.add_subplot(3, 2, 3)
    ax2.imshow(spec_stft, aspect="auto", origin="lower")
    ax2.set_title("STFT")

    ax3 = fig.add_subplot(3, 2, 4)
    ax3.imshow(spec_mel, aspect="auto", origin="lower")
    ax3.set_title("Mel Spectrogram")

    ax4 = fig.add_subplot(3, 1, 3)
    ax4.plot(mask_i)
    ax4.set_title("Mask")

    plt.tight_layout()

    # Save instead of show
    fig.savefig(out_path)
    plt.close(fig)

    print(f"Saved visualization to: {out_path}")

    print("=== CHECK FINISHED ===")


if __name__ == "__main__":
    print("Root:", ROOT)

    check_one_dataset("bami1")
    check_one_dataset("bami2")