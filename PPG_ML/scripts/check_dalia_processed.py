import os
import numpy as np
import matplotlib.pyplot as plt

# =============================
#  CONFIG
# =============================
DATA_DIR = "/content/drive/MyDrive/PPG_ML/data/processed/dalia"
PLOT_DIR = "/content/drive/MyDrive/PPG_ML/plots/dalia"

# 自动创建图像存放目录
os.makedirs(PLOT_DIR, exist_ok=True)


# =============================
#  1. 检查文件存在性
# =============================
def check_exists():
    print("== Checking file existence ==")
    files = ["ppg.npy", "stft.npy", "mel.npy", "mask.npy", "hr.npy"]
    ok = True
    for f in files:
        path = os.path.join(DATA_DIR, f)
        if os.path.exists(path):
            print(f"  ✓ {f} found")
        else:
            print(f"  ✗ {f} NOT found!")
            ok = False
    return ok


# =============================
#  2. 加载数据
# =============================
def load_all():
    print("\n== Loading npy files ==")
    PPG  = np.load(os.path.join(DATA_DIR, "ppg.npy"))
    STFT = np.load(os.path.join(DATA_DIR, "stft.npy"))
    MEL  = np.load(os.path.join(DATA_DIR, "mel.npy"))
    MASK = np.load(os.path.join(DATA_DIR, "mask.npy"))
    HR   = np.load(os.path.join(DATA_DIR, "hr.npy"))

    print(f"PPG shape : {PPG.shape}")
    print(f"STFT shape: {STFT.shape}")
    print(f"MEL shape : {MEL.shape}")
    print(f"MASK shape: {MASK.shape}")
    print(f"HR shape  : {HR.shape}")

    return PPG, STFT, MEL, MASK, HR


# =============================
#  3. 维度一致性检查
# =============================
def basic_checks(PPG, STFT, MEL, MASK, HR):
    print("\n== Basic dimension checks ==")
    n = PPG.shape[0]
    ok = True

    if STFT.shape[0] != n: print("✗ STFT batch mismatch"); ok = False
    if MEL.shape[0]  != n: print("✗ MEL batch mismatch"); ok = False
    if MASK.shape[0] != n: print("✗ MASK batch mismatch"); ok = False
    if HR.shape[0]   != n: print("✗ HR batch mismatch"); ok = False

    if ok:
        print("✓ All batch sizes match")
    else:
        print("✗ Batch mismatch detected")

    return ok


# =============================
#  4. 检查 NaN 和 INF
# =============================
def check_nan_inf(name, x):
    print(f"Checking {name} NaN/inf...")
    if np.isnan(x).any():
        print(f" ✗ {name} contains NaN")
    elif np.isinf(x).any():
        print(f" ✗ {name} contains INF")
    else:
        print(f" ✓ {name} clean")


# =============================
#  5. MASK 二值检查
# =============================
def check_mask(mask):
    print("\n== Checking MASK ==")
    uniq = np.unique(mask)
    print(f"Unique values in mask: {uniq}")

    if np.all((mask == 0) | (mask == 1)):
        print("✓ Mask is binary (0/1)")
    else:
        print("✗ Mask contains non-binary values")


# =============================
#  6. HR 检查
# =============================
def check_hr(hr):
    print("\n== Checking HR ==")
    print(f"HR range: {hr.min()} ~ {hr.max()}")

    if hr.min() < 30 or hr.max() > 220:
        print("✗ HR contains unreasonable values")
    else:
        print("✓ HR values reasonable")


# =============================
#  7. 保存可视化图像
# =============================
def save_visual_sample(PPG, STFT, MEL, MASK, HR, idx=0):
    print(f"Saving visualization for sample #{idx}")

    ppg = PPG[idx, 0]
    mask = MASK[idx, 0]
    stft = STFT[idx, 0]
    mel = MEL[idx, 0]
    hr = HR[idx]

    plt.figure(figsize=(14, 10))

    plt.subplot(4, 1, 1)
    plt.plot(ppg)
    plt.title(f"PPG (len=256), HR={hr:.1f}")

    plt.subplot(4, 1, 2)
    plt.imshow(stft, aspect="auto", origin="lower")
    plt.title("STFT (1 × F × T)")

    plt.subplot(4, 1, 3)
    plt.imshow(mel, aspect="auto", origin="lower")
    plt.title("Mel Spectrogram (1 × 64 × T)")

    plt.subplot(4, 1, 4)
    plt.plot(mask)
    plt.title("Mask (0/1)")

    plt.tight_layout()

    out_path = os.path.join(PLOT_DIR, f"sample_{idx}.png")
    plt.savefig(out_path)
    plt.close()

    print(f"✓ Saved to: {out_path}")


# =============================
#  8. 主入口
# =============================
if __name__ == "__main__":
    if not check_exists():
        raise SystemExit("Missing files!")

    PPG, STFT, MEL, MASK, HR = load_all()

    basic_checks(PPG, STFT, MEL, MASK, HR)

    check_nan_inf("PPG", PPG)
    check_nan_inf("STFT", STFT)
    check_nan_inf("MEL", MEL)
    check_nan_inf("MASK", MASK)
    check_nan_inf("HR", HR)

    check_mask(MASK)
    check_hr(HR)

    # 保存 10 个可视化样本
    for i in [0, 10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]:
        if i < len(PPG):
            save_visual_sample(PPG, STFT, MEL, MASK, HR, idx=i)

    print("\nAll checks completed.")