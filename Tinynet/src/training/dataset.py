import os
import sys
import yaml
import torch
import numpy as np
import random
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Dataset
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, confusion_matrix, accuracy_score, f1_score, precision_recall_curve, recall_score
from tqdm import tqdm

current_script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(current_script_path)
PROJECT_ROOT = os.path.dirname(script_dir)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from models import build_model_from_config
from utils.quantization import prepare_model_for_qat, convert_model_to_int8

class ChunkMultiModalDataset(Dataset):
    def __init__(self, file_paths):
        self.files = file_paths
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        try:
            x_path = self.files[idx]
            y_path = x_path.replace('_X.npy', '_y.npy')
            
            x_data = np.load(x_path).astype(np.float32).transpose(0, 2, 1)
            y_raw = np.load(y_path).astype(np.float32)
            N = x_data.shape[0]
            
            y_af_out = np.zeros((N, 2), dtype=np.float32)
            y_hr_out = np.zeros((N, 128), dtype=np.float32)
            mask_af = np.zeros(N, dtype=np.float32)
            mask_hr = np.zeros(N, dtype=np.float32)
            
            if y_raw.shape[1] == 128: # HR Data
                y_hr_out = y_raw
                y_af_out[:, 0] = 1.0
                mask_hr[:] = 1.0
                mask_af[:] = 1.0 
            elif y_raw.shape[1] == 2: # AF Data
                y_af_out = y_raw
                mask_af[:] = 1.0
            
            return torch.from_numpy(x_data), torch.from_numpy(y_af_out), torch.from_numpy(y_hr_out), torch.from_numpy(mask_af), torch.from_numpy(mask_hr)
        except: return None

class LazyMultiModalDataset(Dataset):
    def __init__(self, file_paths):
        self.samples = [] 
        print(f"  [Lazy] Indexing {len(file_paths)} files...")
        for x_f in file_paths:
            y_f = x_f.replace('_X.npy', '_y.npy')
            if not os.path.exists(y_f): continue
            try:
                n_samples = np.load(x_f, mmap_mode='r').shape[0]
                for i in range(n_samples): self.samples.append((x_f, y_f, i))
            except: pass
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        try:
            x_path, y_path, idx = self.samples[idx]
            x_data = np.array(np.load(x_path, mmap_mode='r')[idx]).astype(np.float32).transpose(1, 0)
            y_raw = np.array(np.load(y_path, mmap_mode='r')[idx]).astype(np.float32)
            
            y_af_out = np.zeros(2, dtype=np.float32)
            y_hr_out = np.zeros(128, dtype=np.float32)
            mask_af = 0.0; mask_hr = 0.0
            
            if y_raw.shape[0] == 128:
                y_hr_out = y_raw
                y_af_out[0] = 1.0
                mask_hr = 1.0; mask_af = 1.0
            elif y_raw.shape[0] == 2:
                y_af_out = y_raw
                mask_af = 1.0
            
            return torch.from_numpy(x_data), torch.from_numpy(y_af_out), torch.from_numpy(y_hr_out), torch.tensor(mask_af), torch.tensor(mask_hr)
        except: return None

def chunk_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    t = list(zip(*batch))
    return torch.cat(t[0], 0), torch.cat(t[1], 0), torch.cat(t[2], 0), torch.cat(t[3], 0), torch.cat(t[4], 0)

def lazy_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    t = list(zip(*batch))
    return torch.stack(t[0]), torch.stack(t[1]), torch.stack(t[2]), torch.stack(t[3]), torch.stack(t[4])

def find_data_root():
    candidates = [
        "/content/local_data",
        os.path.join(PROJECT_ROOT, "data/processed")
    ]
    for path in candidates:
        if os.path.exists(os.path.join(path, "dalia")): return path
    return None

def get_all_files(dir_path):
    if not os.path.exists(dir_path): return []
    return sorted(glob.glob(os.path.join(dir_path, "*_X.npy")))

def get_splits(data_root):
    # BAMI
    bami_dir = os.path.join(data_root, "BAMI")
    bami_files = get_all_files(bami_dir)
    bami_test = []
    if bami_files:
        random.seed(42); random.shuffle(bami_files)
        bami_test = bami_files[int(len(bami_files)*0.2):]

    # Simband
    sim_dir = os.path.join(data_root, "UMMCSIGBAND")
    sim_files = get_all_files(sim_dir)
    sim_val, sim_test = [], []
    if sim_files:
        random.seed(42); random.shuffle(sim_files) # 必须先shuffle
        total = len(sim_files)
        train_end = int(total * 0.7)
        val_end = train_end + int(total * 0.15)
        
        sim_val = sim_files[train_end : val_end]
        sim_test = sim_files[val_end :]
    
    # DaLiA
    dalia_files = get_all_files(os.path.join(data_root, "dalia"))
    
    return dalia_files, sim_val, sim_test, bami_test

def find_optimal_threshold(model, loader, device):
    print("Scanning Simband Val for Optimal Threshold...")
    model.eval()
    y_true, y_scores = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Threshold Search"):
            if batch is None: continue
            x, y_af, _, mask_af, _ = [t.to(device) for t in batch]
            out_af, _ = model(x)
            
            probs = torch.softmax(out_af[:, :2], dim=1)[:, 1]
            
            valid = mask_af > 0.5
            if valid.any():
                y_true.extend(torch.argmax(y_af, dim=1)[valid].cpu().numpy())
                y_scores.extend(probs[valid].cpu().numpy())

    # 计算最佳 F1 阈值
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    
    best_idx = np.argmax(np.nan_to_num(f1_scores))
    if best_idx >= len(thresholds): best_idx = len(thresholds) - 1
    
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    print(f"Found Best Threshold: {best_thresh:.4f} (Val F1: {best_f1:.4f})")
    
    # 绘制 P-R 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, label='PR Curve')
    plt.scatter(recalls[best_idx], precisions[best_idx], color='red', label='Optimal')
    plt.title(f'Precision-Recall Curve (Th={best_thresh:.3f})')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.legend()
    plt.savefig("pr_curve.png")
    
    return best_thresh

def evaluate_final(model, loader, device, threshold, dataset_name, task_type):
    print(f"\n⚡ Evaluating {dataset_name} (Th={threshold:.4f})...")
    model.eval()
    hr_preds, hr_trues = [], []
    af_preds, af_trues = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Eval {dataset_name}"):
            if batch is None: continue
            x, y_af, y_hr, mask_af, mask_hr = [t.to(device) for t in batch]
            out_af, out_hr = model(x)
            
            # --- AF (使用动态阈值) ---
            valid_af = mask_af > 0.5
            if valid_af.any():
                probs = torch.softmax(out_af[:, :2], dim=1)[:, 1]
                preds = (probs >= threshold).long()
                trues = torch.argmax(y_af, dim=1)
                af_preds.extend(preds[valid_af].cpu().numpy())
                af_trues.extend(trues[valid_af].cpu().numpy())
            
            # --- HR (保持不变) ---
            valid_hr = mask_hr > 0.5
            if task_type == "HR" and valid_hr.any():
                probs = torch.softmax(out_hr, dim=1)
                bins = torch.linspace(30, 210, 128).to(device)
                pred = (probs * bins).sum(dim=1)
                true = (y_hr * bins).sum(dim=1)
                hr_preds.extend(pred[valid_hr].cpu().numpy())
                hr_trues.extend(true[valid_hr].cpu().numpy())

    print(f"Results for {dataset_name}:")
    
    if task_type == "AF" and len(af_preds) > 0:
        f1 = f1_score(af_trues, af_preds, average='binary')
        acc = accuracy_score(af_trues, af_preds)
        cm = confusion_matrix(af_trues, af_preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        recall = tp / (tp + fn + 1e-6)
        spec = tn / (tn + fp + 1e-6)
        
        print(f"  [AF] F1: {f1:.4f} | Recall: {recall:.2%} | Spec: {spec:.2%}")
        print(f"     Acc: {acc:.2%} | Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        
    elif task_type == "HR":
        # HR 任务也要报 AF 误报率 (Specificity)
        cm = confusion_matrix(af_trues, af_preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp + 1e-6)
        print(f"  [AF Anti-Noise] Specificity: {spec:.2%} | False Alarms: {fp}")
        
        if len(hr_preds) > 0:
            mae = mean_absolute_error(hr_trues, hr_preds)
            corr, _ = pearsonr(hr_trues, hr_preds)
            print(f"  [HR] MAE: {mae:.2f} BPM | R: {corr:.4f}")

def main():
    USE_GPU = torch.cuda.is_available()
    device = torch.device("cuda" if USE_GPU else "cpu")
    
    if USE_GPU:
        DatasetClass, CollateFunc = ChunkMultiModalDataset, chunk_collate_fn
        BATCH_SIZE, WORKERS = 4, 2
    else:
        DatasetClass, CollateFunc = LazyMultiModalDataset, lazy_collate_fn
        BATCH_SIZE, WORKERS = 64, 0

    data_root = find_data_root()
    if not data_root: print("Data not found"); return
    
    # 加载模型
    config_path = os.path.join(PROJECT_ROOT, "config/config.yaml")
    ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints/tinynet_v2.6.1/model_best.pth")
 
    if not os.path.exists(ckpt_path):
        print("V2.6 not found, looking for others...")
        search = sorted(glob.glob(os.path.join(PROJECT_ROOT, "checkpoints/*/model_best.pth")), key=os.path.getmtime)
        if search: ckpt_path = search[-1]
        else: print("No checkpoint found"); return

    print(f"Loading {os.path.basename(os.path.dirname(ckpt_path))}...")
    
    with open(config_path) as f: config = yaml.safe_load(f)
    model = build_model_from_config(config['model_config_path']).to(device)
    model = prepare_model_for_qat(model)
    
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    
    # 尝试读取训练时保存的阈值，如果没有则重新计算
    saved_threshold = ckpt.get('best_threshold', None)
    
    model.eval()
    model_int8 = convert_model_to_int8(model)
    
    # 获取切分
    dalia_files, sim_val_files, sim_test_files, bami_test_files = get_splits(data_root)
    
    # 1. 确定阈值
    if saved_threshold:
        print(f"Using saved threshold from training: {saved_threshold:.4f}")
        final_threshold = saved_threshold
    elif sim_val_files:
        print("Recalculating threshold on Val set...")
        ds = DatasetClass(sim_val_files)
        ld = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=CollateFunc, num_workers=WORKERS)
        final_threshold = find_optimal_threshold(model_int8, ld, device)
    else:
        final_threshold = 0.5
        
    # 2. 评估 Simband (Test)
    if sim_test_files:
        ds = DatasetClass(sim_test_files)
        ld = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=CollateFunc, num_workers=WORKERS)
        evaluate_final(model_int8, ld, device, final_threshold, "Simband Test", "AF")
        
    # 3. 评估 DaLiA
    if dalia_files:
        ds = DatasetClass(dalia_files)
        ld = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=CollateFunc, num_workers=WORKERS)
        evaluate_final(model_int8, ld, device, final_threshold, "DaLiA", "HR")

    # 4. 评估 BAMI
    if bami_test_files:
        ds = DatasetClass(bami_test_files)
        ld = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=CollateFunc, num_workers=WORKERS)
        evaluate_final(model_int8, ld, device, final_threshold, "BAMI Test", "HR")

if __name__ == "__main__":
    main()