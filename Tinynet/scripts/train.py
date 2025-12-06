import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split, WeightedRandomSampler
from tqdm import tqdm
import numpy as np
import glob
import random
from sklearn.metrics import f1_score, accuracy_score, precision_recall_curve

current_script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(current_script_path)
PROJECT_ROOT = os.path.dirname(script_dir)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from models import build_model_from_config
from utils.quantization import prepare_model_for_qat
from training.dataset import MultiModalDataset, batch_collate_fn 

def get_all_files(dir_path):
    if not os.path.exists(dir_path): return []
    return sorted(glob.glob(os.path.join(dir_path, "*_X.npy")))

def scan_labels_for_sampler(file_list):
    print("⏳ Scanning labels for WeightedRandomSampler...")
    groups = []
    for x_f in tqdm(file_list, desc="Scanning Labels"):
        y_f = x_f.replace('_X.npy', '_y.npy')
        if not os.path.exists(y_f): continue
        try:
            y_mmap = np.load(y_f, mmap_mode='r')
            n = y_mmap.shape[0]
            if y_mmap.ndim == 2 and y_mmap.shape[1] == 128:
                groups.extend([2] * n) # Group 2: HR Data
            elif y_mmap.ndim == 2 and y_mmap.shape[1] == 2:
                y_vals = np.argmax(y_mmap, axis=1)
                groups.extend(y_vals.tolist()) # Group 0/1: Simband
        except: pass
    return np.array(groups)

#在验证集上寻找最佳 F1 对应的阈值
def find_best_threshold_and_score(y_true, y_probs):
    
    # 计算 P-R 曲线
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    
    # 计算 F1，处理分母为0的情况
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    
    # 将 NaN (0/0) 替换为 0.0
    f1_scores = np.nan_to_num(f1_scores, nan=0.0)
    
    # 找到 F1 最高的索引
    best_idx = np.argmax(f1_scores)
    
    # 边界保护：thresholds 长度比 f1/precision/recall 少 1
    # 如果 argmax 选到了最后一个点 (Recall=0, Precision=1)，需要处理
    if best_idx >= len(thresholds):
        best_thresh = thresholds[-1]
    else:
        best_thresh = thresholds[best_idx]
    
    best_f1 = f1_scores[best_idx]
    best_recall = recalls[best_idx]
    
    return best_thresh, best_f1, best_recall

def train(config_path):
    with open(config_path, 'r') as f: config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting TinyNet V2.6 (Robust F1 Calculation) on {device}")

    local_root = "/content/local_data"
    drive_root = os.path.join(PROJECT_ROOT, "data/processed")
    data_root = local_root if os.path.exists(os.path.join(local_root, "dalia")) else drive_root

    dalia_path = os.path.join(data_root, "dalia")
    simband_path = os.path.join(data_root, "UMMCSIGBAND")
    bami_path = os.path.join(data_root, "BAMI")
    
    files_dalia = get_all_files(dalia_path)
    files_simband = get_all_files(simband_path)
    files_bami = get_all_files(bami_path)

    random.seed(42) 
    random.shuffle(files_bami)
    bami_train = files_bami[:int(len(files_bami)*0.2)]
    
    random.shuffle(files_simband)#混合数据
    sb_train_len = int(len(files_simband)*0.7)
    sb_val_len = int(len(files_simband)*0.15)
    sb_train = files_simband[:sb_train_len]
    sb_val = files_simband[sb_train_len : sb_train_len + sb_val_len]

    print(f"  Train Files: DaLiA({len(files_dalia)}) + Simband({len(sb_train)}) + BAMI({len(bami_train)})")

    # Dataset
    train_file_list = files_dalia + sb_train + bami_train
    val_file_list = sb_val

    train_dataset = MultiModalDataset(train_file_list, mode='train')
    val_dataset = MultiModalDataset(val_file_list, mode='val')

    # sampler
    train_groups = scan_labels_for_sampler(train_file_list)
    group_counts = np.bincount(train_groups)
    print(f"  Counts: NSR={group_counts[0]}, AF={group_counts[1]}, HR={group_counts[2]}")
    
    weights = 1. / (group_counts + 1e-6)
    samples_weight = np.array([weights[t] for t in train_groups])
    
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(samples_weight).double(),
        num_samples=len(samples_weight),
        replacement=True
    )

    # Loader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['train']['batch_size'], 
        shuffle=False, 
        sampler=sampler,
        collate_fn=batch_collate_fn, 
        num_workers=4, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['train']['batch_size'], 
        shuffle=False,
        collate_fn=batch_collate_fn,
        num_workers=2
    )

    # --- 2. Model & Loss ---
    model = build_model_from_config(config['model_config_path']).to(device)
    model = prepare_model_for_qat(model)

    optimizer = optim.AdamW(model.parameters(), lr=float(config['train']['learning_rate']), weight_decay=1e-3)
    
    criterion_af = nn.CrossEntropyLoss(reduction='none') 
    criterion_hr = nn.KLDivLoss(reduction='none', log_target=False)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)
    
    best_val_f1 = 0.0
    save_dir = config['train']['checkpoint_dir']
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    # --- 3. Training Loop ---
    epochs = config['train']['epochs']
    torch.cuda.empty_cache()

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            if batch is None: continue
            x, y_af, y_hr, mask_af, mask_hr = [t.to(device) for t in batch]
            
            optimizer.zero_grad()
            out_af, out_hr = model(x)
            
            # AF Loss
            out_af_valid = out_af[:, :2]
            y_af_ind = torch.argmax(y_af, dim=1).long()
            loss_af = (criterion_af(out_af_valid, y_af_ind) * mask_af).sum() / (mask_af.sum() + 1e-6)
            
            # HR Loss
            out_hr_log = torch.log_softmax(out_hr, dim=1)
            loss_hr = (criterion_hr(out_hr_log, y_hr).sum(dim=1) * mask_hr).sum() / (mask_hr.sum() + 1e-6)
            
            loss = loss_af + loss_hr
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({"AF": f"{loss_af.item():.3f}", "HR": f"{loss_hr.item():.3f}"})

        #Validation
        model.eval()
        val_probs, val_trues = [], []
        val_loss = 0.0
        val_count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue
                x, y_af, _, mask_af, _ = [t.to(device) for t in batch]
                out_af, _ = model(x)
                out_af_valid = out_af[:, :2]
                y_af_ind = torch.argmax(y_af, dim=1)
                
                l = criterion_af(out_af_valid, y_af_ind)
                val_loss += (l * mask_af).sum().item()
                val_count += mask_af.sum().item()
                
                if mask_af.any():
                    # 取 AF 类的概率
                    probs = torch.softmax(out_af_valid, dim=1)[:, 1] 
                    val_probs.extend(probs[mask_af>0.5].cpu().numpy())
                    val_trues.extend(y_af_ind[mask_af>0.5].cpu().numpy())

        #动态阈值搜索
        y_true_np = np.array(val_trues)
        y_prob_np = np.array(val_probs)
        
        best_thresh, best_f1, best_recall = find_best_threshold_and_score(y_true_np, y_prob_np)

        y_pred_bin = (y_prob_np >= best_thresh).astype(int)
        val_acc = accuracy_score(y_true_np, y_pred_bin)
        
        avg_loss = val_loss / (val_count + 1e-6)
        print(f"Ep {epoch+1} | Val Loss: {avg_loss:.4f} | Best F1: {best_f1:.4f} (Th={best_thresh:.3f}) | Rec: {best_recall:.2%}")

        scheduler.step(best_f1)

        if best_f1 > best_val_f1:
            best_val_f1 = best_f1
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_f1': best_f1,
                'best_threshold': best_thresh
            }, os.path.join(save_dir, "model_best.pth"))
            print(f"  Saved Best (F1: {best_f1:.4f})")

if __name__ == "__main__":
    config_path = os.path.join(PROJECT_ROOT, "config/config.yaml")
    train(config_path)