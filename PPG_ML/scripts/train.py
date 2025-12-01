import sys
import os
import hydra
from omegaconf import DictConfig
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加 src 到路径，确保能 import
sys.path.append(os.path.join(os.getcwd(), 'src'))

from models import ArtifactAwareBeliefNetwork
from data.dataset import PPGDataset
from loss.multi_task_loss import CombinedLoss

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(f"开始训练: {cfg.experiment_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 准备数据
    # data_path 设为 None 以触发 Mock 模式
    train_dataset = PPGDataset(data_path=None, config=cfg, mode='train')
    train_loader = DataLoader(train_dataset, 
                              batch_size=cfg.dataset.batch_size, 
                              shuffle=True, 
                              num_workers=cfg.dataset.num_workers)

    # 2. 初始化模型
    model = ArtifactAwareBeliefNetwork().to(device)
    
    # 3. 优化器与损失
    optimizer = optim.AdamW(model.parameters(), 
                            lr=cfg.training.learning_rate, 
                            weight_decay=cfg.training.weight_decay)
    
    criterion = CombinedLoss(w_seg=cfg.loss_weights.segmentation, 
                             w_dist=cfg.loss_weights.distribution)

    # 4. 训练循环
    for epoch in range(cfg.training.epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training.epochs}")
        for batch in pbar:
            # 数据搬运到 GPU
            raw_ppg = batch['raw_ppg'].to(device)
            spectrogram = batch['spectrogram'].to(device)
            mask_target = batch['mask'].to(device)
            label_dist = batch['label_dist'].to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            # 模型 forward 接收 (raw_ppg, spectrogram)
            outputs = model(raw_ppg, spectrogram)
            
            # 计算损失
            loss, loss_dict = criterion(outputs, {
                'mask': mask_target, 
                'label_dist': label_dist
            })
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            
            # 更新参数
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'hr_loss': loss_dict['hr_loss']})
        
        print(f"Epoch {epoch+1} Average Loss: {epoch_loss / len(train_loader):.4f}")
        
        # 保存模型
        if (epoch + 1) % 10 == 0:
            os.makedirs(cfg.training.save_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{cfg.training.save_dir}/model_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    main()