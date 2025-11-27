import torch
import torch.nn as nn
import torch.nn.functional as F

class BeliefCNN(nn.Module):
    def __init__(self, time_channels=1, freq_channels=2, bottleneck_dim=48, output_dim=128):
        super().__init__()
        self.output_dim = output_dim # [修复] 显式记录输出维度，供 Fusion 模块读取
        
        # --- 时频分支 (Spectrogram 输入) ---
        # 类似于图像处理，处理 2D 时频图
        self.tf_conv1 = nn.Conv2d(freq_channels, 32, kernel_size=3, padding='same')
        # 在频率轴上降维，保留时间轴
        self.tf_pool = nn.AdaptiveAvgPool2d((1, None)) 
        self.enc1 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        
        # --- 时域分支 (Raw Waveform 输入) ---
        # 类似于 CorNET，处理 1D 原始波形
        self.t_conv1 = nn.Conv1d(time_channels, 32, kernel_size=10, padding=4)
        self.t_pool1 = nn.MaxPool1d(4)
        self.t_conv2 = nn.Conv1d(32, 64, kernel_size=10, padding=4)
        self.t_pool2 = nn.MaxPool1d(4)
        
        # LSTM 捕捉长时依赖
        self.lstm = nn.LSTM(64, 64, num_layers=2, batch_first=True)
        self.time_proj = nn.Linear(64, bottleneck_dim)

        # --- 融合层 ---
        # 输入维度 = 时频特征(64) + 时域特征(bottleneck_dim)
        self.final_proj = nn.Conv1d(64 + bottleneck_dim, output_dim, kernel_size=1)

    def forward(self, x_time, x_freq):
        # 1. 时域处理 (Time Branch)
        t = self.t_pool1(F.relu(self.t_conv1(x_time)))
        t = self.t_pool2(F.relu(self.t_conv2(t)))
        
        # 调整维度给 LSTM: (Batch, Channels, Time) -> (Batch, Time, Channels)
        t = t.permute(0, 2, 1) 
        t_out, _ = self.lstm(t)
        
        # 取 LSTM 最后一个时间步的输出，代表整个窗口的“全局时域特征”
        t_vec = t_out[:, -1, :] # (B, 64)
        t_latent = self.time_proj(t_vec) # (B, bottleneck_dim)

        # 2. 时频处理 (Freq Branch)
        f = F.relu(self.tf_conv1(x_freq)) # (B, 32, Freq, Time)
        f = self.tf_pool(f).squeeze(2)    # (B, 32, Time)
        f = F.relu(self.enc1(f))          # (B, 64, Time)

        # 3. [修复] 特征融合逻辑
        # 我们的目标是把全局的时域特征 t_latent 加到每一个时间步的频域特征 f 上
        B, _, T = f.shape
        
        # Expand: 把 (B, dim) -> (B, dim, 1) -> (B, dim, T)
        t_expand = t_latent.unsqueeze(-1).expand(B, -1, T)
        
        # 拼接: 在通道维度 (dim 1) 拼接
        fused = torch.cat([f, t_expand], dim=1) # (B, 64+48, T)
        
        # 投影到目标维度
        out = self.final_proj(fused) # (B, 128, T)
        
        # 全局平均池化，输出一个定长的特征向量供 DistributionHead 使用
        return F.adaptive_avg_pool1d(out, 1).squeeze(-1) # (B, 128)