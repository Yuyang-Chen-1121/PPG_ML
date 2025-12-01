import torch
import torch.nn as nn
import torch.nn.functional as F

class DistributionHead(nn.Module):
    def __init__(self, in_features, num_classes=180):
        super().__init__()
        # 简单的 MLP，将特征映射到 180 个心率类别概率
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # 输出 Log Softmax，配合 KL 散度 Loss 使用
        return F.log_softmax(self.net(x), dim=1)