import torch
import torch.nn as nn
import torch.nn.functional as F
from..layers import ASPP

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes=1):
        super().__init__()
        # 接收 Encoder 的特征，通过 ASPP 增强多尺度能力
        self.aspp = ASPP(in_channels, out_channels=64)
        self.classifier = nn.Conv1d(64, num_classes, kernel_size=1)
        
    def forward(self, x, original_len):
        x = self.aspp(x)
        logits = self.classifier(x)
        
        # [修复] 使用 'linear' (线性插值) 或 'nearest' 适配 1D 数据
        # 避免了 'bilinear' 导致的维度报错
        if logits.shape[-1]!= original_len:
            logits = F.interpolate(logits, size=original_len, mode='linear', align_corners=False)
            
        return logits