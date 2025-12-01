import torch
import torch.nn as nn
from..layers import DepthwiseSeparableConv1d

class TinyEncoder(nn.Module):
    def __init__(self, in_channels=1, base_filters=32):
        super().__init__()
        self.out_dim = base_filters * 8 # 记录输出维度 (256)
        
        # Tiny-PPG 特色：第一层使用超大卷积核 (Kernel=40~80)
        # 目的：为了直接看清 PPG 波形的完整周期形态
        self.entry = nn.Sequential(
            DepthwiseSeparableConv1d(in_channels, base_filters, kernel_size=40, stride=2, padding='same'),
            nn.MaxPool1d(2) 
        )
        
        self.block1 = DepthwiseSeparableConv1d(base_filters, base_filters*2, kernel_size=20, stride=2, padding='same')
        self.block2 = DepthwiseSeparableConv1d(base_filters*2, base_filters*4, kernel_size=10, stride=2, padding='same')
        self.block3 = DepthwiseSeparableConv1d(base_filters*4, base_filters*8, kernel_size=5, stride=1, padding='same')

    def forward(self, x):
        c1 = self.entry(x)   
        c2 = self.block1(c1) 
        c3 = self.block2(c2) 
        c4 = self.block3(c3) 
        return c4 # (B, 256, L_small)