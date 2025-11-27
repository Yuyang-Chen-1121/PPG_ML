import torch
import torch.nn as nn
from.backbones.tiny_encoder import TinyEncoder
from.backbones.belief_cnn import BeliefCNN
from.heads.segmentation import SegmentationHead
from.heads.distribution import DistributionHead
from.layers import DifferentiableBeliefPropagation

class ArtifactAwareBeliefNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. 实例化骨干网络
        self.artifact_backbone = TinyEncoder() 
        self.hr_backbone = BeliefCNN()
        
        # [修复] 动态获取维度，不再硬编码 "112"
        enc_out_dim = self.artifact_backbone.out_dim # 256
        hr_out_dim = self.hr_backbone.output_dim     # 128
        
        # 2. 实例化预测头
        self.seg_head = SegmentationHead(in_channels=enc_out_dim)
        
        # 3. 融合层 (Gating Mechanism)
        # 将伪影特征压缩成一个 "Gate Vector"
        self.gate_dim = 64
        self.gate_proj = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(enc_out_dim, self.gate_dim),
            nn.Sigmoid() # 输出 0~1 之间的权重
        )
        
        # 心率头输入 = 心率特征 + 伪影Gate
        self.dist_head = DistributionHead(in_features=hr_out_dim + self.gate_dim)
        
        # 4. 后处理 (HMM)
        self.belief_prop = DifferentiableBeliefPropagation(num_classes=180)

    def forward(self, raw_ppg, spectrogram):
        """
        raw_ppg: (Batch, 1, Time_L)
        spectrogram: (Batch, 2, Freq, Time_S)
        """
        # --- Branch A: 伪影检测 ---
        art_feat = self.artifact_backbone(raw_ppg)
        seg_logits = self.seg_head(art_feat, original_len=raw_ppg.shape[-1])
        
        # --- Branch B: 心率特征 ---
        hr_feat = self.hr_backbone(raw_ppg, spectrogram) # (B, 128)
        
        # --- Fusion: 伪影感知门控 ---
        # 计算伪影严重程度向量
        gate_vec = self.gate_proj(art_feat) # (B, 64)
        
        # 拼接特征：让心率预测头知道当前的伪影状态
        # 理论：如果 gate_vec 显示强伪影，MLP 应该学会输出更平坦的分布（高不确定性）
        fused_feat = torch.cat([hr_feat, gate_vec], dim=1)
        
        # 预测心率分布 (Pre-BP)
        unary_logits = self.dist_head(fused_feat) # (B, 180)
        
        # --- HMM 平滑 ---
        # 模拟时序数据 (Batch, Time, Class)
        # 实际训练中，输入通常是一段长视频切片，这里演示单帧扩展
        unary_seq = unary_logits.unsqueeze(1) # (B, 1, 180)
        refined_marginals = self.belief_prop(unary_seq)
        
        return {
            "segmentation": seg_logits,
            "hr_distribution": unary_logits,
            "hr_refined": refined_marginals.squeeze(1)
        }