import torch
import torch.nn as nn
import torch.nn.functional as F

class KLDivergenceLoss(nn.Module):
    """
    计算预测分布 P 和目标分布 Q 之间的 KL 散度。
    Belief-PPG 论文中用于衡量预测心率分布与真实心率分布（高斯平滑后）的差异。
    """
    def __init__(self, reduction='batchmean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_log_probs, target_probs):
        """
        Args:
            pred_log_probs: (B, C) 模型输出的 log_softmax
            target_probs: (B, C) 真实标签生成的高斯概率分布
        """
        # KLDivLoss 期望输入是 log-probabilities
        loss = F.kl_div(pred_log_probs, target_probs, reduction=self.reduction)
        return loss

class DistributionSmoothLabelLoss(nn.Module):
    """
    如果 DataLoader 没有生成高斯标签，可以在 Loss 内部动态生成。
    辅助类，用于将标量 HR 转换为分布后再计算 Loss。
    """
    def __init__(self, num_classes=180, min_hr=30, max_hr=210, sigma=1.5, device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.bins = torch.linspace(min_hr, max_hr, num_classes).to(device)
        self.sigma = sigma
        self.kl_loss = KLDivergenceLoss()

    def forward(self, pred_log_probs, target_hr_scalar):
        # 动态生成高斯标签
        # target_hr_scalar: (B, 1)
        target_dist = torch.exp(-0.5 * ((self.bins - target_hr_scalar) / self.sigma) ** 2)
        target_dist = target_dist / (target_dist.sum(dim=1, keepdim=True) + 1e-8)
        
        return self.kl_loss(pred_log_probs, target_dist)