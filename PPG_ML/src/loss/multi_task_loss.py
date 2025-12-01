import torch
import torch.nn as nn
import torch.nn.functional as F

class KL_Divergence_Loss(nn.Module):
    """
    用于分布回归的损失函数。
    衡量预测分布 P(y|x) 和高斯平滑后的真实分布 Q(y) 之间的距离。
    """
    def __init__(self):
        super().__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=False)

    def forward(self, pred_log_probs, target_probs):
        """
        pred_log_probs: 模型输出的 Log Softmax (Batch, Classes)
        target_probs: 真实概率分布 (Batch, Classes)
        """
        return self.kl(pred_log_probs, target_probs)

class Segmentation_Loss(nn.Module):
    """
    用于伪影检测的损失函数。
    结合 BCE (二元交叉熵) 和 Dice Loss 以处理类别不平衡。
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred_logits, target_mask):
        # BCE Loss
        bce_loss = self.bce(pred_logits, target_mask)
        return bce_loss

class CombinedLoss(nn.Module):
    def __init__(self, w_seg=1.0, w_dist=5.0):
        super().__init__()
        self.loss_dist = KL_Divergence_Loss()
        self.loss_seg = Segmentation_Loss()
        self.w_seg = w_seg
        self.w_dist = w_dist

    def forward(self, outputs, targets):
        """
        outputs: 模型输出的字典
        targets: DataLoader 加载的字典
        """
        # 1. 心率分布损失
        # 注意：模型输出可能是 (Batch, Classes) 或 (Batch, Time, Classes)
        # 这里假设模型最后做了全局池化，输出 (Batch, Classes)
        # 如果是时序输出，需要对 targets['label_dist'] 进行 unsqueeze 处理
        hr_loss = self.loss_dist(outputs['hr_distribution'], targets['label_dist'])
        
        # 2. 伪影分割损失
        seg_loss = self.loss_seg(outputs['segmentation'], targets['mask'])
        
        total_loss = self.w_dist * hr_loss + self.w_seg * seg_loss
        
        return total_loss, {"hr_loss": hr_loss.item(), "seg_loss": seg_loss.item()}