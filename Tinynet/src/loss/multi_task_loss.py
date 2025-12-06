import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5, af_class_weights=None):
        """
        Args:
            alpha (float): 房颤检测 Loss 的权重
            beta (float): 心率分布 Loss 的权重
            af_class_weights (list): 针对 AF 类别不平衡的权重 [weight_nsr, weight_af]
        """
        super(MultiTaskLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        
        # AF 使用 CrossEntropy 处理类别不平衡
        if af_class_weights is not None:
            weights = torch.tensor(af_class_weights).float()
            # 如果使用 GPU，weights 需要移动到 device，这里在 forward 中处理或由外部传入
            self.af_criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self.af_criterion = nn.CrossEntropyLoss()
            
        # HR 使用 KL 散度 (比较预测分布与真实分布的相似性)
        # reduction='batchmean' 是 KLDivLoss 的推荐用法
        self.hr_criterion = nn.KLDivLoss(reduction='batchmean')

    def forward(self, af_pred, hr_pred, af_target, hr_target):
        """
        Args:
            af_pred: (Batch, 16) - 模型输出
            hr_pred: (Batch, 128) - 模型输出
            af_target: (Batch, 2) - One-hot Label
            hr_target: (Batch, 128) - Gaussian Distribution Label
        """
        # --- 1. 处理 AF 分支 ---
        # 硬件约束：AF Head 输出 16 通道，但我们只训练前 2 个
        valid_af_logits = af_pred[:, :2] 
        
        # CrossEntropyLoss 期望 target 是 class index (LongTensor)，而不是 One-hot
        # af_target 是 One-hot [1, 0] or [0, 1]，argmax 后变成 0 或 1
        af_label_indices = torch.argmax(af_target, dim=1).long()
        
        loss_af = self.af_criterion(valid_af_logits, af_label_indices)

        # --- 2. 处理 HR 分支 ---
        # KLDivLoss 要求输入是 Log-Probabilities (经过 LogSoftmax)
        # hr_target 应该是概率分布 (Probabilities)
        hr_log_probs = F.log_softmax(hr_pred, dim=1)
        
        loss_hr = self.hr_criterion(hr_log_probs, hr_target)

        # --- 3. 总 Loss ---
        total_loss = self.alpha * loss_af + self.beta * loss_hr
        
        return total_loss, loss_af, loss_hr