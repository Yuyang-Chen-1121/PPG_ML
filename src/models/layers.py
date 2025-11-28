import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        # [修复] 自动计算 padding 以保持输出长度不变 (Same Padding)
        if padding == 'same':
            padding = (kernel_size - 1) * dilation // 2

        # 1. Depthwise: 关键在于 groups=in_channels
        # 这意味着每个输入通道有自己独立的卷积核，互不干扰
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size, 
            stride=stride, padding=padding, dilation=dilation,
            groups=in_channels, bias=False # 偏差通常由 BatchNorm 处理
        )
        
        # 2. Pointwise: 1x1 卷积，用于通道间的信息融合
        self.pointwise = nn.Conv1d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)

class ASPP(nn.Module):
    # [修复] 修正了 rates= 语法错误，提供了默认值
    def __init__(self, in_channels, out_channels, rates=(6, 12, 18)):
        super().__init__()
        self.modules_list = nn.ModuleList()
        
        # 分支 1: 1x1 卷积 (保留原始分辨率特征)
        self.modules_list.append(
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            )
        )
        
        # 分支 2-4: 多尺度空洞卷积
        for rate in rates:
            self.modules_list.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, 3, 
                              # [理论] padding=rate 保证在 dilation 下输出长度不变
                              padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU()
                )
            )
            
        # 分支 5: 全局平均池化 (捕捉全图上下文)
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
            
        # 融合层: 将所有分支拼接后降维
        # 输入通道数 = (1个1x1分支 + N个空洞分支 + 1个全局分支) * out_channels
        self.project = nn.Sequential(
            nn.Conv1d((len(rates) + 2) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        res = []
        for conv in self.modules_list:
            res.append(conv(x))
            
        # 处理全局池化分支，需要上采样回原始长度 (Broadcast)
        gap = self.global_avg_pool(x)
        # [技巧] F.interpolate 在 1D 数据上使用 nearest 模式
        gap = F.interpolate(gap, size=x.shape[-1], mode='nearest')
        res.append(gap)
        
        # 拼接
        res = torch.cat(res, dim=1)
        return self.project(res)

class DifferentiableBeliefPropagation(nn.Module):
    def __init__(self, num_classes, transition_init=None):
        super().__init__()
        self.num_classes = num_classes
        # [修复] 建议：转移矩阵应在 Log 域中定义，并进行归一化
        if transition_init is not None:
            # 加上一个小数值 1e-9 防止 log(0)
            self.log_trans = nn.Parameter(torch.log(transition_init + 1e-9), requires_grad=True)
        else:
            # 随机初始化 (实际使用时建议用高斯矩阵初始化)
            self.log_trans = nn.Parameter(torch.randn(num_classes, num_classes), requires_grad=True)

    def forward(self, unary_logits):
        """
        unary_logits: (Batch, Time, Classes) - CNN 输出的原始 Log 概率
        """
        # [修复] 归一化转移矩阵，确保每行 sum(prob) = 1 (在 log 域中是 LogSumExp)
        # log_trans[i, j] 表示从状态 i 转移到 j 的 log 概率
        log_trans = F.log_softmax(self.log_trans, dim=1)
        
        B, T, C = unary_logits.shape
        alpha = torch.zeros_like(unary_logits)
        
        # --- 前向传递 (Forward / Alpha) ---
        # 这一步计算：考虑到过去的所有信息，当前时刻处于某个状态的概率
        alpha[:, 0, :] = unary_logits[:, 0, :]
        for t in range(1, T):
            prev_alpha = alpha[:, t-1, :].unsqueeze(2) # (B, C, 1)
            trans = log_trans.unsqueeze(0)             # (1, C, C)
            # LogSumExp 技巧实现概率乘法 (log 域加法) 和求和
            msg_fwd = torch.logsumexp(prev_alpha + trans, dim=1) 
            alpha[:, t, :] = unary_logits[:, t, :] + msg_fwd

        # --- 后向传递 (Backward / Beta) ---
        # 这一步计算：考虑到未来的所有信息...
        beta = torch.zeros_like(unary_logits)
        for t in range(T - 2, -1, -1):
            next_beta = beta[:, t+1, :].unsqueeze(1)    # (B, 1, C)
            next_unary = unary_logits[:, t+1, :].unsqueeze(1)
            trans = log_trans.unsqueeze(0)
            msg_bwd = torch.logsumexp(next_beta + next_unary + trans, dim=2)
            beta[:, t, :] = msg_bwd

        # --- 合并边缘概率 ---
        log_marginals = alpha + beta
        # [修复] 再次归一化，输出真正的 Log Probability
        return F.log_softmax(log_marginals, dim=2)