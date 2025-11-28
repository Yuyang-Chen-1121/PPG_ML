import torch

def create_mask_from_length(lengths, max_len=None):
    """
    生成序列掩码。
    Args:
        lengths: (B,) 每个样本的实际长度
    Returns:
        mask: (B, max_len) Boolean mask, True 表示有效位置
    """
    batch_size = lengths.size(0)
    if max_len is None:
        max_len = lengths.max().item()
    
    ids = torch.arange(0, max_len, device=lengths.device).unsqueeze(0).expand(batch_size, -1)
    mask = ids < lengths.unsqueeze(1)
    return mask

def normalize_batch(x):
    """
    Batch-wise Z-score 标准化
    x: (B, C, T)
    """
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True) + 1e-8
    return (x - mean) / std