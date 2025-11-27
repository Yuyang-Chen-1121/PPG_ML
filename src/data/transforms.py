import torch
import numpy as np

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

class RandomGaussianNoise:
    """加入随机高斯噪声，模拟传感器热噪声"""
    def __init__(self, mean=0., std=0.05, p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, x):
        if np.random.rand() < self.p:
            noise = torch.randn_like(x) * self.std + self.mean
            return x + noise
        return x

class RandomScale:
    """随机缩放信号幅度，模拟佩戴松紧度变化"""
    def __init__(self, scale_range=(0.8, 1.2), p=0.5):
        self.min_scale, self.max_scale = scale_range
        self.p = p

    def __call__(self, x):
        if np.random.rand() < self.p:
            scale = np.random.uniform(self.min_scale, self.max_scale)
            return x * scale
        return x

class RandomMask:
    """随机掩码（丢弃一段信号），模拟接触不良"""
    def __init__(self, mask_ratio=0.1, p=0.3):
        self.mask_ratio = mask_ratio
        self.p = p

    def __call__(self, x):
        if np.random.rand() < self.p:
            seq_len = x.shape[-1]
            mask_len = int(seq_len * self.mask_ratio)
            start = np.random.randint(0, seq_len - mask_len)
            mask = torch.ones_like(x)
            mask[..., start:start+mask_len] = 0
            return x * mask
        return x