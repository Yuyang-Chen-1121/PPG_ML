from .multi_task_loss import CombinedLoss
from .uncertainty_loss import KLDivergenceLoss, DistributionSmoothLabelLoss

__all__ = [
    "CombinedLoss",
    "KLDivergenceLoss",
    "DistributionSmoothLabelLoss",
]