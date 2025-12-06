# src/models/heads/hr_head.py
import torch
import torch.nn as nn

class HRHead(nn.Module):
    def __init__(self, input_dim, config):
        """
        Distribution Regression Head for Heart Rate.
        Hardware Constraint: Softmax supports max 128 inputs.
        """
        super(HRHead, self).__init__()
        
        output_dim = config['output_dim'] # Must be 128
        
        self.regressor = nn.Linear(input_dim, output_dim, bias=True)
        
    def forward(self, x):
        logits = self.regressor(x)
        return logits