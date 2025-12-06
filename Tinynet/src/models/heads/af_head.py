# src/models/heads/af_head.py
import torch
import torch.nn as nn

class AFHead(nn.Module):
    def __init__(self, input_dim, config):
        """
        Binary Classification Head for Atrial Fibrillation.
        Hardware Constraint: Output channels must be multiples of 16.
        """
        super(AFHead, self).__init__()
        
        output_dim = config['output_dim'] # Should be 16
        
        self.classifier = nn.Linear(input_dim, output_dim, bias=True)
        
    def forward(self, x):
        # x is flattened vector
        logits = self.classifier(x)
        return logits