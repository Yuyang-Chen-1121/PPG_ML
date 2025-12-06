import torch
import torch.nn as nn

class SimpleCNNBackbone(nn.Module):
    def __init__(self, config):
        super(SimpleCNNBackbone, self).__init__()
        
        cfg = config['backbone']
        input_ch = config['data_input_channels']
        
        self.features = nn.Sequential(
            # --- Layer 1 ---
            nn.Conv1d(input_ch, cfg['conv1']['out_channels'],
                      kernel_size=cfg['conv1']['kernel_size'],
                      stride=cfg['conv1']['stride'],
                      padding=cfg['conv1']['padding'], bias=False),
            nn.BatchNorm1d(cfg['conv1']['out_channels']),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=cfg['pool1']['kernel_size'], 
                         stride=cfg['pool1']['stride']),
            
            # --- Layer 2 ---
            nn.Conv1d(cfg['conv1']['out_channels'], cfg['conv2']['out_channels'],
                      kernel_size=cfg['conv2']['kernel_size'],
                      stride=cfg['conv2']['stride'],
                      padding=cfg['conv2']['padding'], bias=False),
            nn.BatchNorm1d(cfg['conv2']['out_channels']),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=cfg['pool2']['kernel_size'], 
                         stride=cfg['pool2']['stride']),
            
            # --- Layer 3 ---
            nn.Conv1d(cfg['conv2']['out_channels'], cfg['conv3']['out_channels'],
                      kernel_size=cfg['conv3']['kernel_size'],
                      stride=cfg['conv3']['stride'],
                      padding=cfg['conv3']['padding'], bias=False),
            nn.BatchNorm1d(cfg['conv3']['out_channels']),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=cfg['pool3']['kernel_size'], 
                         stride=cfg['pool3']['stride']),
            
            # --- Layer 4 ---
            nn.Conv1d(cfg['conv3']['out_channels'], cfg['conv4']['out_channels'],
                      kernel_size=cfg['conv4']['kernel_size'],
                      stride=cfg['conv4']['stride'],
                      padding=cfg['conv4']['padding'], bias=False),
            nn.BatchNorm1d(cfg['conv4']['out_channels']),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(kernel_size=cfg['pool4']['kernel_size'], 
                         stride=cfg['pool4']['stride'])
        )
        self.output_channels = cfg['conv4']['out_channels']

    def forward(self, x):
        return self.features(x)