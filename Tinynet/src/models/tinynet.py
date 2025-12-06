import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub
import yaml

from .backbones.simple_cnn import SimpleCNNBackbone
from .heads.af_head import AFHead
from .heads.hr_head import HRHead

class TinyNet(nn.Module):
    def __init__(self, config):
        super(TinyNet, self).__init__()
        
        config['model']['data_input_channels'] = config['data']['input_channels']

        dropout_rate = config['model'].get('dropout', 0.0)
        
        self.quant = QuantStub()
        self.backbone = SimpleCNNBackbone(config['model'])
        
        self.flatten_dim = 64 * 10 
        
        #在 Head 之前加入 Dropout 层
        self.dropout = nn.Dropout(p=dropout_rate)
        
        self.af_head = AFHead(self.flatten_dim, config['model']['af_head'])
        self.hr_head = HRHead(self.flatten_dim, config['model']['hr_head'])
        
        self.dequant_af = DeQuantStub()
        self.dequant_hr = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        features = self.backbone(x)
        features_flat = torch.flatten(features, 1)

        features_drop = self.dropout(features_flat)
        
        out_af = self.af_head(features_drop)
        out_hr = self.hr_head(features_drop)
        
        out_af = self.dequant_af(out_af)
        out_hr = self.dequant_hr(out_hr)
        
        return out_af, out_hr

def build_model_from_config(config_path="./config/config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    model = TinyNet(config)
    return model