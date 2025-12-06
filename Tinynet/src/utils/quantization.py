import torch
import torch.nn as nn
import torch.quantization

def fuse_modules(model):
    """
    显式融合 Conv + BN + ReLU 模块。
    """
    # SimpleCNN 结构: 
    # L1: Conv(0), BN(1), ReLU(2), Pool(3)
    # L2: Conv(4), BN(5), ReLU(6), Pool(7)
    # L3: Conv(8), BN(9), ReLU(10), Pool(11)
    # L4: Conv(12), BN(13), ReLU(14), Pool(15)
    
    fusion_groups = [
        ['0', '1', '2'],    # Conv1 + BN + ReLU
        ['4', '5', '6'],    # Conv2 + BN + ReLU
        ['8', '9', '10'],   # Conv3 + BN + ReLU
        ['12', '13', '14']  # Conv4 + BN + ReLU
    ]

    model.eval()
    torch.quantization.fuse_modules(model.backbone.features, fusion_groups, inplace=True)
    return model

def prepare_model_for_qat(model, backend='qnnpack'):
    """
    准备 QAT 环境：
    1. 设置后端
    2. 切换 Eval -> 融合算子 -> 切换 Train
    3. 插入伪量化节点
    """
    # 1. 设置配置
    model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
    
    # 2. 算子融合
    # 融合后，Conv+BN+ReLU 会变成一个单独的 ConvBnReLU 模块
    model.eval()
    fuse_modules(model)
    
    # 3. 切换回 Train 模式进行 QAT 准备
    # prepare_qat 会自动识别 fused 模块并插入 fake_quantize 节点
    model.train()
    torch.quantization.prepare_qat(model, inplace=True)
    
    return model

def convert_model_to_int8(model):
    """
    转换为 Int8 模型
    """
    model.eval()
    return torch.quantization.convert(model, inplace=False)