
import torch
from math import inf
from math import nan
NoneType = type(None)
import torch
from torch import device
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree


import sys
import os
import collections

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上一级目录（父目录）
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from torch.nn import *

# Purpose: Implement `traverse_and_patch_modules` for the TinyNet workflow.
# Inputs: Parameters defined in `traverse_and_patch_modules` signature.
# Outputs: Return value produced by `traverse_and_patch_modules`.
# Assumptions: Caller provides valid types/shapes for this operation.
def traverse_and_patch_modules(module, prefix=''):
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        try:
            _ = child._modules
        except AttributeError:
            if type(child).__name__ != "Module":
                child._buffers = collections.OrderedDict()
                child._modules = collections.OrderedDict()
                child._parameters = collections.OrderedDict()
                child._load_state_dict_post_hooks = collections.OrderedDict()
                child._state_dict_pre_hooks = collections.OrderedDict()
                child._state_dict_hooks = collections.OrderedDict()
                child._backward_hooks = collections.OrderedDict()
                child._backward_pre_hooks = collections.OrderedDict()
                child._forward_hooks = collections.OrderedDict()
                child._forward_pre_hooks = collections.OrderedDict()
                child._load_state_dict_pre_hooks = collections.OrderedDict()
                child._forward_hooks_with_kwargs = collections.OrderedDict()
                child._backward_hooks_with_kwargs = collections.OrderedDict()
                child._forward_hooks_always_called = collections.OrderedDict()
        traverse_and_patch_modules(child, full_name)

class TinyNetQuant(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = torch.load(r'/Volumes/SamsungT7/Workspace/ZJU_GRAD/PPG_ML/TinyNet/output/int8_model/quant.pt', weights_only=False) # Quantize(scale=tensor([0.0617]), zero_point=tensor([112]), dtype=torch.quint8)
        self.stem = torch.load(r'/Volumes/SamsungT7/Workspace/ZJU_GRAD/PPG_ML/TinyNet/output/int8_model/stem.pt', weights_only=False) # Module(   (0): QuantizedConvReLU1d(16, 32, kernel_size=(7,), stride=(1,), scale=0.12069303542375565, zero_point=0, padding=(3,))   (1): Identity()   (2): Identity() )
        self.hr_block1 = torch.load(r'/Volumes/SamsungT7/Workspace/ZJU_GRAD/PPG_ML/TinyNet/output/int8_model/hr_block1.pt', weights_only=False) # Module(   (downsample): Module(     (0): QuantizedConv1d(32, 48, kernel_size=(1,), stride=(2,), scale=0.08399784564971924, zero_point=197)     (1): Identity()   )   (conv1): QuantizedConv1d(32, 48, kernel_size=(7,), stride=(2,), scale=0.16046154499053955, zero_point=168, padding=(3,))   (bn1): Identity()   (relu): ReLU(inplace=True)   (dropout): QuantizedDropout(p=0.5, inplace=False)   (conv2): QuantizedConv1d(48, 48, kernel_size=(7,), stride=(1,), scale=0.04807814583182335, zero_point=198, padding=(3,))   (bn2): Identity()   (skip_add): Module(     (activation_post_process): Identity()   ) )
        self.hr_block2 = torch.load(r'/Volumes/SamsungT7/Workspace/ZJU_GRAD/PPG_ML/TinyNet/output/int8_model/hr_block2.pt', weights_only=False) # Module(   (conv1): QuantizedConv1d(48, 48, kernel_size=(7,), stride=(1,), scale=0.0630272924900055, zero_point=167, padding=(3,))   (bn1): Identity()   (relu): ReLU(inplace=True)   (dropout): QuantizedDropout(p=0.5, inplace=False)   (conv2): QuantizedConv1d(48, 48, kernel_size=(7,), stride=(1,), scale=0.028484439477324486, zero_point=185, padding=(3,))   (bn2): Identity()   (skip_add): Module(     (activation_post_process): Identity()   ) )
        self.hr_block3 = torch.load(r'/Volumes/SamsungT7/Workspace/ZJU_GRAD/PPG_ML/TinyNet/output/int8_model/hr_block3.pt', weights_only=False) # Module(   (conv1): QuantizedConv1d(48, 48, kernel_size=(7,), stride=(1,), scale=0.04828866571187973, zero_point=159, padding=(3,))   (bn1): Identity()   (relu): ReLU(inplace=True)   (dropout): QuantizedDropout(p=0.5, inplace=False)   (conv2): QuantizedConv1d(48, 48, kernel_size=(7,), stride=(1,), scale=0.029658526182174683, zero_point=205, padding=(3,))   (bn2): Identity()   (skip_add): Module(     (activation_post_process): Identity()   ) )
        self.hr_head_conv = torch.load(r'/Volumes/SamsungT7/Workspace/ZJU_GRAD/PPG_ML/TinyNet/output/int8_model/hr_head_conv.pt', weights_only=False) # QuantizedConv1d(48, 128, kernel_size=(1,), stride=(1,), scale=0.016917049884796143, zero_point=130)
        self.hr_gap = torch.load(r'/Volumes/SamsungT7/Workspace/ZJU_GRAD/PPG_ML/TinyNet/output/int8_model/hr_gap.pt', weights_only=False) # Module(   (pool1): AvgPool1d(kernel_size=(8,), stride=(7,), padding=(0,))   (pool2): AvgPool1d(kernel_size=(8,), stride=(7,), padding=(0,))   (pool3): AvgPool1d(kernel_size=(3,), stride=(1,), padding=(0,)) )
        self.hr_dropout = torch.load(r'/Volumes/SamsungT7/Workspace/ZJU_GRAD/PPG_ML/TinyNet/output/int8_model/hr_dropout.pt', weights_only=False) # QuantizedDropout(p=0.75, inplace=False)
        self.hr_fc = torch.load(r'/Volumes/SamsungT7/Workspace/ZJU_GRAD/PPG_ML/TinyNet/output/int8_model/hr_fc.pt', weights_only=False) # QuantizedLinear(in_features=128, out_features=106, scale=0.04498226195573807, zero_point=75, qscheme=torch.per_tensor_affine)
        self.af_spatial_block1 = torch.load(r'/Volumes/SamsungT7/Workspace/ZJU_GRAD/PPG_ML/TinyNet/output/int8_model/af_spatial_block1.pt', weights_only=False) # Module(   (downsample): Module(     (0): QuantizedConv1d(32, 64, kernel_size=(1,), stride=(2,), scale=0.3654225468635559, zero_point=179)     (1): Identity()   )   (conv1): QuantizedConv1d(32, 64, kernel_size=(3,), stride=(2,), scale=0.2272118330001831, zero_point=149, padding=(1,))   (bn1): Identity()   (relu): ReLU(inplace=True)   (dropout): QuantizedDropout(p=0.5, inplace=False)   (conv2): QuantizedConv1d(64, 64, kernel_size=(3,), stride=(1,), scale=0.1407506912946701, zero_point=191, padding=(1,))   (bn2): Identity()   (skip_add): Module(     (activation_post_process): Identity()   ) )
        self.af_spatial_block2 = torch.load(r'/Volumes/SamsungT7/Workspace/ZJU_GRAD/PPG_ML/TinyNet/output/int8_model/af_spatial_block2.pt', weights_only=False) # Module(   (conv1): QuantizedConv1d(64, 64, kernel_size=(3,), stride=(1,), scale=0.14242352545261383, zero_point=144, padding=(1,))   (bn1): Identity()   (relu): ReLU(inplace=True)   (dropout): QuantizedDropout(p=0.5, inplace=False)   (conv2): QuantizedConv1d(64, 64, kernel_size=(3,), stride=(1,), scale=0.09598856419324875, zero_point=172, padding=(1,))   (bn2): Identity()   (skip_add): Module(     (activation_post_process): Identity()   ) )
        self.af_spatial_block3 = torch.load(r'/Volumes/SamsungT7/Workspace/ZJU_GRAD/PPG_ML/TinyNet/output/int8_model/af_spatial_block3.pt', weights_only=False) # Module(   (conv1): QuantizedConv1d(64, 64, kernel_size=(3,), stride=(1,), scale=0.14164289832115173, zero_point=122, padding=(1,))   (bn1): Identity()   (relu): ReLU(inplace=True)   (dropout): QuantizedDropout(p=0.5, inplace=False)   (conv2): QuantizedConv1d(64, 64, kernel_size=(3,), stride=(1,), scale=0.2544431686401367, zero_point=164, padding=(1,))   (bn2): Identity()   (skip_add): Module(     (activation_post_process): Identity()   ) )
        self.af_spatial_bn = torch.load(r'/Volumes/SamsungT7/Workspace/ZJU_GRAD/PPG_ML/TinyNet/output/int8_model/af_spatial_bn.pt', weights_only=False) # Module(   (dequant): DeQuantize()   (bn): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)   (quant): Quantize(scale=tensor([0.1172]), zero_point=tensor([34]), dtype=torch.quint8) )
        self.af_temporal_pool1 = torch.load(r'/Volumes/SamsungT7/Workspace/ZJU_GRAD/PPG_ML/TinyNet/output/int8_model/af_temporal_pool1.pt', weights_only=False) # AvgPool1d(kernel_size=(5,), stride=(5,), padding=(0,))
        self.af_temporal_pool2 = torch.load(r'/Volumes/SamsungT7/Workspace/ZJU_GRAD/PPG_ML/TinyNet/output/int8_model/af_temporal_pool2.pt', weights_only=False) # AvgPool1d(kernel_size=(4,), stride=(4,), padding=(0,))
        self.af_temporal_block1 = torch.load(r'/Volumes/SamsungT7/Workspace/ZJU_GRAD/PPG_ML/TinyNet/output/int8_model/af_temporal_block1.pt', weights_only=False) # Module(   (downsample): Module(     (0): QuantizedConv1d(32, 64, kernel_size=(1,), stride=(1,), scale=0.17476387321949005, zero_point=160)     (1): Identity()   )   (conv1): QuantizedConv1d(32, 64, kernel_size=(7,), stride=(1,), scale=0.1567246913909912, zero_point=116, padding=(3,))   (bn1): Identity()   (relu): ReLU(inplace=True)   (dropout): QuantizedDropout(p=0.5, inplace=False)   (conv2): QuantizedConv1d(64, 64, kernel_size=(7,), stride=(1,), scale=0.08962813764810562, zero_point=203, padding=(3,))   (bn2): Identity()   (skip_add): Module(     (activation_post_process): Identity()   ) )
        self.af_temporal_block2 = torch.load(r'/Volumes/SamsungT7/Workspace/ZJU_GRAD/PPG_ML/TinyNet/output/int8_model/af_temporal_block2.pt', weights_only=False) # Module(   (conv1): QuantizedConv1d(64, 64, kernel_size=(7,), stride=(1,), scale=0.06418642401695251, zero_point=134, padding=(3,))   (bn1): Identity()   (relu): ReLU(inplace=True)   (dropout): QuantizedDropout(p=0.5, inplace=False)   (conv2): QuantizedConv1d(64, 64, kernel_size=(7,), stride=(1,), scale=0.027402659878134727, zero_point=155, padding=(3,))   (bn2): Identity()   (skip_add): Module(     (activation_post_process): Identity()   ) )
        self.af_temporal_data_conv = torch.load(r'/Volumes/SamsungT7/Workspace/ZJU_GRAD/PPG_ML/TinyNet/output/int8_model/af_temporal_data_conv.pt', weights_only=False) # QuantizedConvReLU1d(64, 64, kernel_size=(3,), stride=(1,), scale=0.060786955058574677, zero_point=0, padding=(1,))
        self.af_temporal_data_bn = torch.load(r'/Volumes/SamsungT7/Workspace/ZJU_GRAD/PPG_ML/TinyNet/output/int8_model/af_temporal_data_bn.pt', weights_only=False) # Identity()
        self.af_temporal_data_relu = torch.load(r'/Volumes/SamsungT7/Workspace/ZJU_GRAD/PPG_ML/TinyNet/output/int8_model/af_temporal_data_relu.pt', weights_only=False) # Identity()
        self.af_temporal_gate_conv = torch.load(r'/Volumes/SamsungT7/Workspace/ZJU_GRAD/PPG_ML/TinyNet/output/int8_model/af_temporal_gate_conv.pt', weights_only=False) # QuantizedConv1d(64, 64, kernel_size=(3,), stride=(1,), scale=0.22584262490272522, zero_point=146, padding=(1,))
        self.af_temporal_gate_bn = torch.load(r'/Volumes/SamsungT7/Workspace/ZJU_GRAD/PPG_ML/TinyNet/output/int8_model/af_temporal_gate_bn.pt', weights_only=False) # Identity()
        self.af_temporal_gate_sigmoid = torch.load(r'/Volumes/SamsungT7/Workspace/ZJU_GRAD/PPG_ML/TinyNet/output/int8_model/af_temporal_gate_sigmoid.pt', weights_only=False) # Sigmoid()
        self.af_temporal_gate_mul = torch.load(r'/Volumes/SamsungT7/Workspace/ZJU_GRAD/PPG_ML/TinyNet/output/int8_model/af_temporal_gate_mul.pt', weights_only=False) # Module(   (activation_post_process): Identity() )
        self.af_temporal_global_pool = torch.load(r'/Volumes/SamsungT7/Workspace/ZJU_GRAD/PPG_ML/TinyNet/output/int8_model/af_temporal_global_pool.pt', weights_only=False) # AvgPool1d(kernel_size=(16,), stride=(1,), padding=(0,))
        self.af_fusion_add = torch.load(r'/Volumes/SamsungT7/Workspace/ZJU_GRAD/PPG_ML/TinyNet/output/int8_model/af_fusion_add.pt', weights_only=False) # Module(   (activation_post_process): Identity() )
        self.af_fusion_se = torch.load(r'/Volumes/SamsungT7/Workspace/ZJU_GRAD/PPG_ML/TinyNet/output/int8_model/af_fusion_se.pt', weights_only=False) # Module(   (squeeze): Module(     (pool1): AvgPool1d(kernel_size=(8,), stride=(7,), padding=(0,))     (pool2): AvgPool1d(kernel_size=(8,), stride=(7,), padding=(0,))     (pool3): AvgPool1d(kernel_size=(3,), stride=(1,), padding=(0,))   )   (fc1): QuantizedLinear(in_features=64, out_features=16, scale=0.1811181604862213, zero_point=106, qscheme=torch.per_tensor_affine)   (relu): ReLU(inplace=True)   (fc2): QuantizedLinear(in_features=16, out_features=64, scale=0.8897940516471863, zero_point=160, qscheme=torch.per_tensor_affine)   (sigmoid): Sigmoid()   (q_mul): Module(     (activation_post_process): Identity()   ) )
        self.af_head_conv = torch.load(r'/Volumes/SamsungT7/Workspace/ZJU_GRAD/PPG_ML/TinyNet/output/int8_model/af_head_conv.pt', weights_only=False) # QuantizedConvReLU1d(64, 128, kernel_size=(1,), stride=(1,), scale=0.028597867116332054, zero_point=0)
        self.af_head_bn = torch.load(r'/Volumes/SamsungT7/Workspace/ZJU_GRAD/PPG_ML/TinyNet/output/int8_model/af_head_bn.pt', weights_only=False) # Identity()
        self.af_head_relu = torch.load(r'/Volumes/SamsungT7/Workspace/ZJU_GRAD/PPG_ML/TinyNet/output/int8_model/af_head_relu.pt', weights_only=False) # Identity()
        self.af_gap = torch.load(r'/Volumes/SamsungT7/Workspace/ZJU_GRAD/PPG_ML/TinyNet/output/int8_model/af_gap.pt', weights_only=False) # Module(   (pool1): AvgPool1d(kernel_size=(8,), stride=(7,), padding=(0,))   (pool2): AvgPool1d(kernel_size=(8,), stride=(7,), padding=(0,))   (pool3): AvgPool1d(kernel_size=(3,), stride=(1,), padding=(0,)) )
        self.af_dropout = torch.load(r'/Volumes/SamsungT7/Workspace/ZJU_GRAD/PPG_ML/TinyNet/output/int8_model/af_dropout.pt', weights_only=False) # QuantizedDropout(p=0.75, inplace=False)
        self.af_fc = torch.load(r'/Volumes/SamsungT7/Workspace/ZJU_GRAD/PPG_ML/TinyNet/output/int8_model/af_fc.pt', weights_only=False) # QuantizedLinear(in_features=128, out_features=1, scale=0.11174407601356506, zero_point=183, qscheme=torch.per_tensor_affine)
        self.dequant = torch.load(r'/Volumes/SamsungT7/Workspace/ZJU_GRAD/PPG_ML/TinyNet/output/int8_model/dequant.pt', weights_only=False) # DeQuantize()
        traverse_and_patch_modules(self)
        self.load_state_dict(torch.load(r'/Volumes/SamsungT7/Workspace/ZJU_GRAD/PPG_ML/TinyNet/output/int8_model/state_dict.pt', weights_only=True))

    
    
    def forward(self, x : _torch_Tensor_) -> _tuple_torch_Tensor_torch_Tensor_:
        quant = self.quant(x);  x = None
        stem_0 = getattr(self.stem, "0")(quant);  quant = None
        stem_1 = getattr(self.stem, "1")(stem_0);  stem_0 = None
        stem_2 = getattr(self.stem, "2")(stem_1);  stem_1 = None
        hr_block1_downsample_0 = getattr(self.hr_block1.downsample, "0")(stem_2)
        hr_block1_downsample_1 = getattr(self.hr_block1.downsample, "1")(hr_block1_downsample_0);  hr_block1_downsample_0 = None
        hr_block1_conv1 = self.hr_block1.conv1(stem_2)
        hr_block1_bn1 = self.hr_block1.bn1(hr_block1_conv1);  hr_block1_conv1 = None
        hr_block1_relu = self.hr_block1.relu(hr_block1_bn1);  hr_block1_bn1 = None
        hr_block1_dropout = self.hr_block1.dropout(hr_block1_relu);  hr_block1_relu = None
        hr_block1_conv2 = self.hr_block1.conv2(hr_block1_dropout);  hr_block1_dropout = None
        hr_block1_bn2 = self.hr_block1.bn2(hr_block1_conv2);  hr_block1_conv2 = None
        add = torch.ops.quantized.add(hr_block1_bn2, hr_block1_downsample_1, scale = 0.09806133061647415, zero_point = 214);  hr_block1_bn2 = hr_block1_downsample_1 = None
        hr_block1_skip_add_activation_post_process = self.hr_block1.skip_add.activation_post_process(add);  add = None
        hr_block1_relu_1 = self.hr_block1.relu(hr_block1_skip_add_activation_post_process);  hr_block1_skip_add_activation_post_process = None
        hr_block2_conv1 = self.hr_block2.conv1(hr_block1_relu_1)
        hr_block2_bn1 = self.hr_block2.bn1(hr_block2_conv1);  hr_block2_conv1 = None
        hr_block2_relu = self.hr_block2.relu(hr_block2_bn1);  hr_block2_bn1 = None
        hr_block2_dropout = self.hr_block2.dropout(hr_block2_relu);  hr_block2_relu = None
        hr_block2_conv2 = self.hr_block2.conv2(hr_block2_dropout);  hr_block2_dropout = None
        hr_block2_bn2 = self.hr_block2.bn2(hr_block2_conv2);  hr_block2_conv2 = None
        add_1 = torch.ops.quantized.add(hr_block2_bn2, hr_block1_relu_1, scale = 0.03269081190228462, zero_point = 155);  hr_block2_bn2 = hr_block1_relu_1 = None
        hr_block2_skip_add_activation_post_process = self.hr_block2.skip_add.activation_post_process(add_1);  add_1 = None
        hr_block2_relu_1 = self.hr_block2.relu(hr_block2_skip_add_activation_post_process);  hr_block2_skip_add_activation_post_process = None
        hr_block3_conv1 = self.hr_block3.conv1(hr_block2_relu_1)
        hr_block3_bn1 = self.hr_block3.bn1(hr_block3_conv1);  hr_block3_conv1 = None
        hr_block3_relu = self.hr_block3.relu(hr_block3_bn1);  hr_block3_bn1 = None
        hr_block3_dropout = self.hr_block3.dropout(hr_block3_relu);  hr_block3_relu = None
        hr_block3_conv2 = self.hr_block3.conv2(hr_block3_dropout);  hr_block3_dropout = None
        hr_block3_bn2 = self.hr_block3.bn2(hr_block3_conv2);  hr_block3_conv2 = None
        add_2 = torch.ops.quantized.add(hr_block3_bn2, hr_block2_relu_1, scale = 0.028587879613041878, zero_point = 184);  hr_block3_bn2 = hr_block2_relu_1 = None
        hr_block3_skip_add_activation_post_process = self.hr_block3.skip_add.activation_post_process(add_2);  add_2 = None
        hr_block3_relu_1 = self.hr_block3.relu(hr_block3_skip_add_activation_post_process);  hr_block3_skip_add_activation_post_process = None
        hr_head_conv = self.hr_head_conv(hr_block3_relu_1);  hr_block3_relu_1 = None
        hr_gap_pool1 = self.hr_gap.pool1(hr_head_conv);  hr_head_conv = None
        hr_gap_pool2 = self.hr_gap.pool2(hr_gap_pool1);  hr_gap_pool1 = None
        hr_gap_pool3 = self.hr_gap.pool3(hr_gap_pool2);  hr_gap_pool2 = None
        squeeze = hr_gap_pool3.squeeze(-1);  hr_gap_pool3 = None
        hr_dropout = self.hr_dropout(squeeze);  squeeze = None
        hr_fc = self.hr_fc(hr_dropout);  hr_dropout = None
        af_spatial_block1_downsample_0 = getattr(self.af_spatial_block1.downsample, "0")(stem_2)
        af_spatial_block1_downsample_1 = getattr(self.af_spatial_block1.downsample, "1")(af_spatial_block1_downsample_0);  af_spatial_block1_downsample_0 = None
        af_spatial_block1_conv1 = self.af_spatial_block1.conv1(stem_2)
        af_spatial_block1_bn1 = self.af_spatial_block1.bn1(af_spatial_block1_conv1);  af_spatial_block1_conv1 = None
        af_spatial_block1_relu = self.af_spatial_block1.relu(af_spatial_block1_bn1);  af_spatial_block1_bn1 = None
        af_spatial_block1_dropout = self.af_spatial_block1.dropout(af_spatial_block1_relu);  af_spatial_block1_relu = None
        af_spatial_block1_conv2 = self.af_spatial_block1.conv2(af_spatial_block1_dropout);  af_spatial_block1_dropout = None
        af_spatial_block1_bn2 = self.af_spatial_block1.bn2(af_spatial_block1_conv2);  af_spatial_block1_conv2 = None
        add_3 = torch.ops.quantized.add(af_spatial_block1_bn2, af_spatial_block1_downsample_1, scale = 0.35451966524124146, zero_point = 189);  af_spatial_block1_bn2 = af_spatial_block1_downsample_1 = None
        af_spatial_block1_skip_add_activation_post_process = self.af_spatial_block1.skip_add.activation_post_process(add_3);  add_3 = None
        af_spatial_block1_relu_1 = self.af_spatial_block1.relu(af_spatial_block1_skip_add_activation_post_process);  af_spatial_block1_skip_add_activation_post_process = None
        af_spatial_block2_conv1 = self.af_spatial_block2.conv1(af_spatial_block1_relu_1)
        af_spatial_block2_bn1 = self.af_spatial_block2.bn1(af_spatial_block2_conv1);  af_spatial_block2_conv1 = None
        af_spatial_block2_relu = self.af_spatial_block2.relu(af_spatial_block2_bn1);  af_spatial_block2_bn1 = None
        af_spatial_block2_dropout = self.af_spatial_block2.dropout(af_spatial_block2_relu);  af_spatial_block2_relu = None
        af_spatial_block2_conv2 = self.af_spatial_block2.conv2(af_spatial_block2_dropout);  af_spatial_block2_dropout = None
        af_spatial_block2_bn2 = self.af_spatial_block2.bn2(af_spatial_block2_conv2);  af_spatial_block2_conv2 = None
        add_4 = torch.ops.quantized.add(af_spatial_block2_bn2, af_spatial_block1_relu_1, scale = 0.15212903916835785, zero_point = 103);  af_spatial_block2_bn2 = af_spatial_block1_relu_1 = None
        af_spatial_block2_skip_add_activation_post_process = self.af_spatial_block2.skip_add.activation_post_process(add_4);  add_4 = None
        af_spatial_block2_relu_1 = self.af_spatial_block2.relu(af_spatial_block2_skip_add_activation_post_process);  af_spatial_block2_skip_add_activation_post_process = None
        af_spatial_block3_conv1 = self.af_spatial_block3.conv1(af_spatial_block2_relu_1)
        af_spatial_block3_bn1 = self.af_spatial_block3.bn1(af_spatial_block3_conv1);  af_spatial_block3_conv1 = None
        af_spatial_block3_relu = self.af_spatial_block3.relu(af_spatial_block3_bn1);  af_spatial_block3_bn1 = None
        af_spatial_block3_dropout = self.af_spatial_block3.dropout(af_spatial_block3_relu);  af_spatial_block3_relu = None
        af_spatial_block3_conv2 = self.af_spatial_block3.conv2(af_spatial_block3_dropout);  af_spatial_block3_dropout = None
        af_spatial_block3_bn2 = self.af_spatial_block3.bn2(af_spatial_block3_conv2);  af_spatial_block3_conv2 = None
        add_5 = torch.ops.quantized.add(af_spatial_block3_bn2, af_spatial_block2_relu_1, scale = 0.2804664969444275, zero_point = 142);  af_spatial_block3_bn2 = af_spatial_block2_relu_1 = None
        af_spatial_block3_skip_add_activation_post_process = self.af_spatial_block3.skip_add.activation_post_process(add_5);  add_5 = None
        af_spatial_block3_relu_1 = self.af_spatial_block3.relu(af_spatial_block3_skip_add_activation_post_process);  af_spatial_block3_skip_add_activation_post_process = None
        af_spatial_bn_dequant = self.af_spatial_bn.dequant(af_spatial_block3_relu_1);  af_spatial_block3_relu_1 = None
        af_spatial_bn_bn = self.af_spatial_bn.bn(af_spatial_bn_dequant);  af_spatial_bn_dequant = None
        af_spatial_bn_quant = self.af_spatial_bn.quant(af_spatial_bn_bn);  af_spatial_bn_bn = None
        af_temporal_pool1 = self.af_temporal_pool1(stem_2);  stem_2 = None
        af_temporal_pool2 = self.af_temporal_pool2(af_temporal_pool1);  af_temporal_pool1 = None
        af_temporal_block1_downsample_0 = getattr(self.af_temporal_block1.downsample, "0")(af_temporal_pool2)
        af_temporal_block1_downsample_1 = getattr(self.af_temporal_block1.downsample, "1")(af_temporal_block1_downsample_0);  af_temporal_block1_downsample_0 = None
        af_temporal_block1_conv1 = self.af_temporal_block1.conv1(af_temporal_pool2);  af_temporal_pool2 = None
        af_temporal_block1_bn1 = self.af_temporal_block1.bn1(af_temporal_block1_conv1);  af_temporal_block1_conv1 = None
        af_temporal_block1_relu = self.af_temporal_block1.relu(af_temporal_block1_bn1);  af_temporal_block1_bn1 = None
        af_temporal_block1_dropout = self.af_temporal_block1.dropout(af_temporal_block1_relu);  af_temporal_block1_relu = None
        af_temporal_block1_conv2 = self.af_temporal_block1.conv2(af_temporal_block1_dropout);  af_temporal_block1_dropout = None
        af_temporal_block1_bn2 = self.af_temporal_block1.bn2(af_temporal_block1_conv2);  af_temporal_block1_conv2 = None
        add_6 = torch.ops.quantized.add(af_temporal_block1_bn2, af_temporal_block1_downsample_1, scale = 0.13639073073863983, zero_point = 196);  af_temporal_block1_bn2 = af_temporal_block1_downsample_1 = None
        af_temporal_block1_skip_add_activation_post_process = self.af_temporal_block1.skip_add.activation_post_process(add_6);  add_6 = None
        af_temporal_block1_relu_1 = self.af_temporal_block1.relu(af_temporal_block1_skip_add_activation_post_process);  af_temporal_block1_skip_add_activation_post_process = None
        af_temporal_block2_conv1 = self.af_temporal_block2.conv1(af_temporal_block1_relu_1)
        af_temporal_block2_bn1 = self.af_temporal_block2.bn1(af_temporal_block2_conv1);  af_temporal_block2_conv1 = None
        af_temporal_block2_relu = self.af_temporal_block2.relu(af_temporal_block2_bn1);  af_temporal_block2_bn1 = None
        af_temporal_block2_dropout = self.af_temporal_block2.dropout(af_temporal_block2_relu);  af_temporal_block2_relu = None
        af_temporal_block2_conv2 = self.af_temporal_block2.conv2(af_temporal_block2_dropout);  af_temporal_block2_dropout = None
        af_temporal_block2_bn2 = self.af_temporal_block2.bn2(af_temporal_block2_conv2);  af_temporal_block2_conv2 = None
        add_7 = torch.ops.quantized.add(af_temporal_block2_bn2, af_temporal_block1_relu_1, scale = 0.04705817624926567, zero_point = 83);  af_temporal_block2_bn2 = af_temporal_block1_relu_1 = None
        af_temporal_block2_skip_add_activation_post_process = self.af_temporal_block2.skip_add.activation_post_process(add_7);  add_7 = None
        af_temporal_block2_relu_1 = self.af_temporal_block2.relu(af_temporal_block2_skip_add_activation_post_process);  af_temporal_block2_skip_add_activation_post_process = None
        af_temporal_data_conv = self.af_temporal_data_conv(af_temporal_block2_relu_1)
        af_temporal_data_bn = self.af_temporal_data_bn(af_temporal_data_conv);  af_temporal_data_conv = None
        af_temporal_data_relu = self.af_temporal_data_relu(af_temporal_data_bn);  af_temporal_data_bn = None
        af_temporal_gate_conv = self.af_temporal_gate_conv(af_temporal_block2_relu_1);  af_temporal_block2_relu_1 = None
        af_temporal_gate_bn = self.af_temporal_gate_bn(af_temporal_gate_conv);  af_temporal_gate_conv = None
        af_temporal_gate_sigmoid = self.af_temporal_gate_sigmoid(af_temporal_gate_bn);  af_temporal_gate_bn = None
        mul = torch.ops.quantized.mul(af_temporal_data_relu, af_temporal_gate_sigmoid, scale = 0.04535887390375137, zero_point = 0);  af_temporal_data_relu = af_temporal_gate_sigmoid = None
        af_temporal_gate_mul_activation_post_process = self.af_temporal_gate_mul.activation_post_process(mul);  mul = None
        af_temporal_global_pool = self.af_temporal_global_pool(af_temporal_gate_mul_activation_post_process);  af_temporal_gate_mul_activation_post_process = None
        add_8 = torch.ops.quantized.add(af_spatial_bn_quant, af_temporal_global_pool, scale = 0.1172984316945076, zero_point = 34);  af_spatial_bn_quant = af_temporal_global_pool = None
        af_fusion_add_activation_post_process = self.af_fusion_add.activation_post_process(add_8);  add_8 = None
        af_fusion_se_squeeze_pool1 = self.af_fusion_se.squeeze.pool1(af_fusion_add_activation_post_process)
        af_fusion_se_squeeze_pool2 = self.af_fusion_se.squeeze.pool2(af_fusion_se_squeeze_pool1);  af_fusion_se_squeeze_pool1 = None
        af_fusion_se_squeeze_pool3 = self.af_fusion_se.squeeze.pool3(af_fusion_se_squeeze_pool2);  af_fusion_se_squeeze_pool2 = None
        squeeze_1 = af_fusion_se_squeeze_pool3.squeeze(-1);  af_fusion_se_squeeze_pool3 = None
        af_fusion_se_fc1 = self.af_fusion_se.fc1(squeeze_1);  squeeze_1 = None
        af_fusion_se_relu = self.af_fusion_se.relu(af_fusion_se_fc1);  af_fusion_se_fc1 = None
        af_fusion_se_fc2 = self.af_fusion_se.fc2(af_fusion_se_relu);  af_fusion_se_relu = None
        af_fusion_se_sigmoid = self.af_fusion_se.sigmoid(af_fusion_se_fc2);  af_fusion_se_fc2 = None
        unsqueeze = af_fusion_se_sigmoid.unsqueeze(-1);  af_fusion_se_sigmoid = None
        mul_1 = torch.ops.quantized.mul(af_fusion_add_activation_post_process, unsqueeze, scale = 0.06966402381658554, zero_point = 48);  af_fusion_add_activation_post_process = unsqueeze = None
        af_fusion_se_q_mul_activation_post_process = self.af_fusion_se.q_mul.activation_post_process(mul_1);  mul_1 = None
        af_head_conv = self.af_head_conv(af_fusion_se_q_mul_activation_post_process);  af_fusion_se_q_mul_activation_post_process = None
        af_head_bn = self.af_head_bn(af_head_conv);  af_head_conv = None
        af_head_relu = self.af_head_relu(af_head_bn);  af_head_bn = None
        af_gap_pool1 = self.af_gap.pool1(af_head_relu);  af_head_relu = None
        af_gap_pool2 = self.af_gap.pool2(af_gap_pool1);  af_gap_pool1 = None
        af_gap_pool3 = self.af_gap.pool3(af_gap_pool2);  af_gap_pool2 = None
        squeeze_2 = af_gap_pool3.squeeze(-1);  af_gap_pool3 = None
        af_dropout = self.af_dropout(squeeze_2);  squeeze_2 = None
        af_fc = self.af_fc(af_dropout);  af_dropout = None
        dequant = self.dequant(af_fc);  af_fc = None
        dequant_1 = self.dequant(hr_fc);  hr_fc = None
        return (dequant, dequant_1)
        
