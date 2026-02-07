# Author: Yuyang Chen
# School: Zhejiang University
# Last Modified: 2026.2.7
# Description: Hardware-oriented quantized export helper functions and hooks.

"""Quantized model export helpers for TinyNet hardware deployment."""

import math
import os
import re
import shutil

import numpy as np
import torch
from torch import nn
from torch.fx import GraphModule
from torch.fx.passes.graph_drawer import FxGraphDrawer

NPU_NEED_EXPORT_NONPARAM_MODULES = (
    nn.quantized.Softmax,
    nn.Softmax,
    nn.Sigmoid,
    nn.ReLU,
    nn.AvgPool1d,
    nn.MaxPool1d,
    torch.nn.quantized.Quantize,
)
NPU_NEED_EXPORT_PARAM_MODULES = (nn.quantized.Conv1d, nn.quantized.Linear)
NPU_NEED_EXPORT_MODULES = NPU_NEED_EXPORT_PARAM_MODULES + NPU_NEED_EXPORT_NONPARAM_MODULES
NPU_NEED_EXPORT_FUNCTIONS = (torch.quantize_per_tensor, torch.ops.quantized.add_relu)


class ExportActHook(nn.Module):
    """Hook module that exports quantized activations and per-layer params."""

    # Purpose: Initialize class state and runtime configuration.
    # Inputs: Parameters defined in `__init__` signature.
    # Outputs: Return value produced by `__init__`.
    # Assumptions: Caller provides valid types/shapes for this operation.
    def __init__(self, export_dir:str, name:str, mod_type, *args, **kwargs):
        super().__init__()
        self.export_dir = export_dir
        self.name = name
        self.type = mod_type

    # Purpose: Execute forward computation for this module.
    # Inputs: Parameters defined in `forward` signature.
    # Outputs: Return value produced by `forward`.
    # Assumptions: Caller provides valid types/shapes for this operation.
    def forward(self, x, *args, **kwargs):
        # Only export the activation
        x = self.__check_input(x, *args, **kwargs)
        for i, t in enumerate(x):
            if "quantize_per_" in self.name or self.name == "quant":
                self.export_activation(t, f"input_{i}")
            else:
                self.export_activation(t, f"{self.name}")
            print(f"[Export Act] {self.name}, shape={t.shape}, dtype={t.dtype}")
        if "add_relu" in str(self.type).lower():
            assert len(args) >= 2, "add_relu should have two inputs tensors"
            with open(os.path.join(self.export_dir, "other_params.txt"), "a+", encoding="utf-8") as f:
                f.write(f"{self.name} scale: {self.get_addrelu_scale_param(args[0].q_scale(), args[1].q_scale(), x[0].q_scale())}\n")
                # print("==========================================================")
                # print(f"scale0:{args[0].q_scale()},scale1:{args[1].q_scale()},scale2:{x[0].q_scale()}")
        elif "sigmoid" in str(self.type).lower() or "softmax" in str(self.type).lower():
            with open(os.path.join(self.export_dir, "other_params.txt"), "a+", encoding="utf-8") as f:
                f.write(f"{self.name} scale: {self.get_nl_scale_param(x[0].q_scale())}\n")
        return x

    # Purpose: Implement `__check_input` for the TinyNet workflow.
    # Inputs: Parameters defined in `__check_input` signature.
    # Outputs: Return value produced by `__check_input`.
    # Assumptions: Caller provides valid types/shapes for this operation.
    def __check_input(self, x, *args, **kwargs):
        valid_act = []
        if isinstance(x, torch.Tensor):
            valid_act.append(x)
        else:
            if hasattr(x, '__iter__'):
                for i, t in enumerate(x):
                    valid_act.append(t)
            else:
                print(f"Warning: Output of {self.name} is not a tensor, nor an iterable of tensors.")
        return valid_act

    # Purpose: Implement `export_params_hook` for the TinyNet workflow.
    # Inputs: Parameters defined in `export_params_hook` signature.
    # Outputs: Return value produced by `export_params_hook`.
    # Assumptions: Caller provides valid types/shapes for this operation.
    def export_params_hook(self, module, inputs, outputs, *args, **kwargs):
        # Only export the parameters
        print(f"[Exporting Params] {self.name}, module={module}, mod_type={self.type}")
        if isinstance(inputs, (tuple, list)):
            if len(inputs) == 1:
                inputs = inputs[0]
            else:
                raise ValueError(f"Multiple inputs are not supported in {self.name}")
        if isinstance(outputs, (tuple, list)):
            if len(outputs) == 1:
                outputs = outputs[0]
            else:
                raise ValueError(f"Multiple outputs are not supported in {self.name}")
        if hasattr(module, "kernel_size"):
            if isinstance(module.kernel_size, int):
                ks = module.kernel_size
            elif len(module.kernel_size)==1:
                ks = module.kernel_size[0]
            else:
                assert module.kernel_size[0] == module.kernel_size[1], "Only support square kernel size"
                ks = module.kernel_size[0]
            if ks == 1:
                layer_type = "conv1"
            elif ks in (3, 5, 7):
                layer_type = "conv"
            else:
                raise ValueError(f"Unsupported kernel size {ks} in {self.name}")
        elif isinstance(module, nn.quantized.Linear):
            if inputs.ndim >= 2:
                layer_type = "linear"
            else:
                raise ValueError(f"Unsupported input shape {inputs.ndim} in {self.name}")
        else:
            raise ValueError(f"Unsupported layer type in {self.name}")
        self.export_layer(module, self.name, inputs.q_scale(), layer_type)

        if hasattr(module, "weight"):
            with open(os.path.join(self.export_dir, "other_params.txt"), "a+", encoding="utf-8") as f:
                f.write(f"{self.name} scale: {self.get_scale_params(inputs.q_scale(),outputs.q_scale(), module.weight().q_scale(), layer_type)}\n")

    # Purpose: Implement `_save_hex` for the TinyNet workflow.
    # Inputs: Parameters defined in `_save_hex` signature.
    # Outputs: Return value produced by `_save_hex`.
    # Assumptions: Caller provides valid types/shapes for this operation.
    def _save_hex(self, filename, hex_lines):
        path = os.path.join(self.export_dir, filename)
        with open(path, 'w') as f:
            f.write('\n'.join(hex_lines) + '\n')
        print(f"[Exported] {path}")

    # Purpose: Implement `_pack_128bit_segments` for the TinyNet workflow.
    # Inputs: Parameters defined in `_pack_128bit_segments` signature.
    # Outputs: Return value produced by `_pack_128bit_segments`.
    # Assumptions: Caller provides valid types/shapes for this operation.
    def _pack_128bit_segments(self, data_list, reverse_bytes=False):
        """核心打包逻辑：补齐16字节 -> 转bytes -> 按32bit切分"""
        # 1. 补齐到 16 个元素 (不足补0)
        if len(data_list) < 16:
            data_list.extend([0] * (16 - len(data_list)))

        # 2. 转为 int8 字节流
        bytes_128 = bytes(np.array(data_list[:16], dtype=np.int8))

        segments = []
        # 3. 切分为 4 个 32-bit 段
        # 你的旧代码逻辑中：
        # - pack_linear_weights (旧): 倒序切分 (seg_idx 3,2,1,0)
        # - pack_fc_big/small (旧): 正序切分 (seg_idx 0,1,2,3)
        # 为了兼容你的 fc_big/small，这里需要默认使用正序 (0,1,2,3)
        # 并在 Conv/Linear 特殊情况手动倒序，或者统一接口。
        #
        # 经过仔细核对你的旧代码：
        # pack_conv_weights 使用了 reversed(range(4)) -> 倒序
        # pack_linear_weights 使用了 reversed(range(4)) -> 倒序
        # pack_fc_small_weights 使用了 range(4) -> 正序 !!!
        # pack_fc_big_weights 使用了 range(4) -> 正序 !!!

        # 因此，我们需要一个参数来控制切分顺序
        pass

    # Purpose: Implement `_pack_segments_smart` for the TinyNet workflow.
    # Inputs: Parameters defined in `_pack_segments_smart` signature.
    # Outputs: Return value produced by `_pack_segments_smart`.
    # Assumptions: Caller provides valid types/shapes for this operation.
    def _pack_segments_smart(self, data_list, reverse_bytes=False, split_order="reversed"):
        """
        split_order:
          - 'reversed': 3, 2, 1, 0 (用于 Conv, Conv1x1)
          - 'forward': 0, 1, 2, 3 (用于 FC Small, FC Big)
        """
        if len(data_list) < 16:
            data_list.extend([0] * (16 - len(data_list)))

        bytes_128 = bytes(np.array(data_list[:16], dtype=np.int8))
        segments = []

        range_iter = range(4) if split_order == 'forward' else reversed(range(4))

        for i in range_iter:
            start = i * 4
            chunk = bytes_128[start: start + 4]
            if reverse_bytes:
                chunk = chunk[::-1]
            segments.append(''.join(f'{b:02X}' for b in chunk))
        return segments

    # Purpose: Implement `get_nl_scale_param` for the TinyNet workflow.
    # Inputs: Parameters defined in `get_nl_scale_param` signature.
    # Outputs: Return value produced by `get_nl_scale_param`.
    # Assumptions: Caller provides valid types/shapes for this operation.
    def get_nl_scale_param(self, scale, max_bits=8):

        """
        计算 Softmax 或 Sigmoid 等非线性层的量化缩放参数 (M, N)。

        将浮点 scale 近似转换为 M * 2^(-N) 的形式。

        Args:
            scale (float or torch.Tensor): 层的量化比例值。
                                           通常为 Softmax 或 Sigmoid 层的输入/输出 q_scale()。
            max_bits (int, optional): M 和 N 的位宽限制. Defaults to 8.

        Returns:
            str: 8位宽的十六进制字符串（例如 "0000063F"）。（0000NNMM)
        """
        x = scale
        if isinstance(x, torch.Tensor): x = x.item()
        if x <= 0: return "00", "00"

        best_error = float('inf')
        best_M, best_N = 0, 0
        for n in range(32):
            M = round(x * (2 ** n))
            if 0 < M < (1 << max_bits):
                error = abs(x - M * (2 ** -n))
                if error < best_error:
                    best_error, best_M, best_N = error, M, n

        combined_val = (best_N << 8) | best_M
        return f"{combined_val:08X}"

    # Purpose: Implement `get_addrelu_scale_param` for the TinyNet workflow.
    # Inputs: Parameters defined in `get_addrelu_scale_param` signature.
    # Outputs: Return value produced by `get_addrelu_scale_param`.
    # Assumptions: Caller provides valid types/shapes for this operation.
    def get_addrelu_scale_param(self, opt1_in_scale, opt2_in_scale, out_scale, max_bits=8):
        """
                计算addrelu 的硬件缩放参数 opt1(M, N),opt2(M,N)。

                Args:
                    opt1_in_scale (float): 作为opt1上一层ouotput scale。
                    opt2_in_scale (float): 作为opt2上一层output scale
                    out_scale (float): 当前层输出scale。
                    max_bits (int, optional): M 和 N 的位宽限制。默认为 8。

                Returns:
                    str: opt1_scale,opt2_scale (两个8位宽的十六进制字符串（例如 "0000063F"））。
                """
        x_opt1 = opt1_in_scale / out_scale
        x_opt2 = opt2_in_scale / out_scale
        if isinstance(x_opt1, torch.Tensor): x_opt1 = x_opt1.item()
        if x_opt1 <= 0: return "00", "00"

        if isinstance(x_opt2, torch.Tensor): x_opt2 = x_opt2.item()
        if x_opt2 <= 0: return "00", "00"

        best_error_opt1 = float('inf')
        best_M_opt1, best_N_opt1 = 0, 0
        for n in range(32):
            M = round(x_opt1 * (2 ** n))
            if 0 < M < (1 << max_bits):
                error = abs(x_opt1 - M * (2 ** -n))
                if error < best_error_opt1:
                    best_error_opt1, best_M_opt1, best_N_opt1 = error, M, n
        combined_val1 = (best_N_opt1 << 8) | best_M_opt1
        best_error_opt2 = float('inf')
        best_M_opt2, best_N_opt2 = 0, 0
        for n in range(32):
            M = round(x_opt2 * (2 ** n))
            if 0 < M < (1 << max_bits):
                error = abs(x_opt2 - M * (2 ** -n))
                if error < best_error_opt2:
                    best_error_opt2, best_M_opt2, best_N_opt2 = error, M, n
        combined_val2 = (best_N_opt2 << 8) | best_M_opt2
        return f"{combined_val1:08X}", f"{combined_val2:08X}"

    # Purpose: Implement `get_scale_params` for the TinyNet workflow.
    # Inputs: Parameters defined in `get_scale_params` signature.
    # Outputs: Return value produced by `get_scale_params`.
    # Assumptions: Caller provides valid types/shapes for this operation.
    def get_scale_params(self, in_scale, out_scale, w_scale, type, max_bits=8):
        """
        计算线性层 (Conv/Linear) 的硬件缩放参数 (M, N)。

        公式: Real_Scale = (Input_Scale * Weight_Scale) / Output_Scale
        近似: Real_Scale ≈ M * 2^(-N)

        Args:
            in_scale (float): 上一层的 Output Scale (当前层的 Input Scale)。
            out_scale (float): 当前层的 Output Scale (q_scale)。
            w_scale (float): 当前层的权重 Scale (weight().q_scale())。
            type (str): 层类型，决定打包顺序。
                        - "conv": 输出 0000NNMM (N在高, M在低)
                        - 其他  : 输出 0000MMNN (M在高, N在低)
            max_bits (int, optional): M 和 N 的位宽限制。默认为 8。

        Returns:
            str: 8位宽的十六进制字符串（例如 "0000063F"）。
        """
        x = (in_scale * w_scale) / out_scale
        if isinstance(x, torch.Tensor): x = x.item()
        if x <= 0: return "00", "00"

        best_error = float('inf')
        best_M, best_N = 0, 0
        for n in range(32):
            M = round(x * (2 ** n))
            if 0 < M < (1 << max_bits):
                error = abs(x - M * (2 ** -n))
                if error < best_error:
                    best_error, best_M, best_N = error, M, n

        if type == "conv":

            combined_val = (best_N << 8) | best_M
        else:
            combined_val = (best_M << 16) | best_N

        return f"{combined_val:08X}"

    # Purpose: Implement `export_input_signal` for the TinyNet workflow.
    # Inputs: Parameters defined in `export_input_signal` signature.
    # Outputs: Return value produced by `export_input_signal`.
    # Assumptions: Caller provides valid types/shapes for this operation.
    def export_input_signal(self, activation, name):
        """
        导出量化后的输入信号 (如 ECG 信号)。

        逻辑：
        1. 读取量化整数值 (int_repr)
        2. 减去 Zero Point (zp)
        3. 转换为 8-bit Hex (无符号/补码格式，如 -1 -> FF)
        4. 不进行 128-bit 打包，按顺序逐个写入，每行一个字节。

        Args:
            activation: 量化的输入 Tensor
            name: 导出文件名
        """
        if not hasattr(activation, 'int_repr'): return

        # 获取 int8 数据 (Batch, Channel, Length)
        data = activation.int_repr().cpu().numpy().astype(np.int32)
        zp = activation.q_zero_point()

        # 维度处理: 假设 Batch Size = 1
        # (1, 1, 1500) -> (1, 1500)
        if data.ndim == 3:
            data = data[0]
        elif data.ndim == 1:
            data = data.reshape(1, -1)

        hex_lines = []

        # 遍历通道 (ECG通常是单通道，即 data 只有一行)
        for channel_data in data:
            # 减去零点 (De-zero)
            adjusted_data = channel_data - zp

            for val in adjusted_data:
                # 转换为 8-bit Hex (处理负数: -1 -> FF)
                # (val + 256) % 256 确保结果在 0-255 之间
                hex_val = f"{(val + 256) % 256:02X}"
                hex_lines.append(hex_val)

        self._save_hex(f"{name}.txt", hex_lines)

    # Purpose: Implement `export_activation` for the TinyNet workflow.
    # Inputs: Parameters defined in `export_activation` signature.
    # Outputs: Return value produced by `export_activation`.
    # Assumptions: Caller provides valid types/shapes for this operation.
    def export_activation(self, activation, name, tile_size=16):
        """
                导出激活值 (Feature Maps)。

                自动处理数据维度展平、Padding 以及硬件所需的通道倒序排列。

                Args:
                    activation (torch.Tensor): 量化的激活张量 (需具备 int_repr 方法)。
                    name (str): 导出文件名标识 (不含后缀)。
                    tile_size (int, optional): 硬件并行处理的通道块大小。默认为 16。
        """
        if not hasattr(activation, 'int_repr'): return
        data = activation.int_repr()[0].cpu().numpy().astype(np.int32)
        zp = activation.q_zero_point()

        # 维度修正逻辑
        if data.ndim == 3:
            C, H, W = data.shape
            data = data.reshape(C, -1)
        elif data.ndim == 1:
            data = data.reshape(1, -1)

        C, T = data.shape
        hex_lines = []
        num_tiles = (C + tile_size - 1) // tile_size

        for tile in range(num_tiles):
            start_ch = tile * tile_size
            end_ch = min(start_ch + tile_size, C)
            for t in range(T):
                group = []
                for ch in reversed(range(start_ch, start_ch + 16)):
                    val = (data[ch, t] - zp) if ch < end_ch else 0
                    val = max(-128, min(127, val))
                    group.append(val)
                # 激活导出通常使用倒序切分
                hex_lines.extend(self._pack_segments_smart(group, reverse_bytes=False, split_order="reversed"))

        self._save_hex(f"{name}.txt", hex_lines)

    # Purpose: Implement `export_layer` for the TinyNet workflow.
    # Inputs: Parameters defined in `export_layer` signature.
    # Outputs: Return value produced by `export_layer`.
    # Assumptions: Caller provides valid types/shapes for this operation.
    def export_layer(self, module, name, input_scale, layer_type="conv"):
        """
        导出层的权重 (Weights) 和偏置 (Bias)。

        根据 layer_type 自动选择不同的 Bias 缩放策略和权重打包顺序。

        Args:
            module (torch.nn.Module): 量化层模块 (需包含 weight() 和 bias())。
            name (str): 导出文件名标识。
            input_scale (float): 上一层的输出 Scale，用于计算 Bias 的定点缩放。
            layer_type (str, optional): 层类型，决定导出策略。默认为 "conv"。
                - 'conv':    标准卷积 (3x3 等)。Bias 缩放含 ceil。权重按 Shift 寄存器顺序打包。
                - 'conv1':   Pointwise卷积 (1x1) 或 Shortcut。Bias 缩放为浮点除。权重按 Input-major 打包。
                - 'fc':      Small FC (输入特征遍历)。权重按 Input-major 且带字节反转打包。
                - 'linear':  Big FC (分块输出遍历)。权重按 Tile Output-major 且带字节反转打包。
        """
        weight = module.weight()
        bias = module.bias()
        w_int = weight.int_repr().cpu().numpy().astype(np.int8)

        # --- 1. Bias 处理 ---
        if layer_type in ["conv", "conv1"]:
            scale_factor = math.ceil(w_int.shape[1] / 16)
        else:
            scale_factor = 1.0

        bias_scale = input_scale * weight.q_scale() * scale_factor
        if bias_scale == 0: bias_scale = 1.0

        b_vals = bias.detach().cpu().numpy()
        b_int = np.round(b_vals / bias_scale).astype(np.int32)

        # FC类型Bias补齐
        pad = (16 - (len(b_int) % 16)) % 16
        if pad > 0: b_int = np.concatenate([b_int, np.zeros(pad, dtype=np.int32)])

        b_lines = [int(x).to_bytes(4, 'big', signed=True).hex().upper() for x in b_int]
        self._save_hex(f"{name}_bias.txt", b_lines)

        # --- 2. Weight 处理 ---
        w_lines = []

        if layer_type == "conv1":
            # Conv1x1 / Pointwise (类似 Linear 结构，但旧代码使用 pack_linear_weights)
            out_ch, in_ch, _ = w_int.shape
            w_2d = w_int.reshape(out_ch, in_ch)
            padded_in = ((in_ch + 15) // 16) * 16
            padded_out = ((out_ch + 15) // 16) * 16
            w_pad = np.zeros((padded_out, padded_in), dtype=np.int8)
            w_pad[:out_ch, :in_ch] = w_2d

            for ic in range(0, padded_in, 16):
                for oc in range(padded_out):  # 必须遍历到 padded_out
                    # 取 16 个输入通道 (倒序)
                    group = [w_pad[oc, ic + i] for i in reversed(range(16))]

                    w_lines.extend(self._pack_segments_smart(group, reverse_bytes=False, split_order="reversed"))

        elif layer_type == "conv":
            out_ch, in_ch, k_sz = w_int.shape
            padded_out = ((out_ch + 15) // 16) * 16
            padded_in = ((in_ch + 15) // 16) * 16
            w_pad = np.zeros((padded_out, padded_in, k_sz), dtype=np.int8)
            w_pad[:out_ch, :in_ch, :] = w_int

            for ob in range(0, padded_out, 16):
                for ib in range(0, padded_in, 16):
                    for shift in range(16):
                        for k in range(k_sz):
                            group = []
                            for i in reversed(range(16)):
                                group.append(w_pad[ob + i, ib + (i + shift) % 16, k])
                            # 旧代码 pack_conv_weights 使用 reversed 切分
                            w_lines.extend(
                                self._pack_segments_smart(group, reverse_bytes=False, split_order="reversed"))


        elif layer_type == "fc":
            out_ch, in_ch = w_int.shape
            for i in range(in_ch):
                # 遍历所有输出，步长 16
                for j in range(0, out_ch, 16):
                    group = []
                    # 收集接下来的 16 个输出通道的权重
                    for k in range(16):
                        if (j + k) < out_ch:
                            group.append(w_int[j + k, i])
                        else:
                            group.append(0)  # Padding
                    w_lines.extend(self._pack_segments_smart(group, reverse_bytes=True, split_order="forward"))


        elif layer_type == "linear":
            # 对应旧代码: fc_big (Blocked Column Major)
            out_ch, in_ch = w_int.shape
            tiles = math.ceil(out_ch / 16)
            for t in range(tiles):
                is_last = (t == tiles - 1)
                r_start, r_end = t * 16, out_ch if is_last else (t + 1) * 16
                for i in range(in_ch):
                    group = [w_int[j, i] for j in range(r_start, r_end)]
                    w_lines.extend(self._pack_segments_smart(group, reverse_bytes=True, split_order="forward"))

        self._save_hex(f"{name}_weight.txt", w_lines)


class HexExporter(nn.Module):
    """Inject export hooks into an FX graph and dump activation/parameter hex files."""

    # Purpose: Initialize class state and runtime configuration.
    # Inputs: Parameters defined in `__init__` signature.
    # Outputs: Return value produced by `__init__`.
    # Assumptions: Caller provides valid types/shapes for this operation.
    def __init__(self, export_dir, quantized_model: GraphModule):
        super().__init__()
        self.hook_modules = {}
        self.hook_handles = {}
        self.hook_nodes = []
        self.export_dir = export_dir
        if os.path.exists(export_dir):
            shutil.rmtree(export_dir)
        os.makedirs(export_dir, exist_ok=True)
        self.qmodel = quantized_model
        self.qmodel.eval()
        self.__inject_export_nodes(self.qmodel)
        self.exported_activations = {}
        self.exported_weights = {}
        self.exported_params = {}

    # Purpose: Implement `__inject_export_nodes` for the TinyNet workflow.
    # Inputs: Parameters defined in `__inject_export_nodes` signature.
    # Outputs: Return value produced by `__inject_export_nodes`.
    # Assumptions: Caller provides valid types/shapes for this operation.
    def __inject_export_nodes(self, gm: GraphModule):
        # 备一份 modules dict，方便通过 node.target 找到对应 module
        modules = dict(gm.named_modules())
        suffix = 0

        for node in list(gm.graph.nodes):
            # 这里只举一个 conv 的例子，其他 linear / fc / softmax / sigmoid 一样的套路
            if node.op == "call_module":
                mod = modules[node.target]
                if isinstance(mod, NPU_NEED_EXPORT_MODULES):
                    layer_name = node.target.replace(".", "_")
                    hook_module = ExportActHook(self.export_dir, layer_name, type(mod))
                    hook_module_name = f"hook_export_{layer_name}"
                    self.qmodel.register_module(hook_module_name, hook_module)
                    print(f"[Hook] {hook_module_name} for {node.target} activation")
                    self.hook_modules[layer_name] = (hook_module_name, hook_module,)
                    if isinstance(mod, NPU_NEED_EXPORT_PARAM_MODULES):
                        self.hook_handles[layer_name] = mod.register_forward_hook(hook_module.export_params_hook)
                        print(f"[Hook] {hook_module_name} for {node.target} params")
                    with gm.graph.inserting_after(node):
                        new_node = gm.graph.call_module(
                            hook_module_name,
                            args=(node, *node.args),
                        )
                    self.hook_nodes.append(new_node)
                    self.__supplement_node_info(node, mod)
                else:
                    print(f"[Skip] {node.target}")
            elif node.op == "call_function" or node.op == "call_method":
                func = node.target
                if node.target in NPU_NEED_EXPORT_FUNCTIONS:
                    func_name = node.name.replace(".", "_")
                    if func_name in self.hook_modules:
                        suffix += 1
                        func_name = f"{func_name}_{suffix}"
                    hook_module = ExportActHook(self.export_dir, func_name, str(node.target))
                    hook_module_name = f"hook_export_{func_name}"
                    self.qmodel.register_module(hook_module_name, hook_module)
                    print(f"[Inject] {hook_module_name} for {node.target} activation")
                    self.hook_modules[func_name] = (hook_module_name, hook_module,)
                    self.__supplement_node_info(node, func)
                    with gm.graph.inserting_after(node):
                        new_node = gm.graph.call_module(
                            hook_module_name,
                            args=(node,*node.args),
                        )
                    self.hook_nodes.append(new_node)
                else:
                    print(f"[Skip] {node.target}")
            else:
                print(f"[Skip] {node.target}")
        gm.recompile()
        return gm

    # Purpose: Implement `__supplement_node_info` for the TinyNet workflow.
    # Inputs: Parameters defined in `__supplement_node_info` signature.
    # Outputs: Return value produced by `__supplement_node_info`.
    # Assumptions: Caller provides valid types/shapes for this operation.
    def __supplement_node_info(self, node, target):
        if "extra_label" not in node.meta:
            node.meta["extra_label"] = {}
        if hasattr(target, "stride"):
            node.meta["extra_label"].update({"stride": target.stride})
        if hasattr(target, "padding"):
            node.meta["extra_label"].update({"padding": target.padding})
        if hasattr(target, "kernel_size"):
            node.meta["extra_label"].update({"kernel_size": target.kernel_size})
        if hasattr(target, "in_channels"):
            node.meta["extra_label"].update({"in_channels": target.in_channels})
        if hasattr(target, "out_channels"):
            node.meta["extra_label"].update({"out_channels": target.out_channels})
        return node


    # Purpose: Implement `export_` for the TinyNet workflow.
    # Inputs: Parameters defined in `export_` signature.
    # Outputs: Return value produced by `export_`.
    # Assumptions: Caller provides valid types/shapes for this operation.
    def export_(self, inputs, *args, **kwargs):
        self.qmodel.eval()
        if isinstance(inputs, (list, tuple)):
            self.qmodel(*inputs)
        elif isinstance(inputs, dict):
            self.qmodel(**inputs)
        else:
            self.qmodel(inputs)
        print(f"[Export Finish] all activations and weights are saved in {self.export_dir}")

    # Purpose: Implement `remove_hooks` for the TinyNet workflow.
    # Inputs: Parameters defined in `remove_hooks` signature.
    # Outputs: Return value produced by `remove_hooks`.
    # Assumptions: Caller provides valid types/shapes for this operation.
    def remove_hooks(self):
        for node in self.hook_nodes:
            self.qmodel.graph.erase_node(node)
        for name, handle in self.hook_handles.items():
            handle.remove()
        for name, (module_name, module) in self.hook_modules.items():
            delattr(self.qmodel, module_name)
        self.hook_handles.clear()
        self.hook_nodes.clear()
        self.hook_modules.clear()
        self.qmodel.recompile()

class CustomFxGraphDrawer(FxGraphDrawer):
    """Graph drawer that appends module meta labels to node text."""

    # Purpose: Return derived values required by downstream steps.
    # Inputs: Parameters defined in `_get_node_label` signature.
    # Outputs: Return value produced by `_get_node_label`.
    # Assumptions: Caller provides valid types/shapes for this operation.
    def _get_node_label(self, module, node, skip_node_names_in_args, parse_stack_trace):
        # 先调用父类方法获取原始标签
        label:str = super()._get_node_label(module, node, skip_node_names_in_args, parse_stack_trace)
        extra_label = node.meta.get("extra_label", None)
        if isinstance(extra_label, dict):
            insert_label_list = []
            for k, v in extra_label.items():
                insert_label_list.append(f"{k}={v}\n")
            insert_label_str = "|".join(insert_label_list)
            if label.endswith("\n}"):
                label = label[:-2]
            else:
                label = label[:-1]
            label = label + insert_label_str + "\n}"
        return label

# Purpose: Implement `patch_module_file` for the TinyNet workflow.
# Inputs: Parameters defined in `patch_module_file` signature.
# Outputs: Return value produced by `patch_module_file`.
# Assumptions: Caller provides valid types/shapes for this operation.
def patch_module_file(path):
    """Patch generated FX `module.py` for compatibility with target runtime."""

    with open(path, 'r', encoding='utf-8') as f:
        code = f.read()

    # -----------------------------
    # 1. 在 from torch.nn import * 前插入内容
    # -----------------------------
    insert_before_import = """
import sys
import os
import collections

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上一级目录（父目录）
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

"""

    code = re.sub(
        r"(?=from\s+torch\.nn\s+import\s+\*)",
        insert_before_import,
        code,
        count=1
    )

    # -----------------------------
    # 2. 在 class xxxx 定义前插入 traverse_and_patch_modules
    # -----------------------------
    traverse_fn = r"""
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

"""

    code = re.sub(
        r"(?=class\s+\w+\s*\()",
        traverse_fn,
        code,
        count=1
    )

    # -----------------------------
    # 3. 在 self.load_state_dict 前插入 traverse_and_patch_modules(self)
    # -----------------------------
    code = re.sub(
        r"(?=self\.load_state_dict)",
        "traverse_and_patch_modules(self)\n        ",
        code
    )

    # -----------------------------
    # 4. 给所有 torch.load(...) 添加 weights_only=True
    # -----------------------------
    # Purpose: Implement `add_weights_only` for the TinyNet workflow.
    # Inputs: Parameters defined in `add_weights_only` signature.
    # Outputs: Return value produced by `add_weights_only`.
    # Assumptions: Caller provides valid types/shapes for this operation.
    def add_weights_only(match):
        text = match.group()
        if "weights_only" in text:
            return text
        return text[:-1] + ", weights_only=True)"

    code = re.sub(
        r"torch\.load\s*\([^)]*\)",
        add_weights_only,
        code
    )

    # -----------------------------
    # 保存文件
    # -----------------------------
    with open(path, 'w', encoding='utf-8') as f:
        f.write(code)

    print("module.py patch completed!")
