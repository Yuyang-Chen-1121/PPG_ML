# Author: Yuyang Chen
# School: Zhejiang University
# Last Modified: 2026.2.7
# Description: Configuration accessor and runtime device utilities.

"""Configuration loading and typed access helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import yaml
import torch

PathLike = Union[str, Sequence[str]]


@dataclass
class ConfigAccessor:
    """Typed accessor over a nested configuration dictionary."""

    data: Dict[str, Any]

    # Purpose: Resolve a nested path in config.
    # Inputs: path as "a.b.c" or sequence of keys; default fallback value.
    # Outputs: the resolved value or default if missing.
    # Assumptions: dictionary-like hierarchy.
    def get(self, path: PathLike, default: Any = None) -> Any:
        keys: Iterable[str] = path.split(".") if isinstance(path, str) else path
        node: Any = self.data
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    # Purpose: Read an integer config field.
    # Inputs: path, default value.
    # Outputs: int value.
    # Assumptions: path value is int-castable.
    def get_int(self, path: PathLike, default: int) -> int:
        return int(self.get(path, default))

    # Purpose: Read a floating-point config field.
    # Inputs: path, default value.
    # Outputs: float value.
    # Assumptions: path value is float-castable.
    def get_float(self, path: PathLike, default: float) -> float:
        return float(self.get(path, default))

    # Purpose: Read a boolean config field.
    # Inputs: path, default value.
    # Outputs: bool value.
    # Assumptions: path value is bool-castable.
    def get_bool(self, path: PathLike, default: bool) -> bool:
        return bool(self.get(path, default))

    # Purpose: Read a list config field.
    # Inputs: path, default list.
    # Outputs: list value.
    # Assumptions: path value is list-like.
    def get_list(self, path: PathLike, default: Optional[List[Any]] = None) -> List[Any]:
        value = self.get(path, default if default is not None else [])
        return list(value)


# Purpose: Load YAML config file into accessor wrapper.
# Inputs: config file path.
# Outputs: ConfigAccessor with parsed dictionary.
# Side effects: reads file from disk.
def load_config(config_path: str) -> ConfigAccessor:
    with open(config_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return ConfigAccessor(payload)


# Purpose: Build BPM bin centers from label config.
# Inputs: ConfigAccessor.
# Outputs: numpy array of BPM bins.
# Assumptions: labels.bpm_step > 0 and bins fit hardware softmax limit.
def get_bpm_bins(cfg: ConfigAccessor) -> np.ndarray:
    bpm_min = cfg.get_int("labels.bpm_min", 30)
    bpm_max = cfg.get_int("labels.bpm_max", 242)
    bpm_step = cfg.get_int("labels.bpm_step", 2)
    return np.arange(bpm_min, bpm_max, bpm_step, dtype=np.float32)

# Purpose: Implement `get_device` for the TinyNet workflow.
# Inputs: Parameters defined in `get_device` signature.
# Outputs: Return value produced by `get_device`.
# Assumptions: Caller provides valid types/shapes for this operation.
def get_device(cfg):
    # Read preferences from config, default to sensible order
    preferences = cfg.get_list("device.preference", ["cuda", "mps", "cpu"])
    
    for dev_type in preferences:
        dev_type = dev_type.lower()
        if dev_type == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if dev_type == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if dev_type == "cpu":
            return torch.device("cpu")
            
    # Fallback
    return torch.device("cpu")
