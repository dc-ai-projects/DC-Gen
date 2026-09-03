import torch
from torch import nn


def get_dtype_from_str(dtype: str) -> torch.dtype:
    if dtype == "fp32":
        return torch.float32
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "int32":
        return torch.int32
    raise NotImplementedError(f"dtype {dtype} is not supported")


def get_dtype(model: nn.Module) -> torch.dtype:
    return model.parameters().__next__().dtype


def convert_to_dtype_recursive(data, dtype: torch.dtype):
    if isinstance(data, torch.Tensor):
        if data.is_floating_point():
            return data.to(dtype)
        else:
            return data
    elif isinstance(data, dict):
        return {k: convert_to_dtype_recursive(v, dtype) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(convert_to_dtype_recursive(v, dtype) for v in data)
    else:
        return data
