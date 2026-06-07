import functools

import numpy as np
import onnxruntime as ort

MAX_SEQ_LEN = 32


def available_providers():
    available = ort.get_available_providers()
    preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return [p for p in preferred if p in available]


def to_numpy(obj):
    try:
        import torch
    except ImportError:
        return obj
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy()
    return obj


def pad_to_length(arr, target_length, pad_value=-1):
    current = arr.shape[1]
    if current < target_length:
        if arr.ndim == 2:
            return np.pad(
                arr, ((0, 0), (0, target_length - current)), constant_values=pad_value
            )
        elif arr.ndim == 3:
            return np.pad(
                arr,
                ((0, 0), (0, target_length - current), (0, 0)),
                constant_values=pad_value,
            )
    return arr[:, :target_length]


def onnx_call(func):
    @functools.wraps(func)
    def wrapper(self, *args):
        try:
            import torch

            is_torch = any(isinstance(a, torch.Tensor) for a in args)
        except ImportError:
            is_torch = False

        numpy_args = tuple(to_numpy(a) for a in args)

        result = func(self, *numpy_args)

        if is_torch:
            return torch.from_numpy(result)

        return result

    return wrapper


class OnnxMeter:
    def __init__(self, onnx_path: str):
        self.session = ort.InferenceSession(onnx_path, providers=available_providers())

    @onnx_call
    def __call__(self, accent_input, pos_input):
        accent_input = pad_to_length(accent_input, MAX_SEQ_LEN)
        pos_input = pad_to_length(pos_input, MAX_SEQ_LEN)
        outputs = self.session.run(
            None,
            {
                "accent_input": accent_input,
                "pos_input": pos_input,
            },
        )
        return outputs[0]


class OnnxAccent:
    def __init__(self, onnx_path: str):
        self.session = ort.InferenceSession(onnx_path, providers=available_providers())

    @onnx_call
    def __call__(self, accent_input, pos_input, meter_input):
        orig_T = accent_input.shape[1]
        accent_input = pad_to_length(accent_input, MAX_SEQ_LEN)
        pos_input = pad_to_length(pos_input, MAX_SEQ_LEN)
        outputs = self.session.run(
            None,
            {
                "accent_input": accent_input,
                "pos_input": pos_input,
                "meter_target": meter_input,
            },
        )
        accent_logits = outputs[0]
        if accent_logits.shape[1] > orig_T:
            accent_logits = accent_logits[:, :orig_T]
        return accent_logits


def load_onnx_models():
    from velimir.settings import METER_ONNX_MODEL, ACCENT_ONNX_MODEL

    return OnnxMeter(METER_ONNX_MODEL), OnnxAccent(ACCENT_ONNX_MODEL)
