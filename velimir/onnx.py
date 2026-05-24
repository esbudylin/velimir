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


class OnnxAccent:
    def __init__(self, onnx_path: str):
        self.session = ort.InferenceSession(onnx_path, providers=available_providers())

    def __call__(self, accent_input, meter_class, pos_input):
        orig_T = accent_input.shape[1]
        try:
            import torch

            is_torch = isinstance(accent_input, torch.Tensor)
        except ImportError:
            is_torch = False

        accent_input = pad_to_length(accent_input, MAX_SEQ_LEN)
        pos_input = pad_to_length(pos_input, MAX_SEQ_LEN)

        outputs = self.session.run(
            None,
            {
                "accent_input": to_numpy(accent_input),
                "meter_class": to_numpy(meter_class),
                "pos_input": to_numpy(pos_input),
            },
        )

        result = outputs[0]
        if result.shape[1] > orig_T:
            result = result[:, :orig_T]

        if is_torch:
            return torch.from_numpy(result)
        return result


class OnnxMeter:
    def __init__(self, onnx_path: str):
        self.session = ort.InferenceSession(onnx_path, providers=available_providers())

    def __call__(self, accent_input, pos_input):
        try:
            import torch

            is_torch = isinstance(accent_input, torch.Tensor)
        except ImportError:
            is_torch = False

        accent_input = pad_to_length(accent_input, MAX_SEQ_LEN)
        pos_input = pad_to_length(pos_input, MAX_SEQ_LEN)

        outputs = self.session.run(
            None,
            {
                "accent_input": to_numpy(accent_input),
                "pos_input": to_numpy(pos_input),
            },
        )

        if is_torch:
            return torch.from_numpy(outputs[0])
        return outputs[0]


def load_onnx_models():
    from velimir.settings import ACCENT_ONNX_MODEL, METER_ONNX_MODEL

    return OnnxAccent(ACCENT_ONNX_MODEL), OnnxMeter(METER_ONNX_MODEL)
