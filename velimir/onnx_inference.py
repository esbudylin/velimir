import torch
import torch.nn.functional as F
import onnxruntime as ort

MAX_SEQ_LEN = 128


def _pad_to_length(tensor, target_length, pad_value=-1):
    if tensor.ndim == 2:
        current = tensor.shape[1]
        if current < target_length:
            return F.pad(tensor, (0, target_length - current), value=pad_value)
        return tensor[:, :target_length]
    elif tensor.ndim == 3:
        current = tensor.shape[1]
        if current < target_length:
            return F.pad(tensor, (0, 0, 0, target_length - current), value=pad_value)
        return tensor[:, :target_length, :]
    return tensor


class OnnxAccentWrapper:
    def __init__(self, onnx_path: str):
        self.session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        self._device = torch.device("cpu")

    def eval(self):
        return self

    def parameters(self):
        return iter([torch.tensor(0.0)])

    def to(self, device):
        return self

    def __call__(self, accent_input, meter_class, pos_input):
        orig_T = accent_input.shape[1]

        accent_input = _pad_to_length(accent_input, MAX_SEQ_LEN)
        pos_input = _pad_to_length(pos_input, MAX_SEQ_LEN)

        inputs = {
            "accent_input": accent_input.cpu().numpy(),
            "meter_class": meter_class.cpu().numpy(),
            "pos_input": pos_input.cpu().numpy(),
        }

        outputs = self.session.run(None, inputs)

        result = torch.from_numpy(outputs[0])

        if result.shape[1] > orig_T:
            result = result[:, :orig_T]

        return result


class OnnxMeterWrapper:
    def __init__(self, onnx_path: str):
        self.session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        self._device = torch.device("cpu")

    def eval(self):
        return self

    def parameters(self):
        return iter([torch.tensor(0.0)])

    def to(self, device):
        return self

    def __call__(self, accent_input, pos_input):
        accent_input = _pad_to_length(accent_input, MAX_SEQ_LEN)
        pos_input = _pad_to_length(pos_input, MAX_SEQ_LEN)

        inputs = {
            "accent_input": accent_input.cpu().numpy(),
            "pos_input": pos_input.cpu().numpy(),
        }

        outputs = self.session.run(None, inputs)

        return torch.from_numpy(outputs[0])


def load_onnx_models():
    from velimir.settings import ACCENT_ONNX_MODEL, METER_ONNX_MODEL

    return OnnxAccentWrapper(ACCENT_ONNX_MODEL), OnnxMeterWrapper(METER_ONNX_MODEL)
