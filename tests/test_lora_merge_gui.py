import json

import torch
from torch import nn

from musubi_tuner_gui.lora_merge_gui import MODEL_PROFILE_MAP
from musubi_tuner_gui.lora_merge_worker import _state_dict_for_save


class _TinyConvRotModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 8, bias=False)
        self.linear.register_buffer("scale_weight", torch.ones(8, 1, dtype=torch.float32))
        self.linear._convrot_groupsize = 16
        self.linear.requires_grad_(False)
        self.linear.weight = nn.Parameter(torch.zeros(8, 16, dtype=torch.int8), requires_grad=False)


def test_krea2_merger_profile_uses_native_loader_type():
    profile = MODEL_PROFILE_MAP["Krea 2 ConvRot INT8 (16 channels)"]

    assert profile == {"label": "Krea 2 ConvRot INT8 (16 channels)", "type": "krea2", "channels": 16}


def test_krea2_save_restores_portable_comfy_convrot_layout():
    state_dict = _state_dict_for_save(_TinyConvRotModel(), "krea2")

    assert state_dict["linear.weight"].dtype is torch.int8
    assert "linear.scale_weight" not in state_dict
    assert state_dict["linear.weight_scale"].dtype is torch.float32
    spec = json.loads(bytes(state_dict["linear.comfy_quant"].tolist()).decode("utf-8"))
    assert spec == {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 16}