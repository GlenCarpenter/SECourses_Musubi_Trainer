import json

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from musubi_tuner.modules.convrot_int8_kernels import (
    dequantize_int8_convrot_weight,
    quantize_int8_convrot_weight,
)
from musubi_tuner_gui.krea2_checkpoint_merge_worker import merge_krea2_checkpoints


def _quant_spec(groupsize: int) -> torch.Tensor:
    payload = json.dumps(
        {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": groupsize},
        separators=(",", ":"),
    ).encode("utf-8")
    return torch.tensor(list(payload), dtype=torch.uint8)


def _checkpoint(path, weight, other, *, prefix="", include_control=True):
    quantized, scale = quantize_int8_convrot_weight(weight, 4)
    tensors = {
        prefix + "blocks.0.linear.weight": quantized,
        prefix + "blocks.0.linear.weight_scale": scale,
        prefix + "head.weight": other,
        prefix + "constant": torch.tensor([1], dtype=torch.int64),
    }
    if include_control:
        tensors[prefix + "blocks.0.linear.comfy_quant"] = _quant_spec(4)
    save_file(tensors, str(path))


def test_merges_krea2_convrot_checkpoints_and_preserves_portable_layout(tmp_path):
    torch.manual_seed(0)
    weight_a = torch.randn(8, 16)
    weight_b = torch.randn(8, 16)
    checkpoint_a = tmp_path / "a.safetensors"
    checkpoint_b = tmp_path / "b.safetensors"
    output = tmp_path / "merged.safetensors"
    _checkpoint(checkpoint_a, weight_a, torch.full((2, 2), 2.0))
    _checkpoint(checkpoint_b, weight_b, torch.full((2, 2), 6.0))

    status, _message = merge_krea2_checkpoints(
        str(checkpoint_a), str(checkpoint_b), str(output), weight_a=0.75, device_choice="cpu"
    )

    assert status == "success"
    with safe_open(str(output), framework="pt", device="cpu") as reader:
        merged_quant = reader.get_tensor("blocks.0.linear.weight")
        merged_scale = reader.get_tensor("blocks.0.linear.weight_scale")
        merged = dequantize_int8_convrot_weight(merged_quant, merged_scale, 4)
        source_a = dequantize_int8_convrot_weight(*quantize_int8_convrot_weight(weight_a, 4), 4)
        source_b = dequantize_int8_convrot_weight(*quantize_int8_convrot_weight(weight_b, 4), 4)
        expected_quant, expected_scale = quantize_int8_convrot_weight(source_a * 0.75 + source_b * 0.25, 4)
        expected = dequantize_int8_convrot_weight(expected_quant, expected_scale, 4)
        assert torch.equal(merged, expected)
        assert torch.equal(reader.get_tensor("head.weight"), torch.full((2, 2), 3.0))
        assert torch.equal(reader.get_tensor("constant"), torch.tensor([1]))
        assert "blocks.0.linear.comfy_quant" in reader.keys()
        assert reader.metadata()["merge_weight_a"] == "0.75"


def test_rejects_different_checkpoint_keys(tmp_path):
    checkpoint_a = tmp_path / "a.safetensors"
    checkpoint_b = tmp_path / "b.safetensors"
    output = tmp_path / "merged.safetensors"
    save_file({"a": torch.ones(1)}, str(checkpoint_a))
    save_file({"b": torch.ones(1)}, str(checkpoint_b))

    status, message = merge_krea2_checkpoints(
        str(checkpoint_a), str(checkpoint_b), str(output), weight_a=0.5, device_choice="cpu"
    )

    assert status == "error"
    assert "layouts do not match" in message
    assert not output.exists()


def test_normalizes_comfy_prefix_and_uses_control_from_other_checkpoint(tmp_path):
    torch.manual_seed(1)
    checkpoint_a = tmp_path / "a.safetensors"
    checkpoint_b = tmp_path / "b.safetensors"
    output = tmp_path / "merged.safetensors"
    _checkpoint(checkpoint_a, torch.randn(8, 16), torch.ones(2, 2))
    _checkpoint(
        checkpoint_b,
        torch.randn(8, 16),
        torch.full((2, 2), 3.0),
        prefix="model.diffusion_model.",
        include_control=False,
    )

    status, message = merge_krea2_checkpoints(
        str(checkpoint_a), str(checkpoint_b), str(output), weight_a=0.5, device_choice="cpu"
    )

    assert status == "success", message
    with safe_open(str(output), framework="pt", device="cpu") as reader:
        assert "blocks.0.linear.weight" in reader.keys()
        assert "blocks.0.linear.comfy_quant" in reader.keys()
        assert not any(key.startswith("model.diffusion_model.") for key in reader.keys())
        assert torch.equal(reader.get_tensor("head.weight"), torch.full((2, 2), 2.0))


def test_preserves_a_full_precision_scope_when_b_quantizes_extra_layer(tmp_path):
    torch.manual_seed(2)
    checkpoint_a = tmp_path / "a.safetensors"
    checkpoint_b = tmp_path / "b.safetensors"
    output = tmp_path / "merged.safetensors"
    base_a = torch.randn(8, 16)
    base_b = torch.randn(8, 16)
    extra_a = torch.randn(8, 16, dtype=torch.bfloat16)
    extra_b = torch.randn(8, 16)
    _checkpoint(checkpoint_a, base_a, torch.ones(2, 2))
    _checkpoint(checkpoint_b, base_b, torch.ones(2, 2), prefix="model.diffusion_model.", include_control=False)

    tensors_a = load_file(str(checkpoint_a))
    tensors_b = load_file(str(checkpoint_b))
    tensors_a["txtfusion.layerwise_blocks.0.attn.gate.weight"] = extra_a
    extra_quant, extra_scale = quantize_int8_convrot_weight(extra_b, 4)
    tensors_b["txtfusion.layerwise_blocks.0.attn.gate.weight"] = extra_quant
    tensors_b["txtfusion.layerwise_blocks.0.attn.gate.weight_scale"] = extra_scale
    save_file(tensors_a, str(checkpoint_a))
    quantization_metadata = {
        "format_version": "1.0",
        "layers": {
            "model.diffusion_model.blocks.0.linear": {
                "format": "int8_tensorwise",
                "convrot": True,
                "convrot_groupsize": 4,
            },
            "txtfusion.layerwise_blocks.0.attn.gate": {
                "format": "int8_tensorwise",
                "convrot": True,
                "convrot_groupsize": 4,
            },
        },
    }
    save_file(
        tensors_b,
        str(checkpoint_b),
        metadata={"_quantization_metadata": json.dumps(quantization_metadata)},
    )

    status, message = merge_krea2_checkpoints(
        str(checkpoint_a), str(checkpoint_b), str(output), weight_a=0.25, device_choice="cpu"
    )

    assert status == "success", message
    with safe_open(str(output), framework="pt", device="cpu") as reader:
        key = "txtfusion.layerwise_blocks.0.attn.gate.weight"
        dense_b = dequantize_int8_convrot_weight(extra_quant, extra_scale, 4)
        expected = (extra_a.float() * 0.25 + dense_b * 0.75).to(torch.bfloat16)
        assert torch.equal(reader.get_tensor(key), expected)
        assert key.replace(".weight", ".weight_scale") not in reader.keys()
        assert key.replace(".weight", ".comfy_quant") not in reader.keys()