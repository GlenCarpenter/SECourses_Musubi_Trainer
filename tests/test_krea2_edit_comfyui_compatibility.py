import pytest
import torch
from safetensors.torch import save_file

from musubi_tuner.krea2.krea2_edit_compatibility import (
    validate_comfyui_lora_files,
    validate_comfyui_lora_state_dict,
)
from musubi_tuner.krea2.krea2_mmdit import SingleStreamDiT
from musubi_tuner.krea2.krea2_utils import single_mmdit_large_wide
from musubi_tuner.networks import lora_krea2
from musubi_tuner.networks.lora import LoRAModule


def test_musubi_lora_export_maps_to_comfyui_standard_loader_alias():
    module = LoRAModule(
        "lora_unet_blocks_0_attn_wq",
        torch.nn.Linear(8, 12, bias=False),
        lora_dim=4,
        alpha=4,
    )

    report = validate_comfyui_lora_state_dict(
        module.export_state_dict(),
        {"blocks.0.attn.wq.weight": torch.empty(12, 8)},
    )

    assert report.adapter_count == 1
    assert report.tensor_count == 3
    assert report.mapped_modules == ("blocks.0.attn.wq",)


def test_full_krea2_lora_export_maps_all_modules_without_conversion():
    with torch.device("meta"):
        model = SingleStreamDiT(single_mmdit_large_wide)
        network = lora_krea2.create_arch_network(1.0, 4, 4.0, None, [], model)

    report = validate_comfyui_lora_state_dict(network.build_export_state_dict(), model.state_dict())

    assert report.adapter_count == 264
    assert report.tensor_count == 792
    assert "blocks.0.attn.wq" in report.mapped_modules
    assert "txtfusion.projector" in report.mapped_modules


def test_comfyui_compatibility_accepts_diffusion_model_base_prefix():
    lora_state = {
        "lora_unet_txtfusion_projector.lora_down.weight": torch.empty(2, 8),
        "lora_unet_txtfusion_projector.lora_up.weight": torch.empty(12, 2),
        "lora_unet_txtfusion_projector.alpha": torch.tensor(2.0),
    }
    base_state = {"diffusion_model.txtfusion.projector.weight": torch.empty(12, 8)}

    report = validate_comfyui_lora_state_dict(lora_state, base_state)

    assert report.mapped_modules == ("txtfusion.projector",)


def test_comfyui_compatibility_rejects_unmapped_or_incomplete_adapters():
    base_state = {"blocks.0.attn.wq.weight": torch.empty(12, 8)}

    with pytest.raises(ValueError, match="missing.*lora_up.weight"):
        validate_comfyui_lora_state_dict(
            {
                "lora_unet_blocks_0_attn_wq.lora_down.weight": torch.empty(4, 8),
                "lora_unet_blocks_0_attn_wq.alpha": torch.tensor(4.0),
            },
            base_state,
        )

    with pytest.raises(ValueError, match="does not map"):
        validate_comfyui_lora_state_dict(
            {
                "lora_unet_unknown.lora_down.weight": torch.empty(4, 8),
                "lora_unet_unknown.lora_up.weight": torch.empty(12, 4),
                "lora_unet_unknown.alpha": torch.tensor(4.0),
            },
            base_state,
        )


def test_comfyui_compatibility_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        validate_comfyui_lora_state_dict(
            {
                "lora_unet_blocks_0_attn_wq.lora_down.weight": torch.empty(4, 7),
                "lora_unet_blocks_0_attn_wq.lora_up.weight": torch.empty(12, 4),
                "lora_unet_blocks_0_attn_wq.alpha": torch.tensor(4.0),
            },
            {"blocks.0.attn.wq.weight": torch.empty(12, 8)},
        )


def test_comfyui_compatibility_validates_safetensors_without_loading_weights(tmp_path):
    lora_path = tmp_path / "edit_lora.safetensors"
    base_path = tmp_path / "krea2.safetensors"
    save_file(
        {
            "lora_unet_blocks_0_attn_wq.lora_down.weight": torch.empty(4, 8),
            "lora_unet_blocks_0_attn_wq.lora_up.weight": torch.empty(12, 4),
            "lora_unet_blocks_0_attn_wq.alpha": torch.tensor(4.0),
        },
        lora_path,
    )
    save_file({"blocks.0.attn.wq.weight": torch.empty(12, 8)}, base_path)

    report = validate_comfyui_lora_files(lora_path, base_path)

    assert report.adapter_count == 1
    assert report.tensor_count == 3