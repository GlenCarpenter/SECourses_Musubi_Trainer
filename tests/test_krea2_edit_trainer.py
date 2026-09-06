from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from musubi_tuner.krea2_edit_train_network import Krea2EditNetworkTrainer
from musubi_tuner.krea2_train_network import Krea2NetworkTrainer
from musubi_tuner.utils import sai_model_spec


class _TinyEditTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(patch=2)
        self.target_scale = torch.nn.Parameter(torch.tensor(0.5))
        self.reference_bias = torch.nn.Parameter(torch.tensor(1.0))
        self.last_inputs = None

    def forward(self, *, img, context, t, pos, mask):
        self.last_inputs = {"img": img.detach().clone(), "pos": pos.detach().clone(), "mask": mask.detach().clone()}
        reference_mask = (pos[:, : img.shape[1], 0] > 0).unsqueeze(-1)
        return img * self.target_scale + reference_mask * self.reference_bias


class _Accelerator:
    device = torch.device("cpu")

    @staticmethod
    def unwrap_model(model, keep_fp32_wrapper=False):
        return model

    @staticmethod
    def autocast():
        return nullcontext()


def _model_args(**overrides):
    values = {
        "fp8_base": False,
        "fp8_scaled": False,
        "convrot_int8": False,
        "convrot_int8_bwd": "bf16",
        "turbo_dit": None,
        "turbo_dit_cache": False,
        "blocks_to_swap": 0,
        "compile": False,
        "sample_prompts": None,
        "network_module": "networks.lora_krea2",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _run_tiny_edit_step(trainer):
    transformer = _TinyEditTransformer()
    target = torch.full((1, 1, 1, 4, 4), 2.0)
    output = trainer.call_dit(
        SimpleNamespace(gradient_checkpointing=False),
        _Accelerator(),
        transformer,
        target,
        {
            "latents": target,
            "latents_control_0": torch.full((1, 1, 1, 2, 2), 10.0),
            "krea2_vl_embed": [torch.zeros((3, 2, 4))],
        },
        torch.full_like(target, 5.0),
        torch.full_like(target, 3.0),
        torch.tensor([500.0]),
        torch.float32,
    )
    torch.nn.functional.mse_loss(output.pred, output.target).backward()
    return transformer


@pytest.mark.parametrize("reference_count", [1, 2])
def test_edit_forward_backward_keeps_references_clean_and_supervises_target_only(reference_count):
    trainer = Krea2EditNetworkTrainer()
    transformer = _TinyEditTransformer()
    target = torch.full((1, 1, 1, 4, 4), 2.0)
    noisy_target = torch.full_like(target, 3.0)
    noise = torch.full_like(target, 5.0)
    references = [torch.full((1, 1, 1, 2, 2), 10.0 + index) for index in range(reference_count)]
    batch = {
        "latents": target,
        "krea2_vl_embed": [torch.zeros((3, 2, 4))],
        **{f"latents_control_{index}": reference for index, reference in enumerate(references)},
    }

    output = trainer.call_dit(
        SimpleNamespace(gradient_checkpointing=False),
        _Accelerator(),
        transformer,
        target,
        batch,
        noise,
        noisy_target,
        torch.tensor([500.0]),
        torch.float32,
    )

    assert output.pred.shape == target.shape
    assert torch.equal(output.target, noise - target)
    assert torch.equal(output.pred, noisy_target * transformer.target_scale)
    assert transformer.last_inputs["img"].shape[1] == reference_count + 4
    for index, reference in enumerate(references):
        assert torch.equal(transformer.last_inputs["img"][:, index], reference.reshape(1, -1))
        assert torch.all(transformer.last_inputs["pos"][:, index, 0] == index + 1)
    assert torch.all(transformer.last_inputs["pos"][:, reference_count : reference_count + 4, 0] == 0)

    torch.nn.functional.mse_loss(output.pred, output.target).backward()
    assert transformer.target_scale.grad is not None
    assert transformer.target_scale.grad.abs().item() > 0
    assert transformer.reference_bias.grad is not None
    assert transformer.reference_bias.grad.item() == 0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"sample_prompts": "samples.txt"}, "previews"),
        ({"network_module": "networks.lora"}, "networks.lora_krea2"),
    ],
)
def test_edit_preflight_rejects_unsupported_training_options(overrides, message):
    with pytest.raises(ValueError, match=message):
        Krea2EditNetworkTrainer().handle_model_specific_args(_model_args(**overrides))


def test_edit_dataset_preflight_validates_each_cache_once(monkeypatch):
    trainer = Krea2EditNetworkTrainer()
    item = SimpleNamespace(
        item_key="example",
        latent_cache_path="example_kr2e.safetensors",
        text_encoder_output_cache_path="example_kr2e_te.safetensors",
    )
    dataset_group = SimpleNamespace(
        datasets=[SimpleNamespace(batch_manager=SimpleNamespace(buckets={(4, 4): [item, item]}))]
    )
    monkeypatch.setattr(Krea2NetworkTrainer, "_build_dataset", lambda self, args: (dataset_group, "collator", "epoch"))
    latent_calls = []
    text_calls = []
    monkeypatch.setattr(
        "musubi_tuner.krea2_edit_train_network.load_krea2_edit_latent_cache",
        lambda path: (latent_calls.append(path) or torch.zeros(1), [torch.zeros(1), torch.zeros(1)], {}),
    )
    monkeypatch.setattr(
        "musubi_tuner.krea2_edit_train_network.validate_krea2_edit_text_encoder_cache",
        lambda path, **kwargs: text_calls.append((path, kwargs)),
    )

    result = trainer._build_dataset(SimpleNamespace())

    assert result == (dataset_group, "collator", "epoch")
    assert latent_calls == ["example_kr2e.safetensors"]
    assert text_calls == [("example_kr2e_te.safetensors", {"expected_reference_count": 2})]


def test_edit_forward_requires_cached_grounded_text():
    trainer = Krea2EditNetworkTrainer()
    target = torch.zeros((1, 1, 1, 4, 4))
    batch = {"latents": target, "latents_control_0": torch.zeros((1, 1, 1, 2, 2))}

    with pytest.raises(ValueError, match="image-grounded text embeddings"):
        trainer.call_dit(
            SimpleNamespace(gradient_checkpointing=False),
            _Accelerator(),
            _TinyEditTransformer(),
            target,
            batch,
            torch.ones_like(target),
            target,
            torch.tensor([500.0]),
            torch.float32,
        )


def test_edit_forward_rejects_non_contiguous_reference_keys():
    with pytest.raises(ValueError, match="contiguous from zero"):
        Krea2EditNetworkTrainer()._reference_latents(
            {"latents_control_1": torch.zeros((1, 1, 1, 2, 2))}
        )


@pytest.mark.parametrize(
    ("overrides", "mode_identifier"),
    [
        ({}, "base=BF16"),
        ({"fp8_base": True, "fp8_scaled": True}, "base=scaled FP8"),
        ({"convrot_int8": True}, "base=ConvRot INT8 (backward=bf16)"),
        ({"convrot_int8": True, "convrot_int8_bwd": "int8"}, "base=ConvRot INT8 (backward=int8)"),
        ({"blocks_to_swap": 4}, "block_swap=4 blocks"),
        ({"compile": True}, "torch_compile=enabled"),
    ],
)
def test_edit_modes_emit_identifier_and_complete_tiny_backward(capsys, overrides, mode_identifier):
    trainer = Krea2EditNetworkTrainer()

    trainer.handle_model_specific_args(_model_args(**overrides))

    assert mode_identifier in capsys.readouterr().out
    transformer = _run_tiny_edit_step(trainer)
    assert transformer.target_scale.grad is not None


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, {"fp8_scaled": False, "convrot_int8": False, "convrot_int8_bwd": "bf16"}),
        (
            {"fp8_base": True, "fp8_scaled": True},
            {"fp8_scaled": True, "convrot_int8": False, "convrot_int8_bwd": "bf16"},
        ),
        (
            {"convrot_int8": True, "convrot_int8_bwd": "bf16"},
            {"fp8_scaled": False, "convrot_int8": True, "convrot_int8_bwd": "bf16"},
        ),
        (
            {"convrot_int8": True, "convrot_int8_bwd": "int8"},
            {"fp8_scaled": False, "convrot_int8": True, "convrot_int8_bwd": "int8"},
        ),
    ],
)
def test_edit_loader_forwards_quantization_and_attention_mode(monkeypatch, overrides, expected):
    calls = []
    loaded_model = object()
    monkeypatch.setattr(
        "musubi_tuner.krea2.krea2_utils.load_krea2_dit",
        lambda path, **kwargs: (calls.append((path, kwargs)) or loaded_model),
    )
    args = _model_args(**overrides)

    result = Krea2EditNetworkTrainer().load_transformer(
        _Accelerator(),
        args,
        "dit.safetensors",
        "flash_auto",
        False,
        "cpu",
        torch.bfloat16,
    )

    assert result is loaded_model
    assert calls[0][0] == "dit.safetensors"
    for key, value in expected.items():
        assert calls[0][1][key] == value
    assert calls[0][1]["attn_mode"] == "flash_auto"
    assert calls[0][1]["loading_device"] == "cpu"


@pytest.mark.parametrize(
    ("convrot_int8", "blocks_to_swap", "expected_disable_linear", "expects_offloader"),
    [
        (False, 0, False, False),
        (True, 0, True, False),
        (False, 4, True, True),
    ],
)
def test_edit_compile_uses_krea_quantized_and_block_swap_policy(
    monkeypatch, convrot_int8, blocks_to_swap, expected_disable_linear, expects_offloader
):
    calls = []
    model = SimpleNamespace(blocks=[object()], offloader=object())
    monkeypatch.setattr(
        "musubi_tuner.krea2_train_network.model_utils.compile_transformer",
        lambda args, transformer, blocks, **kwargs: (calls.append((transformer, blocks, kwargs)) or transformer),
    )
    trainer = Krea2EditNetworkTrainer()
    trainer.blocks_to_swap = blocks_to_swap

    result = trainer.compile_transformer(SimpleNamespace(convrot_int8=convrot_int8), model)

    assert result is model
    assert calls[0][2]["disable_linear"] is expected_disable_linear
    assert (calls[0][2]["offloaders"] == [model.offloader]) is expects_offloader


def test_edit_lora_uses_krea_compatible_model_spec_metadata():
    metadata = sai_model_spec.build_metadata(
        None,
        Krea2EditNetworkTrainer().architecture,
        timestamp=0,
        is_lora=True,
    )

    assert metadata["modelspec.architecture"] == "Krea-2/lora"
    assert metadata["modelspec.implementation"] == "https://github.com/krea-ai/krea-2"
    assert metadata["modelspec.resolution"] == "1024x1024"