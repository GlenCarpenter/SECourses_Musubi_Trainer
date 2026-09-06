import argparse
import logging
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import load_file

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "musubi-tuner", "src")))

from musubi_tuner.dataset.cache_io import validate_krea2_edit_text_encoder_cache
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.krea2.krea2_encoder import (
    KREA2_EDIT_VISION_PLACEHOLDER,
    Qwen3VLConditioner,
    prepare_grounding_image,
    select_grounding_size,
)
from musubi_tuner.krea2_edit_cache_text_encoder_outputs import (
    discard_incompatible_existing_caches,
    encode_and_save_batch,
    setup_parser,
)


class FakeInputs(dict):
    def to(self, *args, **kwargs):
        return self


class FakeProcessor:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        sequence_length = 40 + len(kwargs["images"])
        return FakeInputs(
            input_ids=torch.zeros((1, sequence_length), dtype=torch.long),
            attention_mask=torch.ones((1, sequence_length), dtype=torch.long),
        )


class FakeQwen(torch.nn.Module):
    device = torch.device("cpu")

    def forward(self, input_ids, attention_mask, output_hidden_states):
        shape = (*input_ids.shape, 3)
        hidden_states = tuple(torch.full(shape, float(index)) for index in range(4))
        return SimpleNamespace(hidden_states=hidden_states)


def _conditioner():
    processor = FakeProcessor()
    encoder = Qwen3VLConditioner(
        FakeQwen(),
        tokenizer=None,
        processor=None,
        select_layers=(1, 3),
        multimodal_processor=processor,
    )
    return encoder, processor


def test_grounding_resize_preserves_aspect_ratio_and_does_not_upscale():
    wide = np.zeros((100, 200, 3), dtype=np.uint8)
    small = Image.new("RGB", (32, 24))

    assert prepare_grounding_image(wide, 50).size == (50, 25)
    assert prepare_grounding_image(small, 768).size == (32, 24)


def test_grounding_size_selection_is_injectable_and_validated():
    calls = []

    def choose(low, high):
        calls.append((low, high))
        return 517

    assert select_grounding_size(384, 768, randint=choose) == 517
    assert calls == [(384, 768)]
    assert select_grounding_size(768, 768, randint=choose) == 768
    assert select_grounding_size(0, 0, randint=choose) == 0
    with pytest.raises(ValueError, match="exceeds maximum"):
        select_grounding_size(769, 768, randint=choose)


def test_multimodal_conditioner_preserves_reference_order_and_slices_prefix():
    encoder, processor = _conditioner()
    first = np.full((100, 200, 3), (255, 0, 0), dtype=np.uint8)
    second = np.full((120, 80, 3), (0, 255, 0), dtype=np.uint8)

    hidden, mask = encoder.forward_with_images(
        ["place the subject on a beach"],
        [[first, second]],
        grounding_size_selector=lambda _low, _high: 50,
    )

    call = processor.calls[0]
    assert [image.size for image in call["images"]] == [(50, 25), (33, 50)]
    assert call["images"][0].getpixel((0, 0)) == (255, 0, 0)
    assert call["images"][1].getpixel((0, 0)) == (0, 255, 0)
    assert call["text"][0].count(KREA2_EDIT_VISION_PLACEHOLDER) == 2
    assert call["text"][0].endswith("place the subject on a beach<|im_end|>\n<|im_start|>assistant\n")
    assert hidden.shape == (1, 8, 2, 3)
    assert mask.shape == (1, 8)
    assert torch.all(hidden[:, :, 0] == 1)
    assert torch.all(hidden[:, :, 1] == 3)


def test_multimodal_conditioner_rejects_invalid_reference_counts():
    encoder, _processor = _conditioner()
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="one or two"):
        encoder.forward_with_images(["instruction"], [[]])
    with pytest.raises(ValueError, match="one or two"):
        encoder.forward_with_images(["instruction"], [[image, image, image]])


def test_fixed_scale_cache_records_and_validates_grounding_contract(tmp_path):
    encoder, _processor = _conditioner()
    cache_path = tmp_path / "sample_kr2e_te.safetensors"
    item = ItemInfo("sample.png", "instruction", (128, 128))
    item.text_encoder_output_cache_path = str(cache_path)
    item.control_content = [
        np.zeros((100, 200, 3), dtype=np.uint8),
        np.zeros((120, 80, 3), dtype=np.uint8),
    ]

    encode_and_save_batch(encoder, [item], grounding_pixels=640)

    tensors = load_file(str(cache_path))
    assert len(tensors) == 1
    assert next(iter(tensors)).startswith("varlen_krea2_vl_embed_")
    metadata = validate_krea2_edit_text_encoder_cache(
        str(cache_path), expected_grounding_pixels=640, expected_reference_count=2
    )
    assert metadata["architecture"] == "krea2_edit"
    assert metadata["grounding_mode"] == "fixed"
    assert metadata["reference_0_pixel_height"] == "100"
    assert metadata["reference_1_pixel_width"] == "80"

    with pytest.raises(ValueError, match="scale mismatch"):
        validate_krea2_edit_text_encoder_cache(str(cache_path), expected_grounding_pixels=768)

    existing = [{os.path.normpath(str(cache_path))}]
    discard_incompatible_existing_caches(existing, grounding_pixels=768)
    assert existing == [set()]


def test_grounding_memory_profile_reports_online_delta_and_cache_payload(tmp_path, monkeypatch, caplog):
    encoder, _processor = _conditioner()
    item = ItemInfo("sample.png", "instruction", (128, 128))
    item.text_encoder_output_cache_path = str(tmp_path / "sample_kr2e_te.safetensors")
    item.control_content = [np.zeros((100, 200, 3), dtype=np.uint8)]
    mib = 1024**2
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 100 * mib)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 140 * mib)

    with caplog.at_level(logging.INFO, logger="musubi_tuner.krea2_edit_cache_text_encoder_outputs"):
        encode_and_save_batch(encoder, [item], grounding_pixels=640, profile_memory=True)

    assert "online_baseline=100.00MB" in caplog.text
    assert "online_peak=140.00MB" in caplog.text
    assert "online_delta=40.00MB" in caplog.text
    assert "fixed_cache_payload=0.00MB" in caplog.text


def test_grounding_memory_profile_cli_is_opt_in():
    parser = setup_parser(argparse.ArgumentParser())

    assert parser.parse_args(["--text_encoder", "encoder.safetensors"]).profile_grounding_memory is False
    assert parser.parse_args(
        ["--text_encoder", "encoder.safetensors", "--profile_grounding_memory"]
    ).profile_grounding_memory is True