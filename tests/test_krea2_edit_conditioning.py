import pytest
import torch

from musubi_tuner.dataset.architectures import ARCHITECTURE_KREA2_EDIT, ARCHITECTURE_KREA2_EDIT_FULL
from musubi_tuner.krea2.krea2_edit_conditioning import (
    fit_reference_pixels,
    pack_edit_image_tokens,
    unpack_target_prediction,
)


def test_krea2_edit_architecture_names_are_distinct_from_t2i():
    assert ARCHITECTURE_KREA2_EDIT == "kr2e"
    assert ARCHITECTURE_KREA2_EDIT_FULL == "krea2_edit"


def test_pack_edit_tokens_assigns_frames_and_fractional_center_offsets():
    target = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8)
    reference = torch.arange(24, dtype=torch.float32).reshape(1, 1, 4, 6)

    packed = pack_edit_image_tokens(target, [reference], patch_size=2)

    assert packed.tokens.shape == (1, 22, 4)
    assert packed.reference_token_count == 6
    assert packed.target_grid == (4, 4)
    assert packed.mask.all()
    assert torch.all(packed.positions[:, :6, 0] == 1)
    assert torch.all(packed.positions[:, 6:, 0] == 0)
    assert packed.positions[0, 0].tolist() == [1.0, 1.0, 0.5]
    assert packed.positions[0, 5].tolist() == [1.0, 2.0, 2.5]


def test_pack_edit_tokens_preserves_two_reference_order():
    target = torch.zeros(1, 1, 4, 4)
    first = torch.ones(1, 1, 2, 2)
    second = torch.full((1, 1, 2, 2), 2.0)

    packed = pack_edit_image_tokens(target, [first, second], patch_size=2)

    assert packed.reference_token_count == 2
    assert packed.tokens[0, 0].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert packed.tokens[0, 1].tolist() == [2.0, 2.0, 2.0, 2.0]
    assert packed.positions[0, 0, 0].item() == 1
    assert packed.positions[0, 1, 0].item() == 2


def test_pack_edit_tokens_rejects_invalid_reference_counts_and_shapes():
    target = torch.zeros(1, 1, 4, 4)
    reference = torch.zeros(1, 1, 2, 2)

    with pytest.raises(ValueError, match="at least one"):
        pack_edit_image_tokens(target, [], patch_size=2)
    with pytest.raises(ValueError, match="at most 2"):
        pack_edit_image_tokens(target, [reference, reference, reference], patch_size=2)
    with pytest.raises(ValueError, match="batch size"):
        pack_edit_image_tokens(target, [torch.zeros(2, 1, 2, 2)], patch_size=2)


def test_unpack_target_prediction_discards_reference_outputs():
    target = torch.zeros(1, 1, 4, 4)
    reference = torch.zeros(1, 1, 2, 2)
    packed = pack_edit_image_tokens(target, [reference], patch_size=2)
    prediction = torch.arange(20, dtype=torch.float32).reshape(1, 5, 4)

    restored = unpack_target_prediction(prediction, packed, channels=1, patch_size=2)

    expected = prediction[:, 1:].reshape(1, 2, 2, 1, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(1, 1, 4, 4)
    assert torch.equal(restored, expected)


def test_fit_reference_pixels_preserves_aspect_ratio_on_wide_reference():
    reference = torch.zeros(1, 3, 100, 200)

    fitted = fit_reference_pixels(reference, 128, 128)

    assert fitted.shape == (1, 3, 64, 128)


def test_fit_reference_pixels_matches_v125_crop_before_resize_protocol():
    source_height, source_width = 753, 512
    reference = torch.arange(source_height, dtype=torch.float32).view(1, 1, source_height, 1)
    reference = reference.expand(1, 1, source_height, source_width)

    fitted = fit_reference_pixels(reference, 1024, 1024)

    scale = min(1024 / source_height, 1024 / source_width)
    output_height = int(source_height * scale) // 16 * 16
    output_width = int(source_width * scale) // 16 * 16
    crop_height = round(output_height / scale)
    crop_width = round(output_width / scale)
    top = (source_height - crop_height) // 2
    left = (source_width - crop_width) // 2
    expected = torch.nn.functional.interpolate(
        reference[:, :, top : top + crop_height, left : left + crop_width],
        size=(output_height, output_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )

    direct_resize = torch.nn.functional.interpolate(
        reference,
        size=(output_height, output_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    assert fitted.shape[-2:] == (1008, 688)
    assert torch.equal(fitted, expected)
    assert not torch.equal(fitted, direct_resize)


def test_fit_reference_pixels_fills_target_for_near_matching_aspect_ratio():
    reference = torch.zeros(1, 3, 100, 108)

    fitted = fit_reference_pixels(reference, 128, 128)

    assert fitted.shape == (1, 3, 128, 128)


def test_fit_reference_pixels_rejects_unaligned_target():
    with pytest.raises(ValueError, match="divisible"):
        fit_reference_pixels(torch.zeros(1, 3, 100, 200), 127, 128)