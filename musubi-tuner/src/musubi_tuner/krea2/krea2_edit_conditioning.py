"""Pure geometry and token-packing helpers for Krea 2 image editing.

The fit and positioning contract is adapted from the Apache-2.0 project
https://github.com/lbouaraba/krea2edit-trainer.
"""

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from einops import rearrange, repeat


MAX_REFERENCE_IMAGES = 2
REFERENCE_PIXEL_ALIGNMENT = 16


@dataclass(frozen=True)
class PackedEditImageTokens:
    tokens: torch.Tensor
    positions: torch.Tensor
    mask: torch.Tensor
    reference_token_count: int
    target_grid: tuple[int, int]


def _validate_latent(latent: torch.Tensor, patch_size: int, name: str) -> None:
    if latent.ndim != 4:
        raise ValueError(f"{name} must have shape (B, C, H, W), got {tuple(latent.shape)}")
    if patch_size < 1:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    if latent.shape[-2] % patch_size or latent.shape[-1] % patch_size:
        raise ValueError(
            f"{name} spatial dimensions must be divisible by patch_size={patch_size}, "
            f"got {tuple(latent.shape[-2:])}"
        )


def _image_tokens_and_positions(
    latent: torch.Tensor,
    patch_size: int,
    frame_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, _, height, width = latent.shape
    grid_height, grid_width = height // patch_size, width // patch_size
    positions = torch.zeros((grid_height, grid_width, 3), device=latent.device, dtype=torch.float32)
    positions[..., 0] = frame_index
    positions[..., 1] = torch.arange(grid_height, device=latent.device)[:, None]
    positions[..., 2] = torch.arange(grid_width, device=latent.device)[None, :]
    positions = repeat(positions, "h w axes -> b (h w) axes", b=batch_size)
    mask = torch.ones(batch_size, grid_height * grid_width, device=latent.device, dtype=torch.bool)
    tokens = rearrange(
        latent,
        "b c (h ph) (w pw) -> b (h w) (c ph pw)",
        ph=patch_size,
        pw=patch_size,
    )
    return tokens, positions, mask


def pack_edit_image_tokens(
    target_latents: torch.Tensor,
    reference_latents: Sequence[torch.Tensor],
    patch_size: int,
    *,
    center_references: bool = True,
) -> PackedEditImageTokens:
    """Pack clean references before a noisy target using Krea 2's image-first layout."""
    _validate_latent(target_latents, patch_size, "target_latents")
    references = list(reference_latents)
    if not references:
        raise ValueError("Krea 2 edit conditioning requires at least one reference latent")
    if len(references) > MAX_REFERENCE_IMAGES:
        raise ValueError(
            f"Krea 2 edit conditioning supports at most {MAX_REFERENCE_IMAGES} reference latents, "
            f"got {len(references)}"
        )

    batch_size, channels, target_height, target_width = target_latents.shape
    target_grid = (target_height // patch_size, target_width // patch_size)
    target_tokens, target_positions, target_mask = _image_tokens_and_positions(
        target_latents, patch_size, frame_index=0
    )

    reference_tokens = []
    reference_positions = []
    reference_masks = []
    reference_token_count = 0
    for index, reference in enumerate(references):
        _validate_latent(reference, patch_size, f"reference_latents[{index}]")
        if reference.shape[0] != batch_size:
            raise ValueError(
                f"reference_latents[{index}] batch size {reference.shape[0]} does not match target batch size {batch_size}"
            )
        if reference.shape[1] != channels:
            raise ValueError(
                f"reference_latents[{index}] channel count {reference.shape[1]} does not match target channel count {channels}"
            )

        tokens, positions, mask = _image_tokens_and_positions(reference, patch_size, frame_index=index + 1)
        if center_references:
            reference_grid_height = reference.shape[-2] // patch_size
            reference_grid_width = reference.shape[-1] // patch_size
            positions[..., 1] += max(0.0, (target_grid[0] - reference_grid_height) / 2)
            positions[..., 2] += max(0.0, (target_grid[1] - reference_grid_width) / 2)
        reference_tokens.append(tokens)
        reference_positions.append(positions)
        reference_masks.append(mask)
        reference_token_count += tokens.shape[1]

    return PackedEditImageTokens(
        tokens=torch.cat(reference_tokens + [target_tokens], dim=1),
        positions=torch.cat(reference_positions + [target_positions], dim=1),
        mask=torch.cat(reference_masks + [target_mask], dim=1),
        reference_token_count=reference_token_count,
        target_grid=target_grid,
    )


def unpack_target_prediction(
    prediction: torch.Tensor,
    packed: PackedEditImageTokens,
    channels: int,
    patch_size: int,
) -> torch.Tensor:
    """Discard reference predictions and restore the target latent layout."""
    target_prediction = prediction[:, packed.reference_token_count :]
    expected_tokens = packed.target_grid[0] * packed.target_grid[1]
    if target_prediction.shape[1] != expected_tokens:
        raise ValueError(
            f"target prediction has {target_prediction.shape[1]} tokens after removing references; "
            f"expected {expected_tokens}"
        )
    expected_features = channels * patch_size * patch_size
    if target_prediction.shape[2] != expected_features:
        raise ValueError(
            f"target prediction feature size is {target_prediction.shape[2]}; expected {expected_features}"
        )
    return rearrange(
        target_prediction,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=packed.target_grid[0],
        w=packed.target_grid[1],
        c=channels,
        ph=patch_size,
        pw=patch_size,
    )


def fit_reference_pixels(
    reference: torch.Tensor,
    target_height: int,
    target_width: int,
    *,
    alignment: int = REFERENCE_PIXEL_ALIGNMENT,
    near_match_ratio: float = 0.92,
) -> torch.Tensor:
    """Fit a BCHW reference inside the target using the inference-compatible geometry."""
    if reference.ndim != 4:
        raise ValueError(f"reference must have shape (B, C, H, W), got {tuple(reference.shape)}")
    if target_height < alignment or target_width < alignment:
        raise ValueError(f"target dimensions must be at least {alignment}, got {(target_height, target_width)}")
    if target_height % alignment or target_width % alignment:
        raise ValueError(
            f"target dimensions must be divisible by alignment={alignment}, got {(target_height, target_width)}"
        )

    source_height, source_width = reference.shape[-2:]
    scale = min(target_height / source_height, target_width / source_width)
    if source_height * scale >= target_height * near_match_ratio and source_width * scale >= target_width * near_match_ratio:
        fill_scale = max(target_height / source_height, target_width / source_width)
        crop_height = min(source_height, int(round(target_height / fill_scale)))
        crop_width = min(source_width, int(round(target_width / fill_scale)))
        top = (source_height - crop_height) // 2
        left = (source_width - crop_width) // 2
        reference = reference[:, :, top : top + crop_height, left : left + crop_width]
        output_height, output_width = target_height, target_width
    else:
        output_height = min(
            max(alignment, int(source_height * scale) // alignment * alignment),
            target_height,
        )
        output_width = min(
            max(alignment, int(source_width * scale) // alignment * alignment),
            target_width,
        )
        crop_height = min(source_height, max(1, int(round(output_height / scale))))
        crop_width = min(source_width, max(1, int(round(output_width / scale))))
        top = (source_height - crop_height) // 2
        left = (source_width - crop_width) // 2
        reference = reference[:, :, top : top + crop_height, left : left + crop_width]

    if reference.shape[-2:] != (output_height, output_width):
        reference = F.interpolate(
            reference,
            size=(output_height, output_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    return reference