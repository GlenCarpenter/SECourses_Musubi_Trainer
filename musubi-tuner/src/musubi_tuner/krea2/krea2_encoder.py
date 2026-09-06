"""Krea 2 (K2) text encoder: Qwen3-VL-4B conditioner.

Returns the stacked selected hidden states (b, seq, num_select_layers, dim) plus the
attention mask; the layerwise fusion lives inside the DiT (TextFusionTransformer), so
the raw stack is what gets cached during training.

Loading follows musubi conventions (cf. qwen_image's load_qwen2_5_vl): the model config
is vendored here so it is built without fetching config.json from the Hub, weights are
loaded directly from a local safetensors file (ComfyUI-style `model.`/`visual.` keys are
accepted as well as the official HF layout), and only the tokenizer is still pulled by
repo id. This lets K2 share the same Qwen3-VL-4B weights a user already has for ComfyUI,
instead of requiring a separate transformers/Diffusers checkpoint.

The edit-mode image grounding contract is adapted from the Apache-2.0 project
https://github.com/lbouaraba/krea2edit-trainer.
"""

import logging
import random
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from accelerate import init_empty_weights
from PIL import Image
from torch import Tensor
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2TokenizerFast,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
)

from musubi_tuner.utils.safetensors_utils import load_split_weights

logger = logging.getLogger(__name__)


# Only the tokenizer is still fetched by repo id (small, HF-cached after first use).
QWEN3_VL_4B_INSTRUCT_REPO_ID = "Qwen/Qwen3-VL-4B-Instruct"
KREA2_EDIT_GROUNDING_MAX_PIXELS = 768
KREA2_EDIT_GROUNDING_MIN_PIXELS = 384
KREA2_EDIT_VISION_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"

# Vendored copy of the Qwen3-VL-4B-Instruct config.json so the text encoder is built
# without fetching the config from the Hugging Face Hub. Qwen3-VL is natively supported by
# transformers (no auto_map / remote code), so Qwen3VLConfig.from_dict reproduces
# AutoConfig.from_pretrained exactly. Mirror upstream config.json if Qwen ever revises it.
QWEN3_VL_4B_INSTRUCT_CONFIG = {
    "architectures": ["Qwen3VLForConditionalGeneration"],
    "image_token_id": 151655,
    "model_type": "qwen3_vl",
    "text_config": {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "dtype": "bfloat16",
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 2560,
        "initializer_range": 0.02,
        "intermediate_size": 9728,
        "max_position_embeddings": 262144,
        "model_type": "qwen3_vl_text",
        "num_attention_heads": 32,
        "num_hidden_layers": 36,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-06,
        "rope_scaling": {"mrope_interleaved": True, "mrope_section": [24, 20, 20], "rope_type": "default"},
        "rope_theta": 5000000,
        "tie_word_embeddings": True,
        "use_cache": True,
        "vocab_size": 151936,
    },
    "tie_word_embeddings": True,
    "transformers_version": "4.57.0.dev0",
    "video_token_id": 151656,
    "vision_config": {
        "deepstack_visual_indexes": [5, 11, 17],
        "depth": 24,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1024,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "model_type": "qwen3_vl",
        "num_heads": 16,
        "num_position_embeddings": 2304,
        "out_hidden_size": 2560,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
    "vision_end_token_id": 151653,
    "vision_start_token_id": 151652,
}


@dataclass
class TextEncoderConfig:
    max_length: int = 512
    select_layers: tuple[int, ...] = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
    tokenizer_repo: str = QWEN3_VL_4B_INSTRUCT_REPO_ID


def select_grounding_size(
    minimum: int = KREA2_EDIT_GROUNDING_MIN_PIXELS,
    maximum: int = KREA2_EDIT_GROUNDING_MAX_PIXELS,
    *,
    randint: Callable[[int, int], int] = random.randint,
) -> int:
    """Select the longest-side cap for one semantic-grounding reference."""
    if maximum <= 0:
        return 0
    if minimum < 0:
        raise ValueError(f"grounding minimum must be non-negative, got {minimum}")
    if minimum > maximum:
        raise ValueError(f"grounding minimum {minimum} exceeds maximum {maximum}")
    return randint(minimum, maximum) if 0 < minimum < maximum else maximum


def prepare_grounding_image(image, longest_side: int) -> Image.Image:
    """Convert an RGB image to PIL and optionally downscale it without upscaling."""
    if isinstance(image, Image.Image):
        result = image.convert("RGB")
    else:
        if isinstance(image, Tensor):
            pixels = image.detach().float().cpu()
            if pixels.ndim == 4 and pixels.shape[0] == 1:
                pixels = pixels[0]
            if pixels.ndim != 3:
                raise ValueError(f"grounding tensor must have shape (C,H,W) or (1,C,H,W), got {tuple(pixels.shape)}")
            if pixels.shape[0] not in (3, 4):
                raise ValueError(f"grounding tensor must have 3 or 4 channels, got {pixels.shape[0]}")
            pixels = pixels[:3]
            if pixels.numel() and pixels.max().item() <= 1.0:
                pixels = pixels * 255
            array = pixels.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
        else:
            array = np.asarray(image)
            if array.ndim != 3 or array.shape[-1] not in (3, 4):
                raise ValueError(f"grounding image must have shape (H,W,3|4), got {tuple(array.shape)}")
            array = array[..., :3]
            if np.issubdtype(array.dtype, np.floating) and array.size and float(array.max()) <= 1.0:
                array = array * 255
            array = np.clip(array, 0, 255).astype(np.uint8)
        result = Image.fromarray(array, mode="RGB")

    if longest_side > 0 and max(result.size) > longest_side:
        scale = longest_side / max(result.size)
        result = result.resize(
            (max(1, round(result.size[0] * scale)), max(1, round(result.size[1] * scale))),
            Image.Resampling.LANCZOS,
        )
    return result


def _convert_comfyui_qwen3vl_state_dict(sd: dict[str, Tensor]) -> dict[str, Tensor]:
    """Map a ComfyUI-style (bare ``model.`` / ``visual.``) Qwen3-VL state dict onto the HF
    ``Qwen3VLForConditionalGeneration`` layout. Official HF checkpoints already use the
    ``model.language_model.`` / ``model.visual.`` layout and pass through unchanged.
    """
    converted: dict[str, Tensor] = {}
    for key, value in sd.items():
        if key.startswith("model.language_model.") or key.startswith("model.visual."):
            new_key = key
        elif key.startswith("visual."):
            new_key = "model.visual." + key[len("visual.") :]
        elif key.startswith("language_model."):
            new_key = "model." + key
        elif key.startswith("model."):
            new_key = "model.language_model." + key[len("model.") :]
        else:
            new_key = key
        converted[new_key] = value
    return converted


def _load_qwen3_vl_model(
    model_path: str,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
    disable_mmap: bool = True,
) -> Qwen3VLForConditionalGeneration:
    """Build Qwen3-VL-4B from the vendored config and load weights from a local safetensors."""
    config = Qwen3VLConfig.from_dict(QWEN3_VL_4B_INSTRUCT_CONFIG)
    with init_empty_weights():
        model = Qwen3VLForConditionalGeneration._from_config(config)

    logger.info(f"Loading Krea 2 text encoder (Qwen3-VL) weights from {model_path}")
    sd = load_split_weights(model_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)
    sd = _convert_comfyui_qwen3vl_state_dict(sd)

    info = model.load_state_dict(sd, strict=False, assign=True)
    # Qwen3-VL-4B ties the LM head to the input embeddings (tie_word_embeddings=true), so the
    # checkpoint omits lm_head.weight; re-tie after loading to materialize it.
    model.tie_weights()

    unexpected = list(info.unexpected_keys)
    missing = [k for k in info.missing_keys if k != "lm_head.weight"]
    if unexpected or missing:
        raise RuntimeError(
            f"Qwen3-VL text encoder checkpoint did not match the model: missing={missing[:10]}, unexpected={unexpected[:10]}"
        )

    model.to(device)
    if dtype is not None:
        model.to(dtype)
    return model.eval().requires_grad_(False)


def load_qwen3_vl_conditioner(
    model_path: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cpu",
    max_length: int = TextEncoderConfig.max_length,
    select_layers: tuple[int, ...] = TextEncoderConfig.select_layers,
    tokenizer_repo: str = QWEN3_VL_4B_INSTRUCT_REPO_ID,
    disable_mmap: bool = True,
) -> "Qwen3VLConditioner":
    """Load the Qwen3-VL-4B conditioner used by K2: weights from ``model_path`` (safetensors),
    tokenizer from ``tokenizer_repo`` (Hub id or local dir)."""
    qwen = _load_qwen3_vl_model(model_path, dtype=dtype, device=device, disable_mmap=disable_mmap)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo, max_length=max_length)
    processor = Qwen2TokenizerFast.from_pretrained(tokenizer_repo, max_length=max_length)
    conditioner = Qwen3VLConditioner(
        qwen,
        tokenizer,
        processor,
        max_length=max_length,
        select_layers=select_layers,
        multimodal_processor_repo=tokenizer_repo,
    )
    return conditioner.eval().requires_grad_(False)


class Qwen3VLConditioner(torch.nn.Module):
    def __init__(
        self,
        qwen: Qwen3VLForConditionalGeneration,
        tokenizer,
        processor,
        max_length: int = 512,
        select_layers: tuple[int, ...] = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35),
        multimodal_processor=None,
        multimodal_processor_repo: str = QWEN3_VL_4B_INSTRUCT_REPO_ID,
    ):
        super().__init__()
        self.qwen = qwen.eval().requires_grad_(False)
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        self.select_layers = select_layers
        self.multimodal_processor = multimodal_processor
        self.multimodal_processor_repo = multimodal_processor_repo
        self.prompt_template_encode_prefix = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
        self.prompt_template_encode_suffix = "<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_encode_start_idx = 34
        self.prompt_template_encode_suffix_start_idx = 5

    def forward(self, text: list[str]) -> tuple[Tensor, Tensor]:
        prefix_idx = self.prompt_template_encode_start_idx
        text = [self.prompt_template_encode_prefix + item for item in text]
        suffix_text = [self.prompt_template_encode_suffix] * len(text)
        suffix_inputs = self.processor(text=suffix_text, return_tensors="pt").to(self.qwen.device, non_blocking=True)
        suffix_ids, suffix_mask = (
            suffix_inputs["input_ids"],
            suffix_inputs["attention_mask"].bool(),
        )

        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                truncation=True,
                return_length=False,
                return_overflowing_tokens=False,
                padding="max_length",
                max_length=self.max_length + prefix_idx - self.prompt_template_encode_suffix_start_idx,
                return_tensors="pt",
            ).to(self.qwen.device, non_blocking=True)
            input_ids = torch.cat([inputs["input_ids"], suffix_ids], dim=1)
            mask = torch.cat([inputs["attention_mask"].bool(), suffix_mask], dim=1)
            states = self.qwen(input_ids=input_ids, attention_mask=mask, output_hidden_states=True)

            hiddens = torch.stack([states.hidden_states[i] for i in self.select_layers], dim=2)
            hiddens = hiddens[:, prefix_idx:]
            mask = mask[:, prefix_idx:]

            return hiddens, mask

    def _get_multimodal_processor(self):
        if self.multimodal_processor is None:
            self.multimodal_processor = AutoProcessor.from_pretrained(self.multimodal_processor_repo)
        return self.multimodal_processor

    def forward_with_images(
        self,
        text: list[str],
        images: Sequence[Sequence],
        *,
        grounding_min_pixels: int = KREA2_EDIT_GROUNDING_MIN_PIXELS,
        grounding_max_pixels: int = KREA2_EDIT_GROUNDING_MAX_PIXELS,
        grounding_size_selector: Optional[Callable[[int, int], int]] = None,
    ) -> tuple[Tensor, Tensor]:
        """Encode edit instructions grounded by one or two ordered reference images."""
        if len(text) != len(images):
            raise ValueError(f"text/image batch mismatch: {len(text)} prompts and {len(images)} image groups")
        if not text:
            raise ValueError("Krea 2 edit conditioning requires at least one prompt")

        processor = self._get_multimodal_processor()
        features = []
        for prompt, item_images in zip(text, images):
            references = list(item_images)
            if not 1 <= len(references) <= 2:
                raise ValueError(f"Krea 2 edit conditioning requires one or two reference images, got {len(references)}")

            grounding_images = []
            for reference in references:
                grounding_size = select_grounding_size(
                    grounding_min_pixels,
                    grounding_max_pixels,
                    randint=grounding_size_selector or random.randint,
                )
                grounding_images.append(prepare_grounding_image(reference, grounding_size))

            vision_prefix = KREA2_EDIT_VISION_PLACEHOLDER * len(grounding_images)
            encoded_text = self.prompt_template_encode_prefix + vision_prefix + str(prompt) + self.prompt_template_encode_suffix
            inputs = processor(text=[encoded_text], images=grounding_images, return_tensors="pt").to(
                self.qwen.device, non_blocking=True
            )
            with torch.no_grad():
                states = self.qwen(**inputs, output_hidden_states=True)
                hiddens = torch.stack([states.hidden_states[index] for index in self.select_layers], dim=2)
            valid_length = int(inputs["attention_mask"][0].sum().item())
            features.append(hiddens[0, self.prompt_template_encode_start_idx : valid_length])

        max_length = max(feature.shape[0] for feature in features)
        hidden_shape = features[0].shape[1:]
        padded = features[0].new_zeros((len(features), max_length, *hidden_shape))
        mask = torch.zeros((len(features), max_length), dtype=torch.bool, device=features[0].device)
        for index, feature in enumerate(features):
            padded[index, : feature.shape[0]] = feature
            mask[index, : feature.shape[0]] = True
        return padded, mask
