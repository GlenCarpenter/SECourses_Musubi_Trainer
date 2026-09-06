"""Cache fixed-scale image-grounded Qwen3-VL outputs for Krea 2 edit."""

import argparse
import logging
import os

import torch

import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.architectures import ARCHITECTURE_KREA2_EDIT
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import (
    ItemInfo,
    save_text_encoder_output_cache_krea2_edit,
    validate_krea2_edit_text_encoder_cache,
)
from musubi_tuner.krea2 import krea2_utils


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def encode_and_save_batch(encoder, batch: list[ItemInfo], grounding_pixels: int, *, profile_memory: bool = False):
    if grounding_pixels <= 0:
        raise ValueError("Fixed-scale Krea 2 edit text caching requires --grounding_pixels greater than zero")

    prompts = [item.caption for item in batch]
    references = []
    for item in batch:
        if not isinstance(item.control_content, list) or not 1 <= len(item.control_content) <= 2:
            raise ValueError(f"Krea 2 edit item {item.item_key!r} must contain one or two reference images")
        references.append(item.control_content)

    if profile_memory:
        if not torch.cuda.is_available():
            raise ValueError("Krea 2 Edit grounding memory profiling requires CUDA")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        online_baseline_bytes = torch.cuda.memory_allocated()

    hiddens, mask = krea2_utils.get_krea2_edit_prompt_embeds(
        encoder,
        prompts,
        references,
        grounding_min_pixels=grounding_pixels,
        grounding_max_pixels=grounding_pixels,
        grounding_size_selector=lambda _minimum, _maximum: grounding_pixels,
    )
    valid_hiddens = [hidden[item_mask.bool()] for hidden, item_mask in zip(hiddens, mask)]
    if profile_memory:
        torch.cuda.synchronize()
        online_peak_bytes = torch.cuda.max_memory_allocated()
        fixed_cache_bytes = sum(_tensor_bytes(hidden) for hidden in valid_hiddens)
        logger.info(
            "Krea 2 Edit grounding memory: online_baseline=%.2fMB, online_peak=%.2fMB, "
            "online_delta=%.2fMB, fixed_cache_payload=%.2fMB, grounding_pixels=%d",
            online_baseline_bytes / 1024**2,
            online_peak_bytes / 1024**2,
            max(0, online_peak_bytes - online_baseline_bytes) / 1024**2,
            fixed_cache_bytes / 1024**2,
            grounding_pixels,
        )

    for item, hidden, item_references in zip(batch, valid_hiddens, references):
        reference_pixel_sizes = [(reference.shape[0], reference.shape[1]) for reference in item_references]
        save_text_encoder_output_cache_krea2_edit(
            item,
            hidden,
            grounding_pixels=grounding_pixels,
            reference_pixel_sizes=reference_pixel_sizes,
        )


def discard_incompatible_existing_caches(all_cache_files: list[set[str]], grounding_pixels: int):
    """Ensure --skip_existing only reuses caches built for the requested grounding scale."""
    for cache_files in all_cache_files:
        incompatible = set()
        for cache_path in cache_files:
            try:
                validate_krea2_edit_text_encoder_cache(
                    cache_path,
                    expected_grounding_pixels=grounding_pixels,
                )
            except (OSError, ValueError) as exc:
                logger.info("Rebuilding incompatible Krea 2 edit text cache %s: %s", cache_path, exc)
                incompatible.add(os.path.normpath(cache_path))
        cache_files.difference_update(incompatible)


def setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--text_encoder", type=str, required=True, help="Qwen3-VL-4B text encoder safetensors path")
    parser.add_argument("--text_encoder_dtype", type=str, default=None, help="text encoder dtype; default is bfloat16")
    parser.add_argument(
        "--grounding_pixels",
        type=int,
        default=768,
        help="fixed longest-side grounding scale stored in the cache; default is 768",
    )
    parser.add_argument(
        "--profile_grounding_memory",
        action="store_true",
        help="measure online grounding peak VRAM and fixed-cache tensor payload for the first batch",
    )
    return parser


def main():
    parser = setup_parser(cache_text_encoder_outputs.setup_parser_common())
    args = parser.parse_args()
    if args.grounding_pixels <= 0:
        parser.error("--grounding_pixels must be greater than zero")

    device = torch.device(args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu")
    if args.profile_grounding_memory and device.type != "cuda":
        parser.error("--profile_grounding_memory requires a CUDA device")
    text_encoder_dtype = torch.bfloat16
    if args.text_encoder_dtype is not None:
        from musubi_tuner.utils.model_utils import str_to_dtype

        text_encoder_dtype = str_to_dtype(args.text_encoder_dtype)

    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = BlueprintGenerator(ConfigSanitizer()).generate(user_config, args, architecture=ARCHITECTURE_KREA2_EDIT)
    datasets = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group).datasets
    all_cache_files, all_cache_paths = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)
    discard_incompatible_existing_caches(all_cache_files, args.grounding_pixels)

    encoder = krea2_utils.load_krea2_text_encoder(args.text_encoder, dtype=text_encoder_dtype, device=device)
    logger.info("Encoding image-grounded Krea 2 edit prompts at a fixed %dpx scale", args.grounding_pixels)

    profile_next_batch = args.profile_grounding_memory

    def encode_batch(batch):
        nonlocal profile_next_batch
        encode_and_save_batch(encoder, batch, args.grounding_pixels, profile_memory=profile_next_batch)
        profile_next_batch = False

    cache_text_encoder_outputs.process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_cache_files,
        all_cache_paths,
        encode_batch,
        requires_content=True,
    )
    del encoder
    cache_text_encoder_outputs.post_process_cache_files(datasets, all_cache_files, all_cache_paths, args.keep_cache)


if __name__ == "__main__":
    main()