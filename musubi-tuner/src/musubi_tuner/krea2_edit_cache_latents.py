"""Cache target and fitted reference latents for Krea 2 edit LoRA training."""

import logging
from typing import List

import torch

import musubi_tuner.cache_latents as cache_latents
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.architectures import ARCHITECTURE_KREA2_EDIT
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ItemInfo, save_latent_cache_krea2_edit
from musubi_tuner.krea2.krea2_edit_conditioning import fit_reference_pixels
from musubi_tuner.qwen_image import qwen_image_autoencoder_kl, qwen_image_utils


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _encode_pixels(vae, pixels: torch.Tensor) -> torch.Tensor:
    pixels = pixels.unsqueeze(2).to(vae.device, dtype=vae.dtype)
    return vae.encode_pixels_to_latents(pixels)


def encode_and_save_batch(vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage, batch: List[ItemInfo]):
    if len(batch) != 1:
        raise ValueError(f"Krea 2 edit latent caching requires batch size 1, got {len(batch)}")

    item = batch[0]
    if not isinstance(item.control_content, list) or not 1 <= len(item.control_content) <= 2:
        raise ValueError(f"Krea 2 edit item {item.item_key!r} must contain one or two reference images")

    target_pixels = torch.from_numpy(item.content[..., :3]).permute(2, 0, 1).unsqueeze(0)
    target_pixels = target_pixels / 127.5 - 1.0
    target_height, target_width = target_pixels.shape[-2:]

    fitted_references = []
    for reference in item.control_content:
        reference_pixels = torch.from_numpy(reference[..., :3]).permute(2, 0, 1).unsqueeze(0)
        reference_pixels = reference_pixels / 127.5 - 1.0
        fitted_references.append(fit_reference_pixels(reference_pixels, target_height, target_width))

    with torch.no_grad():
        target_latent = _encode_pixels(vae, target_pixels)[0]
        reference_latents = [_encode_pixels(vae, reference)[0] for reference in fitted_references]

    reference_pixel_sizes = [tuple(reference.shape[-2:]) for reference in fitted_references]
    save_latent_cache_krea2_edit(
        item,
        target_latent,
        reference_latents,
        reference_pixel_sizes=reference_pixel_sizes,
    )


def main():
    parser = cache_latents.setup_parser_common()
    parser = cache_latents.hv_setup_parser(parser)
    args = parser.parse_args()

    if args.vae_dtype is not None:
        raise ValueError("VAE dtype is not supported in Krea 2 edit (uses the Qwen-Image VAE default).")

    device = args.device if getattr(args, "device", None) else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_KREA2_EDIT)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    datasets = train_dataset_group.datasets

    if args.debug_mode is not None:
        cache_latents.show_datasets(
            datasets, args.debug_mode, args.console_width, args.console_back, args.console_num_images, fps=16
        )
        return

    if args.vae is None:
        raise ValueError("VAE checkpoint is required")
    logger.info(f"Loading VAE model from {args.vae}")
    vae = qwen_image_utils.load_vae(args.vae, 3, device=device, disable_mmap=True)
    vae.to(device)

    cache_latents.encode_datasets(datasets, lambda batch: encode_and_save_batch(vae, batch), args)


if __name__ == "__main__":
    main()