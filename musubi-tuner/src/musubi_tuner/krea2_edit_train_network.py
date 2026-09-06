"""LoRA training for Krea 2 Edit with cached reference conditioning."""

import argparse
import torch
import torch.nn.functional as F
from accelerate import Accelerator

from musubi_tuner.dataset.architectures import ARCHITECTURE_KREA2_EDIT, ARCHITECTURE_KREA2_EDIT_FULL
from musubi_tuner.dataset.cache_io import load_krea2_edit_latent_cache, validate_krea2_edit_text_encoder_cache
from musubi_tuner.hv_train_network import read_config_from_file, setup_parser_common
from musubi_tuner.krea2.krea2_edit_conditioning import pack_edit_image_tokens, unpack_target_prediction
from musubi_tuner.krea2_train_network import Krea2NetworkTrainer, krea2_setup_parser
from musubi_tuner.training.outputs import DiTOutput
from musubi_tuner.training.runtime_utils import configure_console_output_for_help


class Krea2EditNetworkTrainer(Krea2NetworkTrainer):
    @property
    def architecture(self) -> str:
        return ARCHITECTURE_KREA2_EDIT

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_KREA2_EDIT_FULL

    def handle_model_specific_args(self, args):
        super().handle_model_specific_args(args)
        if args.sample_prompts:
            raise ValueError("Krea 2 Edit training-time previews are not supported; omit --sample_prompts.")
        if args.network_module != "networks.lora_krea2":
            raise ValueError("Krea 2 Edit requires --network_module networks.lora_krea2.")

        if args.fp8_scaled:
            base_precision = "scaled FP8"
        elif args.convrot_int8:
            base_precision = f"ConvRot INT8 (backward={args.convrot_int8_bwd})"
        else:
            base_precision = "BF16"
        block_swap = f"{args.blocks_to_swap} blocks" if args.blocks_to_swap else "disabled"
        compile_mode = "enabled" if args.compile else "disabled"
        print(
            f"Krea 2 Edit runtime: base={base_precision}, block_swap={block_swap}, "
            f"torch_compile={compile_mode}, grounding=fixed-cache"
        )
        if args.compile and (args.convrot_int8 or args.blocks_to_swap):
            print("Krea 2 Edit torch.compile: quantized or swapped Linear modules remain eager.")

    def _build_dataset(self, args):
        dataset_group, collator, current_epoch = super()._build_dataset(args)
        validated_paths = set()
        for dataset in dataset_group.datasets:
            if dataset.batch_manager is None:
                raise ValueError("Krea 2 Edit dataset was not prepared for training.")
            for bucket in dataset.batch_manager.buckets.values():
                for item in bucket:
                    latent_cache_path = item.latent_cache_path
                    if not latent_cache_path:
                        raise ValueError(f"Krea 2 Edit latent cache is missing for {item.item_key!r}.")
                    if latent_cache_path in validated_paths:
                        continue
                    _, references, _ = load_krea2_edit_latent_cache(latent_cache_path)
                    if not item.text_encoder_output_cache_path:
                        raise ValueError(
                            f"Krea 2 Edit text cache is missing for {item.item_key!r}; "
                            "run krea2_edit_cache_text_encoder_outputs.py before training."
                        )
                    validate_krea2_edit_text_encoder_cache(
                        item.text_encoder_output_cache_path,
                        expected_reference_count=len(references),
                    )
                    validated_paths.add(latent_cache_path)
        return dataset_group, collator, current_epoch

    @staticmethod
    def _reference_latents(batch: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        reference_keys = sorted(
            (key for key in batch if key.startswith("latents_control_")),
            key=lambda key: int(key.rsplit("_", 1)[1]),
        )
        expected_keys = [f"latents_control_{index}" for index in range(len(reference_keys))]
        if reference_keys != expected_keys:
            raise ValueError(f"Krea 2 Edit reference latent keys must be contiguous from zero, got {reference_keys}")
        if len(reference_keys) > 2:
            raise ValueError("Krea 2 Edit supports at most two cached reference latents.")

        references = []
        for key in reference_keys:
            reference = batch[key]
            if reference.ndim != 5 or reference.shape[2] != 1:
                raise ValueError(f"{key} must have shape (B,C,1,H,W), got {tuple(reference.shape)}")
            references.append(reference.squeeze(2))
        if not references:
            raise ValueError("Krea 2 Edit requires one or two cached reference latents.")
        return references

    def call_dit(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        latents: torch.Tensor,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        network_dtype: torch.dtype,
        **kwargs,
    ) -> DiTOutput:
        raw_model = accelerator.unwrap_model(transformer, keep_fp32_wrapper=False)
        patch = raw_model.config.patch
        device = accelerator.device

        if latents.ndim != 5 or latents.shape[2] != 1:
            raise ValueError(f"Krea 2 Edit target latents must have shape (B,C,1,H,W), got {tuple(latents.shape)}")
        if latents.shape[0] != 1:
            raise ValueError(f"Krea 2 Edit training requires batch size 1, got {latents.shape[0]}")

        references = [reference.to(device=device, dtype=network_dtype) for reference in self._reference_latents(batch)]
        packed = pack_edit_image_tokens(
            noisy_model_input.squeeze(2).to(device=device, dtype=network_dtype),
            references,
            patch,
        )

        vl_embed = batch.get("krea2_vl_embed")
        if not vl_embed:
            raise ValueError(
                "Krea 2 Edit requires cached image-grounded text embeddings. "
                "Run krea2_edit_cache_text_encoder_outputs.py before training."
            )
        text_lengths = [embedding.shape[0] for embedding in vl_embed]
        max_text_length = max(text_lengths)
        context = torch.stack(
            [F.pad(embedding, (0, 0, 0, 0, 0, max_text_length - embedding.shape[0])) for embedding in vl_embed],
            dim=0,
        ).to(device=device, dtype=network_dtype)
        text_mask = torch.zeros(latents.shape[0], max_text_length, device=device, dtype=torch.bool)
        for index, length in enumerate(text_lengths):
            text_mask[index, :length] = True
        text_positions = torch.zeros(latents.shape[0], max_text_length, 3, device=device)

        mask = torch.cat((packed.mask, text_mask), dim=1)
        positions = torch.cat((packed.positions, text_positions), dim=1)
        timestep = (timesteps / 1000.0).to(device=device)

        if args.gradient_checkpointing:
            packed.tokens.requires_grad_(True)
            context.requires_grad_(True)

        with accelerator.autocast():
            prediction = transformer(img=packed.tokens, context=context, t=timestep, pos=positions, mask=mask)

        target_prediction = unpack_target_prediction(prediction, packed, latents.shape[1], patch).unsqueeze(2)
        clean_target = latents.to(device=device, dtype=network_dtype)
        return DiTOutput(pred=target_prediction, target=noise - clean_target)


def main():
    configure_console_output_for_help()
    parser = krea2_setup_parser(setup_parser_common())
    args = read_config_from_file(parser.parse_args(), parser)

    args.dit_dtype = "bfloat16"
    if args.vae_dtype is None:
        args.vae_dtype = "bfloat16"

    Krea2EditNetworkTrainer().train(args)


if __name__ == "__main__":
    main()