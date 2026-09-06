# Krea 2 Edit Phase 5 Results

Date: 5 September 2026

## Environment

- GPU: NVIDIA GeForce RTX 4090, 24 GB
- Target fixture: one 256x256 image
- Reference fixture: one 192x256 image
- LoRA: rank 4, alpha 4
- Mixed precision: BF16
- Attention selector: `--sdpa` (`flash_auto` is accepted by the Krea model path)
- Grounding: fixed 256px Qwen3-VL cache

## Grounding memory

Measured by `krea2_edit_cache_text_encoder_outputs.py --profile_grounding_memory` on the first batch:

| Metric | Result |
| --- | ---: |
| Online Qwen3-VL baseline allocation | 8519.61 MB |
| Online Qwen3-VL peak allocation | 8614.41 MB |
| Online grounding activation delta | 94.80 MB |
| Fixed-cache conditioning tensor payload | 4.98 MB |

The fixed-cache figure is the exact tensor payload transferred for training, not a second end-to-end process peak. The online baseline includes the resident Qwen3-VL model.

## Training matrix

Each row used the real Krea 2 base checkpoint, completed at least one optimizer step, saved a 28.05 MB LoRA containing 792 tensors, and reopened with `safetensors.safe_open`. Every checkpoint reported `modelspec.architecture=Krea-2/lora` and `ss_network_module=networks.lora_krea2`.

| Mode | Configuration | Result |
| --- | --- | --- |
| Scaled FP8 | `--fp8_base --fp8_scaled` | Passed; 224 Linear layers quantized |
| ConvRot INT8, BF16 backward | `--convrot_int8 --convrot_int8_bwd bf16` | Passed; fused Triton INT8 GEMM |
| ConvRot INT8, INT8 backward | `--convrot_int8 --convrot_int8_bwd int8` | Passed; fused Triton INT8 GEMM |
| BF16 block swap | `--blocks_to_swap 26 --block_swap_h2d_only` | Passed; 26 of 28 blocks streamed |
| torch.compile | scaled FP8 plus `--compile` | Passed and saved; Dynamo reported a non-fatal graph break at the attention sequence-length `.item()` call |

A separate 20-step, one-image scaled-FP8 run also completed and produced a readable LoRA. Loss values vary with sampled flow timesteps, so this smoke establishes trainability and serialization rather than visual convergence.

## Reproducing the memory measurement

```powershell
venv\Scripts\python.exe musubi-tuner\src\musubi_tuner\krea2_edit_cache_text_encoder_outputs.py `
  --dataset_config <edit-dataset.toml> `
  --text_encoder <qwen3vl-4b.safetensors> `
  --text_encoder_dtype bfloat16 `
  --grounding_pixels 256 `
  --profile_grounding_memory `
  --device cuda `
  --batch_size 1 `
  --num_workers 1 `
  --keep_cache
```
