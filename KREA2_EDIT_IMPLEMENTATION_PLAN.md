# Krea 2 Edit Training Implementation Plan

## Goal

Add a separate `krea2_edit` LoRA training architecture that produces checkpoints compatible with the `comfyui-krea2edit` fit protocol. Existing Krea 2 text-to-image and full-fine-tuning behavior must remain unchanged.

The edit architecture has two conditioning paths:

1. **Appearance path:** one or two clean reference-image latents are inserted before the noisy target tokens in the Krea 2 image-token sequence.
2. **Semantic path:** the instruction and the original reference images are encoded together by Qwen3-VL.

Only target tokens are noised and supervised. Reference tokens remain clean and are excluded from the flow-matching loss.

## Architecture

### Dataset contract

An edit dataset contains target images and one or two stem-matched reference directories:

```text
dataset/
  targets/
    0001.png
    0001.txt
  sources/
    0001.png
  sources_b/
    0001.png
```

- Captions are edit instructions, not descriptions of the target.
- Edit datasets require one or two references per target.
- Targets without references fail validation instead of silently becoming text-to-image examples.
- Batch size is initially fixed to 1 because references can have different native dimensions.
- Flip augmentation is rejected unless target and references are transformed together by a future paired augmentation implementation.
- Plain Krea 2 datasets remain supported through the existing `krea2` architecture, not by making references optional in `krea2_edit`.

### Latent cache

The edit latent cache stores:

- normalized target latent;
- ordered clean reference latents;
- reference count;
- target and fitted-reference geometry;
- fit-protocol version for cache invalidation.

Reference pixels are fitted inside the target grid while preserving aspect ratio. Output dimensions are aligned to 16 pixels because the /8 VAE and 2x2 DiT patching produce one token per 16x16 pixels. RoPE offsets use `(target_grid - reference_grid) / 2` as floating-point values; odd token gaps must retain half-token offsets.

### Text conditioning

The current Krea conditioner remains the text-only path. The edit conditioner adds a Qwen3-VL `AutoProcessor` path using the same fixed prompt template and selected hidden-state layers.

For exact recipe parity, reference images are encoded at training time with a random grounding cap from 384 to 768 pixels. A later low-VRAM mode may cache several grounding scales and select one per step. A single cached scale is allowed only as an explicit quality/VRAM tradeoff.

### DiT input

Musubi's Krea implementation uses image-first ordering so valid tokens remain a contiguous prefix for variable-length attention:

```text
[ref_1 | ref_2 | noisy_target | text | padding]
```

This is a permutation of the reference implementation's joint sequence and retains its positional contract:

- target frame index: `0`;
- reference frame indices: `1..N`;
- text positions: zero;
- fitted references: fractionally centered spatial offsets.

The DiT returns predictions for every image token. The trainer discards the leading reference predictions, unpatchifies only the target suffix, and computes the existing flow-matching target `noise - clean_target`.

### Model and checkpoint compatibility

- Reuse the existing Krea 2 MMDiT loader and `networks.lora_krea2` targets.
- Keep `krea2` and `krea2_edit` as distinct dataset/trainer architectures.
- Start with LoRA only. Full fine-tuning is deferred until LoRA behavior is verified in the inference nodes.
- Validate BF16 first, then scaled FP8, ConvRot INT8, block swap, and torch.compile independently.
- Edit-mode sample generation remains disabled until it accepts reference images and runs the same geometry and multimodal text path as training.

## Phases

### Phase 1: Conditioning primitives

**Status: implemented on 5 September 2026.** The helpers and CPU tests are present and passing.

- Add architecture identifiers without registering a runnable trainer.
- Implement reference fit geometry, token packing, RoPE frame/offset construction, and target-output slicing as pure functions.
- Unit-test shape validation, reference order, two-reference limits, fractional offsets, fit geometry, and target-only slicing.

**Exit criterion:** CPU-only tests prove the token and geometry contract without loading model weights.

### Phase 2: Paired dataset and latent cache

**Status: implemented on 5 September 2026.** Backend dataset discovery and cache serialization are complete; trainer consumption remains Phase 4.

- Add paired target/reference discovery using stem matching.
- Reject missing, partial, duplicate, or more-than-two-reference pairs before model loading.
- Add `krea2_edit_cache_latents.py` and cache schema/version validation.
- Add cache round-trip tests using a fake VAE.

**Exit criterion:** a synthetic paired dataset produces target/reference cache entries with deterministic geometry metadata, and invalid datasets fail before caching.

### Phase 3: Multimodal text conditioning

**Status: implemented on 5 September 2026.** The conditioner supports ordered image-plus-instruction encoding with injectable 384-768 grounding jitter, and the optional text-cache command records a validated fixed grounding scale.

- Extend the Krea Qwen3-VL wrapper with image-plus-instruction encoding.
- Preserve native reference pixels for semantic grounding before appearance-path fitting.
- Add configurable 384-768 grounding jitter and deterministic injection for tests.
- Add an explicit fixed-scale cache mode; defer multi-scale caching unless measurements justify it.

**Exit criterion:** mocked processor/model tests verify image order, prompt template, selected layers, prefix removal, and cache invalidation metadata.

### Phase 4: LoRA trainer

**Status: implemented on 5 September 2026.** The edit trainer requires validated fixed-scale image-grounded text caches, validates all edit cache contracts before model loading, and has CPU forward/backward coverage for one and two references. A 20-step, one-image real-model smoke run produced a readable Krea-compatible LoRA.

- Add `krea2_edit_train_network.py` as a subclass or narrow specialization of `Krea2NetworkTrainer`.
- Load edit caches, pack references and target, run the DiT, and supervise only target tokens.
- Reject sample previews, batch size above 1, unsupported augmentations, and incompatible cache modes during preflight.
- Reuse `networks.lora_krea2` and existing checkpoint conversion.

**Exit criterion:** a tiny synthetic model completes forward/backward; a one-image real-model overfit produces a loadable LoRA.

### Phase 5: Quantization and memory features

**Status: implemented on 5 September 2026.** Real one-step runs completed for scaled FP8, ConvRot INT8 with BF16 backward, ConvRot INT8 with INT8 backward, BF16 block swap, and torch.compile. Each run emitted its active path and produced a readable LoRA. The fixed-cache command now offers `--profile_grounding_memory`; the measured 256px fixture used 8519.61 MB online baseline, 8614.41 MB online peak, 94.80 MB online activation delta, and a 4.98 MB fixed-cache tensor payload. See `benchmarks/krea2_edit/phase5_results.md`.

- Test scaled FP8, ConvRot INT8 BF16 backward, ConvRot INT8 INT8 backward, block swap, and compile separately.
- Measure online-grounding VRAM and fixed-cache VRAM.
- Add guards for unsupported combinations rather than silent fallback.

**Exit criterion:** each advertised mode completes a short training run and emits the active quantization path in stdout.

### Phase 6: GUI integration

**Status: implemented on 5 September 2026.** The mounted Krea 2 tab now has an independent Text-to-Image / Image Edit task selector. Image Edit exposes ordered source directories, the fixed fit protocol, a 384-768 grounding scale, and generate/reuse fixed-cache policy; generates paired batch-size-1 dataset TOMLs; selects all three edit entry points; forces LoRA mode; disables unsupported previews; and restores edit settings from hand-saved and runtime TOMLs. The focused GUI suite passes 93 tests, the full suite passes 233 tests with 1 skipped, and the tab builds headlessly in Image Edit mode.

- Add Text-to-Image / Image Edit selection to the Krea 2 tab.
- Add source directory, optional second source, fit protocol, grounding range, and text-cache policy controls.
- Generate paired dataset TOMLs and chain edit-aware latent caching, optional text caching, and training.
- Preserve edit mode and all edit settings through hand-saved and runtime TOML round trips.

**Exit criterion:** the Gradio tab builds, print-command tests select all three edit scripts, and loading a runtime TOML restores the correct mode and panels.

### Phase 7: End-to-end compatibility

- Compare token geometry against `comfyui-krea2edit` v1.2.4 or newer.
- Overfit identity, outpaint, and two-reference fixtures.
- Load resulting LoRAs with the standard ComfyUI LoRA loader and compare inference behavior.
- Add edit-aware preview sampling only after parity is demonstrated.

**Exit criterion:** saved checkpoints work in the target inference nodes without key conversion and preserve expected reference alignment.

## Testing strategy

- CPU unit tests for geometry, packing, cache schema, pairing, and configuration guards.
- Mocked Qwen3-VL tests for multimodal prompt construction.
- Tiny-module forward/backward tests for target-only loss routing.
- Real-model smoke tests for BF16 first, then one test per memory/quantization feature.
- GUI construction and runtime-TOML round-trip tests before the mode becomes visible to users.

## Attribution

The behavioral contract is based on `lbouaraba/krea2edit-trainer` and its geometry documentation. That project is Apache-2.0 and includes code adapted from the Apache-2.0 Krea 2 reference implementation. Any source code adapted directly must retain the required license and NOTICE attribution.