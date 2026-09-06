# Krea 2 Edit Phase 7 Harness

Status: **in progress**. These scripts verify the integration pipeline; they do not establish visual parity without properly converged training runs.

## Durable checks

- `generate_fixtures.py` creates deterministic identity, outpaint, and ordered two-reference data, plus `dataset.toml` and a 120-step `smoke_training.toml`.
- `verify_comfyui_loader.py` loads a checkpoint through ComfyUI's real Krea model detection and standard LoRA parser/patcher.
- `run_comfyui_inference.py` submits matched-seed workflows to ComfyUI with `comfyui-krea2edit` v1.2.5 or newer.
- `evaluate_outputs.py` records descriptive pixel and landmark metrics for smoke outputs. Its observations are not Phase 7 acceptance thresholds.

Generated data, TOMLs, caches, checkpoints, ComfyUI outputs, and metric files are intentionally ignored. Run the generator to recreate them.

## What the short run proves

The 5 September 2026 smoke used the real Krea 2 base, VAE, and Qwen3-VL encoder. It completed 120 joint training steps, emitted four 792-tensor LoRAs, and every checkpoint mapped all 264 adapters through ComfyUI's standard loader without key conversion. ComfyUI v1.2.5 executed one-reference, outpaint, and two-reference workflows successfully.

This proves cache, training, serialization, loader, node, and token-routing compatibility. It does **not** prove visual convergence. The observed identity and alignment improvements are preliminary; the outpaint and two-reference outputs still contained obvious text artifacts.

## Remaining acceptance work

Run separately converged identity, outpaint, and two-reference experiments for hours as needed. Preserve intermediate checkpoints, compare matched seeds, and establish convergence-aware acceptance criteria before marking Phase 7 complete or enabling edit-aware training previews.