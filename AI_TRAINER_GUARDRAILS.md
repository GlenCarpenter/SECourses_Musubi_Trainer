## AI Trainer Guardrails

Use this checklist whenever adding a new trainer tab or training flow. Each guardrail below documents a mistake already made once in this codebase (and fixed) — they are easy to reintroduce when copy-pasting an existing `*_lora_gui.py` as the starting point for a new one, because the bug only shows up later, when a user reloads a saved/failed run.

---

## Guardrail 1: Logging paths

### Rule

- Never write `logging_dir = ""` into a runtime training TOML.
- Never write root-like logging paths such as `/`, `\`, `F:/`, or `C:/`.
- If logging is disabled, omit both `logging_dir` and `log_with` from the runtime TOML.
- If logging is enabled by UI fields, debug mode, or extra CLI args, default the base logging path to `output_dir/logs`.
- Let the backend create the final timestamped run folder under that base path.

### Why this exists

Empty-string `logging_dir` is treated by the backend as a real path. It gets a timestamp appended and turns into a root-level path:

- Linux: `/20260325100338`
- Windows: `F:\20260325050433`

That causes permission errors on Linux and misplaced TensorBoard folders on Windows.

### Required implementation pattern

- Route new runtime config generation through `musubi_tuner_gui/common_gui.py::SaveConfigFileToRun`.
- Keep `_normalize_logging_fields_for_run_config()` in the path for all new trainers.
- Keep backend protection in `musubi-tuner/src/musubi_tuner/hv_train.py` and `musubi-tuner/src/musubi_tuner/hv_train_network.py` so old/bad configs are still safe.

### Regression check

Before finishing a new trainer:

- Verify disabled logging produces no `logging_dir` and no `log_with` in the generated runtime TOML.
- Verify enabled TensorBoard logging resolves to `output_dir/logs/...`, not a filesystem root path.

---

## Guardrail 2: Training-mode round-trip (LoRA vs. Full Fine-Tuning)

Applies to any trainer that offers both a LoRA mode and a Full Fine-Tuning / DreamBooth mode (currently: FLUX, FLUX.2, FLUX Klein, Qwen Image, Z-Image, Ideogram 4, Krea 2). Skip this guardrail only if the new trainer is LoRA-only (like LTX-2) or has its full-finetune mode fully disabled in the UI (like Wan currently does).

### Rule

- The runtime TOML written to `output_dir` (via `SaveConfigFileToRun`) must include `training_mode`, in addition to correctly omitting `network_module` (and the rest of the LoRA-only keys) for full-finetune runs.
- The "Load"/"Open" config loader must not just do "if key present in file use it, else keep whatever the GUI currently shows" for `training_mode`. That fallback is exactly the bug: it silently trusts stale on-screen state instead of the file being loaded.

### Why this exists

A full-finetune run's runtime TOML correctly omits `network_module` (the backend's `--network_module` default is `None`, which means "full fine-tune the base model" — see `musubi-tuner/src/musubi_tuner/training/parser_common.py`). It used to *also* omit `training_mode`, since that's a GUI-only bookkeeping field with no backend meaning. That combination is the trap: if that run failed and the user reloaded the exact TOML it wrote (to inspect it or retry), the loader had no signal that it had been a full-finetune run. The mode selector silently fell back to "LoRA Training," and clicking Start Training launched a LoRA run instead of resuming the fine-tune — with no error, no warning, nothing on screen indicating anything was wrong.

### Required implementation pattern

Use the shared helpers in `musubi_tuner_gui/full_finetune_gui.py` — do not hand-roll this per trainer:

- `TRAINING_MODE_CHOICES`, `LORA_TRAINING_MODE`, `FULL_FINE_TUNING_MODE` — use these as your Radio's `choices=` when possible. If your trainer needs its own label (e.g. Qwen Image and Z-Image use `"DreamBooth Fine-Tuning"` instead of the canonical `"Full Fine-Tuning"`), that's fine — `normalize_training_mode()` / `is_full_fine_tuning()` accept both via an alias table, but you must tell the load-side helper about your label (see below).
- **Write side**: build your `SaveConfigFileToRun(..., exclusion=...)` list by calling `training_mode_runtime_exclusions(training_mode)` and folding its result in. Do **not** additionally hardcode `"training_mode"` into your own exclusion list — it's deliberately *not* in what that helper returns, specifically so it round-trips. (Persisting it is safe: the backend's `read_config_from_file` merges unknown TOML keys into an `argparse.Namespace` and never uses them — see `musubi-tuner/src/musubi_tuner/training/parser_common.py::read_config_from_file`.)
- **Load side**: in your `open_*_configuration()` function's per-key loop, special-case `training_mode` to resolve via `infer_training_mode_from_loaded_config(data, full_mode_label=<your Radio's actual full-finetune choice string>)` instead of the generic "in data ? file value : current UI value" branch. Pass `full_mode_label=` whenever your Radio doesn't use the canonical `"Full Fine-Tuning"` string. This one call handles three cases correctly: the modern case (`training_mode` present in the file), the legacy/broken case (`training_mode` absent, infers full-finetune from `network_module` also being absent), and the ordinary LoRA case (`training_mode` absent but `network_module` present).
- **Panel visibility**: if a `training_mode.change()` handler shows/hides accordions (e.g. "LoRA Settings" vs "Full Fine-Tuning Settings"), also chain the *same* sync function via `.then()` off both the Open and Load button `.click()` events, reading the now-updated `training_mode` component as input. Gradio does **not** re-fire `.change()` for a component whose value was set programmatically as part of another event's output tuple — see `flux_lora_gui.py`, `flux2_lora_gui.py`, and `modern_image_lora_gui.py` for the established `.then()` pattern.
- **Gotcha — keep the `.then()` sync function narrowly scoped to visibility.** If your `training_mode.change()` handler *also* recomputes some other component's value (e.g. `modern_image_lora_gui.py`'s `blocks_to_swap`, whose allowed maximum shrinks by 1 for Ideogram in full-finetune mode), do **not** reuse that same function for the Load/Open `.then()` chain. Doing so was tried and reverted: a value from the parent Load event and a bound from the `.then()` follow-up can validate against each other in the wrong order across the two server round-trips, producing a spurious `gradio.exceptions.Error: 'Value N is greater than maximum value M.'` that has nothing to do with the actual bug being fixed. Write a second, minimal function that returns only the accordion visibility updates (see `modern_image_lora_gui.py::sync_training_mode_visibility` vs. `toggle_training_mode`) and wire the full one only to the direct `.change()` handler.

### Regression check

Before finishing a new trainer with a LoRA/Full-Fine-Tuning toggle:

- Start (or use "Print Command" to preview) a Full Fine-Tuning run. Confirm the written runtime TOML contains `training_mode` and does **not** contain `network_module`.
- Take that exact TOML, reload it via the "Load" button, and confirm the mode selector shows Full Fine-Tuning again — not LoRA.
- Hand-edit a copy of that TOML to delete the `training_mode` line (simulating a file saved by a pre-fix version, or any other trainer that predates this pattern) and reload it. It must still resolve to Full Fine-Tuning, inferred from the missing `network_module`.
- Do the mirror check: reload an ordinary LoRA runtime TOML (has `network_module`, no `training_mode`) while the GUI currently shows Full Fine-Tuning, and confirm it correctly switches back to LoRA.
- If mode drives panel visibility, confirm the correct panel is visible after Load/Open, not just after manually clicking the mode radio — and confirm no numeric-bounds error appears in the process (see the gotcha above).

---

## Guardrail 3: Attention-mode resolution ("flash_auto") for new architectures

### Rule

- Any new backend model whose attention routes through `musubi-tuner/src/musubi_tuner/modules/attention.py` must accept the attention mode string `"flash_auto"` in whatever `attn_mode` validation it performs.
- Never validate `attn_mode` against only the upstream set `{"torch", "sdpa", "flash", "flash3", "sageattn", "xformers"}`.

### Why this exists

This fork's `trainer_base.py` resolves the `--sdpa` flag through `resolve_sdpa_backend()`, which returns the fork-specific mode `"flash_auto"` on machines where PyTorch's native flash SDPA is unavailable but the external FlashAttention package passes a forward+backward probe (a common Windows configuration). Upstream models validate `attn_mode` against a closed set that does not contain it. The MiniMax H3 tab shipped with `--sdpa` as its default attention and the very first training launch crashed at transformer load with `ValueError: Unsupported MiniMax-H3 attention mode: flash_auto` — after latent and text-encoder caching had already run for minutes. Nothing in the GUI-level tests catches this because the failure happens inside the backend model constructor on a real launch.

### Required implementation pattern

- When merging a new architecture from upstream, grep its model file for `attn_mode not in` and extend the accepted set with `"flash_auto"` (see `minimax_h3/model.py`). The shared `modules.attention` dispatch already implements the runtime behavior; only the closed-set validation needs the extra member.
- Keep the regression test in `musubi-tuner/tests/test_attention_backend_selection.py` (`test_minimax_h3_model_accepts_resolved_flash_auto_mode`) and add an equivalent one for each new architecture.

### Regression check

Before finishing a new trainer: run a real "Start Training" (or at minimum construct the model with `attn_mode="flash_auto"`) on a machine where `resolve_sdpa_backend()` returns `"flash_auto"`, not just `pytest` — the crash only appears at real transformer load time.

---

## Guardrail 4: Canonical checkpoint keys in model mergers

### Rule

- Compare checkpoint structures through canonical model keys, not raw safetensors keys.
- Normalize known wrapper prefixes such as `model.diffusion_model.` and `diffusion_model.` before compatibility checks.
- Treat `.scale_weight` and `.weight_scale` as equivalent storage names where the model format supports both.
- Do not require both inputs to duplicate `.comfy_quant` controls. One-sided controls are valid only when the other checkpoint has the matching INT8 weight and FP32 scale tensors for that canonical layer.
- Preserve checkpoint A's naming convention in the merged output and emit a control tensor for every merged quantized layer.

### Why this exists

The Krea 2 checkpoint merger originally compared raw key sets. Two compatible ConvRot INT8 checkpoints were rejected when one used native Krea keys with per-layer `.comfy_quant` tensors and the other used ComfyUI's `model.diffusion_model.` wrapper without duplicating every control tensor.

### Required implementation pattern

- Build collision-checked canonical-to-raw key maps for both inputs before comparing layouts.
- Compare structural tensors after excluding optional quantization control tensors.
- Resolve each quantized layer's settings from either input, reject conflicting settings, and still require compatible INT8 weight/FP32 scale pairs on both sides.
- Stream output under checkpoint A's raw keys so normalization does not silently change the selected output ecosystem.

### Regression check

- Merge an unprefixed checkpoint containing `.comfy_quant` controls with an otherwise identical `model.diffusion_model.`-prefixed checkpoint that omits those controls.
- Confirm the merge succeeds, ordinary tensors align by canonical key, and the output uses checkpoint A's namespace with a complete ConvRot control triple for every quantized layer.
- Confirm genuinely different canonical tensor layouts still fail before output is installed.

---

## Guardrail 5: Register new architectures for checkpoint metadata

### Rule

- Add every runnable architecture ID to `musubi-tuner/src/musubi_tuner/utils/sai_model_spec.py` before its trainer can save checkpoints.
- When a new mode produces adapters for an existing base model, reuse that base model's ModelSpec architecture and implementation instead of inventing an incompatible adapter identity.

### Why this exists

The first real Krea 2 Edit run completed model loading, forward, backward, and the optimizer step, then failed while saving with `ValueError: Unknown architecture: kr2e`. Unit tests of the training step did not exercise shared checkpoint metadata generation.

### Required implementation pattern

- Import the new short architecture ID in `sai_model_spec.py`.
- Add it to both the architecture/implementation mapping and the default-resolution mapping.
- Add a regression test that calls `build_metadata()` with the trainer's `architecture` property and checks the resulting ModelSpec identity.

### Regression check

- Complete at least one real or synthetic checkpoint save, reopen the safetensors file, and verify `modelspec.architecture`, `modelspec.implementation`, and the expected network metadata.

---

## Guardrail 6: Windows-safe backend help output

### Rule

- New executable backend entry points that use the shared bilingual parser must call `configure_console_output_for_help()` before `parse_args()`.

### Why this exists

The initial Krea 2 Edit trainer entry point crashed on Windows when invoked with `--help` because a CP1252 console could not encode Japanese help text. This prevented basic CLI discovery before any training work began.

### Required implementation pattern

- Import `configure_console_output_for_help` from `musubi_tuner.training.runtime_utils`.
- Call it at the start of `main()`, before constructing or parsing the argument parser.

### Regression check

- Run the new script with `--help` from Windows PowerShell and confirm it exits successfully.

---

## Guardrail 7: Keep task mode separate from training mode

### Rule

- A workflow selector such as Text-to-Image / Image Edit must be persisted independently from the LoRA / Full Fine-Tuning selector.
- Use the task mode to choose the dataset contract and cache/trainer entry points; use the training mode only to choose adapter versus full-model optimization.
- If a task is LoRA-only, force LoRA in normalization as well as in the UI and restore task-specific panel visibility after Load/Open.

### Why this exists

Krea 2 Image Edit shares the Krea model and LoRA implementation with Text-to-Image, but uses a paired dataset and three different backend scripts. Treating Image Edit as another training mode would entangle script selection with the existing full-finetune round trip and could silently launch the plain Krea trainer after loading a saved edit run.

### Required implementation pattern

- Persist a dedicated task key in hand-saved and runtime TOMLs.
- Validate unsupported task/training-mode combinations before model loading.
- Select latent cache, text cache, and trainer scripts from the task key as one coherent set.
- Chain task-specific visibility synchronization from both Open and Load events because programmatic Gradio updates do not trigger `.change()` handlers.
- Clear a generated dataset path when the user changes task so a TOML from the previous dataset contract is not silently reused.

### Regression check

- Print an edit workflow and confirm all three commands use edit entry points.
- Reload its runtime TOML and confirm Image Edit, LoRA, and the edit-only controls are restored.
- Switch back to Text-to-Image and confirm the standard Krea scripts and full-finetune option remain available.

---

## Guardrail 8: Keep GUI bookkeeping out of dataset fallbacks

### Rule

- When a workflow generates a dataset TOML, that file must remain authoritative for task-specific dataset fields.
- GUI bookkeeping persisted in the runtime TOML must not also reach dataset constructors through argparse fallback when the generated dataset TOML owns the corresponding value.
- Preserve bookkeeping needed for GUI Load/Open round trips; exclude it at the dataset blueprint boundary rather than deleting it from runtime persistence.

### Why this exists

Krea 2 Image Edit persists `reference_directory` for GUI round trips and writes the ordered dataset contract as `reference_directories` in its generated dataset TOML. The dataset blueprint previously treated every non-null runtime argument as a fallback, so training supplied both fields to `ImageDataset` and failed with `ValueError: Specify reference_directory or reference_directories, not both`. Cache generation succeeded because the collision appeared only when the training process rebuilt the dataset blueprint.

### Required implementation pattern

- Identify GUI-only or superseded runtime fields before constructing `argparse_config` in `BlueprintGenerator.generate()`.
- Exclude those fields only for the architecture or workflow whose generated dataset TOML replaces them; do not change fallback behavior globally.
- Keep ordered or multi-value dataset contracts in the dataset TOML, including Krea 2 Edit's `reference_directories`.

### Regression check

- Generate a blueprint from a dataset config containing the authoritative plural or structured field and a runtime namespace containing its GUI bookkeeping counterpart.
- Confirm the blueprint keeps the dataset-config value, leaves the conflicting fallback field unset, and can construct the dataset without a mutual-exclusion error.
