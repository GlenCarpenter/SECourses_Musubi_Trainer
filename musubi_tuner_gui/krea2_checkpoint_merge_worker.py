import argparse
import gc
import json
import os
from typing import Dict, Tuple

import torch
from tqdm import tqdm

from musubi_tuner.modules.convrot_int8_kernels import (
    dequantize_int8_convrot_weight,
    quantize_int8_convrot_weight,
)
from musubi_tuner.modules.convrot_int8_utils import parse_comfy_quant_spec
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen, StreamingSafetensorsWriter


COMFY_QUANT_SUFFIX = ".comfy_quant"
WEIGHT_SUFFIX = ".weight"
WEIGHT_SCALE_SUFFIX = ".weight_scale"
MUSUBI_SCALE_SUFFIX = ".scale_weight"
CHECKPOINT_PREFIXES = ("model.diffusion_model.", "diffusion_model.")


def _resolve_device(choice: str) -> torch.device:
    normalized = str(choice or "auto").strip().lower()
    if normalized == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was selected, but CUDA is not available.")
    if normalized == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _validate_paths(checkpoint_a: str, checkpoint_b: str, output_path: str, overwrite: bool) -> None:
    for label, path in (("Checkpoint A", checkpoint_a), ("Checkpoint B", checkpoint_b)):
        if not path or not os.path.isfile(path):
            raise ValueError(f"{label} was not found: {path}")
        if not path.lower().endswith(".safetensors"):
            raise ValueError(f"{label} must be a .safetensors file: {path}")
    if os.path.abspath(checkpoint_a) == os.path.abspath(checkpoint_b):
        raise ValueError("Checkpoint A and Checkpoint B must be different files.")
    if not output_path:
        raise ValueError("Output path must be specified.")
    if os.path.abspath(output_path) in {os.path.abspath(checkpoint_a), os.path.abspath(checkpoint_b)}:
        raise ValueError("Output path must not overwrite either input checkpoint.")
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")


def _canonical_key(key: str) -> str:
    for prefix in CHECKPOINT_PREFIXES:
        if key.startswith(prefix):
            key = key[len(prefix) :]
            break
    if key.endswith(MUSUBI_SCALE_SUFFIX):
        key = key[: -len(MUSUBI_SCALE_SUFFIX)] + WEIGHT_SCALE_SUFFIX
    return key


def _canonical_key_map(reader: MemoryEfficientSafeOpen) -> Dict[str, str]:
    key_map: Dict[str, str] = {}
    for raw_key in reader.keys():
        canonical_key = _canonical_key(raw_key)
        if canonical_key in key_map:
            raise ValueError(
                f"Checkpoint contains duplicate canonical tensor key {canonical_key}: "
                f"{key_map[canonical_key]} and {raw_key}"
            )
        key_map[canonical_key] = raw_key
    return key_map


def _normalize_quant_spec(key: str, spec: object) -> Dict[str, object]:
    if not isinstance(spec, dict):
        raise ValueError(f"Invalid ConvRot quantization metadata for {key}: expected an object.")
    groupsize = spec.get("convrot_groupsize")
    is_power_of_4 = (
        isinstance(groupsize, int)
        and groupsize > 0
        and groupsize & (groupsize - 1) == 0
        and (groupsize.bit_length() - 1) % 2 == 0
    )
    if spec.get("format") != "int8_tensorwise" or spec.get("convrot") is not True or not is_power_of_4:
        raise ValueError(
            f"Unsupported quantization metadata for {key}: {spec}. "
            "Only ConvRot INT8 with a power-of-4 group size is supported."
        )
    return {
        "format": "int8_tensorwise",
        "convrot": True,
        "convrot_groupsize": groupsize,
    }


def _quantized_modules(
    reader: MemoryEfficientSafeOpen,
    key_map: Dict[str, str],
) -> Dict[str, Dict[str, object]]:
    modules: Dict[str, Dict[str, object]] = {}
    raw_metadata = reader.metadata().get("_quantization_metadata")
    if raw_metadata:
        try:
            quantization_metadata = json.loads(raw_metadata)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid _quantization_metadata JSON: {exc}") from exc
        layers = quantization_metadata.get("layers", {})
        if not isinstance(layers, dict):
            raise ValueError("Invalid _quantization_metadata: layers must be an object.")
        for raw_module_path, raw_spec in layers.items():
            module_path = _canonical_key(raw_module_path)
            if module_path.endswith(WEIGHT_SUFFIX):
                module_path = module_path[: -len(WEIGHT_SUFFIX)]
            modules[module_path] = _normalize_quant_spec(raw_module_path, raw_spec)

    for key, raw_key in key_map.items():
        if not key.endswith(COMFY_QUANT_SUFFIX):
            continue
        module_path = key[: -len(COMFY_QUANT_SUFFIX)]
        control_spec = _normalize_quant_spec(raw_key, parse_comfy_quant_spec(raw_key, reader.get_tensor(raw_key)))
        if module_path in modules and modules[module_path] != control_spec:
            raise ValueError(f"ConvRot quantization metadata differs for layer {module_path}.")
        modules[module_path] = control_spec

    for module_path in modules:
        weight_key = module_path + WEIGHT_SUFFIX
        scale_key = module_path + WEIGHT_SCALE_SUFFIX
        missing = [sibling for sibling in (weight_key, scale_key) if sibling not in key_map]
        if missing:
            raise ValueError(f"ConvRot layer {module_path} is missing tensors: {', '.join(missing)}")
    return modules


def _output_quant_key(module_path: str, key_map: Dict[str, str]) -> str:
    canonical_key = module_path + COMFY_QUANT_SUFFIX
    if canonical_key in key_map:
        return key_map[canonical_key]
    raw_weight_key = key_map[module_path + WEIGHT_SUFFIX]
    return raw_weight_key[: -len(WEIGHT_SUFFIX)] + COMFY_QUANT_SUFFIX


def _output_scale_key(module_path: str, key_map: Dict[str, str]) -> str:
    canonical_key = module_path + WEIGHT_SCALE_SUFFIX
    if canonical_key in key_map:
        return key_map[canonical_key]
    raw_weight_key = key_map[module_path + WEIGHT_SUFFIX]
    return raw_weight_key[: -len(WEIGHT_SUFFIX)] + WEIGHT_SCALE_SUFFIX


def _dense_weight(
    reader: MemoryEfficientSafeOpen,
    key_map: Dict[str, str],
    module_path: str,
    groupsize: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.dtype, bool]:
    weight_key = module_path + WEIGHT_SUFFIX
    scale_key = module_path + WEIGHT_SCALE_SUFFIX
    weight = reader.get_tensor(key_map[weight_key], device=device)
    source_dtype = weight.dtype
    if scale_key not in key_map:
        if not weight.is_floating_point():
            raise ValueError(
                f"Layer {module_path} has a non-floating weight without a matching quantization scale."
            )
        return weight.to(torch.float32), source_dtype, False

    scale = reader.get_tensor(key_map[scale_key], device=device)
    if weight.dtype is not torch.int8:
        raise ValueError(f"ConvRot weight for layer {module_path} must use INT8 when a scale tensor is present.")
    if scale.dtype is not torch.float32:
        raise ValueError(f"ConvRot scale for layer {module_path} must use FP32.")
    expected_scale_shape = (weight.shape[0], 1)
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            f"ConvRot scale for layer {module_path} must have shape {expected_scale_shape}, got {tuple(scale.shape)}."
        )
    if weight.shape[1] % groupsize:
        raise ValueError(
            f"ConvRot group size {groupsize} does not divide the input width {weight.shape[1]} "
            f"for layer {module_path}."
        )
    dense = dequantize_int8_convrot_weight(weight, scale, groupsize)
    del weight, scale
    return dense, source_dtype, True


def _blend_float_tensors(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    weight_a: float,
    weight_b: float,
    device: torch.device,
) -> torch.Tensor:
    output_dtype = tensor_a.dtype
    merged = tensor_a.to(device=device, dtype=torch.float32).mul_(weight_a)
    merged.add_(tensor_b.to(device=device, dtype=torch.float32), alpha=weight_b)
    return merged.to(device="cpu", dtype=output_dtype)


def merge_krea2_checkpoints(
    checkpoint_a: str,
    checkpoint_b: str,
    output_path: str,
    weight_a: float,
    device_choice: str = "auto",
    overwrite: bool = False,
) -> Tuple[str, str]:
    try:
        _validate_paths(checkpoint_a, checkpoint_b, output_path, overwrite)
        weight_a = float(weight_a)
        if not 0.0 <= weight_a <= 1.0:
            raise ValueError("Checkpoint A weight must be between 0 and 1.")
        weight_b = 1.0 - weight_a
        device = _resolve_device(device_choice)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        with (
            MemoryEfficientSafeOpen(checkpoint_a) as reader_a,
            MemoryEfficientSafeOpen(checkpoint_b) as reader_b,
        ):
            key_map_a = _canonical_key_map(reader_a)
            key_map_b = _canonical_key_map(reader_b)
            keys_a = set(key_map_a)
            keys_b = set(key_map_b)
            declared_modules_a = _quantized_modules(reader_a, key_map_a)
            declared_modules_b = _quantized_modules(reader_b, key_map_b)
            module_paths = set(declared_modules_a) | set(declared_modules_b)

            structural_keys_a = {
                key
                for key in keys_a
                if not key.endswith(COMFY_QUANT_SUFFIX) and not key.endswith(WEIGHT_SCALE_SUFFIX)
            }
            structural_keys_b = {
                key
                for key in keys_b
                if not key.endswith(COMFY_QUANT_SUFFIX) and not key.endswith(WEIGHT_SCALE_SUFFIX)
            }
            if structural_keys_a != structural_keys_b:
                only_a = sorted(structural_keys_a - structural_keys_b)
                only_b = sorted(structural_keys_b - structural_keys_a)
                raise ValueError(
                    "Checkpoint tensor layouts do not match after normalizing known prefixes. "
                    f"Only in A: {only_a[:3] or 'none'}; only in B: {only_b[:3] or 'none'}"
                )
            if not module_paths:
                raise ValueError("Neither checkpoint contains ComfyUI ConvRot INT8 metadata.")

            modules: Dict[str, Dict[str, object]] = {}
            for module_path in sorted(module_paths):
                spec_a = declared_modules_a.get(module_path)
                spec_b = declared_modules_b.get(module_path)
                if spec_a is not None and spec_b is not None and spec_a != spec_b:
                    raise ValueError(f"ConvRot quantization settings differ for layer {module_path}.")
                for label, key_map in (("A", key_map_a), ("B", key_map_b)):
                    weight_key = module_path + WEIGHT_SUFFIX
                    if weight_key not in key_map:
                        raise ValueError(f"Checkpoint {label} is missing tensor: {weight_key}")
                modules[module_path] = spec_a or spec_b

            declared_scale_modules = {
                key[: -len(WEIGHT_SCALE_SUFFIX)]
                for key in keys_a | keys_b
                if key.endswith(WEIGHT_SCALE_SUFFIX)
            }
            unsupported_scale_modules = sorted(declared_scale_modules - module_paths)
            if unsupported_scale_modules:
                raise ValueError(
                    "Found quantization scales without ConvRot metadata in either checkpoint: "
                    f"{unsupported_scale_modules[:3]}"
                )

            quantized_keys = {
                module_path + suffix
                for module_path in modules
                for suffix in (WEIGHT_SUFFIX, WEIGHT_SCALE_SUFFIX, COMFY_QUANT_SUFFIX)
            }
            metadata = dict(reader_a.metadata())
            metadata.update(
                {
                    "krea2_checkpoint_merge": "true",
                    "merge_checkpoint_a": os.path.basename(checkpoint_a),
                    "merge_checkpoint_b": os.path.basename(checkpoint_b),
                    "merge_weight_a": f"{weight_a:.8g}",
                    "merge_weight_b": f"{weight_b:.8g}",
                }
            )

            with StreamingSafetensorsWriter(output_path, metadata=metadata) as writer:
                ordinary_keys = sorted(structural_keys_a - quantized_keys)
                for key in tqdm(ordinary_keys, desc="Merging Krea 2 tensors", unit="tensor"):
                    tensor_a = reader_a.get_tensor(key_map_a[key])
                    tensor_b = reader_b.get_tensor(key_map_b[key])
                    if tensor_a.shape != tensor_b.shape or tensor_a.dtype != tensor_b.dtype:
                        raise ValueError(
                            f"Tensor {key} differs between checkpoints: "
                            f"A={tuple(tensor_a.shape)} {tensor_a.dtype}, B={tuple(tensor_b.shape)} {tensor_b.dtype}"
                        )
                    if tensor_a.is_floating_point():
                        merged = _blend_float_tensors(tensor_a, tensor_b, weight_a, weight_b, device)
                    else:
                        if not torch.equal(tensor_a, tensor_b):
                            raise ValueError(f"Non-floating tensor {key} differs between checkpoints.")
                        merged = tensor_a
                    writer.write_tensor(key_map_a[key], merged)
                    del tensor_a, tensor_b, merged

                for module_path, spec in tqdm(
                    sorted(modules.items()), desc="Merging ConvRot INT8 layers", unit="layer"
                ):
                    weight_key = module_path + WEIGHT_SUFFIX
                    groupsize = int(spec["convrot_groupsize"])
                    dense_a, dtype_a, quantized_a = _dense_weight(
                        reader_a, key_map_a, module_path, groupsize, device
                    )
                    dense_b, _dtype_b, _quantized_b = _dense_weight(
                        reader_b, key_map_b, module_path, groupsize, device
                    )
                    if dense_a.shape != dense_b.shape:
                        raise ValueError(f"Weight shapes differ for layer {module_path}.")
                    merged = dense_a.mul_(weight_a)
                    merged.add_(dense_b, alpha=weight_b)
                    if quantized_a:
                        merged_quant, merged_scale = quantize_int8_convrot_weight(merged, groupsize)
                        writer.write_tensor(key_map_a[weight_key], merged_quant.cpu())
                        writer.write_tensor(_output_scale_key(module_path, key_map_a), merged_scale.cpu())
                        control_reader = reader_a if module_path in declared_modules_a else reader_b
                        control_key_map = key_map_a if module_path in declared_modules_a else key_map_b
                        control_key = module_path + COMFY_QUANT_SUFFIX
                        if control_key in control_key_map:
                            control_tensor = control_reader.get_tensor(control_key_map[control_key])
                        else:
                            payload = json.dumps(spec, separators=(",", ":")).encode("utf-8")
                            control_tensor = torch.tensor(list(payload), dtype=torch.uint8)
                        writer.write_tensor(_output_quant_key(module_path, key_map_a), control_tensor)
                        del merged_quant, merged_scale
                    else:
                        writer.write_tensor(key_map_a[weight_key], merged.to(device="cpu", dtype=dtype_a))
                    writer.mark_group_complete(module_path)
                    del dense_a, dense_b, merged
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                writer.finalize()

        return "success", (
            "Krea 2 checkpoint merge complete!\n"
            f"- Checkpoint A: {checkpoint_a} ({weight_a:.1%})\n"
            f"- Checkpoint B: {checkpoint_b} ({weight_b:.1%})\n"
            f"- ConvRot layers merged: {len(modules)}\n"
            f"- Device: {device}\n"
            f"- Output: {output_path}"
        )
    except FileExistsError as exc:
        return "skip", str(exc)
    except Exception as exc:
        return "error", str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge two Krea 2 ConvRot INT8 checkpoints")
    parser.add_argument("--input", required=True, help="JSON payload path")
    parser.add_argument("--output", required=True, help="JSON result path")
    args = parser.parse_args()

    try:
        with open(args.input, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        status, message = merge_krea2_checkpoints(
            checkpoint_a=str(payload.get("checkpoint_a", "")),
            checkpoint_b=str(payload.get("checkpoint_b", "")),
            output_path=str(payload.get("output_path", "")),
            weight_a=float(payload.get("weight_a", 0.7)),
            device_choice=str(payload.get("device_choice", "auto")),
            overwrite=bool(payload.get("overwrite", False)),
        )
    except Exception as exc:
        status, message = "error", str(exc)

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump({"status": status, "message": message}, handle)
    return 0 if status in {"success", "skip"} else 1


if __name__ == "__main__":
    raise SystemExit(main())