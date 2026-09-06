"""Compatibility checks for Krea 2 Edit LoRAs loaded by standard ComfyUI."""

import argparse
from dataclasses import dataclass
import math
from os import PathLike
from typing import Mapping, Protocol

import torch
from safetensors import safe_open


_DIFFUSION_MODEL_PREFIX = "diffusion_model."
_LORA_PREFIX = "lora_unet_"
_LORA_SUFFIXES = (".lora_down.weight", ".lora_up.weight", ".alpha")


class _TensorShape(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def ndim(self) -> int: ...

    def numel(self) -> int: ...


@dataclass(frozen=True)
class _SafetensorShape:
    shape: tuple[int, ...]

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def numel(self) -> int:
        return math.prod(self.shape)


@dataclass(frozen=True)
class ComfyUILoRACompatibilityReport:
    adapter_count: int
    tensor_count: int
    mapped_modules: tuple[str, ...]


def _native_weight_name(key: str) -> str:
    if key.startswith(_DIFFUSION_MODEL_PREFIX):
        return key[len(_DIFFUSION_MODEL_PREFIX) :]
    return key


def _comfyui_aliases(base_state_dict: Mapping[str, _TensorShape]) -> dict[str, tuple[str, _TensorShape]]:
    aliases: dict[str, tuple[str, _TensorShape]] = {}
    for key, weight in base_state_dict.items():
        native_key = _native_weight_name(key)
        if not native_key.endswith(".weight"):
            continue
        module_name = native_key[: -len(".weight")]
        alias = f"{_LORA_PREFIX}{module_name.replace('.', '_')}"
        if alias in aliases and aliases[alias][0] != module_name:
            raise ValueError(
                f"ComfyUI alias {alias!r} is ambiguous for {aliases[alias][0]!r} and {module_name!r}"
            )
        aliases[alias] = (module_name, weight)
    return aliases


def validate_comfyui_lora_state_dict(
    lora_state_dict: Mapping[str, _TensorShape],
    base_state_dict: Mapping[str, _TensorShape],
) -> ComfyUILoRACompatibilityReport:
    """Validate Musubi LoRA tensors against ComfyUI's standard ``lora_unet`` aliases."""
    aliases = _comfyui_aliases(base_state_dict)
    adapter_tensors: dict[str, dict[str, _TensorShape]] = {}

    for key, tensor in lora_state_dict.items():
        suffix = next((candidate for candidate in _LORA_SUFFIXES if key.endswith(candidate)), None)
        if suffix is None:
            raise ValueError(f"unsupported LoRA tensor key {key!r}")
        adapter_name = key[: -len(suffix)]
        adapter_tensors.setdefault(adapter_name, {})[suffix] = tensor

    mapped_modules = []
    required_suffixes = set(_LORA_SUFFIXES)
    for adapter_name, tensors in adapter_tensors.items():
        missing = required_suffixes.difference(tensors)
        if missing:
            raise ValueError(f"adapter {adapter_name!r} is missing {', '.join(sorted(missing))}")
        if adapter_name not in aliases:
            raise ValueError(f"adapter {adapter_name!r} does not map to a Krea 2 base-model weight in ComfyUI")

        module_name, base_weight = aliases[adapter_name]
        down_weight = tensors[".lora_down.weight"]
        up_weight = tensors[".lora_up.weight"]
        alpha = tensors[".alpha"]
        if base_weight.ndim != 2 or down_weight.ndim != 2 or up_weight.ndim != 2:
            raise ValueError(f"adapter {adapter_name!r} and its base weight must have two-dimensional shapes")
        rank = down_weight.shape[0]
        expected_down_shape = (rank, base_weight.shape[1])
        expected_up_shape = (base_weight.shape[0], rank)
        if tuple(down_weight.shape) != expected_down_shape or tuple(up_weight.shape) != expected_up_shape:
            raise ValueError(
                f"adapter {adapter_name!r} shape mismatch: down={tuple(down_weight.shape)}, "
                f"up={tuple(up_weight.shape)}, base={tuple(base_weight.shape)}"
            )
        if alpha.numel() != 1:
            raise ValueError(f"adapter {adapter_name!r} alpha must be scalar, got shape {tuple(alpha.shape)}")
        mapped_modules.append(module_name)

    if not mapped_modules:
        raise ValueError("LoRA state dict contains no adapters")

    return ComfyUILoRACompatibilityReport(
        adapter_count=len(mapped_modules),
        tensor_count=len(lora_state_dict),
        mapped_modules=tuple(sorted(mapped_modules)),
    )


def _read_safetensor_shapes(path: str | PathLike[str]) -> dict[str, _SafetensorShape]:
    with safe_open(path, framework="pt", device="cpu") as reader:
        return {key: _SafetensorShape(tuple(reader.get_slice(key).get_shape())) for key in reader.keys()}


def validate_comfyui_lora_files(
    lora_path: str | PathLike[str],
    base_model_path: str | PathLike[str],
) -> ComfyUILoRACompatibilityReport:
    """Validate safetensors files without loading their tensor payloads."""
    return validate_comfyui_lora_state_dict(
        _read_safetensor_shapes(lora_path),
        _read_safetensor_shapes(base_model_path),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a Krea 2 LoRA for ComfyUI's standard LoRA loader.")
    parser.add_argument("lora", help="Musubi Krea 2 LoRA safetensors file")
    parser.add_argument("base_model", help="Krea 2 diffusion-model safetensors file")
    args = parser.parse_args()

    report = validate_comfyui_lora_files(args.lora, args.base_model)
    print(
        f"ComfyUI-compatible: {report.adapter_count} adapters, "
        f"{report.tensor_count} tensors, no key conversion required."
    )


if __name__ == "__main__":
    main()