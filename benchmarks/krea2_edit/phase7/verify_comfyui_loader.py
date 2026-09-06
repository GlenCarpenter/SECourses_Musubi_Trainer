"""Load a Musubi Krea 2 Edit LoRA through ComfyUI's standard loader path."""

import argparse
from pathlib import Path
import sys

from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[3]
COMFYUI_ROOT = ROOT / "ComfyUI"
sys.path.insert(0, str(COMFYUI_ROOT))

import comfy.lora  # noqa: E402
import comfy.lora_convert  # noqa: E402
import comfy.sd  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_model")
    parser.add_argument("lora")
    args = parser.parse_args()

    model = comfy.sd.load_diffusion_model(args.base_model)
    if model is None:
        raise RuntimeError("ComfyUI did not recognize the base checkpoint as a diffusion model")

    lora = comfy.lora_convert.convert_lora(load_file(args.lora, device="cpu"))
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    patches = comfy.lora.load_lora(lora, key_map)
    patched_model = model.clone()
    attached = set(patched_model.add_patches(patches, 1.0))

    if len(patches) != 264:
        raise RuntimeError(f"expected 264 parsed adapters, got {len(patches)}")
    if attached != set(patches):
        missing = sorted(set(patches).difference(attached))
        raise RuntimeError(f"ComfyUI did not attach {len(missing)} adapters: {missing[:5]}")

    print(
        f"ComfyUI standard loader accepted and attached all {len(attached)} adapters "
        f"to {type(model.model).__name__}; no key conversion required."
    )


if __name__ == "__main__":
    main()