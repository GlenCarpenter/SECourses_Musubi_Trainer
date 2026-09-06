"""Run the Phase 7 fixtures through ComfyUI and comfyui-krea2edit v1.2.5."""

import argparse
import asyncio
import json
import uuid

import aiohttp


FIXTURES = {
    "identity": {
        "image": "phase7_identity.png",
        "prompt": "Reproduce the reference image exactly, preserving every shape, color, and registration mark.",
    },
    "outpaint": {
        "image": "phase7_outpaint.png",
        "prompt": "Extend the reference vertically, keeping the center strip unchanged and aligned.",
    },
    "two_reference": {
        "image": "phase7_two_reference_a.png",
        "image_b": "phase7_two_reference_b.png",
        "prompt": (
            "Place the red-circle subject from the first reference on the left and the "
            "blue-triangle subject from the second reference on the right."
        ),
    },
}


def _workflow(fixture_name: str, seed: int, strength: float, label: str, lora_name: str) -> dict:
    fixture = FIXTURES[fixture_name]
    workflow = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "Krea_2_Raw_Base.safetensors", "weight_dtype": "default"}},
        "2": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["1", 0],
                "lora_name": lora_name,
                "strength_model": strength,
            },
        },
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_4b_bf16.safetensors", "type": "krea2"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "5": {"class_type": "LoadImage", "inputs": {"image": fixture["image"]}},
        "6": {"class_type": "VAEEncode", "inputs": {"pixels": ["5", 0], "vae": ["4", 0]}},
        "7": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 256, "height": 256, "batch_size": 1}},
        "8": {
            "class_type": "Krea2EditGroundedEncode",
            "inputs": {"clip": ["3", 0], "prompt": fixture["prompt"], "image": ["5", 0], "grounding_px": 384},
        },
        "9": {
            "class_type": "Krea2EditGroundedEncode",
            "inputs": {"clip": ["3", 0], "prompt": "", "image": ["5", 0], "grounding_px": 384},
        },
        "10": {
            "class_type": "Krea2EditModelPatch",
            "inputs": {
                "model": ["2", 0],
                "source_latent": ["6", 0],
                "ref_boost": 1.0,
                "ref_boost_a": 1.0,
                "fit_mode": "fit",
                "vae": ["4", 0],
                "source_image": ["5", 0],
                "target_latent": ["7", 0],
            },
        },
        "11": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["10", 0],
                "seed": seed,
                "steps": 40,
                "cfg": 3.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "positive": ["8", 0],
                "negative": ["9", 0],
                "latent_image": ["7", 0],
                "denoise": 1.0,
            },
        },
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["4", 0]}},
        "13": {
            "class_type": "SaveImage",
            "inputs": {"images": ["12", 0], "filename_prefix": f"phase7/{label}_{fixture_name}"},
        },
    }

    if "image_b" in fixture:
        workflow["14"] = {"class_type": "LoadImage", "inputs": {"image": fixture["image_b"]}}
        workflow["15"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["14", 0], "vae": ["4", 0]}}
        workflow["8"]["inputs"]["image_b"] = ["14", 0]
        workflow["9"]["inputs"]["image_b"] = ["14", 0]
        workflow["10"]["inputs"]["source_latent_b"] = ["15", 0]
        workflow["10"]["inputs"]["source_image_b"] = ["14", 0]

    return workflow


async def _run(
    server: str,
    fixture_name: str,
    seed: int,
    strength: float,
    label: str,
    lora_name: str,
) -> list[str]:
    client_id = uuid.uuid4().hex
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(f"ws://{server}/ws?clientId={client_id}") as socket:
            async with session.post(
                f"http://{server}/prompt",
                json={"prompt": _workflow(fixture_name, seed, strength, label, lora_name), "client_id": client_id},
            ) as response:
                response.raise_for_status()
                prompt_id = (await response.json())["prompt_id"]

            async for message in socket:
                if message.type != aiohttp.WSMsgType.TEXT:
                    continue
                event = json.loads(message.data)
                if event.get("type") == "execution_error" and event.get("data", {}).get("prompt_id") == prompt_id:
                    raise RuntimeError(json.dumps(event["data"], indent=2))
                if event.get("type") == "execution_success" and event.get("data", {}).get("prompt_id") == prompt_id:
                    break

        async with session.get(f"http://{server}/history/{prompt_id}") as response:
            response.raise_for_status()
            history = (await response.json())[prompt_id]
    return [image["filename"] for image in history["outputs"]["13"]["images"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="127.0.0.1:8188")
    parser.add_argument("--fixture", choices=["all", *FIXTURES], default="all")
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--label", default="lora")
    parser.add_argument("--lora-name", default="krea2_edit_phase7_smoke.safetensors")
    args = parser.parse_args()

    fixture_names = FIXTURES if args.fixture == "all" else [args.fixture]
    for index, fixture_name in enumerate(fixture_names):
        outputs = asyncio.run(
            _run(args.server, fixture_name, args.seed + index, args.strength, args.label, args.lora_name)
        )
        print(f"{fixture_name}: {', '.join(outputs)}")


if __name__ == "__main__":
    main()