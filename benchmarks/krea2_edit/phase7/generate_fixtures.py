"""Generate deterministic Krea 2 Edit parity fixtures and their dataset TOML."""

from pathlib import Path

from PIL import Image, ImageDraw
import toml


ROOT = Path(__file__).resolve().parent
IMAGE_SIZE = 256


def _canvas(width: int = IMAGE_SIZE, height: int = IMAGE_SIZE) -> Image.Image:
    return Image.new("RGB", (width, height), "#f4f0e8")


def _draw_registration_marks(image: Image.Image) -> None:
    draw = ImageDraw.Draw(image)
    draw.line((IMAGE_SIZE // 2, 0, IMAGE_SIZE // 2, IMAGE_SIZE - 1), fill="#202020", width=2)
    draw.line((0, IMAGE_SIZE // 2, IMAGE_SIZE - 1, IMAGE_SIZE // 2), fill="#202020", width=2)
    for x, y in ((8, 8), (232, 8), (8, 232), (232, 232)):
        draw.rectangle((x, y, x + 15, y + 15), outline="#202020", width=3)


def _identity_fixture() -> tuple[Image.Image, list[Image.Image], str]:
    image = _canvas()
    draw = ImageDraw.Draw(image)
    _draw_registration_marks(image)
    draw.rectangle((36, 44, 104, 112), fill="#e84a3c", outline="#202020", width=4)
    draw.ellipse((145, 55, 217, 127), fill="#19a974", outline="#202020", width=4)
    draw.polygon(((74, 157), (112, 226), (36, 226)), fill="#f2c94c", outline="#202020")
    draw.rounded_rectangle((151, 164, 224, 222), radius=8, fill="#3273dc", outline="#202020", width=4)
    return image, [image.copy()], "Reproduce the reference image exactly, preserving every shape, color, and registration mark."


def _outpaint_fixture() -> tuple[Image.Image, list[Image.Image], str]:
    target = _canvas()
    draw = ImageDraw.Draw(target)
    draw.rectangle((0, 0, 255, 63), fill="#9fd8f2")
    draw.rectangle((0, 192, 255, 255), fill="#315b38")
    draw.rectangle((0, 64, 255, 191), fill="#f4f0e8")
    draw.line((0, 64, 255, 64), fill="#202020", width=3)
    draw.line((0, 191, 255, 191), fill="#202020", width=3)
    draw.rectangle((23, 91, 73, 164), fill="#e84a3c", outline="#202020", width=4)
    draw.ellipse((102, 88, 166, 152), fill="#f2c94c", outline="#202020", width=4)
    draw.polygon(((211, 88), (239, 167), (183, 167)), fill="#3273dc", outline="#202020")
    reference = target.crop((0, 64, 256, 192))
    return target, [reference], "Extend the reference vertically, keeping the center strip unchanged and aligned."


def _two_reference_fixture() -> tuple[Image.Image, list[Image.Image], str]:
    first = _canvas(128, 256)
    first_draw = ImageDraw.Draw(first)
    first_draw.rectangle((0, 0, 127, 255), outline="#202020", width=4)
    first_draw.ellipse((23, 55, 105, 137), fill="#e84a3c", outline="#202020", width=5)
    first_draw.rectangle((50, 154, 78, 218), fill="#f2c94c", outline="#202020", width=4)

    second = _canvas(128, 256)
    second_draw = ImageDraw.Draw(second)
    second_draw.rectangle((0, 0, 127, 255), outline="#202020", width=4)
    second_draw.polygon(((64, 48), (108, 133), (20, 133)), fill="#3273dc", outline="#202020")
    second_draw.rounded_rectangle((31, 155, 97, 218), radius=10, fill="#19a974", outline="#202020", width=4)

    target = _canvas()
    target.paste(first, (0, 0))
    target.paste(second, (128, 0))
    ImageDraw.Draw(target).line((127, 0, 127, 255), fill="#202020", width=2)
    return target, [first, second], "Place the red-circle subject from the first reference on the left and the blue-triangle subject from the second reference on the right."


def _write_fixture(name: str, fixture: tuple[Image.Image, list[Image.Image], str]) -> dict:
    target, references, instruction = fixture
    target_dir = ROOT / "data" / name / "target"
    reference_dirs = [ROOT / "data" / name / f"reference_{index}" for index in range(1, len(references) + 1)]
    target_dir.mkdir(parents=True, exist_ok=True)
    for directory in reference_dirs:
        directory.mkdir(parents=True, exist_ok=True)

    target.save(target_dir / "fixture.png")
    (target_dir / "fixture.txt").write_text(instruction, encoding="utf-8")
    for reference, directory in zip(references, reference_dirs):
        reference.save(directory / "fixture.png")

    return {
        "image_directory": str(target_dir),
        "reference_directories": [str(directory) for directory in reference_dirs],
        "cache_directory": str(ROOT / "cache" / name),
        "num_repeats": 20,
    }


def main() -> None:
    datasets = [
        _write_fixture("identity", _identity_fixture()),
        _write_fixture("outpaint", _outpaint_fixture()),
        _write_fixture("two_reference", _two_reference_fixture()),
    ]
    config = {
        "general": {
            "resolution": [IMAGE_SIZE, IMAGE_SIZE],
            "caption_extension": ".txt",
            "batch_size": 1,
            "enable_bucket": False,
            "bucket_no_upscale": False,
        },
        "datasets": datasets,
    }
    with (ROOT / "dataset.toml").open("w", encoding="utf-8") as file:
        toml.dump(config, file)

    model_dir = ROOT.parents[3] / "Training_Models_Krea_2"
    smoke_training_config = {
        "dataset_config": str(ROOT / "dataset.toml"),
        "dit": str(model_dir / "Krea_2_Raw_Base.safetensors"),
        "mixed_precision": "bf16",
        "fp8_base": True,
        "fp8_scaled": True,
        "sdpa": True,
        "gradient_checkpointing": True,
        "blocks_to_swap": 0,
        "network_module": "networks.lora_krea2",
        "network_dim": 4,
        "network_alpha": 4,
        "network_args": [],
        "optimizer_type": "AdaFactor",
        "optimizer_args": [
            "scale_parameter=False",
            "relative_step=False",
            "warmup_init=False",
            "weight_decay=0.0",
        ],
        "learning_rate": 1e-3,
        "lr_scheduler": "constant",
        "max_train_steps": 120,
        "max_data_loader_n_workers": 1,
        "persistent_data_loader_workers": True,
        "seed": 42,
        "save_precision": "bf16",
        "save_every_n_steps": 40,
        "output_dir": str(ROOT / "output"),
        "output_name": "krea2_edit_phase7_smoke",
        "metadata_description": "Krea 2 Edit Phase 7 pipeline smoke; not a visual-parity checkpoint",
        "timestep_sampling": "krea2_shift",
        "weighting_scheme": "none",
        "discrete_flow_shift": 2.5,
        "sigmoid_scale": 1.0,
        "min_timestep": 0,
        "max_timestep": 1000,
        "max_grad_norm": 0.0,
        "gradient_accumulation_steps": 1,
    }
    with (ROOT / "smoke_training.toml").open("w", encoding="utf-8") as file:
        toml.dump(smoke_training_config, file)
    print(f"Generated {len(datasets)} fixtures, dataset.toml, and smoke_training.toml in {ROOT}")


if __name__ == "__main__":
    main()