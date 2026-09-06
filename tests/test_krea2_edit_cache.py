import os
import sys
from argparse import Namespace

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "musubi-tuner", "src")))

from musubi_tuner.dataset.architectures import ARCHITECTURE_KREA2_EDIT
from musubi_tuner.dataset.cache_io import load_krea2_edit_latent_cache
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ImageDataset, ItemInfo
from musubi_tuner.krea2_edit_cache_latents import encode_and_save_batch


class FakeVae:
    device = torch.device("cpu")
    dtype = torch.float32

    def encode_pixels_to_latents(self, pixels):
        return pixels[:, :1, :, ::8, ::8].contiguous()


def _write_rgb(path, size):
    Image.new("RGB", size, color=(64, 128, 192)).save(path)


def _dataset_kwargs(tmp_path, **overrides):
    values = {
        "resolution": (128, 128),
        "caption_extension": ".txt",
        "batch_size": 1,
        "num_repeats": 1,
        "enable_bucket": False,
        "bucket_no_upscale": False,
        "image_directory": str(tmp_path / "targets"),
        "cache_directory": str(tmp_path / "cache"),
        "reference_directories": [str(tmp_path / "sources")],
        "architecture": ARCHITECTURE_KREA2_EDIT,
    }
    values.update(overrides)
    return values


def test_krea2_edit_blueprint_ignores_runtime_reference_bookkeeping(tmp_path):
    sources = str(tmp_path / "sources")
    user_config = {
        "general": {"resolution": [128, 128], "batch_size": 1},
        "datasets": [
            {
                "image_directory": str(tmp_path / "targets"),
                "reference_directories": [sources],
            }
        ],
    }
    runtime_args = Namespace(
        reference_directory=sources,
        reference_directories=[sources],
        reference_directory_2="",
    )

    blueprint = BlueprintGenerator(ConfigSanitizer()).generate(
        user_config,
        runtime_args,
        architecture=ARCHITECTURE_KREA2_EDIT,
    )
    params = blueprint.dataset_group.datasets[0].params

    assert params.reference_directory is None
    assert params.reference_directories == [sources]


def test_krea2_edit_directory_dataset_resolves_ordered_stem_matches(tmp_path):
    targets = tmp_path / "targets"
    sources = tmp_path / "sources"
    sources_b = tmp_path / "sources_b"
    for directory in (targets, sources, sources_b):
        directory.mkdir()
    _write_rgb(targets / "0001.png", (128, 128))
    (targets / "0001.txt").write_text("place the subject on a beach", encoding="utf-8")
    _write_rgb(sources / "0001.jpg", (200, 100))
    _write_rgb(sources_b / "0001.webp", (80, 120))

    dataset = ImageDataset(
        **_dataset_kwargs(tmp_path, reference_directories=[str(sources), str(sources_b)])
    )
    _key, _images, _caption, controls, _mask = dataset.datasource.get_image_data(0)

    assert len(controls) == 2
    assert controls[0].size == (200, 100)
    assert controls[1].size == (80, 120)


def test_krea2_edit_directory_dataset_rejects_missing_pair(tmp_path):
    targets = tmp_path / "targets"
    sources = tmp_path / "sources"
    targets.mkdir()
    sources.mkdir()
    _write_rgb(targets / "0001.png", (128, 128))
    (targets / "0001.txt").write_text("edit instruction", encoding="utf-8")

    with pytest.raises(ValueError, match="without a stem-matched reference"):
        ImageDataset(**_dataset_kwargs(tmp_path))


def test_krea2_edit_directory_dataset_rejects_duplicate_stem(tmp_path):
    targets = tmp_path / "targets"
    sources = tmp_path / "sources"
    targets.mkdir()
    sources.mkdir()
    _write_rgb(targets / "0001.png", (128, 128))
    (targets / "0001.txt").write_text("edit instruction", encoding="utf-8")
    _write_rgb(sources / "0001.png", (128, 128))
    _write_rgb(sources / "0001.jpg", (128, 128))

    with pytest.raises(ValueError, match="duplicate stem"):
        ImageDataset(**_dataset_kwargs(tmp_path))


def test_krea2_edit_directory_dataset_rejects_more_than_two_references(tmp_path):
    directories = []
    for name in ("sources", "sources_b", "sources_c"):
        directory = tmp_path / name
        directory.mkdir()
        directories.append(str(directory))
    (tmp_path / "targets").mkdir()

    with pytest.raises(ValueError, match="one or two reference directories"):
        ImageDataset(**_dataset_kwargs(tmp_path, reference_directories=directories))


def test_krea2_edit_dataset_rejects_unsafe_batch_size(tmp_path):
    with pytest.raises(ValueError, match="batch_size=1"):
        ImageDataset(**_dataset_kwargs(tmp_path, batch_size=2))


def test_krea2_edit_cache_round_trip_with_fake_vae(tmp_path):
    cache_path = tmp_path / "cache" / "0001_0128x0128_kr2e.safetensors"
    item = ItemInfo(
        "0001.png",
        "place the subject on a beach",
        (128, 128),
        content=np.full((128, 128, 3), 127, dtype=np.uint8),
        latent_cache_path=str(cache_path),
    )
    item.control_content = [
        np.full((100, 200, 3), 64, dtype=np.uint8),
        np.full((120, 80, 3), 192, dtype=np.uint8),
    ]

    encode_and_save_batch(FakeVae(), [item])
    target, references, metadata = load_krea2_edit_latent_cache(str(cache_path))

    assert target.shape == (1, 1, 16, 16)
    assert [tuple(reference.shape) for reference in references] == [(1, 1, 8, 16), (1, 1, 16, 10)]
    assert metadata["architecture"] == "krea2_edit"
    assert metadata["krea2_edit_cache_schema"] == "1"
    assert metadata["fit_protocol_version"] == "1.0.0"
    assert metadata["reference_count"] == "2"
    assert metadata["target_grid_height"] == "8"
    assert metadata["reference_0_offset_height"] == "2.0"
    assert metadata["reference_1_offset_width"] == "1.5"