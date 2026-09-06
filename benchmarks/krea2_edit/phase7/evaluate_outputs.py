"""Evaluate and persist matched-seed Phase 7 ComfyUI outputs."""

import json
import math
from pathlib import Path
import shutil

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
COMFY_OUTPUT = ROOT.parents[2] / "ComfyUI" / "output" / "phase7"
RESULTS = ROOT / "results"
COLORS = {
    "red": lambda image: (image[..., 0] > 0.65) & (image[..., 1] < 0.5) & (image[..., 2] < 0.5),
    "blue": lambda image: (image[..., 2] > 0.55) & (image[..., 0] < 0.5),
    "green": lambda image: (image[..., 1] > 0.45) & (image[..., 0] < 0.45) & (image[..., 2] < 0.6),
    "yellow": lambda image: (image[..., 0] > 0.65) & (image[..., 1] > 0.5) & (image[..., 2] < 0.45),
}


def _pixels(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _centroids(image: np.ndarray) -> dict[str, tuple[float, float]]:
    result = {}
    for name, selector in COLORS.items():
        rows, columns = np.where(selector(image))
        if len(columns):
            result[name] = (float(columns.mean()), float(rows.mean()))
    return result


def _metrics(target: np.ndarray, output: np.ndarray) -> dict[str, float]:
    error = output - target
    mse = float(np.square(error).mean())
    return {"mae": float(np.abs(error).mean()), "psnr": float(-10 * np.log10(mse))}


def _mean_centroid_error(target: np.ndarray, output: np.ndarray) -> float:
    target_centroids = _centroids(target)
    output_centroids = _centroids(output)
    shared = target_centroids.keys() & output_centroids.keys()
    if shared != target_centroids.keys():
        raise RuntimeError(f"output is missing color landmarks: {sorted(target_centroids.keys() - shared)}")
    return float(
        np.mean(
            [
                math.dist(target_centroids[name], output_centroids[name])
                for name in shared
            ]
        )
    )


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    report = {}
    images = {}
    for fixture in ("identity", "outpaint", "two_reference"):
        target = _pixels(ROOT / "data" / fixture / "target" / "fixture.png")
        report[fixture] = {}
        images[fixture] = {"target": target}
        for label in ("baseline", "lora"):
            source = COMFY_OUTPUT / f"{label}_{fixture}_00001_.png"
            destination = RESULTS / source.name
            shutil.copy2(source, destination)
            output = _pixels(destination)
            images[fixture][label] = output
            report[fixture][label] = _metrics(target, output)

    for label in ("baseline", "lora"):
        report["outpaint"][label]["center_mae"] = float(
            np.abs(images["outpaint"][label][64:192] - images["outpaint"]["target"][64:192]).mean()
        )
        report["two_reference"][label]["mean_color_centroid_error"] = _mean_centroid_error(
            images["two_reference"]["target"], images["two_reference"][label]
        )

    two_reference_centroids = _centroids(images["two_reference"]["lora"])
    report["two_reference"]["lora"]["ordered_left_to_right"] = bool(
        two_reference_centroids["red"][0] < 128
        and two_reference_centroids["yellow"][0] < 128
        and two_reference_centroids["blue"][0] > 128
        and two_reference_centroids["green"][0] > 128
    )

    smoke_observations = {
        "identity_improves_mae": report["identity"]["lora"]["mae"] < report["identity"]["baseline"]["mae"],
        "identity_mae_below_0_03": report["identity"]["lora"]["mae"] < 0.03,
        "outpaint_improves_center_mae": (
            report["outpaint"]["lora"]["center_mae"] < report["outpaint"]["baseline"]["center_mae"]
        ),
        "outpaint_center_mae_below_0_05": report["outpaint"]["lora"]["center_mae"] < 0.05,
        "two_reference_improves_centroid_error": (
            report["two_reference"]["lora"]["mean_color_centroid_error"]
            < report["two_reference"]["baseline"]["mean_color_centroid_error"]
        ),
        "two_reference_preserves_order": report["two_reference"]["lora"]["ordered_left_to_right"],
    }
    report["smoke_observations"] = smoke_observations
    (RESULTS / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()