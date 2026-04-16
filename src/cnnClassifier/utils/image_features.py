from pathlib import Path

import numpy as np
from PIL import Image

INFERENCE_PROFILE_IMAGE_SIZE = 128
INFERENCE_FEATURE_NAMES = [
    "grayscale_mean",
    "grayscale_std",
    "grayscale_p10",
    "grayscale_p90",
    "red_mean",
    "green_mean",
    "blue_mean",
    "red_std",
    "green_std",
    "blue_std",
    "edge_horizontal",
    "edge_vertical",
    "dark_pixel_ratio",
    "bright_pixel_ratio",
    "border_dark_ratio",
    "center_mean",
    "center_std",
    "channel_delta",
    "entropy",
]
OOD_BOUND_CONFIG = {
    "grayscale_std": {"lower_pad": 0.01, "upper_pad": 0.02},
    "edge_horizontal": {"lower_pad": 0.002, "upper_pad": 0.003},
    "edge_vertical": {"lower_pad": 0.002, "upper_pad": 0.003},
    "entropy": {"lower_pad": 0.1, "upper_pad": 0.05},
    "channel_delta": {"lower_pad": 0.0, "upper_pad": 0.002},
}


def _load_resized_rgb_image(input_image_path: str | Path, image_size: int) -> np.ndarray:
    with Image.open(input_image_path) as opened_image:
        image_array = np.asarray(
            opened_image.convert("RGB").resize((image_size, image_size)),
            dtype=np.float32,
        ) / 255.0
    return image_array


def extract_image_statistics(
    input_image_path: str | Path,
    image_size: int = INFERENCE_PROFILE_IMAGE_SIZE,
) -> dict[str, float]:
    image_array = _load_resized_rgb_image(input_image_path, image_size)
    grayscale = image_array.mean(axis=2)
    border = np.concatenate([grayscale[0, :], grayscale[-1, :], grayscale[:, 0], grayscale[:, -1]])
    center_slice = slice(image_size // 4, image_size - (image_size // 4))
    center = grayscale[center_slice, center_slice]
    histogram, _ = np.histogram(grayscale, bins=16, range=(0, 1), density=True)
    histogram = histogram / max(histogram.sum(), 1e-8)
    entropy = float(-(histogram * np.log(histogram + 1e-12)).sum())
    channel_delta = float(
        np.abs(image_array[:, :, 0] - image_array[:, :, 1]).mean()
        + np.abs(image_array[:, :, 1] - image_array[:, :, 2]).mean()
        + np.abs(image_array[:, :, 0] - image_array[:, :, 2]).mean()
    )

    return {
        "grayscale_mean": float(grayscale.mean()),
        "grayscale_std": float(grayscale.std()),
        "grayscale_p10": float(np.percentile(grayscale, 10)),
        "grayscale_p90": float(np.percentile(grayscale, 90)),
        "red_mean": float(image_array[:, :, 0].mean()),
        "green_mean": float(image_array[:, :, 1].mean()),
        "blue_mean": float(image_array[:, :, 2].mean()),
        "red_std": float(image_array[:, :, 0].std()),
        "green_std": float(image_array[:, :, 1].std()),
        "blue_std": float(image_array[:, :, 2].std()),
        "edge_horizontal": float(np.abs(np.diff(grayscale, axis=1)).mean()),
        "edge_vertical": float(np.abs(np.diff(grayscale, axis=0)).mean()),
        "dark_pixel_ratio": float((grayscale < 0.1).mean()),
        "bright_pixel_ratio": float((grayscale > 0.9).mean()),
        "border_dark_ratio": float((border < 0.1).mean()),
        "center_mean": float(center.mean()),
        "center_std": float(center.std()),
        "channel_delta": channel_delta,
        "entropy": entropy,
    }


def feature_vector(
    feature_dict: dict[str, float],
    feature_names: list[str] | tuple[str, ...] = INFERENCE_FEATURE_NAMES,
) -> np.ndarray:
    return np.array([feature_dict[name] for name in feature_names], dtype=np.float32)


def build_ood_profile(feature_dicts: list[dict[str, float]]) -> dict:
    feature_matrix = np.vstack([feature_vector(features) for features in feature_dicts])
    mean_vector = feature_matrix.mean(axis=0)
    std_vector = feature_matrix.std(axis=0) + 1e-8
    anomaly_scores = np.sqrt((((feature_matrix - mean_vector) / std_vector) ** 2).sum(axis=1))
    score_threshold = float(np.percentile(anomaly_scores, 99.5) + 0.25)

    bounds = {}
    for feature_name, padding in OOD_BOUND_CONFIG.items():
        feature_values = np.array([features[feature_name] for features in feature_dicts], dtype=np.float32)
        lower_bound = max(0.0, float(feature_values.min()) - padding["lower_pad"])
        upper_bound = float(feature_values.max()) + padding["upper_pad"]
        bounds[feature_name] = {"min": lower_bound, "max": upper_bound}

    return {
        "feature_names": INFERENCE_FEATURE_NAMES,
        "feature_mean": mean_vector.tolist(),
        "feature_std": std_vector.tolist(),
        "score_threshold": score_threshold,
        "feature_bounds": bounds,
    }


def evaluate_ood_profile(feature_dict: dict[str, float], ood_profile: dict) -> tuple[bool, float, list[str]]:
    feature_names = ood_profile.get("feature_names", INFERENCE_FEATURE_NAMES)
    feature_values = feature_vector(feature_dict, feature_names)
    mean_vector = np.asarray(ood_profile["feature_mean"], dtype=np.float32)
    std_vector = np.asarray(ood_profile["feature_std"], dtype=np.float32)
    anomaly_score = float(np.sqrt((((feature_values - mean_vector) / std_vector) ** 2).sum()))

    violations: list[str] = []
    for feature_name, feature_bounds in ood_profile.get("feature_bounds", {}).items():
        feature_value = float(feature_dict[feature_name])
        if feature_value < feature_bounds["min"] or feature_value > feature_bounds["max"]:
            violations.append(
                f"{feature_name}={feature_value:.4f} outside [{feature_bounds['min']:.4f}, {feature_bounds['max']:.4f}]"
            )

    if anomaly_score > float(ood_profile.get("score_threshold", float("inf"))):
        violations.append(
            f"anomaly_score={anomaly_score:.4f} above threshold={ood_profile['score_threshold']:.4f}"
        )

    return len(violations) == 0, anomaly_score, violations
