"""Private, streamed map artwork cache and tiled viewer helpers."""
from __future__ import annotations

import json
import math
import re
import shutil
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path
from typing import Any

MAP_ATTRIBUTION = "Mapa: The Embraced One — hallownest.net · Hollow Knight © Team Cherry"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def estimate_tile_bytes(width: int, height: int) -> int:
    """Conservative estimate for a Q85 JPEG pyramid plus a small preview."""
    if width < 1 or height < 1:
        raise ValueError("map dimensions must be positive")
    return math.ceil(width * height * 1.7)


def load_pyvips() -> Any | None:
    try:
        import pyvips  # type: ignore[import-untyped]
    except (ImportError, OSError):
        return None
    return pyvips


def generate_map_tiles(
    source: Path,
    cache_dir: Path,
    *,
    progress: Callable[[float], None] | None = None,
) -> dict[str, Any] | None:
    """Create/restart a Deep Zoom pyramid without decoding the full image in Python.

    Existing tile files are kept until libvips atomically overwrites them, so a
    cancelled job is restartable and never loses already downloaded source data.
    """
    pyvips = load_pyvips()
    if pyvips is None:
        return None
    width, height = _png_dimensions(source)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dzi_path = cache_dir / "tiles" / "map.dzi"
    tile_dir = cache_dir / "tiles" / "map_files"
    metadata_path = cache_dir / "tiles.json"
    if metadata_path.is_file() and dzi_path.is_file():
        try:
            ready = json.loads(metadata_path.read_text("utf-8"))
            if isinstance(ready, dict) and ready.get("width") == width and ready.get("height") == height and ready.get("complete") is True:
                return dict(ready)
        except (OSError, json.JSONDecodeError):
            pass
    tile_dir.mkdir(parents=True, exist_ok=True)
    if progress:
        progress(0.0)
    image = pyvips.Image.new_from_file(str(source), access="sequential", fail=True).autorot()
    image.set_progress(True)

    def update(_image: Any, pointer: Any) -> None:
        if not progress:
            return
        try:
            state = pyvips.ffi.cast("VipsProgress *", pointer)
            progress(min(99.0, max(0.0, float(state.percent))))
        except (AttributeError, TypeError, ValueError):
            return

    image.signal_connect("eval", update)
    image.dzsave(str(dzi_path.with_suffix("")), layout="dz", tile_size=256, overlap=0,
                 suffix=".jpg", Q=85, depth="onepixel")
    thumb_source = pyvips.Image.new_from_file(str(source), access="sequential", fail=True).autorot()
    thumb = thumb_source.thumbnail_image(1800, height=1100, size="down")
    thumb.write_to_file(str(cache_dir / "thumbnail.jpg"), Q=85, optimize_coding=True)
    tree = ET.parse(dzi_path)
    root = tree.getroot()
    image_element = root.find("{*}Size")
    if image_element is None:
        raise ValueError("libvips generated Deep Zoom without image dimensions")
    out_width, out_height = int(image_element.attrib["Width"]), int(image_element.attrib["Height"])
    if (out_width, out_height) != (width, height):
        raise ValueError("Deep Zoom dimensions differ from the validated source PNG")
    max_level = math.ceil(math.log2(max(width, height)))
    result = {"complete": True, "width": width, "height": height, "tile_size": 256,
              "overlap": 0, "max_level": max_level, "format": "jpg",
              "tile_path": "tiles/map_files", "dzi_path": "tiles/map.dzi",
              "thumbnail": "thumbnail.jpg", "estimated_tile_bytes": estimate_tile_bytes(width, height)}
    metadata_path.write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8", newline="\n")
    if progress:
        progress(100.0)
    return result


def _png_dimensions(path: Path) -> tuple[int, int]:
    from hk_agent.run_journey import validate_png_header

    with path.open("rb") as stream:
        return validate_png_header(stream)


def read_map_tile(cache_dir: Path, level: int, column: int, row: int) -> bytes:
    """Read one generated tile after validating all components and its cache root."""
    if min(level, column, row) < 0 or max(level, column, row) > 100_000:
        raise ValueError("tile coordinates are outside the permitted range")
    metadata = json.loads((cache_dir / "tiles.json").read_text("utf-8"))
    if metadata.get("complete") is not True or level > int(metadata["max_level"]):
        raise ValueError("map tile pyramid is not ready")
    tile_root = (cache_dir / "tiles" / "map_files").resolve(strict=True)
    tile = (tile_root / str(level) / f"{column}_{row}.jpg").resolve(strict=True)
    try:
        tile.relative_to(tile_root)
    except ValueError as error:
        raise ValueError("tile path escaped its cache directory") from error
    if not tile.is_file() or tile.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("map tile is missing or exceeds the size limit")
    return tile.read_bytes()


def validate_map_hash(digest: str) -> str:
    if not _SHA256.fullmatch(digest):
        raise ValueError("invalid map SHA-256")
    return digest


def map_cache_directory(cache_root: Path, digest: str) -> Path:
    return cache_root / validate_map_hash(digest)


def free_disk_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def template_match_ncc(map_gray: Any, scene_gray: Any, *, scales: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)) -> dict[str, float] | None:
    """Small-image normalized cross-correlation using NumPy FFTs only.

    Inputs must already be reduced grayscale arrays. Returns the best map pixel
    rectangle and a [0,1] confidence; callers must treat low scores as unplaced.
    """
    import numpy as np

    image = np.asarray(map_gray, dtype=np.float32)
    template = np.asarray(scene_gray, dtype=np.float32)
    if image.ndim != 2 or template.ndim != 2 or min(image.shape) < 2 or min(template.shape) < 2:
        raise ValueError("template matching requires non-empty grayscale images")
    best: tuple[float, int, int, int, int] | None = None
    for scale in scales:
        tw, th = max(2, round(template.shape[1] * scale)), max(2, round(template.shape[0] * scale))
        if th > image.shape[0] or tw > image.shape[1]:
            continue
        ys = np.minimum((np.arange(th) * template.shape[0] / th).astype(int), template.shape[0] - 1)
        xs = np.minimum((np.arange(tw) * template.shape[1] / tw).astype(int), template.shape[1] - 1)
        sample = template[ys[:, None], xs[None, :]]
        centered = sample - float(sample.mean())
        template_energy = float(np.sum(centered * centered))
        if template_energy <= 1e-8:
            continue
        fft_shape = (image.shape[0] + th - 1, image.shape[1] + tw - 1)
        correlation = np.fft.irfft2(
            np.fft.rfft2(image, s=fft_shape)
            * np.fft.rfft2(centered[::-1, ::-1], s=fft_shape),
            s=fft_shape,
        )[th - 1:image.shape[0], tw - 1:image.shape[1]]
        integral = np.pad(image, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        square = np.pad(image * image, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        patch_sum = integral[th:, tw:] - integral[:-th, tw:] - integral[th:, :-tw] + integral[:-th, :-tw]
        patch_sq = square[th:, tw:] - square[:-th, tw:] - square[th:, :-tw] + square[:-th, :-tw]
        patch_energy = np.maximum(patch_sq - patch_sum * patch_sum / (tw * th), 1e-8)
        scores = correlation / np.sqrt(patch_energy * template_energy)
        index = int(np.argmax(scores))
        y_raw, x_raw = np.unravel_index(index, scores.shape)
        y, x = int(y_raw), int(x_raw)
        score = float(scores[y, x])
        if best is None or score > best[0]:
            best = (score, int(x), int(y), tw, th)
    if best is None:
        return None
    score, x, y, width, height = best
    return {"x": float(x), "y": float(y), "width": float(width), "height": float(height),
            "confidence": min(1.0, max(0.0, (score + 1.0) / 2.0))}


def _small_gray(image: Any, max_width: int, max_height: int) -> Any:
    pyvips = load_pyvips()
    if pyvips is None:
        raise RuntimeError("pyvips is required for automatic scene matching")
    reduced = image.thumbnail_image(max_width, height=max_height, size="down")
    if reduced.bands > 1:
        reduced = reduced.colourspace("b-w")
    import numpy as np

    return np.frombuffer(reduced.write_to_memory(), dtype=np.uint8).reshape(reduced.height, reduced.width).copy()


def auto_register_capture(
    map_path: Path,
    capture_path: Path,
    cache_dir: Path,
    *,
    scene: str,
    scene_bounds: Any,
    min_confidence: float = 0.75,
) -> dict[str, Any]:
    """Match one captured scene against the map preview, refine in a full-res crop,
    then convert game world bounds to map pixels. Weak matches stay unpositioned.
    """
    from hk_agent.run_journey import save_scene_calibration

    pyvips = load_pyvips()
    if pyvips is None:
        return {"scene": scene, "positioned": False, "confidence": 0.0}
    map_image = pyvips.Image.new_from_file(str(map_path), access="sequential", fail=True).autorot()
    frame_image = pyvips.Image.new_from_file(str(capture_path), access="sequential", fail=True).autorot()
    map_small = _small_gray(map_image, 1500, 1100)
    frame_small = _small_gray(frame_image, 280, 200)
    first = template_match_ncc(map_small, frame_small)
    bounds = scene_bounds
    if (first is None or not isinstance(bounds, (list, tuple)) or len(bounds) != 4
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in bounds)):
        save_scene_calibration(cache_dir / "calibration.json", map_sha256=cache_dir.name,
                               scene=scene, transform=(0, 0, 0, 0, 0, 0), confidence=0.0)
        return {"scene": scene, "positioned": False, "confidence": first["confidence"] if first else 0.0}
    scale_x, scale_y = map_image.width / map_small.shape[1], map_image.height / map_small.shape[0]
    # Refine the preview match on the corresponding full-resolution map crop.
    crop_x = max(0, round(first["x"] * scale_x))
    crop_y = max(0, round(first["y"] * scale_y))
    crop_w = min(map_image.width - crop_x, max(frame_image.width, round(first["width"] * scale_x * 1.2)))
    crop_h = min(map_image.height - crop_y, max(frame_image.height, round(first["height"] * scale_y * 1.2)))
    refined_score = first["confidence"]
    refined = first
    if crop_w >= frame_image.width and crop_h >= frame_image.height:
        # The overview thumbnail consumes the first sequential PNG decoder. Reopen
        # the source for refinement so the crop is a fresh top-to-bottom pass.
        crop_source = pyvips.Image.new_from_file(
            str(map_path), access="sequential", fail=True
        ).autorot()
        crop = crop_source.crop(crop_x, crop_y, crop_w, crop_h)
        crop_small = _small_gray(crop, 1500, 1100)
        candidate = template_match_ncc(crop_small, frame_small)
        if candidate is not None:
            refined = {**candidate,
                       "x": crop_x + candidate["x"] * crop_w / crop_small.shape[1],
                       "y": crop_y + candidate["y"] * crop_h / crop_small.shape[0],
                       "width": candidate["width"] * crop_w / crop_small.shape[1],
                       "height": candidate["height"] * crop_h / crop_small.shape[0]}
            refined_score = candidate["confidence"]
    min_x, min_y, max_x, max_y = (float(value) for value in bounds)
    if max_x - min_x <= 1e-6 or max_y - min_y <= 1e-6:
        confidence = 0.0
        transform = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    else:
        # The matched frame footprint corresponds to the collidable scene bounds.
        sx, sy = refined["width"] / (max_x - min_x), refined["height"] / (max_y - min_y)
        transform = (sx, 0.0, refined["x"] - min_x * sx,
                     0.0, -sy, refined["y"] + max_y * sy)
        confidence = min(1.0, max(0.0, refined_score))
    save_scene_calibration(cache_dir / "calibration.json", map_sha256=cache_dir.name,
                           scene=scene, transform=transform, confidence=confidence)
    return {"scene": scene, "positioned": confidence >= min_confidence, "confidence": confidence,
            "affine": list(transform)}
