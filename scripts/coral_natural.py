#!/usr/bin/env python3
"""Build a coral catalog entry from natural colour clips on a key colour.

build_atlas.py expects coral clips painted in pure R/G/B zones. This script
starts from photo-like clips (natural colours, flat magenta background):

1. cut out with build_atlas.chroma_key (key found on the borders);
2. colour drift corrected: per channel gain bringing the skeleton base
   (bottom rows, always skeleton) back to its colour on the first day frame;
3. zones, soft weights: 1 = tentacles, 2 = tentacle tips (bright green),
   3 = skeleton. The skeleton has the colour of some tentacles, so it is
   found by its shape and its stillness: the cool solid part connected to
   the bottom on the last close frame (polyps closed), kept in every frame
   only where the picture does not move (tentacles sway, the skeleton never);
4. shade = brightness, weights and shade resized separately (premultiplied by
   alpha), so that zone borders are not darkened;
5. frames laid out with build_atlas.build_atlas, saved as the repo does
   (<id>.shade.webp, <id>.mask.png, thumb.webp, species.json).

    coral_natural.py euphyllia_glabrescens \
        --day ~/clips/euphyllia_glabrescens/day.mp4 \
        --close ~/clips/euphyllia_glabrescens/close.mp4 \
        --meta ~/clips/euphyllia_glabrescens/meta.json --out corals

The meta file gives names, scientific, size_cm and the palette (fluo flags);
a palette colour without "default" gets the one measured on the clips.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_atlas import (
    Clip,
    build_atlas,
    chroma_key,
    detect_key,
    read_frames,
    save_webp,
    thumbnail,
)

#: Rows at the bottom of the clip holding only skeleton (colour reference)
BASE_ROWS = 70
#: Frame width of the 1x atlas, pixels
FRAME_WIDTH = 256


def smoothstep(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """0 below lo, 1 above hi, smooth between."""
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur of a 2D map (scipy when available)."""
    try:
        from scipy import ndimage
    except ImportError:  # pragma: no cover - optional
        return a
    return ndimage.gaussian_filter(a, sigma)


def base_colour(rgba: np.ndarray) -> np.ndarray:
    """Mean colour of the opaque skeleton base (bottom rows)."""
    part = rgba[-BASE_ROWS:]
    solid = part[..., 3] > 250
    return part[..., :3][solid].mean(axis=0)


def warmth(rgb: np.ndarray) -> np.ndarray:
    """Red over blue, relative to brightness, smoothed (dark pixels are noisy)."""
    value = rgb.max(axis=-1)
    return blur((rgb[..., 0] - rgb[..., 2]) / np.maximum(value, 20.0), 1.5)


def skeleton_shape(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Skeleton on a frame where the polyps are closed: the cool solid part
    connected to the bottom of the clip, as a soft map."""
    from scipy import ndimage

    cool = (warmth(rgb) < 0.1) & (alpha > 0.5)
    # The skeleton goes on below the clip: rows repeated downwards, so that
    # the morphology (empty outside the picture) does not eat its foot
    pad = 12
    cool = np.pad(cool, ((0, pad), (0, 0)), mode="edge")
    cool = ndimage.binary_opening(cool, iterations=4)
    labels, _ = ndimage.label(cool)
    bottom = set(np.unique(labels[-3:])) - {0}
    solid = np.isin(labels, list(bottom))
    solid = ndimage.binary_closing(solid, iterations=6)
    solid = ndimage.binary_fill_holes(solid)
    return blur(solid.astype(np.float32), 2.0)[:-pad]


def analyse(
    rgb: np.ndarray,
    alpha: np.ndarray,
    skeleton: np.ndarray,
    references: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Brightness (0..1) and zone weights (H, W, 3) of a corrected frame.

    skeleton: where the skeleton can be (it never moves);
    references: pictures of the still parts; a pixel close to one of them
    does not move, so it is skeleton when in the skeleton shape.
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    value = rgb.max(axis=-1)
    # Tips: bright, green over red and blue
    green = g - np.maximum(r, b)
    tips = blur(smoothstep(green, 8, 28) * smoothstep(value, 110, 160), 0.8)
    # Skeleton: in its shape, not warm, and still (moving parts are tentacles)
    diff = np.min([np.abs(rgb - ref).max(axis=-1) for ref in references], axis=0)
    still = blur(1 - smoothstep(diff, 12, 40), 1.5)
    bone = skeleton * still * (1 - smoothstep(warmth(rgb), 0.12, 0.25))
    w3 = (1 - tips) * bone
    w1 = (1 - tips) * (1 - bone)
    return value / 255.0, np.dstack([w1, tips, w3])


def resize_premul(
    planes: np.ndarray, alpha: np.ndarray, box: tuple[int, int, int, int], size
) -> tuple[np.ndarray, np.ndarray]:
    """Crop to box, resize planes premultiplied by alpha, then unpremultiply."""
    x0, y0, x1, y1 = box

    def rs(a: np.ndarray) -> np.ndarray:
        img = Image.fromarray(a[y0:y1, x0:x1].astype(np.float32), "F")
        return np.asarray(img.resize(size, Image.Resampling.LANCZOS))

    a = np.clip(rs(alpha), 0, 1)
    out = np.dstack([rs(planes[..., k] * alpha) for k in range(planes.shape[-1])])
    out = out / np.maximum(a, 1e-3)[..., None]
    return np.clip(out, 0, 1), a


def main(argv: list[str] | None = None) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("id")
    parser.add_argument("--day", type=Path, required=True)
    parser.add_argument("--close", type=Path, required=True)
    parser.add_argument("--day-range", default="0-119:2", help="from-to:step")
    parser.add_argument("--close-range", default="0-203:4", help="from-to:step")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=None)
    args = parser.parse_args(argv)

    def rng(text: str) -> tuple[tuple[int, int], int]:
        span, _, step = text.partition(":")
        first, last = (int(v) for v in span.split("-"))
        return (first, last), int(step or 1)

    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    sources = {
        "day": (args.day, *rng(args.day_range)),
        "close": (args.close, *rng(args.close_range)),
    }

    # 1-2: cut out, colour drift corrected (kept as 8 bit, full resolution)
    cut: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    reference = None
    for name, (source, span, step) in sources.items():
        frames = read_frames(source, span, step)
        key = detect_key(frames[0])
        if key is None:
            raise SystemExit(f"{source}: no key colour on the borders")
        items = []
        for frame in frames:
            rgba = np.asarray(chroma_key(frame, key)).astype(np.float32)
            colour = base_colour(rgba)
            if reference is None:
                reference = colour
            gain = reference / np.maximum(colour, 1.0)
            rgb = np.clip(rgba[..., :3] * gain + 0.5, 0, 255).astype(np.uint8)
            items.append((rgb, rgba[..., 3] / 255.0))
        cut[name] = items
        print(f"{name}: {len(items)} frames, key {key}")

    # 3: zones. Skeleton shape from the last close frame (polyps closed);
    # still parts: the median day picture and the closed picture
    closed_rgb, closed_alpha = cut["close"][-1]
    skeleton = skeleton_shape(closed_rgb.astype(np.float32), closed_alpha)
    day_median = np.median(np.stack([f[0] for f in cut["day"][::3]]), axis=0).astype(
        np.float32
    )
    refs = [day_median, closed_rgb.astype(np.float32)]
    analysed: dict[str, list[tuple[np.ndarray, ...]]] = {}
    for name, items in cut.items():
        analysed[name] = []
        for rgb8, alpha in items:
            rgb = rgb8.astype(np.float32)
            value, weights = analyse(rgb, alpha, skeleton, refs)
            analysed[name].append((rgb, value, weights, alpha))
    del cut

    # Default palette: mean chromaticity of each zone, at full brightness
    top = np.percentile(
        np.concatenate([v[a > 0.9].ravel() for _, v, _, a in analysed["day"][:4]]),
        99,
    )
    palette = []
    for zone in range(3):
        acc, total = np.zeros(3), 0.0
        for rgb, value, weights, alpha in analysed["day"][:8]:
            w = weights[..., zone] * (alpha > 0.9) * (value > 0.08)
            chroma = rgb / np.maximum(value[..., None] * 255.0, 1.0) * top * 255
            acc += (chroma * w[..., None]).reshape(-1, 3).sum(axis=0)
            total += w.sum()
        palette.append(np.clip(acc / max(total, 1.0), 0, 255))
    hexes = ["#" + "".join(f"{round(float(c)):02x}" for c in p) for p in palette]
    print("default palette:", hexes)
    for item, hx in zip(meta["palette"], hexes, strict=False):
        item.setdefault("default", hx)
    # Brightness relative to the brightest coral pixels
    scale = 1.0 / max(top, 1e-3)

    # 4: common box (union of every frame), foot at the bottom
    union = None
    for items in analysed.values():
        for *_, alpha in items:
            ys, xs = np.nonzero(alpha > 0.5)
            box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
            union = (
                box
                if union is None
                else (
                    min(union[0], box[0]),
                    min(union[1], box[1]),
                    max(union[2], box[2]),
                    max(union[3], box[3]),
                )
            )
    x0, y0, x1, y1 = union
    width = FRAME_WIDTH
    height = int(math.ceil(width * (y1 - y0) / (x1 - x0) / 4) * 4)
    # Box enlarged upwards to the frame ratio, the foot stays on the bottom
    y0 = max(0, round(float(y1 - (x1 - x0) * height / width)))
    print(f"box {union} -> frame {width}x{height}")

    shade_clips, mask_clips = [], []
    for name, items in analysed.items():
        shades, masks = [], []
        for rgb, value, weights, alpha in items:
            planes = np.dstack([np.clip(value * scale, 0, 1), weights])
            small, a = resize_premul(planes, alpha, (x0, y0, x1, y1), (width, height))
            v = small[..., 0]
            w = np.round(small[..., 1:] * 32) / 32
            shades.append(
                Image.fromarray(
                    (np.dstack([v, v, v, a]) * 255 + 0.5).astype(np.uint8), "RGBA"
                )
            )
            masks.append(
                Image.fromarray(
                    (np.dstack([w, (a > 0.02)]) * 255 + 0.5).astype(np.uint8), "RGBA"
                )
            )
        loop = name == "day"
        shade_clips.append(Clip(name, shades, args.fps, loop=loop, pingpong=loop))
        mask_clips.append(Clip(name, masks, args.fps, loop=loop, pingpong=loop))

    # 5: atlases, descriptor, thumbnail
    shade_atlas = build_atlas(shade_clips, (width, height), args.columns)
    mask_atlas = build_atlas(mask_clips, (width, height), shade_atlas.columns)
    folder = args.out / args.id
    folder.mkdir(parents=True, exist_ok=True)
    save_webp(shade_atlas.sheet, folder / f"{args.id}.shade.webp")
    mask_atlas.sheet.save(folder / f"{args.id}.mask.png", optimize=True)

    # Thumbnail: first day frame in the default colours
    v = np.asarray(shade_clips[0].frames[0]).astype(np.float32) / 255
    w = np.asarray(mask_clips[0].frames[0]).astype(np.float32)[..., :3] / 255
    cols = np.asarray(palette + [np.array([128.0] * 3)]) / 255
    rest = np.clip(1 - w.sum(-1, keepdims=True), 0, 1)
    rgb = (w @ cols[:3] + rest * cols[3]) * v[..., :1]
    first = Image.fromarray(
        (np.dstack([rgb, v[..., 3]]).clip(0, 1) * 255 + 0.5).astype(np.uint8), "RGBA"
    )
    thumbnail(first).save(folder / "thumb.webp", format="WEBP", quality=85, method=6)

    descriptor = {
        "kind": "coral",
        "frame": [width, height],
        "columns": shade_atlas.columns,
        "clips": shade_atlas.clips,
        "atlas": {"shade": f"{args.id}.shade.webp", "mask": f"{args.id}.mask.png"},
        "thumbnail": "thumb.webp",
        **meta,
    }
    (folder / "species.json").write_text(
        json.dumps(descriptor, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"written {folder}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
