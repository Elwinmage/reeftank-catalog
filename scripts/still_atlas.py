#!/usr/bin/env python3
"""Animate a still photo of a fish into a "degraded" atlas.

When a species has no video clips, a photo of it (side view, head to the
right, on a flat magenta or green background, see `prompts/still_photo.json`)
is enough for a simple animation: the body undulates, the fish idles and
turns. The atlas has the layout of the ones `build_atlas.py` writes (8
columns, `swim`, `idle` and `turn` clips), and its descriptor says
`"degraded": true` and where the photo comes from (`source`), so that a real
animation built from videos later replaces it.

    python scripts/still_atlas.py <photo> fish/<id> --source photo \\
        --author "Jane Doe" --license "CC BY 4.0"
    python scripts/still_atlas.py <photo> fish/<id> --source ai-generated \\
        --model gemini-2.5-flash-image
    python scripts/still_atlas.py --prompt "Betta splendens" [--key green]

The photo is checked first (uniform key colour in the corners, the whole fish
in the picture, large enough): `--force` writes the atlas anyway. An entry that
already has a real (not degraded) animation is left alone without `--force`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_atlas import GENERATED_KEYS, THUMB_SIZE, save_webp, thumbnail
from validate import LICENSES, SOURCE_TYPES

PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / "still_photo.json"
#: Background colours accepted for the photo
KEYS = {"magenta": (255, 0, 255), "green": (0, 255, 0)}
#: Supersampling, frame width, columns of the atlas
SCALE = 4
FRAME_WIDTH = 208
COLUMNS = 8
#: Clips: frames, fps, loop
CLIPS = {"swim": (43, 24, True), "idle": (45, 12, True), "turn": (26, 24, False)}
#: Checks of the photo
CORNER = 20
MIN_WIDTH = 512
MIN_FISH_SHARE = 0.4
EDGE_MARGIN = 2
UNIFORM_STD = 24.0


def render_prompt(species: str, key: str = "magenta", path: Path = PROMPT_FILE) -> str:
    """The prompt that asks an image model for the photo of a species."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if key not in data["keys"]:
        raise ValueError(f"key must be one of {', '.join(data['keys'])}")
    return (
        data["template"]
        .replace("[SPECIE]", species)
        .replace("[KEY_NAME]", key)
        .replace("[KEY]", data["keys"][key])
    )


def _chroma(rgb: np.ndarray, key: Sequence[float]) -> np.ndarray:
    """How far a colour goes towards the key: the weakest channel the key has
    high, minus the strongest it has low (zero or less for any other colour)."""
    high = [i for i, v in enumerate(key) if v >= 128]
    low = [i for i, v in enumerate(key) if v < 128]
    return rgb[..., high].min(axis=-1) - rgb[..., low].max(axis=-1)


def corners(rgb: np.ndarray) -> np.ndarray:
    """The four corner patches of a picture, as a list of pixels."""
    n = CORNER
    return np.concatenate(
        [
            rgb[:n, :n].reshape(-1, 3),
            rgb[:n, -n:].reshape(-1, 3),
            rgb[-n:, :n].reshape(-1, 3),
            rgb[-n:, -n:].reshape(-1, 3),
        ]
    )


def detect_key(rgb: np.ndarray) -> str | None:
    """Name of the key colour of the corners (None when none fits)."""
    median = np.median(corners(rgb), axis=0)
    best, share = None, 0.0
    for name, key in KEYS.items():
        norm = float(_chroma(np.array(key, dtype=np.float64), key))
        value = float(_chroma(median, key)) / norm
        if value > max(share, 0.6):
            best, share = name, value
    return best


def key_out(photo: Image.Image, key: str | None = None) -> Image.Image:
    """Remove the flat key colour, keeping soft edges (RGBA)."""
    rgb = np.asarray(photo.convert("RGB"), dtype=np.float64)
    name = key or detect_key(rgb) or "magenta"
    colour = np.median(corners(rgb), axis=0)
    norm = max(1.0, float(_chroma(colour, KEYS[name])))
    raw = 1.0 - np.clip(_chroma(rgb, KEYS[name]) / norm, 0, 1)
    alpha = np.clip((raw - 0.06) / 0.9, 0, 1)
    solid = np.maximum(raw, 0.05)[..., None]
    pixels = np.clip((rgb - (1 - raw)[..., None] * colour) / solid, 0, 255)
    return Image.fromarray(np.dstack([pixels, alpha * 255]).astype(np.uint8), "RGBA")


def check_photo(photo: Image.Image, key: str | None = None) -> list[str]:
    """Problems that make a photo unfit (empty when fine)."""
    problems: list[str] = []
    if photo.width < MIN_WIDTH:
        problems.append(
            f"photo too small: {photo.width} px wide (at least {MIN_WIDTH})"
        )
    rgb = np.asarray(photo.convert("RGB"), dtype=np.float64)
    if min(rgb.shape[:2]) <= 2 * CORNER:
        return [*problems, "photo too small to check its corners"]
    name = key or detect_key(rgb)
    if name is None:
        return [
            *problems,
            "corners are not a flat magenta or green background",
        ]
    if float(corners(rgb).std(axis=0).max()) > UNIFORM_STD:
        problems.append("background is not uniform in the corners")
    alpha = key_out(photo, name).getchannel("A").point(lambda v: 255 if v > 10 else 0)
    box = alpha.getbbox()
    if box is None:
        return [*problems, "no fish found on the background"]
    left, top, right, bottom = box
    width, height = photo.size
    if (
        left < EDGE_MARGIN
        or top < EDGE_MARGIN
        or right > width - EDGE_MARGIN
        or bottom > height - EDGE_MARGIN
    ):
        problems.append("the fish touches the edge of the picture (cut off)")
    if (right - left) / width < MIN_FISH_SHARE:
        problems.append(
            f"the fish is too small in the picture (under {MIN_FISH_SHARE:.0%})"
        )
    return problems


class Fish:
    """A keyed fish picture, to draw undulating at any phase."""

    def __init__(self, photo: Image.Image, length: int, key: str | None, wag: int = 8):
        sub = key_out(photo, key)
        box = sub.getchannel("A").point(lambda v: 255 if v > 10 else 0).getbbox()
        sub = sub.crop(box)
        scale = length * SCALE / sub.width
        size = (length * SCALE, max(1, round(sub.height * scale)))
        layer = sub.convert("RGBa").resize(size, Image.Resampling.LANCZOS)
        self.width = FRAME_WIDTH
        self.height = int(math.ceil((layer.height / SCALE + 2 * wag + 4) / 4.0) * 4)
        self.size = (self.width * SCALE, self.height * SCALE)
        self.head = (self.width + length) / 2
        self.tail = (self.width - length) / 2
        self.base = Image.new("RGBA", self.size, (0, 0, 0, 0))
        self.base.alpha_composite(
            layer.convert("RGBA"),
            (
                (self.size[0] - layer.width) // 2,
                (self.size[1] - layer.height) // 2,
            ),
        )

    def wave(self, x: float, phase: float, amp: float) -> float:
        """Vertical shift of a column: larger towards the tail."""
        w = min(1.0, max(0.0, (self.head - x) / (self.head - self.tail)))
        return amp * (0.06 + 0.94 * w**1.7) * math.sin(phase - 1.6 * w)

    def warp(self, phase: float, amp: float) -> Image.Image:
        """The fish with every column shifted by the wave."""
        arr = np.asarray(self.base)
        out = np.zeros_like(arr)
        rows = arr.shape[0]
        for column in range(arr.shape[1]):
            shift = round(self.wave(column / SCALE, phase, amp) * SCALE)
            if shift == 0:
                out[:, column] = arr[:, column]
            elif shift > 0:
                out[shift:, column] = arr[: rows - shift, column]
            else:
                out[: rows + shift, column] = arr[-shift:, column]
        return Image.fromarray(out, "RGBA")

    def frame(self, phase: float, amp: float, theta: float = 0.0) -> Image.Image:
        """One frame; `theta` turns the fish (0 to pi) by squeezing it, then
        mirroring it."""
        fish = self.warp(phase, amp)
        cos = math.cos(theta)
        squeeze = math.hypot(cos, 0.2 * math.sin(theta))
        new_width = max(2, round(self.size[0] * squeeze))
        layer = (
            fish.convert("RGBa")
            .resize((new_width, self.size[1]), Image.Resampling.LANCZOS)
            .convert("RGBA")
        )
        if cos < 0:
            layer = layer.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        canvas = Image.new("RGBA", self.size, (0, 0, 0, 0))
        canvas.alpha_composite(layer, ((self.size[0] - new_width) // 2, 0))
        return (
            canvas.convert("RGBa")
            .resize((self.width, self.height), Image.Resampling.LANCZOS)
            .convert("RGBA")
        )


def build_frames(fish: Fish) -> list[Image.Image]:
    """Frames of the swim, idle and turn clips, in this order."""
    frames = []
    count = CLIPS["swim"][0]
    for i in range(count):
        frames.append(fish.frame(2 * math.pi * 2 * i / count, 7.0))
    count = CLIPS["idle"][0]
    for i in range(count):
        frames.append(fish.frame(2 * math.pi * i / count, 2.2))
    count = CLIPS["turn"][0]
    for i in range(count):
        u = i / (count - 1)
        smooth = u * u * (3 - 2 * u)
        frames.append(
            fish.frame(
                2 * math.pi * 1.5 * u, 5.0 * math.sin(math.pi * u), math.pi * smooth
            )
        )
    return frames


def source_info(args: argparse.Namespace) -> dict[str, str]:
    """The `source` field from the command line (without empty values)."""
    info = {
        "type": args.source,
        "model": args.model,
        "author": args.author,
        "license": args.license,
        "date": args.date or datetime.now(UTC).date().isoformat(),
    }
    return {k: v for k, v in info.items() if v}


def write_atlas(
    photo: Image.Image,
    folder: Path,
    source: dict[str, str],
    length: int,
    key: str | None,
) -> dict[str, Any]:
    """Write the atlas, the thumbnail and the descriptor of an entry."""
    folder.mkdir(parents=True, exist_ok=True)
    fish = Fish(photo, length, key)
    frames = build_frames(fish)
    rows = math.ceil(len(frames) / COLUMNS)
    sheet = Image.new("RGBA", (COLUMNS * fish.width, rows * fish.height), (0, 0, 0, 0))
    for i, frame in enumerate(frames):
        sheet.paste(frame, ((i % COLUMNS) * fish.width, (i // COLUMNS) * fish.height))
    name = folder.name
    save_webp(sheet, folder / f"{name}.webp")
    save_webp(thumbnail(frames[0], THUMB_SIZE), folder / "thumb.webp")
    swim = frames[: CLIPS["swim"][0]]
    widths = [b[2] - b[0] for b in (f.getchannel("A").getbbox() for f in swim)]
    clips: dict[str, Any] = {}
    start = 0
    for clip, (count, fps, loop) in CLIPS.items():
        clips[clip] = {"from": start, "to": start + count - 1, "fps": fps, "loop": loop}
        start += count
    generated = {
        "kind": "fish",
        "frame": [fish.width, fish.height],
        "columns": COLUMNS,
        "clips": clips,
        "atlas": {"1x": f"{name}.webp"},
        "facing": "right",
        "length_frac": round(float(np.median(widths)) / fish.width, 4),
        "thumbnail": "thumb.webp",
        "degraded": True,
        "source": source,
    }
    file = folder / "species.json"
    try:
        current = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        current = {}
    current = current if isinstance(current, dict) else {}
    kept = {k: v for k, v in current.items() if k not in GENERATED_KEYS}
    descriptor = {**generated, **kept}
    file.write_text(
        json.dumps(descriptor, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return descriptor


def has_real_animation(folder: Path) -> bool:
    """True when the entry already has an atlas made from videos."""
    try:
        current = json.loads((folder / "species.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        isinstance(current, dict) and "atlas" in current and not current.get("degraded")
    )


def main(argv: list[str] | None = None) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("photo", type=Path, nargs="?")
    parser.add_argument("folder", type=Path, nargs="?", help="entry, ex: fish/betta")
    parser.add_argument("--prompt", metavar="SPECIES", help="print the prompt")
    parser.add_argument("--key", choices=sorted(KEYS), help="background colour")
    parser.add_argument("--source", choices=[s for s in SOURCE_TYPES if s != "video"])
    parser.add_argument("--author", help="who made the photo")
    parser.add_argument("--license", choices=LICENSES)
    parser.add_argument("--model", help="image model, for an ai-generated photo")
    parser.add_argument("--date", help="YYYY-MM-DD (default: today)")
    parser.add_argument("--length", type=int, default=192, help="fish length, px")
    parser.add_argument("--force", action="store_true", help="ignore the checks")
    args = parser.parse_args(argv)

    if args.prompt:
        print(render_prompt(args.prompt, args.key or "magenta"))
        return 0
    if args.photo is None or args.folder is None or args.source is None:
        parser.error("photo, folder and --source are needed")
    if has_real_animation(args.folder) and not args.force:
        print(f"{args.folder}: already has a real animation (--force to replace it)")
        return 1
    photo = Image.open(args.photo)
    problems = check_photo(photo, args.key)
    for problem in problems:
        print(f"{args.photo}: {problem}")
    if problems and not args.force:
        return 1
    descriptor = write_atlas(
        photo, args.folder, source_info(args), args.length, args.key
    )
    print(
        f"{args.folder}: degraded atlas {descriptor['frame']}, {descriptor['source']['type']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
