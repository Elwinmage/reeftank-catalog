"""Degraded atlas of a fish made from a still photo: the shared implementation.

Canonical copy: reeftank-catalog/client/. `scripts/still_atlas.py` (command line)
and the catalog editor (which asks an image model for the photo, then builds the
atlas) use the same code, so that an atlas made in the editor is the one the
command would make. Tools get a copy through `scripts/sync_client.py`; edit it
here, never in a copy.

A photo (side view, head to the right, on a flat magenta or green background,
see `PROMPT`) is enough for a simple animation: the body undulates, the fish
idles and turns. The atlas has the layout of the ones `build_atlas.py` writes (8
columns, `swim`, `idle` and `turn` clips); its descriptor keys say
`"degraded": true` and where the photo comes from (`source`), so that a real
animation built from videos later replaces it.

    problems = check_photo(photo)                 # empty when the photo is fit
    still = build_still(photo, "amphiprion_ocellaris", {"type": "photo", ...})
    still.files   # {"amphiprion_ocellaris.webp": bytes, "thumb.webp": bytes}
    still.keys    # the descriptor keys that come with them

No Home Assistant dependency; needs Pillow and numpy.
"""

from __future__ import annotations

import io
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

#: The prompt that asks an image model for the photo (versioned: `version` changes
#: with the wording; the same text is prompts/still_photo.json, a test checks it)
PROMPT: dict[str, Any] = {
    "version": 2,
    "keys": {"magenta": "flat magenta #FF00FF", "green": "flat green #00FF00"},
    "default_key": "magenta",
    "alternate_key": {
        "green": "pink, red-pink or purple fish: a magenta background would eat them"
    },
    "template": "Realistic side view of [SPECIE], head to the RIGHT, on a [KEY] background (the colour code only tells which colour the background is: never write it, nor any other text, on the image), perfectly flat uniform background without gradient or vignette, neutral white studio lighting, no shadow, no [KEY_NAME] tint on the edges of the animal, highest resolution, the animal fills about 70% of the image width, whole animal visible with all its fins, clean image: no text, no watermark, no mark in the corners.",
}
#: Descriptor keys the atlas comes with (the others describe the species)
GENERATED_KEYS = frozenset(
    {
        "kind",
        "frame",
        "columns",
        "clips",
        "atlas",
        "thumbnail",
        "facing",
        "length_frac",
        "degraded",
        "source",
    }
)
THUMB_SIZE = (96, 64)
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
#: Alpha above 10 counts as the fish (lookup table of `Image.point`)
OPAQUE = [0] * 11 + [255] * 245
#: Specks thinner than this many pixels are not the fish (a stray pixel at the
#: edge of a generated picture), and the soft edge kept around what remains
SPECK = 5
SOFT_EDGE = 7


def render_prompt(
    species: str, key: str = "magenta", template: dict[str, Any] | None = None
) -> str:
    """The prompt that asks an image model for the photo of a species."""
    data = template or PROMPT
    if key not in data["keys"]:
        names = ", ".join(data["keys"])
        raise ValueError(f"key must be one of {names}")
    return (
        data["template"]
        .replace("[SPECIE]", species)
        .replace("[KEY_NAME]", key)
        .replace("[KEY]", data["keys"][key])
    )


def encode_webp(image: Image.Image) -> bytes:
    """Lossy WebP keeping a clean alpha."""
    buffer = io.BytesIO()
    image.save(buffer, format="WEBP", quality=80, alpha_quality=90, method=6)
    return buffer.getvalue()


def thumbnail(frame: Image.Image, size: tuple[int, int] = THUMB_SIZE) -> Image.Image:
    """A small picture of a frame: cropped to its subject, fitted in `size`."""
    box = frame.getchannel("A").getbbox() or (0, 0, frame.width, frame.height)
    subject = frame.crop(box)
    scale = min(size[0] / subject.width, size[1] / subject.height)
    small = subject.convert("RGBa").resize(
        (max(1, round(subject.width * scale)), max(1, round(subject.height * scale))),
        Image.Resampling.LANCZOS,
    )
    out = Image.new("RGBA", size, (0, 0, 0, 0))
    out.paste(
        small.convert("RGBA"),
        ((size[0] - small.width) // 2, (size[1] - small.height) // 2),
    )
    return out


def _chroma(rgb: np.ndarray, key: Sequence[float]) -> np.ndarray:
    """How far a colour goes towards the key: the weakest channel the key has
    high, minus the strongest it has low (zero or less for any other colour)."""
    high = [i for i, v in enumerate(key) if v >= 128]
    low = [i for i, v in enumerate(key) if v < 128]
    return rgb[..., high].min(axis=-1) - rgb[..., low].max(axis=-1)


def _patches(rgb: np.ndarray) -> list[np.ndarray]:
    n = CORNER
    return [
        rgb[:n, :n].reshape(-1, 3),
        rgb[:n, -n:].reshape(-1, 3),
        rgb[-n:, :n].reshape(-1, 3),
        rgb[-n:, -n:].reshape(-1, 3),
    ]


def corners(rgb: np.ndarray) -> np.ndarray:
    """The four corner patches of a picture, as a list of pixels."""
    return np.concatenate(_patches(rgb))


def corner_spread(rgb: np.ndarray) -> float:
    """How far from flat the corners are: the spread of the patches, but the
    two worst ones, which an image model may mark (a smudge, a written colour
    code)."""
    spreads = sorted(float(p.std(axis=0).max()) for p in _patches(rgb))
    return spreads[-3]


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


def nearest_key(rgb: np.ndarray) -> str:
    """The key colour the corners are closest to, however washed out.

    `detect_key` only names a background saturated enough to be keyed out
    safely, and says nothing otherwise. An image model, though, often answers
    with a dusty pink or a sage green: unmistakably one of the two, but far
    from the colour asked for. This says which one it is, so that a prompt
    built around such a picture can name the colour that is really in it —
    telling a video model the background is magenta while it is shown a green
    one leaves it reconciling the two instead of animating the animal.
    """
    median = np.median(corners(rgb), axis=0)
    return max(
        KEYS,
        key=lambda name: (
            float(_chroma(median, KEYS[name]))
            / float(_chroma(np.array(KEYS[name], dtype=np.float64), KEYS[name]))
        ),
    )


#: Blobs are looked for on a grid of BLOCK pixels; fish parts closer than
#: 2 * GAP_BLOCKS * BLOCK pixels count as one fish
BLOCK = 8
GAP_BLOCKS = 2


def _largest_blob(mask: np.ndarray) -> np.ndarray:
    """The biggest group of true pixels of a mask, and nothing else.

    An image model leaves marks away from the fish (a smudge in a corner, the
    colour code written in the picture): the fish is the largest blob. Pieces of
    the fish a few pixels apart (the tips of the fins) still belong to it.
    """
    height, width = mask.shape
    rows, cols = -(-height // BLOCK), -(-width // BLOCK)
    padded = np.zeros((rows * BLOCK, cols * BLOCK), dtype=bool)
    padded[:height, :width] = mask
    cells = np.asarray(
        padded.reshape(rows, BLOCK, cols, BLOCK).any(axis=(1, 3)), dtype=bool
    )
    for _ in range(GAP_BLOCKS):  # grow, so that the gaps close
        grown = cells.copy()
        grown[1:] |= cells[:-1]
        grown[:-1] |= cells[1:]
        grown[:, 1:] |= cells[:, :-1]
        grown[:, :-1] |= cells[:, 1:]
        cells = grown
    label = np.zeros(cells.shape, dtype=np.int32)
    sizes: list[int] = [0]
    for start in zip(*np.nonzero(cells), strict=True):
        if label[start]:
            continue
        sizes.append(0)
        label[start] = len(sizes) - 1
        todo = [start]
        while todo:
            r, c = todo.pop()
            sizes[-1] += 1
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (
                    0 <= nr < rows
                    and 0 <= nc < cols
                    and cells[nr, nc]
                    and not label[nr, nc]
                ):
                    label[nr, nc] = label[r, c]
                    todo.append((nr, nc))
    if len(sizes) == 1:
        return np.zeros_like(mask)
    keep = label == int(np.argmax(sizes))
    big = np.repeat(np.repeat(keep, BLOCK, axis=0), BLOCK, axis=1)
    return big[:height, :width]


def _fish_zone(raw: np.ndarray) -> np.ndarray:
    """1 around the fish, 0 elsewhere: what is left once the marks are dropped.

    An image model leaves a few off-colour pixels, mostly on the last row or
    column; they would stretch the box of the fish to the whole picture. The
    zone is the fish with its soft edge, found by erasing then growing the
    pixels that are not the key, keeping the largest blob.
    """
    fish = Image.fromarray(((raw > 0.1) * 255).astype(np.uint8), "L")
    opened = fish.filter(ImageFilter.MinFilter(SPECK)).filter(
        ImageFilter.MaxFilter(SPECK)
    )
    zone = opened.filter(ImageFilter.MaxFilter(SOFT_EDGE))
    return (_largest_blob(np.asarray(zone) > 0)).astype(np.float64)


def key_out(photo: Image.Image, key: str | None = None) -> Image.Image:
    """Remove the flat key colour, keeping soft edges (RGBA)."""
    rgb = np.asarray(photo.convert("RGB"), dtype=np.float64)
    name = key or detect_key(rgb) or "magenta"
    colour = np.median(corners(rgb), axis=0)
    norm = max(1.0, float(_chroma(colour, KEYS[name])))
    raw = 1.0 - np.clip(_chroma(rgb, KEYS[name]) / norm, 0, 1)
    alpha = np.clip((raw - 0.06) / 0.9, 0, 1) * _fish_zone(raw)
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
    if corner_spread(rgb) > UNIFORM_STD:
        problems.append("background is not uniform in the corners")
    alpha = key_out(photo, name).getchannel("A").point(OPAQUE)
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
        box = sub.getchannel("A").point(OPAQUE).getbbox()
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


@dataclass(frozen=True)
class Still:
    """What a photo gives: the files of the entry and the descriptor keys."""

    files: dict[str, bytes]
    keys: dict[str, Any]


def build_still(
    photo: Image.Image,
    name: str,
    source: dict[str, str],
    length: int = 192,
    key: str | None = None,
) -> Still:
    """The degraded atlas and thumbnail of an entry (`name`: its id)."""
    fish = Fish(photo, length, key)
    frames = build_frames(fish)
    rows = math.ceil(len(frames) / COLUMNS)
    sheet = Image.new("RGBA", (COLUMNS * fish.width, rows * fish.height), (0, 0, 0, 0))
    for i, frame in enumerate(frames):
        sheet.paste(frame, ((i % COLUMNS) * fish.width, (i // COLUMNS) * fish.height))
    swim = frames[: CLIPS["swim"][0]]
    boxes = [f.getchannel("A").getbbox() or (0, 0, 0, 0) for f in swim]
    widths = [b[2] - b[0] for b in boxes]
    clips: dict[str, Any] = {}
    start = 0
    for clip, (count, fps, loop) in CLIPS.items():
        clips[clip] = {"from": start, "to": start + count - 1, "fps": fps, "loop": loop}
        start += count
    keys = {
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
    files = {
        f"{name}.webp": encode_webp(sheet),
        "thumb.webp": encode_webp(thumbnail(frames[0], THUMB_SIZE)),
    }
    return Still(files, keys)


def is_real_animation(descriptor: Any) -> bool:
    """True when a descriptor has an atlas made from videos (not a degraded one)."""
    return (
        isinstance(descriptor, dict)
        and "atlas" in descriptor
        and not descriptor.get("degraded")
    )
