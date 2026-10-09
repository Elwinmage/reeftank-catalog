#!/usr/bin/env python3
"""Build a sprite atlas and its descriptor from animation clips.

The card does not play videos: it draws frames out of an atlas (one picture,
frames on a grid), so every fish has its own phase and a tail beat following
its speed. This script turns source clips into that atlas.

Source clips can be:
- a folder of PNG frames (with alpha), played in name order;
- any file ffmpeg reads (MP4, APNG, MOV with alpha, WebM...), when ffmpeg is
  installed.

Fish
----

Clips generated on a flat key colour (magenta, or green for pink and purple
fish) are cut out: soft alpha from the distance to the key, edge colours
unmixed from the key, key spill removed, specks dropped (needs scipy).
The key is found on the frame borders (`--key auto`), or given
(`--key #ff00ff`), or skipped (`--key none`, sources with alpha).

Every clip is placed on one common scale, so that the fish keeps its size
from one clip to the other: the fish length is measured on the clip (median
over its frames, or over its first frames for a clip played once, like a
turn) and brought to `--length` pixels, centred in the frame. The frame size
fits the largest extent of every clip, and the descriptor tells the card the
share of the frame width the fish length takes (`length_frac`).

    build_atlas.py fish siganus_vulpinus \
        --clip swim=clips/swim.mp4:24:loop:118-154 \
        --clip idle=clips/idle.mp4:12:pingpong:0-72:2 \
        --clip turn=clips/turn.mp4:24:once:95-185:3 \
        --meta clips/siganus_vulpinus.meta.json \
        --out custom_components/reeftank/catalog/fish

or, with the same settings in a recipe file next to the clips:

    build_atlas.py fish siganus_vulpinus --recipe clips/siganus.json --out ...

writes <out>/siganus_vulpinus/{species.json, siganus_vulpinus.webp}.

Automatic mode
--------------

Clips generated from one reference picture (swim and idle start and end on
it, the turn starts on it and ends on its mirror) need no setting: put them
in a folder named after the species, with an optional meta.json,

    clips/siganus_vulpinus/{swim.mp4, idle.mp4, turn.mp4, meta.json}

and run

    build_atlas.py fish siganus_vulpinus --auto clips/siganus_vulpinus --out ...

The clips are analysed: a 1-3 s seamless swim loop, an idle ping-pong or a
2-4 s loop kept at 12 fps, the frames where the turn actually turns, sped up
to about 1.2 s. The choices go to clips/siganus_vulpinus/siganus_vulpinus.json
(a recipe, to edit if needed), then the atlas is built. Clips that do not
start or end on the reference picture are reported.

For a whole batch, one folder per species:

    build_atlas.py batch clips --out build/fish

reuses a folder's recipe when there is one (hand edits stay; `--reanalyse`
rewrites them), and writes it from the clips otherwise.

To choose ranges by hand, `loops` lists the best seamless loops of a clip and
tells whether it is a forward-then-backward (ping-pong) clip:

    build_atlas.py loops clips/swim.mp4

Corals
------

A coral is drawn with its zones painted in **pure red, pure green and pure
blue**, shaded as wanted; grey parts take the fourth colour. From that one
clip the script derives:

- `<id>.shade.webp`: the shading (brightness of every pixel);
- `<id>.mask.png`: lossless, R/G/B = weight of palette colours 1/2/3, the
  fourth colour gets what is left.

The card recolours the coral with the user's palette:
`pixel = shade x (w1.c1 + w2.c2 + w3.c3 + (1 - w1 - w2 - w3).c4)`.

    build_atlas.py coral euphyllia_glabrescens \
        --clip day=clips/euphyllia_day --clip close=clips/euphyllia_close \
        --clip night=clips/euphyllia_night --meta clips/euphyllia.meta.json \
        --out custom_components/reeftank/catalog/corals

Clips
-----

`--clip name=source[:fps[:mode[:from-to[:step]]]]`

- fps: playback rate (default 24);
- mode: `loop`, `once` or `pingpong` (played forward then backward). Default:
  `swim`, `idle`, `day` and `night` loop, the others play once;
- from-to: source frames kept (0-based, both included); default all;
- step: keep one frame out of `step` (a turn played faster, an idle at a
  lower rate).

The meta file (JSON) is merged into the descriptor: size_cm, behavior, night,
feeding_response, palette, names...

A recipe (JSON) holds the same settings; paths are relative to it:

    {
      "key": "auto",
      "length": 192,
      "clips": {
        "swim": {"source": "swim.mp4", "fps": 24, "mode": "loop",
                 "range": [118, 154]},
        "turn": {"source": "turn.mp4", "mode": "once", "range": [95, 185],
                 "step": 3}
      },
      "meta": {"size_cm": [15, 20], "behavior": {"profile": "cruiser"}}
    }

Requires Pillow and numpy, ffmpeg (and ffprobe) for videos; scipy for speck
removal (optional).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import shutil
import subprocess
import sys
import tempfile
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _load_ndimage() -> Any:
    """scipy.ndimage, or None when scipy is missing or unusable.

    A scipy built for another numpy (a system package next to a pip numpy)
    fails on import after printing a long message: kept quiet, and told in
    one line.
    """
    try:
        with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from scipy import ndimage as module
    except ModuleNotFoundError as err:  # pragma: no cover - optional
        if err.name == "scipy":
            return None
        return _scipy_unusable(err)
    except Exception as err:  # pragma: no cover - broken install
        return _scipy_unusable(err)
    return module


def _scipy_unusable(err: Exception) -> None:  # pragma: no cover - broken install
    """Tell in one line why scipy is not used."""
    print(
        f"note: scipy unusable ({type(err).__name__}), specks are kept; "
        "install scipy built for this numpy (in a virtual environment)",
        file=sys.stderr,
    )


ndimage = _load_ndimage()

LOOPING = {"swim", "idle", "day", "night"}
MODES = ("loop", "once", "pingpong")
DEFAULT_FPS = 24
DEFAULT_LENGTH = 192


@dataclass
class Clip:
    """A named sequence of frames."""

    name: str
    frames: list[Image.Image]
    fps: int = DEFAULT_FPS
    loop: bool = False
    pingpong: bool = False


@dataclass
class ClipSpec:
    """Where a clip comes from and how it plays."""

    name: str
    source: Path
    fps: int = DEFAULT_FPS
    mode: str | None = None
    range: tuple[int, int] | None = None
    step: int = 1

    @property
    def resolved_mode(self) -> str:
        """The mode, or the default one of the clip name."""
        if self.mode:
            return self.mode
        return "loop" if self.name in LOOPING else "once"


@dataclass
class Atlas:
    """Frames laid out on a grid, with the clips indexing them."""

    frame: tuple[int, int]
    columns: int
    sheet: Image.Image
    clips: dict[str, dict[str, Any]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Reading sources
# ---------------------------------------------------------------------------


def _select(count: int, spec_range: tuple[int, int] | None, step: int) -> list[int]:
    """Indices of the kept frames."""
    first, last = (0, count - 1) if spec_range is None else spec_range
    first = max(0, first)
    last = min(count - 1, last)
    if last < first:
        raise SystemExit(f"empty frame range {spec_range} ({count} frames)")
    return list(range(first, last + 1, max(1, step)))


def read_frames(
    source: Path, spec_range: tuple[int, int] | None = None, step: int = 1
) -> list[Image.Image]:
    """Frames of a clip: a folder of PNG files, or a file ffmpeg can read."""
    if source.is_dir():
        files = sorted(p for p in source.iterdir() if p.suffix.lower() == ".png")
        if not files:
            raise SystemExit(f"no PNG frame in {source}")
        return [
            Image.open(files[i]).convert("RGBA")
            for i in _select(len(files), spec_range, step)
        ]

    # An animated picture Pillow understands natively (APNG, animated WebP)
    try:
        image = Image.open(source)
        n_frames = int(getattr(image, "n_frames", 1))
        if n_frames > 1:
            frames = []
            for index in _select(n_frames, spec_range, step):
                image.seek(index)
                frames.append(image.convert("RGBA"))
            return frames
    except OSError:
        pass

    if shutil.which("ffmpeg") is None:
        raise SystemExit(f"cannot read {source}: install ffmpeg or give PNG frames")
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-vsync",
                "0",
                "-pix_fmt",
                "rgba",
                str(Path(tmp) / "f%05d.png"),
            ],
            check=True,
        )
        return read_frames(Path(tmp), spec_range, step)


def parse_clip(spec: str) -> ClipSpec:
    """Parse `name=source[:fps[:mode[:from-to[:step]]]]`."""
    if "=" not in spec:
        raise SystemExit(f"bad --clip {spec!r}: expected name=source")
    name, rest = spec.split("=", 1)
    parts = rest.split(":")
    clip = ClipSpec(name, Path(parts[0]))
    try:
        if len(parts) > 1 and parts[1]:
            clip.fps = int(parts[1])
        if len(parts) > 2 and parts[2]:
            clip.mode = _mode(parts[2])
        if len(parts) > 3 and parts[3]:
            first, last = parts[3].split("-", 1)
            clip.range = (int(first), int(last))
        if len(parts) > 4 and parts[4]:
            clip.step = int(parts[4])
    except ValueError as err:
        raise SystemExit(f"bad --clip {spec!r}: {err}") from err
    return clip


def _mode(value: str) -> str:
    """A playback mode; true/false are accepted for loop/once."""
    value = value.lower()
    if value in ("1", "true", "yes"):
        return "loop"
    if value in ("0", "false", "no"):
        return "once"
    if value not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    return value


def read_recipe(path: Path) -> dict[str, Any]:
    """Settings of a recipe file, with clip sources resolved."""
    data = json.loads(path.read_text(encoding="utf-8"))
    clips = []
    for name, item in data.get("clips", {}).items():
        spec = ClipSpec(name, (path.parent / item["source"]).resolve())
        spec.fps = int(item.get("fps", DEFAULT_FPS))
        if item.get("mode"):
            spec.mode = _mode(str(item["mode"]))
        if item.get("range"):
            spec.range = (int(item["range"][0]), int(item["range"][1]))
        spec.step = int(item.get("step", 1))
        clips.append(spec)
    data["clips"] = clips
    return data


# ---------------------------------------------------------------------------
# Cutting out (chroma key)
# ---------------------------------------------------------------------------


def detect_key(frame: Image.Image) -> tuple[int, int, int] | None:
    """Key colour of a frame: its borders, when flat and saturated.

    None when the frame already has transparency, or its borders are not a
    flat vivid colour.
    """
    rgba = np.asarray(frame.convert("RGBA"))
    if int(rgba[..., 3].min()) < 250:
        return None
    border = np.concatenate(
        [
            rgba[:4, :, :3].reshape(-1, 3),
            rgba[-4:, :, :3].reshape(-1, 3),
            rgba[:, :4, :3].reshape(-1, 3),
            rgba[:, -4:, :3].reshape(-1, 3),
        ]
    ).astype(np.float32)
    median = np.median(border, axis=0)
    spread = float(np.median(np.abs(border - median).max(axis=1)))
    if spread > 12 or float(median.max() - median.min()) < 100:
        return None
    r, g, b = (round(float(v)) for v in median)
    return r, g, b


def parse_key(value: str) -> tuple[int, int, int] | str | None:
    """`auto`, `none` or a #rrggbb colour."""
    value = value.strip().lower()
    if value in ("auto", "none"):
        return value if value == "auto" else None
    try:
        if len(value) != 7 or value[0] != "#":
            raise ValueError(value)
        return int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16)
    except ValueError as err:
        raise SystemExit(f"bad --key {value!r}: auto, none or #rrggbb") from err


def chroma_key(
    frame: Image.Image,
    key: tuple[int, int, int],
    inner: float = 40.0,
    outer: float = 110.0,
) -> Image.Image:
    """Cut a frame out of its key colour.

    inner/outer: colour distances fully transparent / fully opaque. Between
    them, the colour is unmixed from the key (an edge pixel is a blend of
    fish and key), then the key spill is removed: the excess of the key's
    strong channels over its weak ones.
    """
    rgb = np.asarray(frame.convert("RGB")).astype(np.float32)
    k = np.asarray(key, dtype=np.float32)
    distance = np.sqrt(((rgb - k) ** 2).sum(axis=-1))
    alpha = np.clip((distance - inner) / (outer - inner), 0.0, 1.0)

    a = alpha[..., None]
    unmixed = (rgb - (1.0 - a) * k) / np.maximum(a, 0.05)
    rgb = np.where((a > 0) & (a < 1), np.clip(unmixed, 0, 255), rgb)

    # Key spill: the excess of the key's strong channels over its weak ones.
    # On a translucent part (a fin over the key), it is the key seen through:
    # turned into transparency, then the colour is unmixed and clamped.
    strong = k >= 128
    if strong.any() and (~strong).any():
        spill = np.clip(
            rgb[..., strong].min(axis=-1) - rgb[..., ~strong].max(axis=-1), 0, None
        )
        contrast = max(1.0, float(k[strong].min() - k[~strong].max()))
        seen = np.clip(spill / contrast, 0.0, 0.8)[..., None]
        rgb = np.clip((rgb - seen * k) / (1.0 - seen), 0, 255)
        alpha = alpha * (1.0 - seen[..., 0])
        spilled = spill > 0
        weak_max = rgb[..., ~strong].max(axis=-1)
        for channel in np.nonzero(strong)[0]:
            rgb[..., channel] = np.where(
                spilled, np.minimum(rgb[..., channel], weak_max), rgb[..., channel]
            )

    alpha = drop_specks(alpha)
    out = np.dstack([np.clip(rgb, 0, 255), alpha * 255.0])
    return Image.fromarray((out + 0.5).astype(np.uint8), "RGBA")


def drop_specks(alpha: np.ndarray, min_share: float = 0.01) -> np.ndarray:
    """Clear the small blobs (dust, particles) around the subject.

    Keeps every blob of at least `min_share` of the largest one, with a
    small margin for its soft edges. Needs scipy; unchanged without it.
    """
    if ndimage is None:  # pragma: no cover - optional
        return alpha
    solid = alpha > 0.5
    labelled: Any = ndimage.label(solid)
    labels, count = labelled[0], int(labelled[1])
    if count <= 1:
        return alpha
    sizes = np.asarray(ndimage.sum(solid, labels, range(1, count + 1)))
    keep_ids = np.nonzero(sizes >= max(16.0, sizes.max() * min_share))[0] + 1
    keep = np.isin(labels, keep_ids)
    keep = ndimage.binary_dilation(keep, iterations=3)
    return alpha * keep


# ---------------------------------------------------------------------------
# Common scale
# ---------------------------------------------------------------------------


def alpha_box(frame: Image.Image) -> tuple[int, int, int, int] | None:
    """Box of the opaque part of a frame."""
    solid = np.asarray(frame.getchannel("A")) > 127
    return Image.fromarray(solid.astype(np.uint8) * 255).getbbox()


@dataclass
class Placement:
    """How a clip maps onto the frame: source centre and scale."""

    cx: float
    cy: float
    scale: float


def measure(frames: Sequence[Image.Image], once: bool) -> tuple[float, float, float]:
    """Length and centre of the subject in a clip (medians).

    A clip played once (a turn) is measured on its first frames, where the
    fish is still in profile.
    """
    sample = frames[: min(8, len(frames))] if once else frames
    boxes = [b for b in (alpha_box(f) for f in sample) if b]
    if not boxes:
        width, height = frames[0].size
        return float(width), width / 2.0, height / 2.0
    arr = np.asarray(boxes, dtype=np.float64)
    length = float(np.median(arr[:, 2] - arr[:, 0]))
    cx = float(np.median((arr[:, 0] + arr[:, 2]) / 2))
    cy = float(np.median((arr[:, 1] + arr[:, 3]) / 2))
    return max(1.0, length), cx, cy


def frame_size(
    clips: Sequence[Clip], placements: Sequence[Placement], pad: int = 2
) -> tuple[int, int]:
    """Smallest frame (multiple of 4) holding every placed frame."""
    half_w = half_h = 1.0
    for clip, place in zip(clips, placements, strict=True):
        for frame in clip.frames:
            box = alpha_box(frame) or frame.getchannel("A").getbbox()
            if not box:
                continue
            half_w = max(
                half_w,
                abs(box[0] - place.cx) * place.scale,
                abs(box[2] - place.cx) * place.scale,
            )
            half_h = max(
                half_h,
                abs(box[1] - place.cy) * place.scale,
                abs(box[3] - place.cy) * place.scale,
            )
    round4 = lambda v: int(math.ceil((2 * v + 2 * pad) / 4.0) * 4)
    return round4(half_w), round4(half_h)


def place(
    frame: Image.Image, where: Placement, size: tuple[int, int], zoom: float = 1.0
) -> Image.Image:
    """Scale a frame and centre its subject in a frame of `size`."""
    scale = where.scale * zoom
    width = max(1, round(frame.width * scale))
    height = max(1, round(frame.height * scale))
    # Resampling premultiplied, so the key colour does not bleed at edges
    resized = frame.convert("RGBa").resize((width, height), Image.Resampling.LANCZOS)
    left = round(where.cx * scale - size[0] / 2)
    top = round(where.cy * scale - size[1] / 2)
    return resized.crop((left, top, left + size[0], top + size[1])).convert("RGBA")


def prepare_fish(
    specs: Sequence[ClipSpec],
    key: tuple[int, int, int] | str | None = "auto",
    length: int = DEFAULT_LENGTH,
) -> tuple[list[Clip], list[Placement]]:
    """Read, cut out and measure the clips of a fish."""
    clips: list[Clip] = []
    placements: list[Placement] = []
    for spec in specs:
        frames = read_frames(spec.source, spec.range, spec.step)
        clip_key = detect_key(frames[0]) if key == "auto" else key
        if isinstance(clip_key, tuple):
            frames = [chroma_key(f, clip_key) for f in frames]
        mode = spec.resolved_mode
        clips.append(
            Clip(
                spec.name,
                frames,
                spec.fps,
                loop=mode != "once",
                pingpong=mode == "pingpong",
            )
        )
        if mode == "once" and spec.range and spec.range[0] > 0:
            # A turn cut out of a longer clip: measured on the source's first
            # frames, where the fish still swims in profile
            sample = read_frames(spec.source, (0, 7))
            if isinstance(clip_key, tuple):
                sample = [chroma_key(f, clip_key) for f in sample]
            size, cx, cy = measure(sample, once=True)
        else:
            size, cx, cy = measure(frames, once=mode == "once")
        placements.append(Placement(cx, cy, length / size))
    return clips, placements


# ---------------------------------------------------------------------------
# Loops
# ---------------------------------------------------------------------------


def _small(frames: Sequence[Image.Image] | np.ndarray, width: int = 160) -> np.ndarray:
    """Frames as small grey arrays, for comparisons."""
    if isinstance(frames, np.ndarray):
        return frames.astype(np.float32)
    first = frames[0]
    height = max(1, round(first.height * width / first.width))
    return np.stack(
        [
            np.asarray(f.convert("L").resize((width, height))).astype(np.float32)
            for f in frames
        ]
    )


def find_loops(
    frames: Sequence[Image.Image] | np.ndarray,
    min_len: int = 24,
    max_len: int = 120,
    count: int = 5,
    min_motion: float = 0.5,
) -> dict[str, Any]:
    """Best seamless loops of a clip, and whether it plays back and forth.

    A loop from..to is seamless when frame `to + 1` would look like frame
    `from` (and `to + 2` like `from + 1`, so that the motion goes on too).
    The score is relative to the change between two following frames: under
    about 1.5 the jump is not visible. Loops moving less than `min_motion`
    times a usual frame change (a pause of the tail) are left out.
    """
    small = _small(frames)
    n = len(small)
    step = [float(((small[i] - small[i + 1]) ** 2).mean()) for i in range(n - 1)]
    unit = max(1e-6, float(np.median(step))) if step else 1.0
    motion = np.concatenate([[0.0], np.cumsum(step)])

    found: list[tuple[float, int, int]] = []
    for first in range(n - 2):
        for length in range(min_len, min(max_len, n - 2 - first) + 1):
            last = first + length - 1
            moving = (motion[last] - motion[first]) / max(1, last - first) / unit
            if moving < min_motion:
                continue
            score = float(
                ((small[first] - small[last + 1]) ** 2).mean()
                + ((small[first + 1] - small[last + 2]) ** 2).mean()
            ) / (2 * unit)
            found.append((score, first, last))
    found.sort()
    loops: list[dict[str, Any]] = []
    for score, first, last in found:
        if all(abs(first - l["from"]) > 4 or abs(last - l["to"]) > 4 for l in loops):
            loops.append({"from": first, "to": last, "score": round(score, 2)})
        if len(loops) >= count:
            break

    # Ping-pong: frame i mirrors frame (pivot - i) all along
    pingpong = None
    if n >= 8:
        best = None
        for total in range(n - 6, n + 6):
            pairs = [
                (i, total - i)
                for i in range(n)
                if 0 <= total - i < n and i < total - i - 4
            ]
            if len(pairs) < n // 4:
                continue
            score = (
                float(np.mean([((small[a] - small[b]) ** 2).mean() for a, b in pairs]))
                / unit
            )
            if best is None or score < best[0]:
                best = (score, total)
        if best is not None and best[0] < 1.5:
            pingpong = {"pivot": best[1] / 2.0, "score": round(best[0], 2)}
    return {"frames": n, "loops": loops, "pingpong": pingpong}


# ---------------------------------------------------------------------------
# Automatic recipe
# ---------------------------------------------------------------------------

CLIP_EXTENSIONS = (".mp4", ".mov", ".webm", ".mkv", ".apng", ".png", ".webp", ".gif")
#: Wanted lengths, seconds: a swim loop, an idle loop, a turn
SWIM_LOOP_S = (1.0, 3.0)
IDLE_LOOP_S = (2.0, 4.0)
TURN_S = 1.2
#: Rate an idle clip is kept at (its motion is slow)
IDLE_FPS = 12
#: Most frames kept for a looping clip: more makes heavy atlases
MAX_LOOP_FRAMES = 72
#: A difference under this many usual frame changes is "the same picture"
SAME_PICTURE = 3.0


def probe_fps(source: Path) -> int:
    """Frame rate of a video (24 when unknown, or for PNG folders)."""
    if source.is_dir() or shutil.which("ffprobe") is None:
        return DEFAULT_FPS
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "csv=p=0",
            str(source),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        num, _, den = result.stdout.strip().partition("/")
        return max(1, round(float(num) / float(den or 1)))
    except ValueError:
        return DEFAULT_FPS


def read_small(source: Path, width: int = 160) -> tuple[np.ndarray, int]:
    """Small grey frames of a clip, for the analysis, and its frame rate.

    Videos are scaled down by ffmpeg, so that a long clip does not fill the
    memory with full size frames.
    """
    fps = probe_fps(source)
    if source.is_dir() or shutil.which("ffmpeg") is None:
        return _small(read_frames(source), width), fps
    try:
        # An animated picture Pillow reads natively
        image = Image.open(source)
        if int(getattr(image, "n_frames", 1)) > 1:
            return _small(read_frames(source), width), fps
    except OSError:
        pass
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-vsync",
                "0",
                "-vf",
                f"scale={width}:-2",
                str(Path(tmp) / "f%05d.png"),
            ],
            check=True,
        )
        return _small(read_frames(Path(tmp)), width), fps


def _diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(((a - b) ** 2).mean())


def _unit(small: np.ndarray) -> float:
    """Usual change between two following frames."""
    steps = [_diff(small[i], small[i + 1]) for i in range(len(small) - 1)]
    return max(1e-6, float(np.median(steps))) if steps else 1.0


def turn_window(small: np.ndarray) -> tuple[int, int]:
    """Frames where a turn clip actually turns.

    The clip starts on the reference picture and ends on its mirror. The
    turn starts at the last calm frame before the picture moves well away
    from the reference, and ends once the picture is close to the mirror
    (the fish may still settle slowly after that).
    """
    n = len(small)
    ref = small[0]
    mirror = ref[:, ::-1]
    span = max(1e-6, _diff(ref, mirror))
    away = np.asarray([_diff(f, ref) for f in small]) / span
    near = np.asarray([_diff(f, mirror) for f in small]) / span
    moving = np.nonzero(away > 0.5)[0]
    if not len(moving):
        return 0, n - 1
    first = int(moving[0])
    # Back to the calm frame the departure grows from
    start = first
    while start > 0 and away[start - 1] < away[start]:
        start -= 1
    arrived = np.nonzero(near[first:] < 0.3)[0]
    end = first + int(arrived[0]) if len(arrived) else n - 1
    return start, max(start + 1, end)


def _loop_choice(
    small: np.ndarray, fps: int, seconds: tuple[float, float]
) -> tuple[tuple[int, int], str]:
    """Range and mode of a looping clip."""
    n = len(small)
    unit = _unit(small)
    same = _diff(small[0], small[-1]) < SAME_PICTURE * unit
    whole = (0, n - 2 if same and n > 2 else n - 1)
    if n < seconds[0] * fps + 3:
        return whole, "loop"
    result = find_loops(
        small,
        min_len=max(2, round(seconds[0] * fps)),
        max_len=max(2, round(seconds[1] * fps)),
        count=1,
    )
    if result["pingpong"]:
        return (0, math.floor(result["pingpong"]["pivot"])), "pingpong"
    if result["loops"] and result["loops"][0]["score"] < 1.5:
        loop = result["loops"][0]
        return (loop["from"], loop["to"]), "loop"
    # The whole clip, without its last frame when it repeats the first one
    return whole, "loop"


def find_clip(folder: Path, name: str) -> Path | None:
    """The clip of that name in a species folder (any known extension)."""
    for path in sorted(folder.iterdir()):
        if path.stem.lower() != name:
            continue
        if path.is_dir() or path.suffix.lower() in CLIP_EXTENSIONS:
            return path
    return None


def auto_recipe(folder: Path, log: Any = print) -> dict[str, Any]:
    """Recipe of a species folder (swim, idle, turn clips), from the clips.

    Every clip is expected to start on the same reference picture; the swim
    and idle clips end on it too, the turn ends on its mirror. Departures
    are reported, not fatal.
    """
    swim = find_clip(folder, "swim")
    if swim is None:
        raise SystemExit(f"{folder}: no swim clip (swim.mp4, swim/ ...)")
    clips: dict[str, dict[str, Any]] = {}
    notes: list[str] = []
    reference: np.ndarray | None = None

    for name in ("swim", "idle", "turn"):
        source = find_clip(folder, name)
        if source is None:
            continue
        small, fps = read_small(source)
        unit = _unit(small)
        item: dict[str, Any] = {"source": source.name}
        if reference is not None and reference.shape == small[0].shape:
            if _diff(small[0], reference) > SAME_PICTURE * 4 * unit:
                notes.append(f"{name}: does not start on the swim's first picture")
        if name == "turn":
            mirror = small[0][:, ::-1]
            if _diff(small[-1], mirror) > SAME_PICTURE * 4 * unit:
                notes.append("turn: does not end on the mirrored first picture")
            start, end = turn_window(small)
            step = max(1, round((end - start + 1) / (TURN_S * fps)))
            item.update(fps=fps, mode="once", range=[start, end], step=step)
        else:
            if _diff(small[0], small[-1]) > SAME_PICTURE * 4 * unit:
                notes.append(f"{name}: does not end on its first picture")
            seconds = SWIM_LOOP_S if name == "swim" else IDLE_LOOP_S
            (first, last), mode = _loop_choice(small, fps, seconds)
            step = 1 if name == "swim" else max(1, round(fps / IDLE_FPS))
            count = last - first + 1
            if math.ceil(count / step) > MAX_LOOP_FRAMES:
                # No short loop found: the whole clip, at a lower rate
                step = math.ceil(count / MAX_LOOP_FRAMES)
                notes.append(
                    f"{name}: no {seconds[0]:g}-{seconds[1]:g} s loop, whole clip "
                    f"kept at {max(1, round(fps / step))} fps (a shorter clip "
                    "would play smoother)"
                )
            item.update(
                fps=max(1, round(fps / step)), mode=mode, range=[first, last], step=step
            )
        if name == "swim":
            reference = small[0]
        clips[name] = item
        frames = len(range(item["range"][0], item["range"][1] + 1, item["step"]))
        log(
            f"  {name}: frames {item['range'][0]}-{item['range'][1]}"
            f" step {item['step']} -> {frames} frames at {item['fps']} fps ({item['mode']})"
        )
    for warning in notes:
        log(f"  warning: {warning}")

    recipe: dict[str, Any] = {"length": DEFAULT_LENGTH, "clips": clips}
    meta_file = folder / "meta.json"
    if meta_file.exists():
        recipe["meta"] = json.loads(meta_file.read_text(encoding="utf-8"))
    return recipe


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


def common_box(frames: list[Image.Image]) -> tuple[int, int, int, int]:
    """Smallest box holding the visible part of every frame."""
    boxes = [f.getchannel("A").getbbox() for f in frames]
    boxes = [b for b in boxes if b]
    if not boxes:
        width, height = frames[0].size
        return (0, 0, width, height)
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def fit(frames: list[Image.Image], size: tuple[int, int]) -> list[Image.Image]:
    """Crop frames to their common visible box and fit them into `size`."""
    box = common_box(frames)
    out = []
    for frame in frames:
        cropped = frame.crop(box)
        scale = min(size[0] / cropped.width, size[1] / cropped.height)
        resized = cropped.resize(
            (
                max(1, round(cropped.width * scale)),
                max(1, round(cropped.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )
        canvas = Image.new("RGBA", size, (0, 0, 0, 0))
        canvas.paste(
            resized,
            ((size[0] - resized.width) // 2, (size[1] - resized.height) // 2),
        )
        out.append(canvas)
    return out


def build_atlas(
    clips: list[Clip], frame: tuple[int, int], columns: int | None = 8
) -> Atlas:
    """Lay the frames of every clip on one grid.

    columns: None picks a count giving a roughly square sheet.
    """
    all_frames: list[Image.Image] = []
    index: dict[str, dict[str, Any]] = {}
    for clip in clips:
        start = len(all_frames)
        all_frames += clip.frames
        entry: dict[str, Any] = {
            "from": start,
            "to": len(all_frames) - 1,
            "fps": clip.fps,
            "loop": clip.loop,
        }
        if clip.pingpong:
            entry["pingpong"] = True
        index[clip.name] = entry
    if columns is None:
        columns = math.ceil(math.sqrt(len(all_frames) * frame[1] / frame[0]))
    columns = max(1, min(columns, len(all_frames)))
    rows = math.ceil(len(all_frames) / columns)
    sheet = Image.new("RGBA", (frame[0] * columns, frame[1] * rows), (0, 0, 0, 0))
    for n, image in enumerate(all_frames):
        sheet.paste(image, ((n % columns) * frame[0], (n // columns) * frame[1]))
    return Atlas(frame=frame, columns=columns, sheet=sheet, clips=index)


def split_coral(sheet: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Split a coral sheet painted in pure R/G/B/grey into shade and mask.

    shade: brightness (max channel) + alpha.
    mask:  R/G/B = weights of palette colours 1/2/3 (saturated parts), the
           grey part (1 - saturation) goes to colour 4.
    """
    rgba = np.asarray(sheet.convert("RGBA")).astype(np.float32) / 255.0
    rgb, alpha = rgba[..., :3], rgba[..., 3]
    value = rgb.max(axis=-1)
    low = rgb.min(axis=-1)
    chroma = value - low
    saturation = np.where(value > 1e-6, chroma / np.maximum(value, 1e-6), 0.0)
    excess = rgb - low[..., None]
    total = excess.sum(axis=-1)
    weights = np.where(
        total[..., None] > 1e-6,
        excess / np.maximum(total, 1e-6)[..., None] * saturation[..., None],
        0.0,
    )

    shade = np.dstack([value, value, value, alpha])
    # 32 weight levels are plenty for a colour mix, and compress far better
    weights = np.round(weights * 32) / 32
    mask = np.dstack([weights, np.where(alpha > 0.02, 1.0, 0.0)])
    to_img = lambda a: Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))
    return to_img(shade), to_img(mask)


def save_webp(image: Image.Image, path: Path) -> None:
    """Lossy WebP keeping a clean alpha."""
    image.save(path, format="WEBP", quality=80, alpha_quality=90, method=6)


THUMB_SIZE = (96, 64)


def _palette_rgb(palette: Any) -> list[tuple[float, float, float]]:
    """Default colours of a coral palette (4 entries, grey when missing)."""
    colours: list[tuple[float, float, float]] = []
    for item in palette if isinstance(palette, list) else []:
        text = item.get("default", "") if isinstance(item, dict) else str(item)
        try:
            colours.append(
                tuple(int(text[i : i + 2], 16) / 255.0 for i in (1, 3, 5))  # type: ignore[arg-type]
            )
        except (ValueError, TypeError):
            colours.append((0.6, 0.6, 0.6))
    while len(colours) < 4:
        colours.append((0.6, 0.6, 0.6))
    return colours[:4]


def coral_preview(frame: Image.Image, palette: Any) -> Image.Image:
    """A coral frame (zones in pure R/G/B/grey) in its default colours."""
    shade, mask = split_coral(frame)
    value = np.asarray(shade).astype(np.float32) / 255.0
    weights = np.asarray(mask).astype(np.float32) / 255.0
    colours = np.asarray(_palette_rgb(palette), dtype=np.float32)
    w = weights[..., :3]
    rest = np.clip(1.0 - w.sum(axis=-1, keepdims=True), 0.0, 1.0)
    mix = w @ colours[:3] + rest * colours[3]
    rgb = mix * value[..., :1]
    out = np.dstack([rgb, value[..., 3]])
    return Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8), "RGBA")


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


def write_entry(
    kind: str,
    entry_id: str,
    clips: list[Clip],
    out: Path,
    frame: tuple[int, int],
    columns: int | None,
    meta: dict[str, Any],
    hd: bool = False,
    placements: Sequence[Placement] | None = None,
) -> Path:
    """Write the atlas pictures and the descriptor of a catalog entry.

    placements: for a fish, how every clip maps onto the frame (common
    scale); without them, frames are fitted into the frame size.
    hd: also write a double resolution atlas (for very large sprites).
    """
    folder = out / entry_id
    folder.mkdir(parents=True, exist_ok=True)

    def scaled(zoom: int) -> list[Clip]:
        size = (frame[0] * zoom, frame[1] * zoom)
        out_clips = []
        for n, c in enumerate(clips):
            frames = (
                [place(f, placements[n], size, zoom) for f in c.frames]
                if placements is not None
                else fit(c.frames, size)
            )
            out_clips.append(Clip(c.name, frames, c.fps, c.loop, c.pingpong))
        return out_clips

    atlas_2x = (
        build_atlas(scaled(2), (frame[0] * 2, frame[1] * 2), columns) if hd else None
    )
    atlas_1x = build_atlas(scaled(1), frame, columns)

    descriptor: dict[str, Any] = {
        "kind": "fish" if kind == "fish" else "coral",
        "frame": list(frame),
        "columns": atlas_1x.columns,
        "clips": atlas_1x.clips,
    }
    if kind == "fish":
        save_webp(atlas_1x.sheet, folder / f"{entry_id}.webp")
        descriptor["atlas"] = {"1x": f"{entry_id}.webp"}
        if atlas_2x is not None:
            save_webp(atlas_2x.sheet, folder / f"{entry_id}@2x.webp")
            descriptor["atlas"]["2x"] = f"{entry_id}@2x.webp"
        descriptor.setdefault("facing", "right")
    else:
        shade, mask = split_coral(atlas_1x.sheet)
        save_webp(shade, folder / f"{entry_id}.shade.webp")
        mask.save(folder / f"{entry_id}.mask.png", optimize=True)
        descriptor["atlas"] = {
            "shade": f"{entry_id}.shade.webp",
            "mask": f"{entry_id}.mask.png",
        }
        if atlas_2x is not None:
            shade2, mask2 = split_coral(atlas_2x.sheet)
            save_webp(shade2, folder / f"{entry_id}.shade@2x.webp")
            mask2.save(folder / f"{entry_id}.mask@2x.png", optimize=True)
            descriptor["atlas"]["shade_2x"] = f"{entry_id}.shade@2x.webp"
            descriptor["atlas"]["mask_2x"] = f"{entry_id}.mask@2x.png"
    descriptor.update(meta)

    # Small picture for the editor's lists: the first frame of the swim (or
    # day) clip, a coral in its default colours
    first_clip = next(
        (c for c in atlas_1x.clips if c in ("swim", "day")), next(iter(atlas_1x.clips))
    )
    index = atlas_1x.clips[first_clip]["from"]
    x = (index % atlas_1x.columns) * frame[0]
    y = (index // atlas_1x.columns) * frame[1]
    first = atlas_1x.sheet.crop((x, y, x + frame[0], y + frame[1]))
    if kind != "fish":
        first = coral_preview(first, descriptor.get("palette"))
    thumbnail(first).save(folder / "thumb.webp", format="WEBP", quality=85, method=6)
    descriptor["thumbnail"] = "thumb.webp"

    (folder / "species.json").write_text(
        json.dumps(descriptor, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return folder


def _print_loops(source: Path, min_len: int, max_len: int) -> int:
    """`loops` command: best loops of a clip."""
    small, _ = read_small(source)
    result = find_loops(small, min_len, max_len)
    print(f"{source}: {result['frames']} frames")
    if result["pingpong"]:
        pivot = result["pingpong"]["pivot"]
        print(
            f"  ping-pong clip (plays forward then backward around frame "
            f"{pivot:g}): keep 0-{math.floor(pivot)} and use mode pingpong"
        )
    print("  best loops (score < 1.5: seamless):")
    for loop in result["loops"]:
        length = loop["to"] - loop["from"] + 1
        print(
            f"    {loop['from']}-{loop['to']}  ({length} frames)  score {loop['score']}"
        )
    return 0


def write_auto_recipe(folder: Path, entry_id: str) -> Path:
    """Analyse a species folder and write its recipe, `<folder>/<id>.json`."""
    print(f"{entry_id}: analysing {folder}")
    recipe = auto_recipe(folder)
    path = folder / f"{entry_id}.json"
    path.write_text(
        json.dumps(recipe, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"  recipe written to {path}")
    return path


def _batch(argv: list[str]) -> int:
    """`batch` command: every species folder of a directory."""
    batch = argparse.ArgumentParser(
        prog="build_atlas.py batch",
        description=(
            "Build every species folder (named after the species id, holding "
            "swim/idle/turn clips). A folder's recipe <id>.json is reused when "
            "present (so that hand edits stay), else written from the clips."
        ),
    )
    batch.add_argument("folder", type=Path, help="directory of species folders")
    batch.add_argument("--out", type=Path, required=True)
    batch.add_argument(
        "--reanalyse", action="store_true", help="rewrite the recipes from the clips"
    )
    batch.add_argument("--hd", action="store_true")
    bargs, extra = batch.parse_known_args(argv)
    failed = []
    done = 0
    for folder in sorted(p for p in bargs.folder.iterdir() if p.is_dir()):
        if find_clip(folder, "swim") is None:
            continue
        entry_id = folder.name
        recipe = folder / f"{entry_id}.json"
        args = ["fish", entry_id, "--out", str(bargs.out), *extra]
        if bargs.hd:
            args.append("--hd")
        try:
            if bargs.reanalyse or not recipe.exists():
                write_auto_recipe(folder, entry_id)
            main([*args, "--recipe", str(recipe)])
            done += 1
        except (SystemExit, subprocess.CalledProcessError, OSError, ValueError) as err:
            print(f"{entry_id}: failed: {err}")
            failed.append(entry_id)
    print(
        f"{done} built, {len(failed)} failed{': ' + ', '.join(failed) if failed else ''}"
    )
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    """Command line entry point."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["batch"]:
        return _batch(argv[1:])
    if argv[:1] == ["loops"]:
        loops = argparse.ArgumentParser(
            prog="build_atlas.py loops", description="Best seamless loops of a clip"
        )
        loops.add_argument("source", type=Path)
        loops.add_argument("--min", type=int, default=24, help="shortest loop")
        loops.add_argument("--max", type=int, default=120, help="longest loop")
        largs = loops.parse_args(argv[1:])
        return _print_loops(largs.source, largs.min, largs.max)

    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("kind", choices=["fish", "coral"])
    parser.add_argument("id", help="catalog id (folder name), ex: chromis_viridis")
    parser.add_argument(
        "--clip",
        action="append",
        default=[],
        help="name=source[:fps[:mode[:from-to[:step]]]]",
    )
    parser.add_argument("--recipe", type=Path, help="JSON recipe (clips, key, meta)")
    parser.add_argument(
        "--auto",
        type=Path,
        help="species folder (swim/idle/turn clips): write its recipe, then build",
    )
    parser.add_argument("--meta", type=Path, help="JSON merged into the descriptor")
    parser.add_argument(
        "--frame",
        default=None,
        help="frame size WxH (default: fitted for a fish, 256x256 for a coral)",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=None,
        help=f"fish length in the 1x atlas, pixels (default {DEFAULT_LENGTH})",
    )
    parser.add_argument(
        "--key", default=None, help="key colour: auto (default), none or #rrggbb"
    )
    parser.add_argument(
        "--columns", type=int, default=None, help="atlas columns (default: square)"
    )
    parser.add_argument(
        "--hd", action="store_true", help="also write a double resolution atlas"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.auto:
        if args.kind != "fish":
            parser.error("--auto is for fish")
        args.recipe = write_auto_recipe(args.auto, args.id)
    recipe = read_recipe(args.recipe) if args.recipe else {}
    specs: list[ClipSpec] = list(recipe.get("clips", []))
    for text in args.clip:
        spec = parse_clip(text)
        specs = [s for s in specs if s.name != spec.name] + [spec]
    if not specs:
        parser.error("no clip: give --clip or --recipe")

    meta: dict[str, Any] = dict(recipe.get("meta", {}))
    if args.meta:
        meta.update(json.loads(args.meta.read_text(encoding="utf-8")))

    frame_spec = args.frame or recipe.get("frame")
    if isinstance(frame_spec, list):
        frame_spec = f"{frame_spec[0]}x{frame_spec[1]}"

    placements: list[Placement] | None = None
    if args.kind == "fish":
        key = parse_key(str(args.key or recipe.get("key", "auto")))
        length = int(args.length or recipe.get("length", DEFAULT_LENGTH))
        clips, placements = prepare_fish(specs, key, length)
        if frame_spec:
            width, height = (int(v) for v in frame_spec.lower().split("x"))
        else:
            width, height = frame_size(clips, placements)
        meta.setdefault("length_frac", round(length / width, 4))
    else:
        width, height = (int(v) for v in (frame_spec or "256x256").lower().split("x"))
        clips = []
        for spec in specs:
            mode = spec.resolved_mode
            clips.append(
                Clip(
                    spec.name,
                    read_frames(spec.source, spec.range, spec.step),
                    spec.fps,
                    loop=mode != "once",
                    pingpong=mode == "pingpong",
                )
            )

    folder = write_entry(
        args.kind,
        args.id,
        clips,
        args.out,
        (width, height),
        args.columns if args.columns is not None else recipe.get("columns"),
        meta,
        args.hd,
        placements,
    )
    print(
        f"written {folder}: frame {width}x{height}, {sum(len(c.frames) for c in clips)} frames"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
