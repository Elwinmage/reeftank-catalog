"""Coral entries from natural colour clips: cut out, zones, atlases."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
coral_natural = importlib.import_module("coral_natural")
validate = importlib.import_module("validate")

KEY = (247, 46, 236)
SIZE = (160, 120)
STALK = (70, 75, 95)  # cool grey-blue skeleton
TENTACLE = (120, 85, 60)  # warm brown
TIP = (170, 230, 170)  # bright green


def coral_frame(
    reach: int, sway: int, cast: tuple[float, float, float] = (1, 1, 1)
) -> Image.Image:
    """A coral on the key colour: a stalk from the bottom, tentacles of
    `reach` pixels around its top (moved by `sway`), green tips."""
    image = Image.new("RGB", SIZE, KEY)
    draw = ImageDraw.Draw(image)
    tint = lambda c: tuple(min(255, round(v * k)) for v, k in zip(c, cast))
    draw.rectangle((65, 60, 95, SIZE[1]), fill=tint(STALK))
    for n in range(7):
        x = 50 + n * 10 + sway
        top = 58 - reach
        draw.line((80, 62, x, top), fill=tint(TENTACLE), width=4)
        draw.ellipse((x - 3, top - 3, x + 3, top + 3), fill=tint(TIP))
    return image


def write_clip(folder: Path, frames: list[Image.Image]) -> Path:
    folder.mkdir(parents=True)
    for n, frame in enumerate(frames):
        frame.save(folder / f"f{n:03d}.png")
    return folder


@pytest.fixture
def clips(tmp_path: Path) -> dict[str, Path]:
    day = write_clip(
        tmp_path / "day", [coral_frame(40, s) for s in (0, 3, 6, 3, 0, -3)]
    )
    # Closing, with a blue cast growing (as the generated clips do)
    close = write_clip(
        tmp_path / "close",
        [coral_frame(40 - 9 * n, 0, (1 - 0.05 * n, 1, 1 + 0.15 * n)) for n in range(5)],
    )
    meta = tmp_path / "meta.json"
    names = {lang: f"Coral {lang}" for lang in validate.LANGUAGES}
    meta.write_text(
        json.dumps(
            {
                "names": names,
                "scientific": "Euphyllia glabrescens",
                "size_cm": [10, 30],
                "palette": [
                    {"fluo": False},
                    {"fluo": True},
                    {"default": "#123456", "fluo": False},
                ],
            }
        ),
        encoding="utf-8",
    )
    return {"day": day, "close": close, "meta": meta, "out": tmp_path / "corals"}


def run(clips: dict[str, Path], *extra: str) -> Path:
    assert (
        coral_natural.main(
            [
                "euphyllia_glabrescens",
                "--day",
                str(clips["day"]),
                "--close",
                str(clips["close"]),
                "--day-range",
                "0-5",
                "--close-range",
                "0-4:1",
                "--meta",
                str(clips["meta"]),
                "--out",
                str(clips["out"]),
                *extra,
            ]
        )
        == 0
    )
    return clips["out"] / "euphyllia_glabrescens"


def test_builds_a_valid_entry(clips: dict[str, Path]) -> None:
    folder = run(clips, "--columns", "4")
    assert validate.check_entry(folder, "corals") == []
    # Only the shipped files, no preview left behind
    assert sorted(p.name for p in folder.iterdir()) == [
        "euphyllia_glabrescens.mask.png",
        "euphyllia_glabrescens.shade.webp",
        "species.json",
        "thumb.webp",
    ]
    desc = json.loads((folder / "species.json").read_text(encoding="utf-8"))
    assert desc["kind"] == "coral"
    assert desc["columns"] == 4
    assert desc["clips"] == {
        "day": {"from": 0, "to": 5, "fps": 12, "loop": True, "pingpong": True},
        "close": {"from": 6, "to": 10, "fps": 12, "loop": False},
    }
    # Frame: 256 wide, the ratio of the coral, multiple of 4
    width, height = desc["frame"]
    assert width == coral_natural.FRAME_WIDTH and height % 4 == 0
    # Measured defaults fill the gaps, a given default is kept
    tentacles, tips, stalk = (p["default"] for p in desc["palette"])
    r, g, b = (int(tentacles[i : i + 2], 16) for i in (1, 3, 5))
    assert r > b  # warm
    r, g, b = (int(tips[i : i + 2], 16) for i in (1, 3, 5))
    assert g > r and g > b  # green
    assert stalk == "#123456"


def test_zones(clips: dict[str, Path]) -> None:
    folder = run(clips)
    desc = json.loads((folder / "species.json").read_text(encoding="utf-8"))
    width, height = desc["frame"]
    mask = np.asarray(Image.open(folder / "euphyllia_glabrescens.mask.png"))
    # First day frame, last close frame (blue cast corrected)
    columns = desc["columns"]
    for index in (desc["clips"]["day"]["from"], desc["clips"]["close"]["to"]):
        x = (index % columns) * width
        y = (index // columns) * height
        frame = mask[y : y + height, x : x + width]
        # The bottom middle is skeleton (colour 3), whatever the cast
        foot = frame[height - 4, width // 2]
        assert foot[2] > 200 and foot[0] < 40, foot
        assert frame[0, 0, 3] == 0  # key cut out
    # Somewhere tentacles (colour 1) and tips (colour 2)
    assert (mask[..., 0] > 200).any()
    assert (mask[..., 1] > 200).any()


def test_no_key(clips: dict[str, Path], tmp_path: Path) -> None:
    grey = write_clip(tmp_path / "grey", [Image.new("RGB", SIZE, (128, 128, 128))])
    clips["day"] = grey
    with pytest.raises(SystemExit, match="no key colour"):
        run(clips)


def test_helpers() -> None:
    values = coral_natural.smoothstep(np.asarray([0.0, 0.5, 1.0]), 0.0, 1.0)
    assert values.tolist() == [0.0, 0.5, 1.0]
    flat = np.ones((5, 5), dtype=np.float32)
    assert np.allclose(coral_natural.blur(flat, 1.0), flat)
