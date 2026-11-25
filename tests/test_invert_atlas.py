"""The atlas of an invertebrate, made from one photo on a flat background."""

from __future__ import annotations

import importlib
import io
import json
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
atlas = importlib.import_module("invert_atlas")
validate = importlib.import_module("validate")

MAGENTA = (255, 0, 255)
GREEN = (0, 255, 0)
SOURCE = {"type": "ai-generated", "model": "gemini-x", "date": "2026-11-24"}
LANGUAGES = ("en", "fr", "de", "es", "it", "nl", "pl", "pt")


def animal(
    key: tuple[int, int, int] = MAGENTA,
    size: tuple[int, int] = (900, 600),
    box: tuple[float, float, float, float] = (0.1, 0.3, 0.9, 0.7),
) -> Image.Image:
    """A flat background with an animal on it (a box of shell)."""
    image = Image.new("RGB", size, key)
    w, h = size
    ImageDraw.Draw(image).rectangle(
        (w * box[0], h * box[1], w * box[2], h * box[3]), fill=(190, 120, 60)
    )
    return image


def entry(folder: Path, movement: str = "ground") -> Path:
    """A described invertebrate, with no picture yet."""
    folder.mkdir(parents=True, exist_ok=True)
    desc = {
        "kind": "invertebrate",
        "names": dict.fromkeys(LANGUAGES, "Nerite"),
        "scientific": "Nerita sp.",
        "water_types": ["marine"],
        "size_cm": [1, 2],
        "role": ["algae_eater"],
        "movement": movement,
    }
    (folder / "species.json").write_text(json.dumps(desc), encoding="utf-8")
    return folder


def test_a_wide_animal_is_judged_on_its_width_not_its_height() -> None:
    # 80 % of the width but 40 % of the height: a shrimp lying across
    assert atlas.check_invert_photo(animal(), None) == []
    assert atlas.check_invert_photo(animal(GREEN), "green") == []


def test_the_problems_are_told_for_an_invertebrate() -> None:
    small = atlas.check_invert_photo(animal(box=(0.45, 0.45, 0.55, 0.55)), None)
    assert small == ["the invertebrate is too small in the picture (under 40%)"]
    cut = atlas.check_invert_photo(animal(box=(0.0, 0.3, 0.9, 0.7)), None)
    assert any("the invertebrate touches the edge" in p for p in cut)
    blank = atlas.check_invert_photo(Image.new("RGB", (900, 600), MAGENTA), None)
    assert any("no invertebrate found" in p for p in blank)


def test_an_animal_seen_from_the_side_stands_on_the_bottom(tmp_path: Path) -> None:
    folder = entry(tmp_path / "invertebrates" / "nerita")
    desc = json.loads((folder / "species.json").read_text())
    still = atlas.build_invert_still(animal(), "nerita", desc, SOURCE)
    assert set(still.files) == {"nerita.webp", "thumb.webp"}
    keys = still.keys
    assert keys["kind"] == "invertebrate" and keys["degraded"] is True
    assert keys["frame"] == [atlas.FRAME, atlas.FRAME] and keys["columns"] == 1
    assert keys["clips"] == {"still": {"from": 0, "to": 0, "fps": 1, "loop": False}}
    assert keys["atlas"] == {"1x": "nerita.webp"} and keys["source"] == SOURCE
    frame = Image.open(io.BytesIO(still.files["nerita.webp"]))
    assert frame.size == (atlas.FRAME, atlas.FRAME)
    box = frame.getchannel("A").point(lambda a: 255 if a > 128 else 0).getbbox()
    assert box is not None
    assert box[3] >= atlas.FRAME - atlas.PAD - 3  # on the bottom of the frame
    assert abs((box[0] + box[2]) / 2 - atlas.FRAME / 2) <= 3  # centred across


def test_an_animal_seen_from_above_lies_in_the_middle() -> None:
    desc = {"movement": "ground"}
    still = atlas.build_invert_still(animal(), "archaster_typicus", desc, SOURCE)
    frame = Image.open(io.BytesIO(still.files["archaster_typicus.webp"]))
    box = frame.getchannel("A").point(lambda a: 255 if a > 128 else 0).getbbox()
    assert box is not None
    # A starfish is shot from above: it floats in its frame, off the bottom
    assert box[3] < atlas.FRAME - atlas.PAD - 3
    assert abs((box[1] + box[3]) / 2 - atlas.FRAME / 2) <= 3


def test_an_empty_picture_still_gives_a_frame() -> None:
    blank = atlas.build_invert_still(
        Image.new("RGB", (900, 600), MAGENTA), "x", {}, SOURCE
    )
    assert blank.keys["frame"] == [atlas.FRAME, atlas.FRAME]


def test_the_atlas_written_passes_the_validation(tmp_path: Path) -> None:
    folder = entry(tmp_path / "invertebrates" / "nerita")
    desc = json.loads((folder / "species.json").read_text())
    desc["atlas"] = {"1x": "old.webp"}  # a generated key is replaced, the rest kept
    (folder / "species.json").write_text(json.dumps(desc), encoding="utf-8")
    written = atlas.write_invert_atlas(animal(), folder, SOURCE)
    assert written["scientific"] == "Nerita sp." and written["atlas"] == {
        "1x": "nerita.webp"
    }
    assert written["role"] == ["algae_eater"] and written["movement"] == "ground"
    # the entry is complete for the catalog: names, picture, source
    assert validate.check_entry(folder, "invertebrates") == []


def test_an_entry_with_no_descriptor_yet_is_fitted_all_the_same(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "invertebrates" / "newcomer"
    written = atlas.write_invert_atlas(animal(), folder, SOURCE)
    assert written["atlas"] == {"1x": "newcomer.webp"}


def test_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    folder = entry(tmp_path / "invertebrates" / "nerita")
    photo = tmp_path / "photo.png"
    animal().save(photo)
    args = [str(photo), str(folder), "--source", "ai-generated", "--model", "gemini-x"]
    assert atlas.main(args) == 0
    assert "degraded atlas [320, 320]" in capsys.readouterr().out
    assert (
        json.loads((folder / "species.json").read_text())["source"]["model"]
        == "gemini-x"
    )
    bad = tmp_path / "bad.png"
    animal(box=(0.45, 0.45, 0.55, 0.55)).save(bad)
    other = tmp_path / "invertebrates" / "other"
    assert atlas.main([str(bad), str(other), "--source", "photo"]) == 1
    assert "too small" in capsys.readouterr().out and not other.exists()
    assert atlas.main([str(bad), str(other), "--source", "photo", "--force"]) == 0
