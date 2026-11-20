"""The atlas of a plant, made from one photo on a flat background."""

from __future__ import annotations

import importlib
import io
import json
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
atlas = importlib.import_module("plant_atlas")
validate = importlib.import_module("validate")

MAGENTA = (255, 0, 255)
GREEN = (0, 255, 0)
SOURCE = {"type": "ai-generated", "model": "gemini-x", "date": "2026-11-20"}


def plant(
    key: tuple[int, int, int] = GREEN,
    size: tuple[int, int] = (600, 800),
    box: tuple[float, float, float, float] = (0.35, 0.1, 0.65, 0.9),
) -> Image.Image:
    """A flat background with a tall plant (a box of leaves)."""
    image = Image.new("RGB", size, key)
    w, h = size
    ImageDraw.Draw(image).rectangle(
        (w * box[0], h * box[1], w * box[2], h * box[3]), fill=(40, 140, 30)
    )
    return image


def test_a_tall_plant_is_judged_on_its_height_not_its_width() -> None:
    # 30 % of the width but 80 % of the height: a fine stem plant
    assert atlas.check_plant_photo(plant(), "green") == []
    assert atlas.check_plant_photo(plant(MAGENTA), None) == []


def test_the_problems_are_told_for_a_plant() -> None:
    small = atlas.check_plant_photo(plant(box=(0.45, 0.45, 0.55, 0.55)), "green")
    assert small == ["the plant is too small in the picture (under 40%)"]
    cut = atlas.check_plant_photo(plant(box=(0.35, 0.0, 0.65, 0.9)), "green")
    assert any("the plant touches the edge" in p for p in cut)
    blank = atlas.check_plant_photo(Image.new("RGB", (600, 800), GREEN), "green")
    assert any("no plant found" in p for p in blank)
    assert atlas.check_plant_photo(Image.new("RGB", (100, 100), GREEN), "green")


def test_a_plant_stands_on_the_bottom_of_a_square_frame() -> None:
    still = atlas.build_plant_still(plant(), "egeria_densa", SOURCE, "green")
    assert set(still.files) == {"egeria_densa.webp", "thumb.webp"}
    keys = still.keys
    assert keys["kind"] == "plant" and keys["degraded"] is True
    assert keys["frame"] == [atlas.FRAME, atlas.FRAME] and keys["columns"] == 1
    assert keys["clips"] == {"still": {"from": 0, "to": 0, "fps": 1, "loop": False}}
    assert keys["atlas"] == {"1x": "egeria_densa.webp"} and keys["source"] == SOURCE
    frame = Image.open(io.BytesIO(still.files["egeria_densa.webp"]))
    assert frame.size == (atlas.FRAME, atlas.FRAME)
    box = frame.getchannel("A").point(lambda a: 255 if a > 128 else 0).getbbox()
    assert box is not None
    assert box[3] >= atlas.FRAME - atlas.PAD - 3 and box[1] <= atlas.PAD + 3
    assert abs((box[0] + box[2]) / 2 - atlas.FRAME / 2) <= 3  # centred
    # an empty picture (no plant found) still gives a frame
    blank = atlas.build_plant_still(
        Image.new("RGB", (600, 800), GREEN), "x", SOURCE, "green"
    )
    assert blank.keys["frame"] == [atlas.FRAME, atlas.FRAME]


def test_the_atlas_written_passes_the_validation(tmp_path: Path) -> None:
    folder = tmp_path / "plants" / "egeria_densa"
    folder.mkdir(parents=True)
    desc = {
        "kind": "plant",
        "names": dict.fromkeys(
            ("en", "fr", "de", "es", "it", "nl", "pl", "pt"), "Egeria"
        ),
        "scientific": "Egeria densa",
        "water_types": ["freshwater"],
        "size_cm": [30, 100],
        "zone": "background",
        "atlas": {"1x": "old.webp"},  # a generated key is replaced, the rest kept
    }
    (folder / "species.json").write_text(json.dumps(desc))
    written = atlas.write_plant_atlas(plant(), folder, SOURCE, "green")
    assert written["scientific"] == "Egeria densa" and written["atlas"] == {
        "1x": "egeria_densa.webp"
    }
    # the entry is complete for the catalog: names, picture, source
    assert validate.check_entry(folder, "plants") == []


def test_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    folder = tmp_path / "plants" / "egeria_densa"
    photo = tmp_path / "photo.png"
    plant().save(photo)
    args = [str(photo), str(folder), "--source", "ai-generated", "--model", "gemini-x"]
    assert atlas.main(args) == 0
    assert "degraded atlas [320, 320]" in capsys.readouterr().out
    assert (
        json.loads((folder / "species.json").read_text())["source"]["model"]
        == "gemini-x"
    )
    bad = tmp_path / "bad.png"
    plant(box=(0.45, 0.45, 0.55, 0.55)).save(bad)
    other = tmp_path / "plants" / "other"
    assert atlas.main([str(bad), str(other), "--source", "photo"]) == 1
    assert "too small" in capsys.readouterr().out and not other.exists()
    assert atlas.main([str(bad), str(other), "--source", "photo", "--force"]) == 0
