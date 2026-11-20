"""Degraded atlas made from a still photo."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
still_atlas = importlib.import_module("still_atlas")
validate = importlib.import_module("validate")

MAGENTA = (255, 0, 255)
GREEN = (0, 255, 0)


def photo(
    key: tuple[int, int, int] = MAGENTA,
    fish: tuple[int, int, int] = (230, 120, 20),
    size: tuple[int, int] = (640, 400),
    margin: int = 100,
) -> Image.Image:
    """A flat background with an ellipse and a tail."""
    image = Image.new("RGB", size, key)
    draw = ImageDraw.Draw(image)
    w, h = size
    draw.ellipse((margin + 60, h * 0.3, w - margin, h * 0.7), fill=fish)
    draw.polygon(
        [(margin + 80, h / 2), (margin, h * 0.3), (margin, h * 0.7)], fill=fish
    )
    return image


def test_prompt() -> None:
    text = still_atlas.render_prompt("Betta splendens")
    assert text.startswith("Realistic side view of Betta splendens, head to the RIGHT")
    assert "flat magenta #FF00FF background" in text
    assert "no magenta tint" in text
    assert "flat green #00FF00" in still_atlas.render_prompt("Guppy", "green")
    assert "[" not in text
    with pytest.raises(ValueError, match="key must be one of"):
        still_atlas.render_prompt("Guppy", "blue")


def test_key_detection_and_removal() -> None:
    import numpy as np

    assert still_atlas.detect_key(np.asarray(photo(MAGENTA), dtype=float)) == "magenta"
    assert still_atlas.detect_key(np.asarray(photo(GREEN), dtype=float)) == "green"
    assert still_atlas.detect_key(np.full((50, 50, 3), 128.0)) is None
    out = still_atlas.key_out(photo())
    assert out.getpixel((2, 2))[3] == 0
    assert out.getpixel((320, 200))[3] == 255
    assert out.getpixel((320, 200))[:3] == (230, 120, 20)
    # a pink fish survives a green background
    pink = still_atlas.key_out(photo(GREEN, fish=(255, 60, 160)))
    assert pink.getpixel((320, 200)) == (255, 60, 160, 255)
    assert pink.getpixel((2, 2))[3] == 0


def test_photo_checks() -> None:
    assert still_atlas.check_photo(photo()) == []
    assert still_atlas.check_photo(photo(GREEN), "green") == []
    small = still_atlas.check_photo(photo(size=(300, 200), margin=40))
    assert small == ["photo too small: 300 px wide (at least 512)"]
    assert still_atlas.check_photo(Image.new("RGB", (30, 30), MAGENTA)) == [
        "photo too small: 30 px wide (at least 512)",
        "photo too small to check its corners",
    ]
    assert still_atlas.check_photo(Image.new("RGB", (600, 600), (90, 90, 90))) == [
        "corners are not a flat magenta or green background"
    ]
    assert still_atlas.check_photo(Image.new("RGB", (600, 600), MAGENTA)) == [
        "no fish found on the background"
    ]
    cut = photo(margin=0)
    assert "touches the edge" in still_atlas.check_photo(cut)[0]
    tiny = photo(margin=250)
    assert "too small in the picture" in still_atlas.check_photo(tiny)[0]
    noisy = photo()
    for x in range(10):
        for y in range(10):  # three corners are not flat (one or two may be marked)
            for dx, dy in ((0, 0), (630, 0), (0, 390)):
                noisy.putpixel((x + dx, y + dy), (255 - 5 * x * y % 200, 0, 0))
    assert "background is not uniform in the corners" in (
        still_atlas.check_photo(noisy, "magenta")
    )


def test_atlas_of_a_photo(tmp_path: Path) -> None:
    folder = tmp_path / "fish" / "betta"
    source = {"type": "ai-generated", "model": "gemini"}
    (folder).mkdir(parents=True)
    (folder / "species.json").write_text(
        json.dumps({"kind": "fish", "names": {"en": "Betta"}, "clips": {"old": 1}})
    )
    descriptor = still_atlas.write_atlas(photo(), folder, source, 160, None)
    assert descriptor["degraded"] is True and descriptor["source"] == source
    assert descriptor["names"] == {"en": "Betta"}  # described fields kept
    assert set(descriptor["clips"]) == {"swim", "idle", "turn"}
    assert descriptor["clips"]["turn"]["to"] == 113
    assert descriptor["columns"] == 8 and descriptor["frame"][0] == 208
    sheet = Image.open(folder / "betta.webp")
    assert sheet.width == 8 * 208 and sheet.height % descriptor["frame"][1] == 0
    assert 0.5 < descriptor["length_frac"] < 1
    # valid in the catalog once described
    problems = validate.check_entry(folder, "fish")
    assert all(p.startswith("to describe") for p in problems)
    (folder / "species.json").write_text("[]")
    assert still_atlas.write_atlas(photo(GREEN), folder, source, 160, "green")


def test_command(tmp_path: Path, capsys: Any) -> None:
    path = tmp_path / "betta.png"
    photo().save(path)
    folder = tmp_path / "fish" / "betta"
    assert still_atlas.main(["--prompt", "Guppy", "--key", "green"]) == 0
    assert "flat green" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        still_atlas.main([str(path)])
    args = [str(path), str(folder), "--source", "photo"]
    assert still_atlas.main(args) == 0
    assert "degraded atlas" in capsys.readouterr().out
    data = json.loads((folder / "species.json").read_text())
    assert data["source"]["type"] == "photo" and data["source"]["date"]
    # the same photo again over the degraded atlas: fine
    full = [*args, "--author", "Jane", "--license", "CC0", "--date", "2026-11-19"]
    assert still_atlas.main(full) == 0
    data = json.loads((folder / "species.json").read_text())
    assert data["source"] == {
        "type": "photo",
        "author": "Jane",
        "license": "CC0",
        "date": "2026-11-19",
    }
    # a real animation is not replaced
    data.pop("degraded")
    (folder / "species.json").write_text(json.dumps(data))
    assert still_atlas.main(args) == 1
    assert "already has a real animation" in capsys.readouterr().out
    assert still_atlas.main([*args, "--force"]) == 0
    # an unfit photo is refused, unless forced
    bad = tmp_path / "bad.png"
    Image.new("RGB", (600, 600), (90, 90, 90)).save(bad)
    other = tmp_path / "fish" / "bad"
    assert still_atlas.main([str(bad), str(other), "--source", "photo"]) == 1
    assert "not a flat magenta" in capsys.readouterr().out
    assert not other.exists()


def test_real_animation_detection(tmp_path: Path) -> None:
    assert not still_atlas.has_real_animation(tmp_path)
    file = tmp_path / "species.json"
    file.write_text("[]")
    assert not still_atlas.has_real_animation(tmp_path)
    file.write_text('{"names": {}}')
    assert not still_atlas.has_real_animation(tmp_path)
    file.write_text('{"atlas": {"1x": "x.webp"}, "degraded": true}')
    assert not still_atlas.has_real_animation(tmp_path)
    file.write_text('{"atlas": {"1x": "x.webp"}}')
    assert still_atlas.has_real_animation(tmp_path)
