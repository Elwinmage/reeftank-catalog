"""Catalog checks and release files."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
validate = importlib.import_module("validate")
make_release = importlib.import_module("make_release")

REPO = Path(__file__).parent.parent


def described(kind: str, entry_id: str) -> dict[str, Any]:
    """A complete description (names in every language, size, behaviour)."""
    names = {lang: f"{entry_id.title()} {lang}" for lang in validate.LANGUAGES}
    names["en"] = entry_id.title()
    scientific = f"{entry_id.split('_')[0].title()} sp."
    if kind == "fish":
        return {
            "names": names,
            "scientific": scientific,
            "size_cm": [5, 8],
            "behavior": {
                "profile": "shoal",
                "band": [0.1, 0.7],
                "depth": [0.2, 0.8],
                "speed_cm_s": [4, 10],
                "turn_rate": 2.5,
            },
            "night": "hide",
            "feeding_response": 0.8,
        }
    return {
        "names": names,
        "scientific": scientific,
        "size_cm": [5, 20],
        "palette": [{"default": "#ff0000"}],
    }


def entry(root: Path, folder_kind: str, entry_id: str, **overrides: Any) -> Path:
    """A valid entry (2 frames of 10x10 on 2 columns)."""
    kind = folder_kind
    folder = root / kind / entry_id
    folder.mkdir(parents=True)
    Image.new("RGBA", (20, 10), (255, 0, 0, 255)).save(folder / "atlas.webp")
    Image.new("RGBA", (8, 8)).save(folder / "thumb.webp")
    desc: dict[str, Any] = {
        "kind": "fish" if kind == "fish" else "coral",
        "frame": [10, 10],
        "columns": 2,
        "clips": {"swim" if kind == "fish" else "day": {"from": 0, "to": 1, "fps": 24}},
        "atlas": {"1x" if kind == "fish" else "shade": "atlas.webp"},
        "thumbnail": "thumb.webp",
        **described(kind, entry_id),
    }
    desc.update(overrides)
    (folder / "species.json").write_text(json.dumps(desc), encoding="utf-8")
    return folder


def test_repository_catalog_is_valid() -> None:
    assert validate.check_catalog(REPO) == {}


def test_valid_entries(tmp_path: Path) -> None:
    entry(tmp_path, "fish", "fox")
    entry(tmp_path, "corals", "euphyllia")
    (tmp_path / "fish" / "fox" / "recipe.json").write_text("{}")  # not shipped
    assert validate.check_catalog(tmp_path) == {}
    names = [p.name for p in validate.shipped_files(tmp_path / "fish" / "fox")]
    assert names == ["atlas.webp", "species.json", "thumb.webp"]
    assert validate.main([str(tmp_path)]) == 0


def test_invalid_entries(tmp_path: Path, capsys: Any) -> None:
    entry(tmp_path, "fish", "Bad_Case")
    entry(tmp_path, "fish", "wrong_kind", kind="coral")
    entry(tmp_path, "fish", "no_atlas", atlas={})
    entry(tmp_path, "fish", "bad_atlas", atlas={"1x": "../x.webp", "2x": "gone.webp"})
    entry(tmp_path, "fish", "no_thumb", thumbnail=3)
    entry(tmp_path, "fish", "bad_frame", frame=[10])
    entry(tmp_path, "fish", "bad_columns", columns=0)
    entry(tmp_path, "fish", "misfit", frame=[7, 10])
    entry(tmp_path, "fish", "no_clips", clips={})
    entry(tmp_path, "fish", "far_clip", clips={"swim": {"from": 0, "to": 9, "fps": 24}})
    entry(tmp_path, "fish", "no_swim", clips={"idle": {"from": 0, "to": 1, "fps": 12}})
    broken = entry(tmp_path, "fish", "not_picture")
    (broken / "atlas.webp").write_text("nope")
    many = entry(tmp_path, "fish", "too_many")
    for n in range(20):
        Image.new("RGBA", (1, 1)).save(many / f"x{n}.png")
    (many / "bad name.png").write_bytes(b"")
    for name, content in (("no_desc", None), ("bad_json", "{"), ("listy", "[]")):
        folder = tmp_path / "corals" / name
        folder.mkdir(parents=True)
        if content is not None:
            (folder / "species.json").write_text(content)
    problems = validate.check_catalog(tmp_path)
    assert set(problems) == {
        "fish/Bad_Case",
        "fish/wrong_kind",
        "fish/no_atlas",
        "fish/bad_atlas",
        "fish/no_thumb",
        "fish/bad_frame",
        "fish/bad_columns",
        "fish/misfit",
        "fish/no_clips",
        "fish/far_clip",
        "fish/no_swim",
        "fish/not_picture",
        "fish/too_many",
        "corals/no_desc",
        "corals/bad_json",
        "corals/listy",
    }
    assert any("files" in e for e in problems["fish/too_many"])
    assert any("bad file name" in e for e in problems["fish/too_many"])
    assert any("not a picture" in e for e in problems["fish/not_picture"])
    assert any("missing" in e for e in problems["fish/bad_atlas"])
    assert validate.main([str(tmp_path)]) == 1
    assert "with problems" in capsys.readouterr().out


def test_size_limit(tmp_path: Path, monkeypatch: Any) -> None:
    entry(tmp_path, "fish", "fat")
    monkeypatch.setattr(validate, "MAX_ENTRY_BYTES", 10)
    assert any("bytes" in e for e in validate.check_catalog(tmp_path)["fish/fat"])


def test_release_files_and_notes(tmp_path: Path) -> None:
    root = tmp_path / "catalog"
    entry(root, "fish", "fox")
    entry(root, "fish", "anthias")
    out = tmp_path / "dist1"
    first = make_release.build(root, "2026.10.0", out)
    assert set(first["entries"]) == {"fish/anthias", "fish/fox"}
    item = first["entries"]["fish/fox"]
    blob = (out / item["file"]).read_bytes()
    assert hashlib.sha256(blob).hexdigest() == item["sha256"]
    assert len(blob) == item["size"]
    with zipfile.ZipFile(out / item["file"]) as archive:
        assert sorted(archive.namelist()) == [
            "atlas.webp",
            "species.json",
            "thumb.webp",
        ]
    assert "**Added**: Anthias (`anthias`), Fox (`fox`)" in first["notes"]
    assert first["min_integration"] == "0.1.0"

    # rebuilt unchanged: same archives, versions kept
    again = make_release.build(root, "2026.11.0", tmp_path / "dist2", first)
    assert again["entries"]["fish/fox"] == {**item, "version": "2026.10.0"}
    assert "No species change" in again["notes"]

    # one changed, one removed, one added
    (root / "fish" / "fox" / "thumb.webp").write_bytes(b"other")
    for path in (root / "fish" / "anthias").iterdir():
        path.unlink()
    (root / "fish" / "anthias").rmdir()
    entry(root, "corals", "euphyllia")
    (root / "fish" / "fox" / "species.json").write_text(
        json.dumps(
            {
                **json.loads((root / "fish" / "fox" / "species.json").read_text()),
                "names": {},
            }
        )
    )
    third = make_release.build(root, "2026.12.0", tmp_path / "dist3", again)
    assert third["entries"]["fish/fox"]["version"] == "2026.12.0"
    assert "**Updated**: `fox`" in third["notes"]
    assert "**Removed**: `anthias`" in third["notes"]
    assert "**Added**: Euphyllia (`euphyllia`)" in third["notes"]
    (root / "fish" / "fox" / "species.json").write_text("{")
    assert make_release.entry_name(root / "fish" / "fox") == "`fox`"


def test_release_command(tmp_path: Path, capsys: Any) -> None:
    root = tmp_path / "catalog"
    entry(root, "fish", "fox")
    out = tmp_path / "dist"
    assert make_release.main(["1.0.0", "--root", str(root), "--out", str(out)]) == 0
    previous = out / "manifest.json"
    args = ["1.1.0", "--root", str(root), "--out", str(tmp_path / "d2")]
    assert (
        make_release.main(
            [*args, "--previous", str(previous), "--min-integration", "0.2.0"]
        )
        == 0
    )
    manifest = json.loads((tmp_path / "d2" / "manifest.json").read_text())
    assert manifest["min_integration"] == "0.2.0"
    assert manifest["entries"]["fish/fox"]["version"] == "1.0.0"
    # an invalid catalog is not released
    entry(root, "fish", "broken", frame=[3])
    assert (
        make_release.main(["1.2.0", "--root", str(root), "--out", str(tmp_path / "d3")])
        == 1
    )
    assert "frame must be" in capsys.readouterr().out


def texture(root: Path, entry_id: str, **overrides: Any) -> Path:
    """A valid rock texture."""
    folder = root / "textures" / entry_id
    folder.mkdir(parents=True)
    Image.new("RGB", (16, 16), (120, 110, 100)).save(folder / f"{entry_id}.webp")
    desc: dict[str, Any] = {
        "kind": "texture",
        "role": "rock",
        "image": f"{entry_id}.webp",
        "scale_cm": 30,
        "names": {"en": "Live rock"},
        **overrides,
    }
    (folder / "texture.json").write_text(json.dumps(desc), encoding="utf-8")
    return folder


def test_textures(tmp_path: Path, monkeypatch: Any) -> None:
    root = tmp_path / "catalog"
    rock = texture(root, "live_rock")
    (rock / "notes.txt").write_text("not shipped")
    texture(root, "white_sand", role="sand", names={})
    assert validate.check_catalog(root) == {}
    assert [p.name for p in validate.shipped_files(rock)] == [
        "live_rock.webp",
        "texture.json",
    ]
    # released like the species
    manifest = make_release.build(root, "1", tmp_path / "dist")
    assert set(manifest["entries"]) == {"textures/live_rock", "textures/white_sand"}
    assert "Live rock (`live_rock`)" in manifest["notes"]

    texture(root, "Bad", role="wall", scale_cm=0, kind="x", names={"en": ""})
    texture(root, "no_image", image=None)
    big = texture(root, "big")
    Image.new("RGB", (4096, 8)).save(big / "big.webp")
    texture(root, "gone", image="missing.webp")
    for name, content in (("no_desc", None), ("bad_json", "{"), ("listy", "[]")):
        folder = root / "textures" / name
        folder.mkdir()
        if content is not None:
            (folder / "texture.json").write_text(content)
    problems = validate.check_catalog(root)
    assert len(problems["textures/Bad"]) == 5
    assert problems["textures/no_image"] == ["no image"]
    assert problems["textures/big"] == ["image: 2048 pixels at most"]
    assert problems["textures/gone"] == ["image: missing.webp missing"]
    assert problems["textures/no_desc"] == ["no texture.json"]
    assert problems["textures/bad_json"][0].startswith("texture.json:")
    assert problems["textures/listy"] == ["texture.json is not an object"]
    monkeypatch.setattr(validate, "MAX_ENTRY_BYTES", 10)
    assert "bytes" in validate.check_texture(rock)[0]


def test_territory(tmp_path: Path) -> None:
    good = {**described("fish", "goby")}
    good["behavior"] = {**good["behavior"], "profile": "sand", "territory_cm": 12}
    assert validate.check_description(good, "fish") == []
    good["behavior"]["burrow"] = True
    assert validate.check_description(good, "fish") == []
    good["behavior"]["burrow"] = "yes"
    assert "behavior.burrow must be true or false" in (
        validate.check_description(good, "fish")
    )
    good["behavior"]["burrow"] = False
    good["behavior"]["territory_cm"] = -1
    assert "behavior.territory_cm must be a positive number" in (
        validate.check_description(good, "fish")
    )
