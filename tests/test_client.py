"""The client of the catalog: schemas, rules shared with the tools, sync, release."""

from __future__ import annotations

import copy
import importlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "client"))
validate = importlib.import_module("validate")
make_release = importlib.import_module("make_release")
sync_client = importlib.import_module("sync_client")
rules = importlib.import_module("reeftank_catalog_rules")
sections = importlib.import_module("reeftank_catalog_sections")

KIND_OF = {folder: kind for folder, kind in validate.KINDS.items()}
VALIDATORS = rules.load_validators(ROOT / "schema")


def species() -> list[tuple[str, str, dict[str, Any]]]:
    """Every species of the repository: folder, id, descriptor."""
    found = []
    for folder in validate.KINDS:
        for entry in sorted((ROOT / folder).glob("*/species.json")):
            found.append((folder, entry.parent.name, json.loads(entry.read_text())))
    return found


def report(folder: str, entry_id: str, data: dict[str, Any]) -> Any:
    return rules.validate_draft(
        KIND_OF[folder],
        entry_id,
        data,
        {"species.json": 900},
        rules.Context(),
        VALIDATORS,
    )


def test_sections_match_the_folders() -> None:
    assert set(sections.SECTIONS) == set(validate.KINDS)
    for section, kinds in sections.SECTIONS.items():
        assert list(kinds) == [KIND_OF[section]]
        assert kinds[KIND_OF[section]] == (section, validate.DESCRIPTOR)
    assert rules.DESCRIPTORS.keys() == rules.FOLDERS.keys() == set(KIND_OF.values())
    assert list(rules.KNOWN_LANGS) == list(validate.LANGUAGES)


def test_every_species_of_the_catalog_passes_the_shared_rules() -> None:
    for folder, entry_id, data in species():
        result = report(folder, entry_id, data)
        assert result.errors == [], (folder, entry_id)


def test_the_picture_keys_are_the_ones_the_build_writes() -> None:
    build = importlib.import_module("build_atlas")
    # `kind` is written by the tools too, and is a fixed key for the editor
    assert set(build.GENERATED_KEYS) - {"kind"} <= set(rules.PICTURE_KEYS)


FOLDER_OF = {kind: folder for folder, kind in validate.KINDS.items()}


def example(kind: str) -> dict[str, Any]:
    """A species of the repository of that kind, to break."""
    return copy.deepcopy(next(data for _, _, data in species() if data["kind"] == kind))


@pytest.mark.parametrize(
    ("kind", "mutate"),
    [
        ("fish", lambda d: d.pop("night")),
        ("fish", lambda d: d.update(night="sleep")),
        ("fish", lambda d: d["behavior"].update(profile="swarm")),
        ("fish", lambda d: d["behavior"].update(band=[0.9, 0.1])),
        ("fish", lambda d: d["behavior"].update(depth=[0, 1.5])),
        ("fish", lambda d: d["behavior"].update(speed_cm_s=[9, 3])),
        ("fish", lambda d: d["behavior"].update(turn_rate=0)),
        ("fish", lambda d: d["behavior"].update(territory_cm=-1)),
        ("fish", lambda d: d["behavior"].update(burrow="yes")),
        ("fish", lambda d: d["behavior"].update(shoal={"cohesion": -1})),
        ("fish", lambda d: d.update(feeding_response=2)),
        ("fish", lambda d: d.update(size_cm=[30, 10])),
        ("fish", lambda d: d.update(size_cm=[1])),
        ("fish", lambda d: d.update(scientific="amphiprion")),
        ("fish", lambda d: d.update(water_types=[])),
        ("fish", lambda d: d.update(water_types=["marine", "marine"])),
        ("fish", lambda d: d.update(water_types=["lava"])),
        ("fish", lambda d: d["names"].pop("nl")),
        ("fish", lambda d: d["names"].update(fr="  ")),
        ("fish", lambda d: d.update(degraded=True)),
        ("fish", lambda d: d.update(degraded=True, source={"type": "video"})),
        ("fish", lambda d: d.update(source={"type": "ai-generated"})),
        ("fish", lambda d: d.update(source={"type": "photo", "author": "A"})),
        ("fish", lambda d: d.update(source={"type": "photo", "license": "CC0"})),
        ("fish", lambda d: d.update(source={"type": "drawing"})),
        ("fish", lambda d: d.update(source={"type": "photo", "date": "May"})),
        ("coral", lambda d: d.pop("palette")),
        ("coral", lambda d: d.update(palette=[])),
        ("coral", lambda d: d.update(palette=[{"default": "red"}] * 5)),
        ("coral", lambda d: d.update(palette=[{"default": "#GG0000"}])),
        ("plant", lambda d: d.update(zone="roof")),
        ("plant", lambda d: d.pop("zone")),
        ("invertebrate", lambda d: d.update(role=[])),
        ("invertebrate", lambda d: d.update(role=["boss"])),
        ("invertebrate", lambda d: d.update(movement="fly")),
    ],
)
def test_the_shared_rules_agree_with_the_ci(kind: str, mutate: Any) -> None:
    """A draft is refused by the shared rules exactly when validate.py refuses it."""
    data = example(kind)
    mutate(data)
    ci = validate.check_description(data, FOLDER_OF[kind]) + validate.check_source(data)
    ours = report(FOLDER_OF[kind], "x", data)
    assert bool(ci) == bool(ours.errors), (ci, ours.errors)


def test_the_example_is_accepted_by_both() -> None:
    for kind in KIND_OF.values():
        data = example(kind)
        assert validate.check_description(data, FOLDER_OF[kind]) == []
        assert report(FOLDER_OF[kind], "x", data).errors == []


def test_a_wrong_kind_is_refused() -> None:
    data = example("fish")
    data["kind"] = "coral"
    assert report("fish", "x", data).errors


def test_a_draft_may_have_no_picture() -> None:
    data = example("plant")
    assert not any(key in data for key in rules.PICTURE_KEYS)
    assert report("plants", "x", data).errors == []


def test_files_of_an_entry() -> None:
    check = rules.Report()
    files = {"x.webp": 10, "bad name.png": 1, "notes.txt": 1}
    rules.check_files(check, "fish/x", files, "species.json")
    assert any("missing species.json" in e for e in check.errors)
    assert any("'bad name.png'" in e for e in check.errors)
    assert any("'notes.txt'" in e for e in check.errors)
    big = rules.Report()
    many = {"species.json": 40_000, **{f"p{n}.png": 40_000_000 for n in range(20)}}
    rules.check_files(big, "fish/x", many, "species.json")
    assert any("limit" in e for e in big.errors)
    assert any("20 files" not in e and "21 files" in e for e in big.errors)
    assert any("bytes (at most" in e for e in big.errors)


def test_a_scientific_name_already_taken_is_a_warning() -> None:
    data = example("fish")
    context = rules.Context(fish={"other": data, "x": data})
    result = rules.validate_draft(
        "fish", "x", data, {"species.json": 900}, context, VALIDATORS
    )
    assert result.errors == []
    assert [w for w in result.warnings if "already the species other" in w]
    assert len(context.fish) == 2


def test_with_fixed_puts_the_kind_first() -> None:
    assert list(rules.with_fixed("plant", {"zone": "midground", "kind": "x"})) == [
        "kind",
        "zone",
    ]


def test_every_schema_is_a_valid_schema() -> None:
    from jsonschema import Draft202012Validator

    for path in (ROOT / "schema").glob("*.json"):
        Draft202012Validator.check_schema(json.loads(path.read_text()))
    assert set(VALIDATORS) == set(KIND_OF.values())


def test_the_release_carries_the_schemas(tmp_path: Path) -> None:
    out = tmp_path / "dist"
    manifest = make_release.build(ROOT, "2026.99.0", out)
    record = manifest["schema"]
    blob = (out / "schema.zip").read_bytes()
    assert record["size"] == len(blob) and record["file"] == "schema.zip"
    with zipfile.ZipFile(out / "schema.zip") as archive:
        assert sorted(archive.namelist()) == sorted(
            p.name for p in (ROOT / "schema").glob("*.schema.json")
        )
    # the entries of the manifest do not change
    assert all("/" in key for key in manifest["entries"])


def test_a_release_without_schemas_has_no_record(tmp_path: Path) -> None:
    (tmp_path / "fish").mkdir()
    manifest = make_release.build(tmp_path, "1", tmp_path / "dist")
    assert "schema" not in manifest
    assert not (tmp_path / "dist" / "schema.zip").exists()


def test_sync_client(tmp_path: Path, capsys: Any) -> None:
    assert sync_client.main([str(tmp_path), "--check"]) == 1
    assert not list(tmp_path.iterdir())
    assert sync_client.main([str(tmp_path)]) == 0
    sections_copy = (tmp_path / "reeftank_catalog_sections.py").read_text()
    assert sections_copy.startswith("# Generated from reeftank-catalog/client/")
    assert sync_client.main([str(tmp_path), "--check"]) == 0
    (tmp_path / "reeftank_catalog_rules.py").write_text("old")
    assert sync_client.main([str(tmp_path), "--check"]) == 1
    assert "stale: reeftank_catalog_rules.py" in capsys.readouterr().out
    assert sync_client.sync(tmp_path) == ["reeftank_catalog_rules.py"]
    assert sync_client.sync(tmp_path) == []


# -- the still atlas shared with the editor ---------------------------------------


still = importlib.import_module("reeftank_catalog_still")


def test_the_prompt_is_the_versioned_file() -> None:
    file = json.loads((ROOT / "prompts" / "still_photo.json").read_text())
    assert still.PROMPT == file
    assert "[" not in still.render_prompt("Betta splendens", "green")
    assert "flat green #00FF00" in still.render_prompt("Betta splendens", "green")
    with pytest.raises(ValueError, match="key must be one of magenta, green"):
        still.render_prompt("Betta splendens", "blue")


def test_the_still_atlas_agrees_with_the_build() -> None:
    build_atlas = importlib.import_module("build_atlas")
    assert still.GENERATED_KEYS == build_atlas.GENERATED_KEYS
    assert still.THUMB_SIZE == build_atlas.THUMB_SIZE
    from PIL import Image

    frame = Image.new("RGBA", (208, 120), (0, 0, 0, 0))
    frame.paste((200, 90, 30, 255), (40, 30, 170, 90))
    assert still.thumbnail(frame).tobytes() == build_atlas.thumbnail(frame).tobytes()


def test_stray_pixels_at_the_edge_are_not_the_fish() -> None:
    """A model leaves a few off-colour pixels on the last row and column."""
    from PIL import Image, ImageDraw

    photo = Image.new("RGB", (640, 400), (240, 14, 231))
    ImageDraw.Draw(photo).ellipse((160, 120, 540, 280), fill=(230, 120, 20))
    for x in (100, 101, 102, 103, 104):
        photo.putpixel((x, 399), (200, 60, 180))
    for y in (50, 51, 52):
        photo.putpixel((639, y), (200, 60, 180))
    assert still.check_photo(photo) == []
    box = still.key_out(photo).getchannel("A").point(still.OPAQUE).getbbox()
    assert box is not None and box[1] > 100 and box[3] < 300 and box[2] < 560
    # a fish really cut by the edge is still refused
    ImageDraw.Draw(photo).rectangle((500, 150, 639, 250), fill=(230, 120, 20))
    assert "the fish touches the edge of the picture (cut off)" in still.check_photo(
        photo
    )


def test_marks_away_from_the_fish_are_not_the_fish() -> None:
    """A smudge in a corner and a written colour code are not part of the photo."""
    from PIL import Image, ImageDraw

    photo = Image.new("RGB", (1248, 832), (210, 80, 150))
    draw = ImageDraw.Draw(photo)
    draw.ellipse((300, 300, 900, 520), fill=(230, 120, 20))
    draw.polygon([(320, 410), (200, 330), (200, 490)], fill=(230, 120, 20))
    draw.ellipse((0, 0, 40, 30), fill=(190, 190, 190))  # a smudge in a corner
    for i in range(6):  # a colour code, written in the opposite corner
        draw.rectangle((1130 + i * 18, 800, 1140 + i * 18, 831), fill=(10, 10, 10))
    assert still.check_photo(photo, "magenta") == []
    box = still.key_out(photo, "magenta").getchannel("A").point(still.OPAQUE).getbbox()
    assert box is not None
    assert box[0] > 150 and box[1] > 250 and box[2] < 950 and box[3] < 560
    # the fins a few pixels apart from the body stay with the fish
    draw.rectangle((910, 400, 960, 405), fill=(230, 120, 20))
    box = still.key_out(photo, "magenta").getchannel("A").point(still.OPAQUE).getbbox()
    assert box is not None and box[2] >= 955
    # the corners are flat when two of them are marked, not when three are
    rgb = np.asarray(photo, dtype=float)
    assert still.corner_spread(rgb) < still.UNIFORM_STD
    for xy in ((0, 810), (1000, 5)):
        draw.ellipse((xy[0], xy[1], xy[0] + 19, xy[1] + 19), fill=(10, 200, 10))
    draw.line((1228, 0, 1247, 19), fill=(10, 10, 10), width=6)
    assert "background is not uniform in the corners" in still.check_photo(
        photo, "magenta"
    )


def test_build_still_gives_the_files_and_the_keys() -> None:
    from PIL import Image, ImageDraw

    photo = Image.new("RGB", (640, 400), (255, 0, 255))
    ImageDraw.Draw(photo).ellipse((160, 120, 540, 280), fill=(230, 120, 20))
    assert still.check_photo(photo) == []
    result = still.build_still(photo, "betta_splendens", {"type": "photo"}, 160)
    assert set(result.files) == {"betta_splendens.webp", "thumb.webp"}
    assert all(blob[:4] == b"RIFF" for blob in result.files.values())
    assert set(result.keys) == still.GENERATED_KEYS
    assert result.keys["atlas"] == {"1x": "betta_splendens.webp"}
    assert result.keys["degraded"] is True and result.keys["kind"] == "fish"
    assert not still.is_real_animation(result.keys)
    assert still.is_real_animation({"atlas": {"1x": "a.webp"}})
    assert not still.is_real_animation([])
