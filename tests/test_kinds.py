"""Plants, invertebrates, water types, pictureless species, default picture."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
validate = importlib.import_module("validate")
make_release = importlib.import_module("make_release")
make_readme = importlib.import_module("make_readme")

from test_release import described, entry


def species(root: Path, folder: str, entry_id: str, **overrides: Any) -> Path:
    """A described species without any picture."""
    path = root / folder / entry_id
    path.mkdir(parents=True)
    kind = {"fish": "fish", "corals": "coral", "plants": "plant"}.get(
        folder, "invertebrate"
    )
    desc = {"kind": kind, **described(folder, entry_id), **overrides}
    (path / "species.json").write_text(json.dumps(desc), encoding="utf-8")
    return path


def default_fish(root: Path, **overrides: Any) -> Path:
    """The default "?" fish: a picture and a light descriptor."""
    folder = entry(root, "fish", "tmp")
    target = root / "defaults" / "unknown_fish"
    target.parent.mkdir(parents=True, exist_ok=True)
    folder.rename(target)
    desc = json.loads((target / "species.json").read_text(encoding="utf-8"))
    for key in ("names", "scientific", "size_cm", "behavior", "night", "water_types"):
        desc.pop(key, None)
    desc.pop("feeding_response", None)
    desc.update(overrides)
    (target / "species.json").write_text(json.dumps(desc), encoding="utf-8")
    return target


def test_species_without_picture(tmp_path: Path) -> None:
    for folder in ("fish", "corals", "plants", "invertebrates"):
        species(tmp_path, folder, "none")
    assert validate.check_catalog(tmp_path) == {}
    # shipped and released like the others
    manifest = make_release.build(tmp_path, "1", tmp_path / "dist")
    assert "plants/none" in manifest["entries"]
    # a picture, once started, must be complete
    species(tmp_path, "fish", "half", atlas={"1x": "gone.webp"})
    problems = validate.check_catalog(tmp_path)
    assert set(problems) == {"fish/half"}
    assert "atlas 1x: gone.webp missing" in problems["fish/half"]


def test_water_types(tmp_path: Path) -> None:
    good = described("plants", "java")
    assert validate.check_description(good, "plants") == []
    for water in ([], ["lake"], ["marine", "marine"], "marine"):
        errors = validate.check_description({**good, "water_types": water}, "plants")
        assert errors == ["water_types: a list of marine, freshwater, brackish"]
    assert "water_types" in validate.missing_fields({}, "fish")


def test_plants_and_invertebrates(tmp_path: Path) -> None:
    plant = described("plants", "java")
    assert validate.check_description({**plant, "zone": "sky"}, "plants") == [
        "zone must be one of " + ", ".join(validate.ZONES)
    ]
    assert validate.missing_fields({}, "plants")[:4] == [
        "scientific",
        "water_types",
        "size_cm",
        "zone",
    ]
    shrimp = described("invertebrates", "shrimp")
    assert validate.check_description(shrimp, "invertebrates") == []
    for role in ([], ["boss"], "cleaner"):
        errors = validate.check_description({**shrimp, "role": role}, "invertebrates")
        assert errors == ["role: a list of " + ", ".join(validate.ROLES)]
    errors = validate.check_description({**shrimp, "movement": "fly"}, "invertebrates")
    assert errors == ["movement must be one of " + ", ".join(validate.MOVEMENTS)]
    assert validate.missing_fields({}, "invertebrates")[3:5] == ["role", "movement"]


def test_source_and_degraded(tmp_path: Path) -> None:
    photo = {
        "type": "photo",
        "author": "Jane",
        "license": "CC BY 4.0",
        "date": "2026-11-19",
    }
    ai = {"type": "ai-generated", "model": "gemini"}
    assert validate.check_source({}) == []
    assert validate.check_source({"degraded": True, "source": photo}) == []
    assert validate.check_source({"source": ai}) == []
    assert validate.check_source({"source": {"type": "video"}}) == []
    assert validate.check_source({"degraded": 1}) == ["degraded must be true or false"]
    assert "source is needed" in validate.check_source({"degraded": True})[0]
    assert "source.type" in validate.check_source({"source": "x"})[0]
    assert "source.type" in validate.check_source({"source": {"type": "dream"}})[0]
    assert (
        "not a video"
        in validate.check_source({"degraded": True, "source": {"type": "video"}})[0]
    )
    assert validate.check_source({"source": {"type": "ai-generated"}}) == [
        "source.model: the model that made the picture"
    ]
    errors = validate.check_source({"source": {"type": "photo", "model": ""}})
    assert "source.model must be a text" in errors
    assert "source.author: who made the photo" in errors
    assert any("source.license" in e for e in errors)
    bad_license = {**photo, "license": "All rights reserved"}
    assert any("license" in e for e in validate.check_source({"source": bad_license}))
    bad_date = {**photo, "date": "yesterday"}
    assert validate.check_source({"source": bad_date}) == [
        "source.date must be YYYY-MM-DD"
    ]


def test_degraded_needs_a_picture(tmp_path: Path) -> None:
    species(tmp_path, "fish", "ghost", degraded=True, source={"type": "photo"})
    entry(tmp_path, "fish", "still", degraded=True, source=photo_source())
    problems = validate.check_catalog(tmp_path)
    assert set(problems) == {"fish/ghost"}
    assert "degraded: there is no atlas" in problems["fish/ghost"]


def photo_source() -> dict[str, str]:
    return {"type": "photo", "author": "Jane", "license": "CC0"}


def test_default_picture(tmp_path: Path) -> None:
    default_fish(tmp_path)
    assert validate.check_catalog(tmp_path) == {}
    manifest = make_release.build(tmp_path, "1", tmp_path / "dist")
    assert "defaults/unknown_fish" in manifest["entries"]
    assert make_release.entry_name(tmp_path / "defaults" / "unknown_fish") == (
        "`unknown_fish`"
    )
    # a default must have its picture, and be a known one
    desc = tmp_path / "defaults" / "unknown_fish" / "species.json"
    data = json.loads(desc.read_text(encoding="utf-8"))
    desc.write_text(json.dumps({**data, "kind": "coral"}), encoding="utf-8")
    other = default_fish_copy(tmp_path, "mystery", data)
    (other / "species.json").write_text(json.dumps(data), encoding="utf-8")
    problems = validate.check_catalog(tmp_path)
    assert "kind must be 'fish'" in problems["defaults/unknown_fish"]
    assert any("unknown default" in e for e in problems["defaults/mystery"])
    bare = tmp_path / "defaults" / "bare"
    bare.mkdir()
    (bare / "species.json").write_text('{"kind": "fish"}', encoding="utf-8")
    assert "no atlas" in validate.check_catalog(tmp_path)["defaults/bare"]


def default_fish_copy(root: Path, name: str, data: dict[str, Any]) -> Path:
    folder = root / "defaults" / name
    folder.mkdir()
    Image.new("RGBA", (20, 10)).save(folder / "atlas.webp")
    Image.new("RGBA", (8, 8)).save(folder / "thumb.webp")
    return folder


def test_readme_tables_of_plants_and_invertebrates() -> None:
    plant = {
        "id": "java",
        "_kind": "plants",
        "names": {"fr": "Fougère"},
        "water_types": ["freshwater", "brackish", "lake"],
        "zone": "epiphyte",
    }
    table = make_readme.species_table([plant], "fr", "plants")
    assert (
        "| <small>Eau douce, Saumâtre</small> | <small>—</small> "
        "| <small>Épiphyte</small> |"
    ) in table
    assert "| <small>Zone</small> |" in table
    shrimp = {
        "id": "cherry",
        "_kind": "invertebrates",
        "role": ["detritivore", "cleaner", "boss"],
        "movement": "glass",
    }
    table = make_readme.species_table([shrimp], "en", "invertebrates")
    assert (
        "| <small>—</small> | <small>—</small> | <small>—</small> "
        "| <small>Detritivore · Cleaner · climbs the glass</small> |"
    ) in table
    assert "| <small>Role</small> |" in table
    assert make_readme.invertebrate_details({"role": "x"}, "en") == "—"
    assert make_readme.plant_details({"zone": "sky"}, "en") == "—"
    assert make_readme.water({}, "en") == "—"
    assert make_readme.species_table([plant], "en", "corals").count("|") > 0
