"""Descriptions of the species: export, import, checks, rebuilds."""

from __future__ import annotations

import importlib
import io
import json
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
species_meta = importlib.import_module("species_meta")
validate = importlib.import_module("validate")
build_atlas = importlib.import_module("build_atlas")

from test_release import described, entry


def undescribed(root: Path, kind: str, entry_id: str, **fields: Any) -> Path:
    """An entry whose description is only `fields`."""
    folder = entry(root, kind, entry_id)
    file = folder / "species.json"
    species = json.loads(file.read_text(encoding="utf-8"))
    for key in described(kind, entry_id):
        species.pop(key, None)
    species.update(fields)
    file.write_text(json.dumps(species), encoding="utf-8")
    return folder


def _read(root: Path, key: str) -> dict[str, Any]:
    return json.loads((root / key / "species.json").read_text(encoding="utf-8"))


def test_export_shows_every_field(tmp_path: Path) -> None:
    root = tmp_path / "catalog"
    undescribed(
        root,
        "fish",
        "fox",
        names={"en": "Fox"},
        size_cm=[15, 22],
        behavior={"profile": "cruiser"},
    )
    undescribed(root, "corals", "euphyllia")
    (root / "fish" / "empty").mkdir()  # no descriptor: skipped
    out = tmp_path / "meta.tgz"
    assert species_meta.export(root, out) == 2
    with tarfile.open(out) as archive:
        names = sorted(archive.getnames())
        fox = json.loads(archive.extractfile("fish/fox/meta.json").read())
        coral = json.loads(archive.extractfile("corals/euphyllia/meta.json").read())
    assert names == ["corals/euphyllia/meta.json", "fish/fox/meta.json"]
    # generated fields are not exported
    assert not set(fox) & build_atlas.GENERATED_KEYS
    assert fox["size_cm"] == [15, 22]
    assert fox["behavior"] == {
        "profile": "cruiser",
        "band": None,
        "depth": None,
        "speed_cm_s": None,
        "turn_rate": None,
    }
    assert fox["night"] is None and fox["feeding_response"] is None
    assert fox["names"]["en"] == "Fox" and fox["names"]["fr"] is None
    assert coral["palette"] is None and "night" not in coral


def test_import_merges_descriptions(tmp_path: Path) -> None:
    root = tmp_path / "catalog"
    undescribed(root, "fish", "fox", names={"en": "Fox"}, demo=True)
    entry(root, "fish", "same")
    out = tmp_path / "meta.tgz"
    species_meta.export(root, out)
    assert set(validate.check_catalog(root)) == {"fish/fox"}  # to describe
    # edit the export as a folder
    edited = tmp_path / "edited"
    with tarfile.open(out) as archive:
        archive.extractall(edited, filter="data")
    fox = json.loads((edited / "fish/fox/meta.json").read_text())
    full = described("fish", "fox")
    fox["names"].update({**full["names"], "en": None, "fr": "Renard"})
    fox["behavior"].update(full["behavior"])
    assert fox["scientific"] is None  # shown, to fill
    fox.update(
        scientific="Vulpes sp.", size_cm=[3, 4], night="hover", feeding_response=0.2
    )
    fox["clips"] = {"swim": "not imported"}  # generated: ignored
    (edited / "fish/fox/meta.json").write_text(json.dumps(fox))
    (edited / "fish" / "ghost").mkdir()
    (edited / "fish/ghost/meta.json").write_text("{}")

    report = species_meta.import_meta(root, edited, dry_run=True)
    assert (
        "fish/fox: behavior, feeding_response, names, night, scientific, size_cm"
        in report
    )
    assert _read(root, "fish/fox")["names"] == {"en": "Fox"}  # dry run

    report = species_meta.import_meta(root, edited)
    assert "fish/same: unchanged" in report
    assert "fish/ghost: no such species, skipped" in report
    fox_after = _read(root, "fish/fox")
    assert fox_after["names"]["en"] == "Fox"  # null: kept
    assert fox_after["names"]["fr"] == "Renard"
    assert fox_after["behavior"]["profile"] == "shoal"
    assert fox_after["night"] == "hover"
    assert fox_after["demo"] is True  # not in the export's edits: kept
    assert fox_after["clips"]["swim"]["from"] == 0
    assert validate.check_catalog(root) == {}


def test_import_from_archive(tmp_path: Path) -> None:
    root = tmp_path / "catalog"
    entry(root, "fish", "fox")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, text in (
            ("./fish/fox/meta.json", '{"names": {"en": "Fox", "de": "Fuchs"}}'),
            ("fish/fox/notes.txt", "ignored"),
            ("plants/x/meta.json", "{}"),
        ):
            blob = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(blob)
            archive.addfile(info, io.BytesIO(blob))
        archive.addfile(tarfile.TarInfo("fish/dir"))  # not a file
    source = tmp_path / "in.tgz"
    source.write_bytes(buffer.getvalue())
    assert species_meta.import_meta(root, source) == ["fish/fox: names"]
    fox = _read(root, "fish/fox")
    assert fox["names"]["de"] == "Fuchs" and fox["names"]["it"] == "Fox it"
    assert fox["behavior"]["profile"] == "shoal"  # untouched


@pytest.mark.parametrize("text", ["{oops", "[1]"])
def test_import_refuses_broken_files(tmp_path: Path, text: str) -> None:
    source = tmp_path / "edited" / "fish" / "fox"
    source.mkdir(parents=True)
    (source / "meta.json").write_text(text)
    with pytest.raises(SystemExit):
        species_meta.import_meta(tmp_path, tmp_path / "edited")


def test_command_line(tmp_path: Path, capsys: Any) -> None:
    root = tmp_path / "catalog"
    entry(root, "fish", "fox")
    out = tmp_path / "m.tgz"
    assert species_meta.main(["--root", str(root), "export", "--out", str(out)]) == 0
    assert "1 species exported" in capsys.readouterr().out
    args = ["--root", str(root), "import", str(out), "--dry-run"]
    assert species_meta.main(args) == 0
    assert "dry run" in capsys.readouterr().out


def test_description_checks(tmp_path: Path) -> None:
    good = {
        "names": {lang: "Fox" for lang in validate.LANGUAGES},
        "scientific": "Siganus vulpinus",
        "size_cm": [15, 22],
        "behavior": {
            "profile": "shoal",
            "band": [0.1, 0.6],
            "depth": [0, 1],
            "speed_cm_s": [3, 8],
            "turn_rate": 3,
            "shoal": {"cohesion": 0.7},
        },
        "night": "hide",
        "feeding_response": 0.5,
    }
    assert validate.check_description(good, "fish") == []
    bad = {
        "names": {"en": ""},
        "scientific": "siganus",
        "size_cm": [5],
        "behavior": {
            "profile": "swimmer",
            "band": [0.5, 2],
            "speed_cm_s": [9, 3],
            "turn_rate": 0,
            "shoal": {"cohesion": "a lot"},
        },
        "night": "sleep",
        "feeding_response": 2,
    }
    errors = validate.check_description(bad, "fish")
    assert len(errors) == 11
    assert "scientific must be" in errors[2]
    assert errors[0].startswith("to describe: behavior.depth, names.en, names.fr")
    assert "behavior must be an object" in validate.check_description(
        {"behavior": []}, "fish"
    )
    coral_ok = {
        "names": good["names"],
        "scientific": "Euphyllia glabrescens",
        "size_cm": [5, 20],
        "palette": [{"default": "#ff0000", "fluo": True}, {"default": "#00ff00"}],
    }
    assert validate.check_description(coral_ok, "corals") == []
    assert validate.missing_fields({}, "corals")[:3] == [
        "scientific",
        "size_cm",
        "palette",
    ]
    for name in ("Amphiprion sp.", "Amphiprion ocellaris", "Acropora millepora x"):
        assert validate.SCIENTIFIC_RE.match(name)
    for name in ("amphiprion sp.", "Amphiprion", "Amphiprion Ocellaris", 3):
        assert validate.check_description({"scientific": name}, "fish")
    for palette in ([], [{"default": "red"}], [{"default": "#ff0000", "fluo": 1}], "x"):
        assert validate.check_description({"palette": palette}, "corals")


def test_rebuild_keeps_the_catalog_description(tmp_path: Path) -> None:
    """A species edited in the catalog keeps its description when rebuilt."""
    clips = tmp_path / "clips" / "fox"
    frames = clips / "swim"
    frames.mkdir(parents=True)
    from test_build_atlas import fish_frame

    for n in range(4):
        fish_frame(tail=n).save(frames / f"{n:03d}.png")
    (clips / "meta.json").write_text(
        json.dumps({"names": {"en": "Seed"}, "size_cm": [5, 8]})
    )
    out = tmp_path / "fish"
    build = ["fish", "fox", "--auto", str(clips), "--out", str(out)]
    assert build_atlas.main(build) == 0
    assert _read(tmp_path, "fish/fox")["names"] == {"en": "Seed"}
    # edited in the catalog, then rebuilt
    species = _read(tmp_path, "fish/fox")
    species["names"] = {"en": "Fox", "fr": "Renard"}
    (out / "fox" / "species.json").write_text(json.dumps(species))
    assert build_atlas.main(build) == 0
    rebuilt = _read(tmp_path, "fish/fox")
    assert rebuilt["names"] == {"en": "Fox", "fr": "Renard"}
    assert rebuilt["size_cm"] == [5, 8]
    # an unreadable descriptor is rebuilt from the meta file
    (out / "fox" / "species.json").write_text("[")
    assert build_atlas.described(out / "fox" / "species.json") == {}
    (out / "fox" / "species.json").write_text("[1]")
    assert build_atlas.described(out / "fox" / "species.json") == {}
