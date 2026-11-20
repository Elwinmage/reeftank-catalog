"""READMEs: species tables in every language."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
make_readme = importlib.import_module("make_readme")
validate = importlib.import_module("validate")

from test_release import entry

HAND = "# reeftank-catalog\n\nIntro\n\n{markers}\n\n## Contents\n\nHand written.\n"


def catalog(root: Path) -> Path:
    """A catalog with a fish, a coral, and an English README."""
    fox = entry(root, "fish", "siganus_vulpinus", scientific="Siganus vulpinus")
    desc = json.loads((fox / "species.json").read_text(encoding="utf-8"))
    desc["behavior"]["territory_cm"] = 20
    (fox / "species.json").write_text(json.dumps(desc), encoding="utf-8")
    entry(root, "corals", "euphyllia")
    (root / "README.md").write_text(
        HAND.format(markers=f"{make_readme.BEGIN}\n{make_readme.END}"),
        encoding="utf-8",
    )
    return root


def test_every_language_has_its_texts() -> None:
    keys = set(make_readme.TEXTS["en"])
    assert set(make_readme.TEXTS) == set(validate.LANGUAGES)
    for lang, texts in make_readme.TEXTS.items():
        assert set(texts) == keys, lang
        assert set(texts["profiles"]) == set(validate.PROFILES), lang
        assert set(texts["night"]) == set(validate.NIGHT_MODES), lang


def test_writes_one_readme_per_language(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = catalog(tmp_path)
    assert make_readme.main(["--root", str(root), "--check"]) == 1
    assert "README.fr.md is out of date" in capsys.readouterr().out
    assert make_readme.main(["--root", str(root)]) == 0
    assert "README.md updated" in capsys.readouterr().out
    names = sorted(p.name for p in root.glob("README*.md"))
    assert names == sorted(make_readme.readme_name(l) for l in validate.LANGUAGES)

    english = (root / "README.md").read_text(encoding="utf-8")
    # the hand written parts are kept around the block
    assert english.startswith("# reeftank-catalog\n\nIntro\n\n<!-- species:begin")
    assert english.endswith("<!-- species:end -->\n\n## Contents\n\nHand written.\n")
    assert (
        '[<img src="https://flagicons.lipis.dev/flags/4x3/gb.svg" width="5%" '
        'alt="English" title="English"/>](README.md) '
        '[<img src="https://flagicons.lipis.dev/flags/4x3/fr.svg" width="5%" '
        'alt="Français" title="Français"/>](README.fr.md)'
    ) in english
    assert "## Fish (1)" in english and "## Corals (1)" in english
    assert (
        '| <img src="fish/siganus_vulpinus/thumb.webp" width="160" alt=""> '
        "| <small>Siganus_Vulpinus<br>`siganus_vulpinus`</small> "
        "| <small>*Siganus vulpinus*</small> | <small>Marine</small> "
        "| <small>5–8 cm</small> "
        "| <small>Shoaling · stays near its home · hides at night</small> |"
    ) in english
    # corals have no details column
    assert (
        "| <small>Name</small> | <small>Scientific name</small> "
        "| <small>Water</small> | <small>Size</small> |\n|---|---|---|---|---|\n"
        in english
    )
    assert "documentation en anglais" not in english

    french = (root / "README.fr.md").read_text(encoding="utf-8")
    assert "flags/4x3/nl.svg" in french
    assert (
        "Siganus_Vulpinus fr" in french
        and "En banc · reste près de son abri · se cache la nuit" in french
    )
    assert "[documentation en anglais](README.md)" in french

    # once written: up to date, nothing rewritten
    assert make_readme.main(["--root", str(root), "--check"]) == 0
    assert make_readme.main(["--root", str(root)]) == 0
    assert "READMEs up to date" in capsys.readouterr().out


def test_table_of_partly_described_species(tmp_path: Path) -> None:
    root = catalog(tmp_path)
    folder = root / "fish" / "plain"
    folder.mkdir()
    (folder / "species.json").write_text(
        json.dumps({"kind": "fish", "names": "x", "size_cm": [4, 4]}),
        encoding="utf-8",
    )
    other = root / "fish" / "odd"
    other.mkdir()
    (other / "species.json").write_text(
        json.dumps(
            {
                "names": {"en": "Odd | fish"},
                "behavior": {"profile": "unknown"},
                "thumbnail": "t.webp",
            }
        ),
        encoding="utf-8",
    )
    (root / "fish" / "list").mkdir()
    (root / "fish" / "list" / "species.json").write_text("[]", encoding="utf-8")
    (root / "fish" / "empty").mkdir()
    species = make_readme.load_species(root)
    assert [d["id"] for d in species["fish"]] == ["odd", "plain", "siganus_vulpinus"]
    table = make_readme.species_table(species["fish"], "de", "fish")
    # no name: the id; no scientific name, size or behaviour: a dash
    assert (
        '| <img src="defaults/unknown_fish/thumb.webp" width="160" alt=""> '
        "| <small>plain<br>`plain`</small> | <small>—</small> | <small>—</small> "
        "| <small>4 cm</small> | <small>—</small> |"
    ) in table
    # English name when the language is missing, pipes escaped
    assert (
        "| <small>Odd \\| fish<br>`odd`</small> | <small>—</small> | <small>—</small> "
        "| <small>—</small> | <small>—</small> |"
    ) in table
    assert make_readme.species_table([], "pl", "fish") == "Brak gatunków."
    goby = {"behavior": {"profile": "sand", "burrow": True, "territory_cm": 9}}
    assert make_readme.behaviour(goby, "en") == "Lives on the sand · lives in a burrow"


def test_kind_folder_missing(tmp_path: Path) -> None:
    entry(tmp_path, "fish", "fox")
    assert make_readme.load_species(tmp_path)["corals"] == []


def test_english_readme_needs_its_markers(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no generated block"):
        make_readme.render(tmp_path)
    (tmp_path / "README.md").write_text(
        HAND.format(markers=f"{make_readme.END}\n{make_readme.BEGIN}"),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="no generated block"):
        make_readme.render(tmp_path)


def test_translations_keep_the_ecosystem_block(tmp_path: Path) -> None:
    root = catalog(tmp_path)
    assert make_readme.main(["--root", str(root)]) == 0
    french = root / "README.fr.md"
    text = french.read_text(encoding="utf-8")
    # written with empty markers, after the generated block
    assert text.endswith(
        f"{make_readme.END}\n\n{make_readme.ECO_BEGIN}\n{make_readme.ECO_END}\n"
    )
    assert "> Fait partie de l'[**écosystème ReefTech**]" in text
    # filled by gen_ecosystem.py: kept as it is
    filled = f"{make_readme.ECO_BEGIN}\n\n## Projets liés\n\n{make_readme.ECO_END}"
    french.write_text(
        text.replace(f"{make_readme.ECO_BEGIN}\n{make_readme.ECO_END}", filled),
        encoding="utf-8",
    )
    assert make_readme.main(["--root", str(root), "--check"]) == 0
    assert filled in french.read_text(encoding="utf-8")


def test_every_language_explains_how_it_works(tmp_path: Path) -> None:
    """The animated pictures come right after the introduction, in every language."""
    assert set(make_readme.DEMO) == set(validate.LANGUAGES)
    root = catalog(tmp_path)
    rendered = make_readme.render(root)
    for lang in validate.LANGUAGES:
        text = rendered[make_readme.readme_name(lang)]
        title, _, names = make_readme.DEMO[lang]
        assert text.index("<!-- species:begin") < text.index(f"## {title}")
        for (src, _), name in zip(make_readme.DEMO_IMAGES, names, strict=True):
            assert f'<img src="{src}"' in text and f'alt="{name}"' in text
        # before the first table
        assert text.index(f"## {title}") < text.index("|---|"), lang


def test_hd_column_marks_the_real_animations_of_the_fish() -> None:
    real = {"id": "a", "_kind": "fish", "atlas": {"1x": "a.webp"}}
    photo = {
        "id": "b",
        "_kind": "fish",
        "atlas": {"1x": "b.webp"},
        "degraded": True,
    }
    none = {"id": "c", "_kind": "fish"}
    assert [make_readme.is_hd(d) for d in (real, photo, none)] == [True, False, False]
    table = make_readme.species_table([real, photo, none], "en", "fish").splitlines()
    assert table[0].endswith("| HD |")
    assert [row.split("|")[-2].strip() for row in table[2:]] == [
        make_readme.HD_MARK,
        "",
        "",
    ]
    # only the fish have the column, and every language explains it
    assert (
        "HD"
        not in make_readme.species_table(
            [{**real, "_kind": "corals"}], "en", "corals"
        ).splitlines()[0]
    )
    assert set(make_readme.HD_NOTE) == set(validate.LANGUAGES)


def test_a_sharper_preview_replaces_the_thumbnail(tmp_path: Path) -> None:
    entry(tmp_path, "fish", "fox")
    preview = tmp_path / "docs" / "thumbs" / "fish" / "fox.webp"
    preview.parent.mkdir(parents=True)
    preview.write_bytes(b"webp")
    (fox,) = make_readme.load_species(tmp_path)["fish"]
    assert fox["_preview"] == "docs/thumbs/fish/fox.webp"
    assert make_readme._thumb(fox) == (
        f'<img src="docs/thumbs/fish/fox.webp" width="{make_readme.PREVIEW_WIDTH}" alt="">'
    )
    preview.unlink()
    (fox,) = make_readme.load_species(tmp_path)["fish"]
    assert fox["_preview"] == ""


def test_the_picture_column_keeps_its_width() -> None:
    table = make_readme.species_table([], "en", "fish")
    assert table == make_readme.TEXTS["en"]["none"]
    desc = {"id": "x", "_kind": "fish", "_preview": "", "behavior": {"profile": "x"}}
    header = make_readme.species_table([desc], "en", "fish").splitlines()[0]
    assert header.startswith("| " + make_readme.COLUMN_SPACER + " |")
