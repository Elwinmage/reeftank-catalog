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

from test_release import REPO, entry

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


def test_repository_readmes_are_up_to_date() -> None:
    assert make_readme.outdated(REPO) == {}


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
        '| <img src="fish/siganus_vulpinus/thumb.webp" width="72" alt=""> '
        "| Siganus_Vulpinus<br>`siganus_vulpinus` | *Siganus vulpinus* | 5–8 cm "
        "| Shoaling · stays near its home · hides at night |"
    ) in english
    # corals have no behaviour column
    assert "| Name | Scientific name | Size |\n|---|---|---|---|\n" in english
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
    table = make_readme.species_table(species["fish"], "de", fish=True)
    # no name: the id; no scientific name, size or behaviour: a dash
    assert "|  | plain<br>`plain` | — | 4 cm | — |" in table
    # English name when the language is missing, pipes escaped
    assert "| Odd \\| fish<br>`odd` | — | — | — |" in table
    assert make_readme.species_table([], "pl", fish=True) == "Brak gatunków."
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
