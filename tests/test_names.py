"""Hints about common names that were never translated."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
validate = importlib.import_module("validate")


def names(**overrides: str) -> dict:
    base = {lang: f"name {lang}" for lang in validate.LANGUAGES}
    return {"names": {**base, **overrides}}


def test_translated_names_are_fine() -> None:
    assert validate.untranslated_names(names()) == []


def test_english_name_copied_everywhere() -> None:
    desc = {"names": {lang: "Powder blue tang" for lang in validate.LANGUAGES}}
    assert validate.untranslated_names(desc) == ["de", "es", "it", "nl", "pl", "pt"]


def test_single_word_is_not_a_hint() -> None:
    desc = {"names": {lang: "Discus" for lang in validate.LANGUAGES}}
    assert validate.untranslated_names(desc) == []


def test_catalog_hints(tmp_path: Path) -> None:
    folder = tmp_path / "fish" / "some_fish"
    folder.mkdir(parents=True)
    desc = {"names": {lang: "Banded goby" for lang in validate.LANGUAGES}}
    (folder / "species.json").write_text(json.dumps(desc), encoding="utf-8")
    hints = validate.warnings_catalog(tmp_path)
    assert hints == {"fish/some_fish": "de, es, it, nl, pl, pt"}


def test_main_prints_the_hints_without_failing(tmp_path: Path, capsys) -> None:
    folder = tmp_path / "fish" / "some_fish"
    folder.mkdir(parents=True)
    desc = {"names": {lang: "Banded goby" for lang in validate.LANGUAGES}}
    (folder / "species.json").write_text(json.dumps(desc), encoding="utf-8")
    validate.main([str(tmp_path)])
    out = capsys.readouterr().out
    # the hint is printed, and it is only a hint (the problems decide the exit code)
    assert "warning: fish/some_fish: name copied from English in de, es" in out
