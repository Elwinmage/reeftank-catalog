"""The prompt of a plant depends on its growth form."""

import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
plant_prompt = importlib.import_module("plant_prompt")

PROMPT = json.loads(plant_prompt.PROMPT_FILE.read_text(encoding="utf-8"))
PLANTS = plant_prompt.load_plants()


def test_every_plant_has_a_form_and_a_complete_prompt() -> None:
    assert PLANTS
    for plant_id, desc in PLANTS.items():
        text = plant_prompt.render_prompt(plant_id, desc, PROMPT)
        assert "[" not in text, plant_id
        assert "no roots" in text


def test_the_form_follows_the_species_then_the_zone() -> None:
    form = plant_prompt.plant_form
    assert form("lemna_minor", PLANTS["lemna_minor"], PROMPT) == "duckweed"
    assert form("pistia_stratiotes", PLANTS["pistia_stratiotes"], PROMPT) == "floating"
    assert (
        form("vallisneria_spiralis", PLANTS["vallisneria_spiralis"], PROMPT) == "ribbon"
    )
    assert form("cabomba_caroliniana", PLANTS["cabomba_caroliniana"], PROMPT) == "stem"


def test_species_overrides_name_real_plants_and_forms() -> None:
    for table in ("species_forms", "species_keys"):
        assert set(PROMPT[table]) <= set(PLANTS)
    assert set(PROMPT["species_forms"].values()) <= set(PROMPT["forms"])
    assert set(PROMPT["zone_forms"].values()) <= set(PROMPT["forms"])


def test_red_plants_are_asked_on_green() -> None:
    desc = PLANTS["rotala_rotundifolia"]
    assert "flat green #00FF00" in plant_prompt.render_prompt(
        "rotala_rotundifolia", desc, PROMPT
    )
    assert "flat magenta #FF00FF" in plant_prompt.render_prompt(
        "rotala_rotundifolia", desc, PROMPT, "magenta"
    )
    with pytest.raises(ValueError, match="key must be one of"):
        plant_prompt.render_prompt("rotala_rotundifolia", desc, PROMPT, "blue")


def test_a_plant_without_a_known_form_is_an_error() -> None:
    with pytest.raises(ValueError, match="no growth form for plant mystery"):
        plant_prompt.plant_form("mystery", {"zone": "nowhere"}, PROMPT)


def test_the_name_falls_back_to_the_id_and_the_common_name_is_added() -> None:
    bare = plant_prompt.render_prompt("some_plant", {"zone": "background"}, PROMPT)
    assert "some plant" in bare
    named = {"zone": "background", "scientific": "Abc def", "names": {"en": "Pondweed"}}
    assert "Abc def (Pondweed)" in plant_prompt.render_prompt("x", named, PROMPT)


def test_main_prints_one_prompt_or_all_of_them(capsys) -> None:
    assert plant_prompt.main(["lemna_minor", "--key", "green"]) == 0
    out = capsys.readouterr().out
    assert "Lemna minor" in out and "flat green #00FF00" in out
    assert plant_prompt.main(["--all"]) == 0
    out = capsys.readouterr().out
    assert all(f"## {plant_id}" in out for plant_id in PLANTS)


def test_main_needs_a_known_plant() -> None:
    with pytest.raises(SystemExit):
        plant_prompt.main(["not_a_plant"])
    with pytest.raises(SystemExit):
        plant_prompt.main([])
