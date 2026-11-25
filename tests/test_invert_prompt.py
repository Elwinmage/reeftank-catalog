"""The prompt of an invertebrate depends on its form, and so on its view."""

import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
invert_prompt = importlib.import_module("invert_prompt")

PROMPT = json.loads(invert_prompt.PROMPT_FILE.read_text(encoding="utf-8"))
INVERTS = invert_prompt.load_inverts()


def test_every_invertebrate_has_a_form_and_a_complete_prompt() -> None:
    assert INVERTS
    for invert_id, desc in INVERTS.items():
        text = invert_prompt.render_prompt(invert_id, desc, PROMPT)
        assert "[" not in text, invert_id
        assert "One single adult specimen" in text


def test_every_species_names_its_own_form() -> None:
    """A new species must say what it looks like: `movement` cannot guess it.

    `ground` holds both the snails and the shrimps, `decor` both the urchins
    and the cleaner shrimps, so the fallback would photograph a shrimp as a
    snail. Naming the form of a new entry is part of adding it.
    """
    assert set(INVERTS) == set(PROMPT["species_forms"])


def test_the_form_follows_the_species_then_the_movement() -> None:
    form = invert_prompt.invert_form
    assert form("nerita", INVERTS["nerita"], PROMPT) == "snail"
    assert form("archaster_typicus", INVERTS["archaster_typicus"], PROMPT) == "star"
    assert form("tridacna_crocea", INVERTS["tridacna_crocea"], PROMPT) == "clam"
    assert (
        form("lysmata_amboinensis", INVERTS["lysmata_amboinensis"], PROMPT) == "shrimp"
    )
    # Unknown species: the commonest form of its movement
    assert form("newcomer", {"movement": "sessile"}, PROMPT) == "worm"
    assert form("newcomer", {"movement": "swim"}, PROMPT) == "shrimp"


def test_the_view_says_where_the_animal_is_seen_from() -> None:
    view = invert_prompt.invert_view
    # The ones lying flat are shot from above, the others from the side
    assert view("archaster_typicus", INVERTS["archaster_typicus"], PROMPT) == "above"
    assert view("ophiolepis_superba", INVERTS["ophiolepis_superba"], PROMPT) == "above"
    assert view("nerita", INVERTS["nerita"], PROMPT) == "side"
    assert view("neocaridina_davidi", INVERTS["neocaridina_davidi"], PROMPT) == "side"
    # An entry whose form is not known yet is fitted the commonest way
    assert view("newcomer", {}, PROMPT) == PROMPT["default_view"]
    assert sorted(set(PROMPT["views"].values())) == ["above", "side"]


def test_the_shape_of_the_picture_follows_the_form() -> None:
    aspect = invert_prompt.invert_aspect
    # A shrimp is longer than it is tall, a snail fits a square
    assert aspect("neocaridina_davidi", INVERTS["neocaridina_davidi"], PROMPT) == "3:2"
    assert aspect("nerita", INVERTS["nerita"], PROMPT) == "1:1"
    assert aspect("sabellastarte", INVERTS["sabellastarte"], PROMPT) == "3:4"


def test_species_overrides_name_real_species_and_forms() -> None:
    for table in ("species_forms", "species_keys"):
        assert set(PROMPT[table]) <= set(INVERTS)
    assert set(PROMPT["species_forms"].values()) <= set(PROMPT["forms"])
    assert set(PROMPT["movement_forms"].values()) <= set(PROMPT["forms"])
    assert set(PROMPT["views"]) == set(PROMPT["forms"])
    assert set(PROMPT["aspects"]) == set(PROMPT["forms"])


def test_red_species_are_asked_on_green() -> None:
    desc = INVERTS["neocaridina_davidi"]
    assert "flat green #00FF00" in invert_prompt.render_prompt(
        "neocaridina_davidi", desc, PROMPT
    )
    assert "flat magenta #FF00FF" in invert_prompt.render_prompt(
        "neocaridina_davidi", desc, PROMPT, "magenta"
    )
    with pytest.raises(ValueError, match="key must be one of"):
        invert_prompt.render_prompt("neocaridina_davidi", desc, PROMPT, "blue")


def test_green_is_warned_against_for_a_green_animal() -> None:
    advice = invert_prompt.key_advice
    # The emerald crab is not on the green list: green would eat it
    assert advice("mithraculus_sculptus", "green", PROMPT)
    assert advice("mithraculus_sculptus", "magenta", PROMPT) is None
    assert advice("neocaridina_davidi", "green", PROMPT) is None


def test_an_invertebrate_without_a_known_form_is_an_error() -> None:
    with pytest.raises(ValueError, match="no form for invertebrate mystery"):
        invert_prompt.invert_form("mystery", {"movement": "nowhere"}, PROMPT)


def test_the_name_falls_back_to_the_id_and_the_common_name_is_added() -> None:
    bare = invert_prompt.render_prompt("some_snail", {"movement": "ground"}, PROMPT)
    assert "some snail" in bare
    named = {
        "movement": "ground",
        "scientific": "Abc def",
        "names": {"en": "Mud snail"},
    }
    assert "Abc def (Mud snail)" in invert_prompt.render_prompt("x", named, PROMPT)


def test_main_prints_one_prompt_or_all_of_them(capsys) -> None:
    assert invert_prompt.main(["nerita", "--key", "green"]) == 0
    out = capsys.readouterr().out
    assert "Nerita" in out and "flat green #00FF00" in out
    assert invert_prompt.main(["--all"]) == 0
    out = capsys.readouterr().out
    assert all(f"## {invert_id}" in out for invert_id in INVERTS)


def test_main_needs_a_known_invertebrate() -> None:
    with pytest.raises(SystemExit):
        invert_prompt.main(["not_an_invertebrate"])
    with pytest.raises(SystemExit):
        invert_prompt.main([])
