"""The invertebrate module shared with the tools that edit the catalog (`client/`)."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "client"))

import reeftank_catalog_invert as invert


def test_the_embedded_prompt_is_the_prompt_file():
    file = json.loads((ROOT / "prompts" / "still_invert.json").read_text("utf-8"))
    assert invert.INVERT_PROMPT == file


def test_an_invertebrate_is_asked_on_magenta_unless_listed():
    assert invert.invert_key("nerita") == "magenta"
    assert invert.invert_key("neocaridina_davidi") == "green"
    text = invert.render_invert_prompt("nerita", {"movement": "glass"})
    assert "flat magenta" in text and "flat green" not in text


def test_green_is_advised_against_for_a_species_that_is_not_listed():
    assert invert.key_advice("mithraculus_sculptus", "green")
    assert invert.key_advice("mithraculus_sculptus", "magenta") is None
    assert invert.key_advice("neocaridina_davidi", "green") is None


def test_the_background_can_be_forced():
    text = invert.render_invert_prompt("nerita", {"movement": "glass"}, "green")
    assert "flat green" in text


def test_the_view_fixes_how_the_animal_is_fitted():
    # The two shot from above lie in their frame; everything else stands on it
    assert invert.invert_view("archaster_typicus", {}) == "above"
    assert invert.invert_view("nerita", {}) == "side"
    assert invert.invert_view("unknown", {"movement": "ground"}) == "side"


def test_the_defaults_hold_for_a_species_nobody_described():
    assert invert.invert_aspect("unknown", {"movement": "ground"}) == "1:1"
    assert invert.INVERT_KEYS == ("magenta", "green")
