"""The plant module shared with the tools that edit the catalog (`client/`)."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "client"))

import reeftank_catalog_plant as plant


def test_the_embedded_prompt_is_the_prompt_file():
    file = json.loads((ROOT / "prompts" / "still_plant.json").read_text("utf-8"))
    assert plant.PLANT_PROMPT == file


def test_a_plant_is_asked_on_magenta_unless_listed():
    assert plant.plant_key("egeria_densa") == "magenta"
    assert plant.plant_key("ludwigia_repens") == "green"
    text = plant.render_plant_prompt("egeria_densa", {"zone": "background"})
    assert "flat magenta" in text and "flat green" not in text


def test_green_is_advised_against_for_a_plant_that_is_not_listed():
    assert plant.key_advice("egeria_densa", "green")
    assert plant.key_advice("egeria_densa", "magenta") is None
    assert plant.key_advice("ludwigia_repens", "green") is None


def test_the_background_can_be_forced():
    text = plant.render_plant_prompt("egeria_densa", {"zone": "background"}, "green")
    assert "flat green" in text
