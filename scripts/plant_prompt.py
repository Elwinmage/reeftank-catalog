#!/usr/bin/env python3
"""Prints the prompt that asks an image model for the photo of a plant.

The wording is `prompts/still_plant.json`: one template, and a "form" sentence that
depends on the growth habit of the plant (stem plant, rosette, ribbon leaves,
carpet, floating, epiphyte, moss, alga...). The form comes from the species itself
(`species_forms`), else from its `zone` (`zone_forms`). Red, pink or purple plants
are asked on a green background (`species_keys`), the others on magenta.

    python scripts/plant_prompt.py vallisneria_spiralis [--key green]
    python scripts/plant_prompt.py --all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
PROMPT_FILE = ROOT / "prompts" / "still_plant.json"


def plant_form(plant_id: str, desc: dict[str, Any], prompt: dict[str, Any]) -> str:
    """The growth form (key of `forms`) of a plant."""
    form = prompt["species_forms"].get(plant_id) or prompt["zone_forms"].get(
        desc.get("zone", "")
    )
    if form not in prompt["forms"]:
        raise ValueError(f"no growth form for plant {plant_id}")
    return form


def plant_key(plant_id: str, prompt: dict[str, Any]) -> str:
    """The background colour advised for a plant."""
    return prompt["species_keys"].get(plant_id, prompt["default_key"])


def plant_aspect(plant_id: str, desc: dict[str, Any], prompt: dict[str, Any]) -> str:
    """The picture ratio to ask for: tall for a stem or a ribbon, else square."""
    form = plant_form(plant_id, desc, prompt)
    return str(prompt["aspects"].get(form, prompt["default_aspect"]))


def render_prompt(
    plant_id: str,
    desc: dict[str, Any],
    prompt: dict[str, Any],
    key: str | None = None,
) -> str:
    """The prompt of a plant (`key` forces the background colour)."""
    key = key or plant_key(plant_id, prompt)
    if key not in prompt["keys"]:
        raise ValueError(f"key must be one of {', '.join(prompt['keys'])}")
    name = desc.get("scientific") or plant_id.replace("_", " ")
    common = (desc.get("names") or {}).get("en")
    specie = f"{name} ({common})" if common and common != name else name
    text = prompt["template"].replace("[SPECIE]", specie)
    text = text.replace("[FORM]", prompt["forms"][plant_form(plant_id, desc, prompt)])
    return text.replace("[KEY]", prompt["keys"][key]).replace("[KEY_NAME]", key)


def load_plants(root: Path = ROOT) -> dict[str, dict[str, Any]]:
    """The descriptors of all the plants, by id."""
    return {
        folder.name: json.loads((folder / "species.json").read_text(encoding="utf-8"))
        for folder in sorted((root / "plants").iterdir())
        if (folder / "species.json").is_file()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("plant", nargs="?", help="plant id (folder of plants/)")
    parser.add_argument("--all", action="store_true", help="print every plant")
    parser.add_argument(
        "--key", choices=("magenta", "green"), help="force the background"
    )
    args = parser.parse_args(argv)
    prompt = json.loads(PROMPT_FILE.read_text(encoding="utf-8"))
    plants = load_plants()
    if args.all:
        for plant_id, desc in plants.items():
            print(f"## {plant_id}\n{render_prompt(plant_id, desc, prompt, args.key)}\n")
        return 0
    if args.plant not in plants:
        parser.error("give a plant id of plants/ (or --all)")
    print(render_prompt(args.plant, plants[args.plant], prompt, args.key))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
