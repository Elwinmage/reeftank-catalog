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
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "client"))
from reeftank_catalog_plant import (  # noqa: F401 - re-exported
    plant_aspect,
    plant_form,
    plant_key,
    render_plant_prompt,
)

ROOT = Path(__file__).resolve().parent.parent
PROMPT_FILE = ROOT / "prompts" / "still_plant.json"


def render_prompt(
    plant_id: str,
    desc: dict[str, Any],
    prompt: dict[str, Any],
    key: str | None = None,
) -> str:
    """The prompt of a plant (`key` forces the background colour)."""
    return render_plant_prompt(plant_id, desc, key, prompt)


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
