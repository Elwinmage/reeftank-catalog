#!/usr/bin/env python3
"""Prints the prompt that asks an image model for the photo of an invertebrate.

The wording is `prompts/still_invert.json`: one template, and a "form" sentence
that says what the animal is and **where it is seen from** (a starfish from
straight above, a snail in three-quarter view, a shrimp from the side, head to
the right). The form comes from the species itself (`species_forms`), else from
its `movement`. Red, pink or purple species are asked on a green background
(`species_keys`), the others on magenta.

    python scripts/invert_prompt.py nerita [--key green]
    python scripts/invert_prompt.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "client"))
from reeftank_catalog_invert import (  # noqa: F401 - re-exported
    invert_aspect,
    invert_form,
    invert_key,
    invert_view,
    key_advice,
    render_invert_prompt,
)

ROOT = Path(__file__).resolve().parent.parent
PROMPT_FILE = ROOT / "prompts" / "still_invert.json"


def render_prompt(
    invert_id: str,
    desc: dict[str, Any],
    prompt: dict[str, Any],
    key: str | None = None,
) -> str:
    """The prompt of an invertebrate (`key` forces the background colour)."""
    return render_invert_prompt(invert_id, desc, key, prompt)


def load_inverts(root: Path = ROOT) -> dict[str, dict[str, Any]]:
    """The descriptors of all the invertebrates, by id."""
    return {
        folder.name: json.loads((folder / "species.json").read_text(encoding="utf-8"))
        for folder in sorted((root / "invertebrates").iterdir())
        if (folder / "species.json").is_file()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "invertebrate", nargs="?", help="invertebrate id (folder of invertebrates/)"
    )
    parser.add_argument("--all", action="store_true", help="print every invertebrate")
    parser.add_argument(
        "--key", choices=("magenta", "green"), help="force the background"
    )
    args = parser.parse_args(argv)
    prompt = json.loads(PROMPT_FILE.read_text(encoding="utf-8"))
    inverts = load_inverts()
    if args.all:
        for invert_id, desc in inverts.items():
            print(
                f"## {invert_id}\n{render_prompt(invert_id, desc, prompt, args.key)}\n"
            )
        return 0
    if args.invertebrate not in inverts:
        parser.error("give an invertebrate id of invertebrates/ (or --all)")
    print(
        render_prompt(args.invertebrate, inverts[args.invertebrate], prompt, args.key)
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
