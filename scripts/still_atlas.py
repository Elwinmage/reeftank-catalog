#!/usr/bin/env python3
"""Animate a still photo of a fish into a "degraded" atlas.

When a species has no video clips, a photo of it (side view, head to the
right, on a flat magenta or green background, see `prompts/still_photo.json`)
is enough for a simple animation: the body undulates, the fish idles and
turns. The atlas has the layout of the ones `build_atlas.py` writes (8
columns, `swim`, `idle` and `turn` clips), and its descriptor says
`"degraded": true` and where the photo comes from (`source`), so that a real
animation built from videos later replaces it.

    python scripts/still_atlas.py <photo> fish/<id> --source photo \\
        --author "Jane Doe" --license "CC BY 4.0"
    python scripts/still_atlas.py <photo> fish/<id> --source ai-generated \\
        --model gemini-2.5-flash-image
    python scripts/still_atlas.py --prompt "Betta splendens" [--key green]

The photo is checked first (uniform key colour in the corners, the whole fish
in the picture, large enough): `--force` writes the atlas anyway. An entry that
already has a real (not degraded) animation is left alone without `--force`.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "client"))
from reeftank_catalog_still import (  # noqa: F401 (the tests and users read these here)
    GENERATED_KEYS,
    KEYS,
    Fish,
    build_frames,
    build_still,
    check_photo,
    detect_key,
    is_real_animation,
    key_out,
    nearest_key,
)
from reeftank_catalog_still import render_prompt as render_template
from validate import LICENSES, SOURCE_TYPES

PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / "still_photo.json"


def render_prompt(species: str, key: str = "magenta", path: Path = PROMPT_FILE) -> str:
    """The prompt that asks an image model for the photo of a species."""
    return render_template(species, key, json.loads(path.read_text(encoding="utf-8")))


def source_info(args: argparse.Namespace) -> dict[str, str]:
    """The `source` field from the command line (without empty values)."""
    info = {
        "type": args.source,
        "model": args.model,
        "author": args.author,
        "license": args.license,
        "date": args.date or datetime.now(UTC).date().isoformat(),
    }
    return {k: v for k, v in info.items() if v}


def write_still(still: Any, folder: Path) -> dict[str, Any]:
    """Write the files and the descriptor of a still (see `build_still`)."""
    folder.mkdir(parents=True, exist_ok=True)
    for name, blob in still.files.items():
        (folder / name).write_bytes(blob)
    file = folder / "species.json"
    try:
        current = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        current = {}
    current = current if isinstance(current, dict) else {}
    kept = {k: v for k, v in current.items() if k not in GENERATED_KEYS}
    descriptor = {**still.keys, **kept}
    file.write_text(
        json.dumps(descriptor, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return descriptor


def write_atlas(
    photo: Image.Image,
    folder: Path,
    source: dict[str, str],
    length: int,
    key: str | None,
) -> dict[str, Any]:
    """Write the atlas, the thumbnail and the descriptor of an entry."""
    return write_still(build_still(photo, folder.name, source, length, key), folder)


def has_real_animation(folder: Path) -> bool:
    """True when the entry already has an atlas made from videos."""
    try:
        current = json.loads((folder / "species.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return is_real_animation(current)


def main(argv: list[str] | None = None) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("photo", type=Path, nargs="?")
    parser.add_argument("folder", type=Path, nargs="?", help="entry, ex: fish/betta")
    parser.add_argument("--prompt", metavar="SPECIES", help="print the prompt")
    parser.add_argument("--key", choices=sorted(KEYS), help="background colour")
    parser.add_argument("--source", choices=[s for s in SOURCE_TYPES if s != "video"])
    parser.add_argument("--author", help="who made the photo")
    parser.add_argument("--license", choices=LICENSES)
    parser.add_argument("--model", help="image model, for an ai-generated photo")
    parser.add_argument("--date", help="YYYY-MM-DD (default: today)")
    parser.add_argument("--length", type=int, default=192, help="fish length, px")
    parser.add_argument("--force", action="store_true", help="ignore the checks")
    args = parser.parse_args(argv)

    if args.prompt:
        print(render_prompt(args.prompt, args.key or "magenta"))
        return 0
    if args.photo is None or args.folder is None or args.source is None:
        parser.error("photo, folder and --source are needed")
    if has_real_animation(args.folder) and not args.force:
        print(f"{args.folder}: already has a real animation (--force to replace it)")
        return 1
    photo = Image.open(args.photo)
    problems = check_photo(photo, args.key)
    for problem in problems:
        print(f"{args.photo}: {problem}")
    if problems and not args.force:
        return 1
    descriptor = write_atlas(
        photo, args.folder, source_info(args), args.length, args.key
    )
    print(
        f"{args.folder}: degraded atlas {descriptor['frame']}, {descriptor['source']['type']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
