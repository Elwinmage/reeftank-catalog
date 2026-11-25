#!/usr/bin/env python3
"""Turn the photo of an invertebrate into its (degraded) atlas.

The photo is the one `invert_prompt.py` asks an image model for: one adult
specimen on a flat magenta or green background, nothing else. The background is
keyed out and the animal is fitted in a square transparent frame — standing on
the bottom when it is seen from the side, in the middle when it is seen from
above: an atlas of a single frame (clip `still`), with its thumbnail, in the
layout the other sections use, so that `validate.py` and the README tables
handle it.

A single frame is also what tells the card to make the motion itself. The 13
shrimps and crabs get real clips later (`wan_clips.py`); the snails, stars,
urchins and sessile species keep their still picture.

    python scripts/invert_atlas.py <photo> invertebrates/<id> --source ai-generated \\
        --model gemini-2.5-flash-image [--key green]

The photo is checked first (flat key colour in the corners, the whole animal in
the picture, large enough): `--force` writes the atlas anyway.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "client"))
from invert_prompt import PROMPT_FILE
from reeftank_catalog_invert import (  # noqa: F401 - re-exported for the callers
    CLIP,
    FRAME,
    MIN_SHARE,
    PAD,
    SHARE_MESSAGE,
    build_invert_still,
    check_invert_photo,
)
from still_atlas import KEYS, source_info, write_still
from validate import LICENSES, SOURCE_TYPES


def write_invert_atlas(
    photo: Image.Image, folder: Path, source: dict[str, str], key: str | None = None
) -> dict:
    """Write the atlas, the thumbnail and the descriptor of an invertebrate."""
    entry = folder / "species.json"
    descriptor = (
        json.loads(entry.read_text(encoding="utf-8")) if entry.is_file() else {}
    )
    prompt = json.loads(PROMPT_FILE.read_text(encoding="utf-8"))
    still = build_invert_still(photo, folder.name, descriptor, source, key, prompt)
    return write_still(still, folder)


def main(argv: list[str] | None = None) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("photo", type=Path)
    parser.add_argument("folder", type=Path, help="entry, ex: invertebrates/nerita")
    parser.add_argument("--key", choices=sorted(KEYS), help="background colour")
    parser.add_argument(
        "--source", required=True, choices=[s for s in SOURCE_TYPES if s != "video"]
    )
    parser.add_argument("--author", help="who made the photo")
    parser.add_argument("--license", choices=LICENSES)
    parser.add_argument("--model", help="image model, for an ai-generated photo")
    parser.add_argument("--date", help="YYYY-MM-DD (default: today)")
    parser.add_argument("--force", action="store_true", help="ignore the checks")
    args = parser.parse_args(argv)
    photo = Image.open(args.photo)
    problems = check_invert_photo(photo, args.key)
    for problem in problems:
        print(f"{args.photo}: {problem}")
    if problems and not args.force:
        return 1
    descriptor = write_invert_atlas(photo, args.folder, source_info(args), args.key)
    print(f"{args.folder}: degraded atlas {descriptor['frame']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
