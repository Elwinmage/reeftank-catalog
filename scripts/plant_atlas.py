#!/usr/bin/env python3
"""Turn the photo of a plant into its (degraded) atlas.

The photo is the one `plant_prompt.py` asks an image model for: one specimen on a
flat magenta or green background, no roots, no pot. The background is keyed out
and the plant is fitted, standing on the bottom of a square transparent frame: an
atlas of a single frame (clip `still`), with its thumbnail, in the layout the
other sections use, so that `validate.py` and the README tables handle it.

    python scripts/plant_atlas.py <photo> plants/<id> --source ai-generated \\
        --model gemini-2.5-flash-image [--key green]

The photo is checked first (flat key colour in the corners, the whole plant in
the picture, large enough): `--force` writes the atlas anyway.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "client"))
from reeftank_catalog_plant import (  # noqa: F401 - re-exported for the callers
    CLIP,
    FRAME,
    MIN_SHARE,
    PAD,
    SHARE_MESSAGE,
    build_plant_still,
    check_plant_photo,
)
from still_atlas import KEYS, source_info, write_still
from validate import LICENSES, SOURCE_TYPES


def write_plant_atlas(
    photo: Image.Image, folder: Path, source: dict[str, str], key: str | None = None
) -> dict:
    """Write the atlas, the thumbnail and the descriptor of a plant."""
    return write_still(build_plant_still(photo, folder.name, source, key), folder)


def main(argv: list[str] | None = None) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("photo", type=Path)
    parser.add_argument("folder", type=Path, help="entry, ex: plants/egeria_densa")
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
    problems = check_plant_photo(photo, args.key)
    for problem in problems:
        print(f"{args.photo}: {problem}")
    if problems and not args.force:
        return 1
    descriptor = write_plant_atlas(photo, args.folder, source_info(args), args.key)
    print(f"{args.folder}: degraded atlas {descriptor['frame']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
