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
from reeftank_catalog_still import (
    OPAQUE,
    THUMB_SIZE,
    Still,
    encode_webp,
    thumbnail,
)
from still_atlas import KEYS, check_photo, key_out, source_info, write_still
from validate import LICENSES, SOURCE_TYPES

#: Side of the square frame, px, and the margin kept around the plant
FRAME = 320
PAD = 12
#: A plant fills at least this share of the picture (its larger side)
MIN_SHARE = 0.4
SHARE_MESSAGE = "too small in the picture"
CLIP = {"from": 0, "to": 0, "fps": 1, "loop": False}


def check_plant_photo(photo: Image.Image, key: str | None = None) -> list[str]:
    """Problems that make a photo unfit (empty when fine).

    `check_photo` judges the width of a fish; a plant is judged on its larger
    side, a stem plant being taller than wide.
    """
    problems = [p.replace("fish", "plant") for p in check_photo(photo, key)]
    kept = [p for p in problems if SHARE_MESSAGE not in p]
    if len(kept) != len(problems):  # the plant was found: measure it our way
        box = key_out(photo, key).getchannel("A").point(OPAQUE).getbbox()
        assert box is not None  # check_photo found it
        width, height = photo.size
        if max((box[2] - box[0]) / width, (box[3] - box[1]) / height) < MIN_SHARE:
            kept.append(f"the plant is {SHARE_MESSAGE} (under {MIN_SHARE:.0%})")
    return kept


def build_plant_still(
    photo: Image.Image, name: str, source: dict[str, str], key: str | None = None
) -> Still:
    """The atlas and the thumbnail of a plant (`name`: its id)."""
    sub = key_out(photo, key)
    box = sub.getchannel("A").point(OPAQUE).getbbox() or (0, 0, *sub.size)
    plant = sub.crop(box)
    room = FRAME - 2 * PAD
    scale = min(room / plant.width, room / plant.height)
    size = (max(1, round(plant.width * scale)), max(1, round(plant.height * scale)))
    small = plant.convert("RGBa").resize(size, Image.Resampling.LANCZOS).convert("RGBA")
    frame = Image.new("RGBA", (FRAME, FRAME), (0, 0, 0, 0))
    frame.paste(small, ((FRAME - size[0]) // 2, FRAME - PAD - size[1]))
    keys = {
        "kind": "plant",
        "frame": [FRAME, FRAME],
        "columns": 1,
        "clips": {"still": dict(CLIP)},
        "atlas": {"1x": f"{name}.webp"},
        "thumbnail": "thumb.webp",
        "degraded": True,
        "source": source,
    }
    files = {
        f"{name}.webp": encode_webp(frame),
        "thumb.webp": encode_webp(thumbnail(frame, THUMB_SIZE)),
    }
    return Still(files, keys)


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
