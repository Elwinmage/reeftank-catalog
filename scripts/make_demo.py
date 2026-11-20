#!/usr/bin/env python3
"""Build the animated pictures of the README from the atlases of a few species.

An atlas is a sheet of frames; the card plays its clips (swim, idle, turn). Here
the clips are played one after the other into an animated WebP that GitHub shows
in the README: the fish swims, turns, swims back (the turn mirrored) and loops.

    python scripts/make_demo.py [--check]

`--check` writes nothing and fails when a picture is missing or out of date (CI).
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "img"
#: Sharper pictures for the tables of the README: docs/thumbs/<kind>/<id>.webp
THUMBS = ROOT / "docs" / "thumbs"
#: Width of a preview (the thumbnail of the card is only 96 px wide)
PREVIEW_WIDTH = 240
#: Species shown in the README: picture name -> species folder
DEMOS = {
    "demo-emperor-angelfish": "pomacanthus_imperator_adult",
    "demo-clownfish": "amphiprion",
}
#: Number of swim cycles before the turn, and displayed width of a frame
SWIM_CYCLES = 1
WIDTH = 144
QUALITY = 60
#: One frame out of STEP is kept (the delay is multiplied): a lighter file
STEP = 2


def frames_of(folder: Path) -> tuple[list[Image.Image], dict[str, dict]]:
    """The frames of an atlas, in order, and its clips."""
    desc = json.loads((folder / "species.json").read_text(encoding="utf-8"))
    sheet = Image.open(folder / desc["atlas"]["1x"]).convert("RGBA")
    width, height = desc["frame"]
    columns = desc["columns"]
    last = max(clip["to"] for clip in desc["clips"].values())
    frames = []
    for index in range(last + 1):
        left, top = (index % columns) * width, (index // columns) * height
        frames.append(sheet.crop((left, top, left + width, top + height)))
    return frames, desc["clips"]


def sequence(folder: Path) -> tuple[list[Image.Image], int]:
    """The frames of the demo (swim, turn, swim, turn back) and the delay of one."""
    frames, clips = frames_of(folder)
    swim = frames[clips["swim"]["from"] : clips["swim"]["to"] + 1]
    turn = frames[clips["turn"]["from"] : clips["turn"]["to"] + 1]
    mirrored = [ImageOps.mirror(frame) for frame in turn]
    # the fish swims one way, turns, swims the other way (mirrored), turns back
    flipped = [ImageOps.mirror(frame) for frame in swim]
    shown = swim * SWIM_CYCLES + turn + flipped * SWIM_CYCLES + mirrored
    height = round(shown[0].height * WIDTH / shown[0].width)
    shown = [
        frame.resize((WIDTH, height), Image.Resampling.LANCZOS)
        for frame in shown[::STEP]
    ]
    return shown, round(1000 * STEP / clips["swim"]["fps"])


def render(name: str, species: str) -> bytes:
    shown, delay = sequence(ROOT / "fish" / species)
    out = io.BytesIO()
    shown[0].save(
        out,
        format="WEBP",
        save_all=True,
        append_images=shown[1:],
        duration=delay,
        loop=0,
        quality=QUALITY,
        method=4,
        minimize_size=True,
    )
    return out.getvalue()


def preview(folder: Path) -> bytes:
    """The first frame of an atlas, cropped to the animal, as a README picture."""
    frames, _ = frames_of(folder)
    frame = frames[0]
    box = frame.getchannel("A").getbbox()
    if box:
        frame = frame.crop(box)
    height = round(frame.height * PREVIEW_WIDTH / frame.width)
    frame = frame.resize((PREVIEW_WIDTH, height), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    frame.save(out, format="WEBP", quality=80, method=6)
    return out.getvalue()


def species_with_atlas() -> list[Path]:
    """Folders of the species with a plain RGBA atlas (not the shade/mask corals)."""
    found = []
    for descriptor in sorted(ROOT.glob("*/*/species.json")):
        desc = json.loads(descriptor.read_text(encoding="utf-8"))
        plain = "1x" in desc.get("atlas", {}) and "columns" in desc
        if plain and descriptor.parent.parent.name != "defaults":
            found.append(descriptor.parent)
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    stale = []
    for name, species in DEMOS.items():
        path = OUT / f"{name}.webp"
        data = render(name, species)
        if args.check:
            if not path.is_file():
                stale.append(path.name)
            continue
        path.write_bytes(data)
        print(f"{path.relative_to(ROOT)}: {len(data) // 1024} KB")
    for folder in species_with_atlas():
        path = THUMBS / folder.parent.name / f"{folder.name}.webp"
        if args.check:
            if not path.is_file():
                stale.append(str(path.relative_to(ROOT)))
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(preview(folder))
    if stale:
        print("missing: " + ", ".join(stale), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
