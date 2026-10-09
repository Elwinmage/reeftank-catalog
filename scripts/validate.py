#!/usr/bin/env python3
"""Check the catalog entries before a release.

For every `fish/<id>/` and `corals/<id>/` folder:

- the id is url-safe;
- `species.json` parses, its `kind` matches the folder;
- the atlas files and the thumbnail it names exist, and are pictures;
- the frame size, the columns and the clips fit the atlas picture;
- only the files a release ships (species.json, webp, png) are counted, and
  they stay under the size and count limits of the integration.

Other files (a recipe, notes) are allowed in a folder: they are not shipped.

    python scripts/validate.py [catalog root]

Exit code 1 when an entry is wrong.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from PIL import Image

KINDS = {"fish": "fish", "corals": "coral"}
DESCRIPTOR = "species.json"
SHIPPED_SUFFIXES = (".webp", ".png")
#: Limits of the integration (pack.py)
MAX_ENTRY_BYTES = 32 * 1024 * 1024
MAX_ENTRY_FILES = 16
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,79}$")
FILE_RE = re.compile(r"^[A-Za-z0-9_@.-]{1,120}$")


def shipped_files(folder: Path) -> list[Path]:
    """Files of an entry that go in its release archive."""
    return sorted(
        p
        for p in folder.iterdir()
        if p.is_file()
        and (p.name == DESCRIPTOR or p.suffix.lower() in SHIPPED_SUFFIXES)
    )


def _picture(path: Path, errors: list[str], what: str) -> Image.Image | None:
    if not path.is_file():
        errors.append(f"{what}: {path.name} missing")
        return None
    try:
        image = Image.open(path)
        image.load()
    except OSError:
        errors.append(f"{what}: {path.name} is not a picture")
        return None
    return image


def check_entry(folder: Path, kind: str) -> list[str]:
    """Problems of one entry (empty when fine)."""
    errors: list[str] = []
    if not ID_RE.match(folder.name):
        errors.append("id must be lowercase letters, digits, - and _")
    try:
        desc: Any = json.loads((folder / DESCRIPTOR).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return [*errors, f"no {DESCRIPTOR}"]
    except ValueError as err:
        return [*errors, f"{DESCRIPTOR}: {err}"]
    if not isinstance(desc, dict):
        return [*errors, f"{DESCRIPTOR} is not an object"]
    if desc.get("kind") != KINDS[kind]:
        errors.append(f"kind must be {KINDS[kind]!r}")

    atlas = desc.get("atlas")
    if not isinstance(atlas, dict) or not atlas:
        errors.append("no atlas")
        atlas = {}
    sheets: dict[str, Image.Image] = {}
    for key, name in atlas.items():
        if not isinstance(name, str) or not FILE_RE.match(name):
            errors.append(f"atlas {key}: bad file name {name!r}")
            continue
        image = _picture(folder / name, errors, f"atlas {key}")
        if image is not None:
            sheets[key] = image
    thumb = desc.get("thumbnail")
    if not isinstance(thumb, str):
        errors.append("no thumbnail")
    else:
        _picture(folder / thumb, errors, "thumbnail")

    frame = desc.get("frame")
    columns = desc.get("columns")
    if (
        not isinstance(frame, list)
        or len(frame) != 2
        or not all(isinstance(v, int) and v > 0 for v in frame)
    ):
        errors.append("frame must be [width, height]")
    elif not isinstance(columns, int) or columns < 1:
        errors.append("columns must be a positive integer")
    else:
        base = sheets.get("1x") or sheets.get("shade")
        if base is not None:
            if base.width != frame[0] * columns or base.height % frame[1]:
                errors.append(
                    f"atlas {base.width}x{base.height} does not fit "
                    f"{columns} columns of {frame[0]}x{frame[1]}"
                )
            count = columns * (base.height // frame[1])
            clips = desc.get("clips")
            if not isinstance(clips, dict) or not clips:
                errors.append("no clips")
            else:
                for name, clip in clips.items():
                    ok = (
                        isinstance(clip, dict)
                        and isinstance(clip.get("from"), int)
                        and isinstance(clip.get("to"), int)
                        and 0 <= clip["from"] <= clip["to"] < count
                        and isinstance(clip.get("fps"), int)
                        and clip["fps"] > 0
                    )
                    if not ok:
                        errors.append(f"clip {name}: frames outside the atlas")
            if kind == "fish" and "swim" not in (clips or {}):
                errors.append("a fish needs a swim clip")

    files = shipped_files(folder)
    if len(files) > MAX_ENTRY_FILES:
        errors.append(f"{len(files)} files (at most {MAX_ENTRY_FILES})")
    size = sum(p.stat().st_size for p in files)
    if size > MAX_ENTRY_BYTES:
        errors.append(f"{size} bytes (at most {MAX_ENTRY_BYTES})")
    for path in files:
        if not FILE_RE.match(path.name):
            errors.append(f"bad file name {path.name!r}")
    return errors


def check_catalog(root: Path) -> dict[str, list[str]]:
    """Problems of every entry, by `kind/id`."""
    problems: dict[str, list[str]] = {}
    for kind in KINDS:
        folder = root / kind
        if not folder.is_dir():
            continue
        for entry in sorted(p for p in folder.iterdir() if p.is_dir()):
            errors = check_entry(entry, kind)
            if errors:
                problems[f"{kind}/{entry.name}"] = errors
    return problems


def main(argv: list[str] | None = None) -> int:
    """Command line entry point."""
    args = sys.argv[1:] if argv is None else argv
    root = Path(args[0]) if args else Path(__file__).resolve().parent.parent
    problems = check_catalog(root)
    for key, errors in problems.items():
        for error in errors:
            print(f"{key}: {error}")
    count = sum(
        1
        for kind in KINDS
        if (root / kind).is_dir()
        for p in (root / kind).iterdir()
        if p.is_dir()
    )
    print(f"{count} entries, {len(problems)} with problems")
    return 1 if problems else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
