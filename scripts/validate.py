#!/usr/bin/env python3
"""Check the catalog entries before a release.

For every `fish/<id>/` and `corals/<id>/` folder:

- the id is url-safe;
- `species.json` parses, its `kind` matches the folder;
- the atlas files and the thumbnail it names exist, and are pictures;
- the frame size, the columns and the clips fit the atlas picture;
- only the files a release ships (species.json, webp, png) are counted, and
  they stay under the size and count limits of the integration.

For every `textures/<id>/` folder: `texture.json` names a role (rock, sand),
a tileable picture that exists, and the tank length it covers (`scale_cm`).

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

#: Species folders, with the `kind` of their descriptor
KINDS = {"fish": "fish", "corals": "coral"}
DESCRIPTOR = "species.json"
TEXTURES = "textures"
TEXTURE_DESCRIPTOR = "texture.json"
TEXTURE_ROLES = ("rock", "sand")
#: Every folder of entries a release ships, with their descriptor
DESCRIPTORS = {"fish": DESCRIPTOR, "corals": DESCRIPTOR, TEXTURES: TEXTURE_DESCRIPTOR}
SHIPPED_SUFFIXES = (".webp", ".png")
#: Limits of the integration (pack.py)
MAX_ENTRY_BYTES = 32 * 1024 * 1024
MAX_ENTRY_FILES = 16
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,79}$")
FILE_RE = re.compile(r"^[A-Za-z0-9_@.-]{1,120}$")


def shipped_files(folder: Path) -> list[Path]:
    """Files of an entry that go in its release archive."""
    descriptor = DESCRIPTORS.get(folder.parent.name, DESCRIPTOR)
    return sorted(
        p
        for p in folder.iterdir()
        if p.is_file()
        and (p.name == descriptor or p.suffix.lower() in SHIPPED_SUFFIXES)
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


PROFILES = ("shoal", "cruiser", "benthic", "hover", "sand")
#: Languages every species is named in
LANGUAGES = ("en", "fr", "de", "es", "it", "nl", "pl", "pt")
#: Fields a species must describe (dotted: inside `behavior`)
REQUIRED = {
    "fish": (
        "scientific",
        "size_cm",
        "behavior.profile",
        "behavior.band",
        "behavior.depth",
        "behavior.speed_cm_s",
        "behavior.turn_rate",
        "night",
        "feeding_response",
    ),
    "corals": ("scientific", "size_cm", "palette"),
}
NIGHT_MODES = ("hide", "hover", "rest_bottom")
_HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
#: "Genus species", "Genus species subspecies", or "Genus sp." for a genus
SCIENTIFIC_RE = re.compile(r"^[A-Z][a-z]+ (sp\.|[a-z]+(-[a-z]+)?( [a-z]+)?)$")


def _number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _pair(value: Any, low: float, high: float | None = None) -> bool:
    """[a, b] with low <= a <= b (<= high)."""
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(_number(v) for v in value)
        and low <= value[0] <= value[1]
        and (high is None or value[1] <= high)
    )


def missing_fields(desc: dict[str, Any], kind: str) -> list[str]:
    """Description fields still to fill (names in every language too)."""
    missing: list[str] = []
    for field in REQUIRED[kind]:
        value: Any = desc
        for part in field.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        if value is None:
            missing.append(field)
    names = desc.get("names")
    names = names if isinstance(names, dict) else {}
    missing += [
        f"names.{lang}"
        for lang in LANGUAGES
        if not isinstance(names.get(lang), str) or not names[lang].strip()
    ]
    return missing


def check_description(desc: dict[str, Any], kind: str) -> list[str]:
    """Problems of the hand written fields (names, size, behaviour...)."""
    errors: list[str] = []
    missing = missing_fields(desc, kind)
    if missing:
        errors.append(
            "to describe: "
            + ", ".join(missing)
            + " (scripts/species_meta.py export, fill, import)"
        )
    names = desc.get("names")
    if names is not None and (
        not isinstance(names, dict)
        or not all(isinstance(v, str) and v.strip() for v in names.values())
    ):
        errors.append("names: language -> non empty text")
    scientific = desc.get("scientific")
    if scientific is not None and (
        not isinstance(scientific, str) or not SCIENTIFIC_RE.match(scientific)
    ):
        errors.append('scientific must be "Genus species" or "Genus sp." (for a genus)')
    if "size_cm" in desc and not _pair(desc["size_cm"], 0.1):
        errors.append("size_cm must be [min, max] in cm")
    if kind == "fish":
        behavior = desc.get("behavior", {})
        if not isinstance(behavior, dict):
            errors.append("behavior must be an object")
            behavior = {}
        if behavior.get("profile", "shoal") not in PROFILES:
            errors.append(f"behavior.profile must be one of {', '.join(PROFILES)}")
        for key in ("band", "depth"):
            if key in behavior and not _pair(behavior[key], 0, 1):
                errors.append(f"behavior.{key} must be [a, b] within 0..1")
        if "speed_cm_s" in behavior and not _pair(behavior["speed_cm_s"], 0.1):
            errors.append("behavior.speed_cm_s must be [min, max] in cm/s")
        territory = behavior.get("territory_cm")
        if territory is not None and (not _number(territory) or territory <= 0):
            errors.append("behavior.territory_cm must be a positive number")
        rate = behavior.get("turn_rate", 1)
        if not _number(rate) or rate <= 0:
            errors.append("behavior.turn_rate must be a positive number")
        shoal = behavior.get("shoal", {})
        if not isinstance(shoal, dict) or not all(
            _number(v) and v >= 0 for v in shoal.values()
        ):
            errors.append("behavior.shoal: weights must be positive numbers")
        if desc.get("night", "hide") not in NIGHT_MODES:
            errors.append(f"night must be one of {', '.join(NIGHT_MODES)}")
        response = desc.get("feeding_response", 1)
        if not _number(response) or not 0 <= response <= 1:
            errors.append("feeding_response must be within 0..1")
    else:
        palette = desc.get("palette")
        if palette is not None and (
            not isinstance(palette, list)
            or not 1 <= len(palette) <= 4
            or not all(
                isinstance(c, dict)
                and isinstance(c.get("default"), str)
                and _HEX_RE.match(c["default"])
                and isinstance(c.get("fluo", False), bool)
                for c in palette
            )
        ):
            errors.append('palette: 1 to 4 {"default": "#rrggbb", "fluo": bool}')
    return errors


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
    errors += check_description(desc, kind)

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


def check_texture(folder: Path) -> list[str]:
    """Problems of one texture (empty when fine)."""
    errors: list[str] = []
    if not ID_RE.match(folder.name):
        errors.append("id must be lowercase letters, digits, - and _")
    try:
        desc: Any = json.loads(
            (folder / TEXTURE_DESCRIPTOR).read_text(encoding="utf-8")
        )
    except FileNotFoundError:
        return [*errors, f"no {TEXTURE_DESCRIPTOR}"]
    except ValueError as err:
        return [*errors, f"{TEXTURE_DESCRIPTOR}: {err}"]
    if not isinstance(desc, dict):
        return [*errors, f"{TEXTURE_DESCRIPTOR} is not an object"]
    if desc.get("kind") != "texture":
        errors.append("kind must be 'texture'")
    if desc.get("role") not in TEXTURE_ROLES:
        errors.append(f"role must be one of {', '.join(TEXTURE_ROLES)}")
    scale = desc.get("scale_cm")
    if not _number(scale) or scale <= 0:
        errors.append("scale_cm must be a positive number (cm the picture covers)")
    image = desc.get("image")
    if not isinstance(image, str) or not FILE_RE.match(image):
        errors.append("no image")
    else:
        picture = _picture(folder / image, errors, "image")
        if picture is not None and max(picture.size) > 2048:
            errors.append("image: 2048 pixels at most")
    names = desc.get("names", {})
    if not isinstance(names, dict) or not all(
        isinstance(v, str) and v.strip() for v in names.values()
    ):
        errors.append("names: language -> non empty text")
    files = shipped_files(folder)
    size = sum(p.stat().st_size for p in files)
    if size > MAX_ENTRY_BYTES:
        errors.append(f"{size} bytes (at most {MAX_ENTRY_BYTES})")
    return errors


def check_catalog(root: Path) -> dict[str, list[str]]:
    """Problems of every entry, by `kind/id`."""
    problems: dict[str, list[str]] = {}
    for kind in DESCRIPTORS:
        folder = root / kind
        if not folder.is_dir():
            continue
        for entry in sorted(p for p in folder.iterdir() if p.is_dir()):
            errors = (
                check_texture(entry) if kind == TEXTURES else check_entry(entry, kind)
            )
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
        for kind in DESCRIPTORS
        if (root / kind).is_dir()
        for p in (root / kind).iterdir()
        if p.is_dir()
    )
    print(f"{count} entries, {len(problems)} with problems")
    return 1 if problems else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
