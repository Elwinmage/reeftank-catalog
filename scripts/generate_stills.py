#!/usr/bin/env python3
"""Make the degraded atlas of every fish without animation, photos from Gemini.

For each species of `fish/` that has no atlas yet, the photo is asked from a
Google Gemini image model (the prompt of `prompts/still_photo.json`), checked
like any photo (`still_atlas.py`), and its degraded atlas is written next to the
descriptor, with `source` saying which model made it. A species that already has
an atlas is left alone: the script can be stopped and started again, it goes on
with the species still lacking one. An animation made from videos is never
replaced.

    export GEMINI_API_KEY=...
    python scripts/generate_stills.py --dry-run
    python scripts/generate_stills.py --limit 5 --photos photos/
    python scripts/generate_stills.py                      # the whole lot
    python scripts/generate_stills.py --only betta_splendens --redo
    python scripts/generate_stills.py --from-photos photos/    # no request at all

A photo that fails the checks is asked again (up to `--attempts`, alternating the
magenta and the green background: a pink fish is lost on magenta). `--photos`
keeps every photo received (the refused ones too) to look at them; they are not
part of the catalog. `--from-photos` builds the atlases from the photos kept by an
earlier run (`photos/<id>-<n>.png`, the refused ones too, checked again): no key
and no request, and the species without a photo are left for a later run. The run
stops at once when the key is refused or the quota is reached (exit 2): start it
again later. Exit 1 when some species failed.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from still_atlas import check_photo, is_real_animation, render_prompt, write_atlas

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
DEFAULT_MODEL = "gemini-2.5-flash-image"
KEY_ENV = "GEMINI_API_KEY"
#: Background colours asked in turn
BACKGROUNDS = ("magenta", "green")
#: HTTP statuses after which no other request can work
FATAL_STATUSES = frozenset({401, 403, 429})
TIMEOUT = 120
SECTION = "fish"
LENGTH = 192  # fish length in the atlas frame, px (see still_atlas.py)


class GenerationError(Exception):
    """A picture that could not be had; `fatal` when asking again is useless."""

    def __init__(self, message: str, *, fatal: bool = False) -> None:
        super().__init__(message)
        self.fatal = fatal


def _detail(err: urllib.error.HTTPError) -> str:
    try:
        error = json.loads(err.read()).get("error") or {}
        return str(error.get("message") or "")
    except (OSError, ValueError, AttributeError):
        return ""


def post(url: str, key: str, body: dict[str, Any]) -> dict[str, Any]:
    """POST a JSON body to the service and read its JSON answer."""
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as err:
        detail = _detail(err)
        fatal = err.code in FATAL_STATUSES or "api key" in detail.lower()
        raise GenerationError(f"HTTP {err.code} {detail}".strip(), fatal=fatal) from err
    except (urllib.error.URLError, TimeoutError, ValueError) as err:
        raise GenerationError(f"service not reachable: {err}") from err
    if not isinstance(payload, dict):
        raise GenerationError("the service did not send a JSON object")
    return payload


def generate(prompt: str, model: str, key: str) -> bytes:
    """The picture Gemini makes of a prompt (3:2, a fish is longer than high)."""
    payload = post(
        GEMINI_URL.format(model=model),
        key,
        {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
                "imageConfig": {"aspectRatio": "3:2"},
            },
        },
    )
    for candidate in payload.get("candidates") or []:
        for part in (candidate.get("content") or {}).get("parts") or []:
            inline = part.get("inlineData") or part.get("inline_data")
            if isinstance(inline, dict) and inline.get("data"):
                try:
                    return base64.b64decode(str(inline["data"]), validate=True)
                except (binascii.Error, ValueError) as err:
                    raise GenerationError("the picture is not readable") from err
    block = (payload.get("promptFeedback") or {}).get("blockReason")
    raise GenerationError(
        f"prompt blocked: {block}" if block else "the service sent no picture"
    )


def read_descriptor(folder: Path) -> dict[str, Any] | None:
    """The descriptor of an entry (None when missing or not an object)."""
    try:
        data = json.loads((folder / "species.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def targets(root: Path, only: list[str], redo: bool) -> list[Path]:
    """The entries to make an atlas for: without one, or degraded with `redo`."""
    found = []
    for folder in sorted((root / SECTION).iterdir()):
        descriptor = read_descriptor(folder)
        if descriptor is None or (only and folder.name not in only):
            continue
        if "atlas" in descriptor and (is_real_animation(descriptor) or not redo):
            continue
        found.append(folder)
    return found


def species_name(folder: Path) -> str:
    """The scientific name to ask for (the folder name when the entry has none)."""
    descriptor = read_descriptor(folder) or {}
    name = descriptor.get("scientific")
    return str(name) if name else folder.name.replace("_", " ").capitalize()


def make_atlas(folder: Path, args: argparse.Namespace, api_key: str) -> str | None:
    """Photo and atlas of an entry; None when done, else why it failed."""
    name = species_name(folder)
    reason = "no attempt"
    for attempt in range(args.attempts):
        background = args.background or BACKGROUNDS[attempt % len(BACKGROUNDS)]
        try:
            blob = generate(render_prompt(name, background), args.model, api_key)
        except GenerationError as err:
            if err.fatal:
                raise
            reason = str(err)
            time.sleep(args.delay)
            continue
        time.sleep(args.delay)
        try:
            photo = Image.open(io.BytesIO(blob))
            photo.load()
        except (OSError, ValueError):
            reason = "the picture is not an image"
            continue
        problems = check_photo(photo, background)
        if args.photos:
            args.photos.mkdir(parents=True, exist_ok=True)
            suffix = "-refused" if problems else ""
            photo.save(args.photos / f"{folder.name}-{attempt + 1}{suffix}.png")
        if problems:
            reason = "; ".join(problems)
            continue
        build(photo, folder, args, background)
        return None
    return reason


def build(
    photo: Image.Image, folder: Path, args: argparse.Namespace, background: str
) -> None:
    """Write the atlas of an entry, the photo being made by the model of the run."""
    source = {
        "type": "ai-generated",
        "model": args.model,
        "date": datetime.now(UTC).date().isoformat(),
    }
    write_atlas(photo, folder, source, LENGTH, background)


def saved_photos(directory: Path, name: str) -> list[tuple[int, Path]]:
    """The photos a run kept for a species, the accepted ones before the refused."""
    found = []
    for path in directory.glob(f"{name}-*.png"):
        match = re.fullmatch(rf"{re.escape(name)}-(\d+)(-refused)?\.png", path.name)
        if match:
            found.append((bool(match[2]), int(match[1]), path))
    return [(number, path) for _, number, path in sorted(found)]


def make_atlas_from_photos(
    folder: Path, photos: list[tuple[int, Path]], args: argparse.Namespace
) -> str | None:
    """Atlas of an entry from the photos kept for it; None when done, else why not.

    The background of a photo is the one that was asked at its attempt (they
    alternate), or the other when that one does not fit.
    """
    reason = "no photo fits"
    for number, path in photos:
        try:
            photo = Image.open(path)
            photo.load()
        except (OSError, ValueError):
            reason = f"{path.name} is not an image"
            continue
        first = (number - 1) % len(BACKGROUNDS)
        order = [BACKGROUNDS[first], BACKGROUNDS[1 - first]]
        for background in [args.background] if args.background else order:
            problems = check_photo(photo, background)
            if not problems:
                build(photo, folder, args, background)
                return None
            reason = f"{path.name}: " + "; ".join(problems)
    return reason


def run(args: argparse.Namespace, root: Path) -> int:
    """Make the atlases; 0 all done, 1 some failed, 2 stopped by the service."""
    todo = targets(root, args.only or [], args.redo)
    if args.limit:
        todo = todo[: args.limit]
    if args.dry_run:
        for folder in todo:
            print(f"{SECTION}/{folder.name}: {species_name(folder)}")
        print(f"{len(todo)} species, {len(todo) * args.attempts} requests at most")
        return 0
    if args.from_photos:
        return run_from_photos(args, todo)
    api_key = os.environ.get(KEY_ENV, "")
    if not api_key:
        print(f"set {KEY_ENV} to the Gemini API key")
        return 2
    failed = []
    for folder in todo:
        try:
            reason = make_atlas(folder, args, api_key)
        except GenerationError as err:
            print(f"stopped on {folder.name}: {err}; start again later")
            return 2
        if reason is None:
            print(f"{SECTION}/{folder.name}: done")
        else:
            failed.append(folder.name)
            print(f"{SECTION}/{folder.name}: failed ({reason})")
    print(f"{len(todo) - len(failed)} done, {len(failed)} failed")
    return 1 if failed else 0


def run_from_photos(args: argparse.Namespace, todo: list[Path]) -> int:
    """Make the atlases from the photos of an earlier run; 0 all done, 1 some failed."""
    done, failed, waiting = 0, [], 0
    for folder in todo:
        photos = saved_photos(args.from_photos, folder.name)
        if not photos:
            waiting += 1
            continue
        reason = make_atlas_from_photos(folder, photos, args)
        if reason is None:
            done += 1
            print(f"{SECTION}/{folder.name}: done")
        else:
            failed.append(folder.name)
            print(f"{SECTION}/{folder.name}: failed ({reason})")
    print(f"{done} done, {len(failed)} failed, {waiting} without photo")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini image model")
    parser.add_argument("--only", nargs="+", metavar="ID", help="these species only")
    parser.add_argument("--limit", type=int, help="at most this many species")
    parser.add_argument("--redo", action="store_true", help="also the degraded ones")
    parser.add_argument("--attempts", type=int, default=3, help="requests per species")
    parser.add_argument("--background", choices=BACKGROUNDS, help="always this one")
    parser.add_argument(
        "--delay", type=float, default=3.0, help="seconds between requests"
    )
    parser.add_argument("--photos", type=Path, help="keep the photos received there")
    parser.add_argument(
        "--from-photos", type=Path, help="build from the photos kept there, no request"
    )
    parser.add_argument("--dry-run", action="store_true", help="list the species")
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parent.parent
    )
    args = parser.parse_args(argv)
    if args.attempts < 1:
        parser.error("--attempts must be at least 1")
    return run(args, args.root)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
