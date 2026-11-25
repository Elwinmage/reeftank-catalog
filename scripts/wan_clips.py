#!/usr/bin/env python3
"""Real animations of the animals that only have a degraded atlas, made with Wan on your GPU.

A fish whose atlas comes from a single photo (`"degraded": true`, see
`generate_stills.py`) gets its three clips from Wan, run by a local ComfyUI, with
a first and a last frame: the swim and the idle start and end on the reference
photo (seamless loops), the turn starts on it and ends on its mirror. That is
what `build_atlas.py --auto` expects; it builds the atlas. The new animation is
staged next to the clips and only goes to the catalog once accepted, after a look
at it played in a loop.

`--section invertebrates` animates the shrimps, the crabs and the hermit crabs
instead of the fish: the same three clips (`swim` is their walk, `idle` and
`turn`), worded for the animal by `prompts/wan_invert.json`, which carries one
motion sentence per form (shrimp, crab, hermit) on top of its clip prompts. The
snails, the stars, the urchins and the sessile species are never animated this
way: their still picture is what the card makes their motion from.

The reference photo of a species is the one `generate_stills.py --photos photos/`
kept (`photos/<id>-<n>.png`, the accepted ones first). With `generate --gemini`
(and `GEMINI_API_KEY`), a fish without one gets a new photo from Gemini, asked
and checked as `generate_stills.py` does, kept in `photos/`; the catalog (its
degraded atlas) is not touched. The prompts are those of
`prompts/wan_clips.json` (one per clip, `[SPECIE]` = scientific name,
`[BACKGROUND]` = the key colour of the photo, magenta or green).

Cutting out: Wan does not keep the key colour (it dulls the magenta, sometimes
with a vignette), so a colour key either keeps the whole frame or eats into
the fish. The frames are cut out by a background removal model instead (rembg,
`--matte-model`, `isnet-general-use` by default; on the CPU, `--matte-gpu` to
use the GPU when ComfyUI is not running), their edges unmixed from the
background colour, into `clips/<id>/matte/<clip>/`; the recipe points
`build_atlas.py` to them (`"key": "none"`). Install it where the script runs:
`pip install "rembg[cpu]"` (or `"rembg[gpu]"`); the model is downloaded on its
first use.

ComfyUI workflow
----------------

Build and tune the workflow in ComfyUI (two Load Image, the Wan model, the
prompts, Save Video), then export it with Workflow > Export (API) and let the
script put its placeholders in it:

    python scripts/wan_clips.py template ~/Downloads/workflow_api.json

It writes `clips/wan_workflow.json`, the copy where the values the script gives
are placeholders, and prints what it changed: `{{FIRST_FRAME}}`,
`{{LAST_FRAME}}` (the images of the two Load Image), `{{PROMPT}}`,
`{{NEGATIVE}}`, `{{SEED}}`, `{{WIDTH}}`, `{{HEIGHT}}`, `{{LENGTH}}` (frames) or
`{{DURATION}}` (seconds), `{{STRENGTH}}` (how closely the model has to follow the
two reference frames), `{{PREFIX}}` (the file name of Save Video). Check its
report (the first and last images in particular) and edit the copy by hand if
needed: any input value written as one of these placeholders is filled.

Commands
--------

    python scripts/wan_clips.py todo                     # the fish to animate
    python scripts/wan_clips.py prompt canthigaster_valentini turn
    python scripts/wan_clips.py generate --limit 3       # clips + staged atlas
    python scripts/wan_clips.py generate --limit 3 --gemini  # missing photos too
    python scripts/wan_clips.py generate --only betta_splendens
    python scripts/wan_clips.py preview betta_splendens  # loop in the browser
    python scripts/wan_clips.py review                   # look, accept or redo
    python scripts/wan_clips.py gallery --no-open        # every new one, one page
    python scripts/wan_clips.py accept betta_splendens   # into fish/
    python scripts/wan_clips.py accept                   # all that is staged
    python scripts/wan_clips.py redo betta_splendens --clips turn
    python scripts/wan_clips.py build --only betta_splendens  # after a recipe edit

Every command takes `--section invertebrates` to work on the shrimps, the crabs
and the hermit crabs instead of the fish:

    python scripts/wan_clips.py --section invertebrates todo
    python scripts/wan_clips.py --section invertebrates prompt neocaridina_davidi swim
    python scripts/wan_clips.py --section invertebrates generate --limit 3 --gemini

`generate` skips the clips already made, so it can be stopped and started again
(ComfyUI killed for want of memory, say); it stops the run when ComfyUI cannot
be reached (exit 2). The work folder (`clips/`, not committed) holds, per
species: `reference.png` and `reference_mirror.png` (the photo fitted to the
size of the video), `swim.mp4`, `idle.mp4`, `turn.mp4`, `meta.json` (the
`source` of the animation), `wan.json` (prompts and seeds), the recipe
`<id>.json` of `build_atlas.py` (edit it and run `build` to change a range),
`build/<id>/` (the staged entry) and `preview.webp` / `preview.html`.

`gallery` writes `clips/gallery.html`: the new animation of every staged species
not accepted yet (`--all`: accepted ones too), in a loop, on one page; then
`accept` the good ones and `redo` the others.

`preview` plays, side by side, the current atlas of the catalog (left) and the
new one (right): swim, idle, turn, swim back, turn back, in a loop, on a dark
blue background. It opens `preview.html` with `xdg-open` (`--player` another
command, `--no-open` only writes it).
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import os
import random
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "client"))
import build_atlas
import generate_stills as stills
from generate_stills import read_descriptor, saved_photos, species_name
from make_demo import frames_of
from reeftank_catalog_invert import invert_form
from reeftank_catalog_still import corners, is_real_animation, nearest_key
from reeftank_catalog_still import detect_key as photo_key
from validate import KINDS

ROOT = Path(__file__).resolve().parent.parent
PROMPT_FILE = ROOT / "prompts" / "wan_clips.json"
#: Wording of the clips, by section (the fish by default)
PROMPT_FILES = {
    "fish": PROMPT_FILE,
    "invertebrates": ROOT / "prompts" / "wan_invert.json",
}
SECTIONS = tuple(PROMPT_FILES)
#: `swim` is the walk of an invertebrate: the name is kept so that the whole
#: chain (this script, `build_atlas.py`, the card) reads one set of clips.
CLIP_NAMES = ("swim", "idle", "turn")
DEFAULT_COMFY = "http://127.0.0.1:8188"
WORKFLOW_NAME = "wan_workflow.json"
#: Background removal model of rembg, and the alpha below which a pixel is dropped
DEFAULT_MATTE = "isnet-general-use"
MATTE_FLOOR = 0.03
VIDEO_EXTENSIONS = (".mp4", ".webm", ".mov", ".mkv", ".gif", ".webp")
#: Placeholders the generation fills, and the ones a workflow cannot do without
PLACEHOLDERS = (
    "FIRST_FRAME",
    "LAST_FRAME",
    "PROMPT",
    "NEGATIVE",
    "SEED",
    "WIDTH",
    "HEIGHT",
    "LENGTH",
    "DURATION",
    "STRENGTH",
    "PREFIX",
)
REQUIRED = ("FIRST_FRAME", "LAST_FRAME", "PROMPT")
#: Preview: background, zoom of a frame, rate of the timeline (frames a second)
PREVIEW_BACKGROUND = (16, 48, 88, 255)
PREVIEW_ZOOM = 2
PREVIEW_FPS = 24
PREVIEW_SWIMS = 3
TIMEOUT = 30


class ComfyError(Exception):
    """A clip ComfyUI did not make; `fatal` when the next ones cannot work either."""

    def __init__(self, message: str, *, fatal: bool = False) -> None:
        super().__init__(message)
        self.fatal = fatal


# ---------------------------------------------------------------------------
# Species and prompts
# ---------------------------------------------------------------------------


def load_spec(path: Path = PROMPT_FILE) -> dict[str, Any]:
    """The Wan settings: model, size, frame rate, clips and their prompts."""
    return json.loads(path.read_text(encoding="utf-8"))


def clip_prompt(
    spec: dict[str, Any],
    clip: str,
    species: str,
    background: str = "magenta",
    form: str | None = None,
) -> str:
    """The prompt of a clip for a species, on its background colour.

    A section whose wording carries `motions` (the invertebrates) puts the
    sentence of the animal's form into its clip prompts: a shrimp walks on its
    legs, a crab sideways, a hermit crab drags its shell.
    """
    prompt = str(spec["clips"][clip]["prompt"]).replace("[SPECIE]", species)
    motions = spec.get("motions", {}).get(clip)
    if motions:
        if form not in motions:
            raise SystemExit(
                f"{species}: no {clip} for a {form or 'species without a form'} "
                f"(animated with clips: {', '.join(motions)})"
            )
        prompt = prompt.replace("[MOTION]", str(motions[form]))
    return prompt.replace("[BACKGROUND]", background)


def entry_form(root: Path, section: str, entry_id: str) -> str | None:
    """The form of an entry (shrimp, crab...), None outside the invertebrates."""
    if section != "invertebrates":
        return None
    return invert_form(entry_id, read_descriptor(root / section / entry_id) or {})


def reference_background(folder: Path) -> tuple[str, tuple[int, int, int]]:
    """Name (magenta, green) and actual colour of the background of a reference.

    The colour an image model gives is rarely the pure key it was asked for,
    and `detect_key` names only the saturated ones. The prompt of a clip has
    to name the colour the video model is actually shown, so a background it
    will not name is read as the nearer of the two keys rather than magenta.
    """
    rgb = np.asarray(Image.open(folder / "reference.png").convert("RGB"))
    colour = tuple(int(v) for v in np.median(corners(rgb), axis=0))
    return photo_key(rgb) or nearest_key(rgb), (colour[0], colour[1], colour[2])


def clip_strength(spec: dict[str, Any]) -> float:
    """How closely the model has to follow the two reference frames, 0 to 1.

    A control model (VACE) told to follow them exactly has only one cheap way to
    reach the same pose it started from: barely move. Lowering this gives the
    prompt room, which the fish do not need (they undulate, a small motion) but
    the walking invertebrates do.
    """
    return float(spec.get("strength", 1.0))


def duration_of(spec: dict[str, Any], clip: str) -> float | int:
    """Seconds of a clip (a whole number when it is one): (frames - 1) / fps."""
    seconds = (int(spec["clips"][clip]["frames"]) - 1) / int(spec["fps"])
    return int(seconds) if seconds.is_integer() else round(seconds, 3)


def animated_forms(spec: dict[str, Any]) -> set[str] | None:
    """The forms a wording makes clips for; None when every entry is animated.

    Only the shrimps, the crabs and the hermit crabs are worth filming: a
    snail, a starfish or an urchin is drawn from its still picture and the
    card gives it its own slow motion.
    """
    motions = spec.get("motions")
    if not motions:
        return None
    return set.intersection(*(set(clip) for clip in motions.values()))


def targets(
    root: Path,
    only: list[str] | None = None,
    section: str = "fish",
    forms: set[str] | None = None,
) -> list[Path]:
    """The entries without a real animation (degraded atlas, or none).

    `forms` keeps only the entries whose form is filmed at all (see
    `animated_forms`); an entry left out is never asked for, even by `--only`.
    """
    found = []
    for folder in sorted((root / section).iterdir()):
        descriptor = read_descriptor(folder)
        if descriptor is None or (only and folder.name not in only):
            continue
        if forms is not None and invert_form(folder.name, descriptor) not in forms:
            continue
        if not is_real_animation(descriptor):
            found.append(folder)
    return found


def reference_photo(photos: Path, entry_id: str) -> Path | None:
    """The photo a species is animated from: the first accepted one kept for it."""
    if not photos.is_dir():
        return None
    for _, path in saved_photos(photos, entry_id):
        if not path.stem.endswith("-refused"):
            return path
    return None


def ask_photo(
    folder: Path,
    photos: Path,
    key: str,
    model: str,
    attempts: int,
    delay: float,
    section: str = "fish",
) -> Path | None:
    """A new reference photo of an entry from Gemini, kept in `photos`.

    The prompt and the checks are those of `generate_stills.py` (magenta, then
    green background); every photo received is kept, the refused ones as
    `<id>-<n>-refused.png`. None when no photo fits; a fatal `GenerationError`
    (key refused, quota) goes up.
    """
    photos.mkdir(parents=True, exist_ok=True)
    number = max((n for n, _ in saved_photos(photos, folder.name)), default=0)
    for attempt in range(attempts):
        background = stills.BACKGROUNDS[attempt % len(stills.BACKGROUNDS)]
        prompt, aspect = stills.request_for(folder, section, background)
        number += 1
        try:
            blob = stills.generate(prompt, model, key, aspect)
            photo = Image.open(io.BytesIO(blob))
            photo.load()
        except stills.GenerationError as err:
            if err.fatal:
                raise
            print(f"{folder.name}: no photo ({err})")
            time.sleep(delay)
            continue
        except (OSError, ValueError):
            print(f"{folder.name}: the picture is not an image")
            continue
        time.sleep(delay)
        problems = stills.problems_of(photo, background, section)
        path = photos / f"{folder.name}-{number}{'-refused' if problems else ''}.png"
        photo.save(path)
        if not problems:
            return path
        print(f"{folder.name}: photo refused ({'; '.join(problems)})")
    return None


def fit_reference(photo: Image.Image, size: tuple[int, int]) -> Image.Image:
    """The photo at the size of the video, padded with its own background colour.

    The ratio is kept (a fish is not squashed): the photo is padded on the sides
    (or above and below) with the median colour of its corners, then resized.
    """
    rgb = photo.convert("RGB")
    colour = tuple(int(v) for v in np.median(corners(np.asarray(rgb)), axis=0))
    ratio = size[0] / size[1]
    width, height = rgb.size
    if width / height < ratio:
        canvas = Image.new("RGB", (round(height * ratio), height), colour)
    else:
        canvas = Image.new("RGB", (width, round(width / ratio)), colour)
    canvas.paste(rgb, ((canvas.width - width) // 2, (canvas.height - height) // 2))
    return canvas.resize(size, Image.Resampling.LANCZOS)


def prepare_references(
    photo_path: Path, folder: Path, size: tuple[int, int]
) -> tuple[Path, Path]:
    """`reference.png` and `reference_mirror.png` of a species folder."""
    folder.mkdir(parents=True, exist_ok=True)
    reference = fit_reference(Image.open(photo_path), size)
    first, mirror = folder / "reference.png", folder / "reference_mirror.png"
    reference.save(first)
    ImageOps.mirror(reference).save(mirror)
    return first, mirror


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


def _token(name: str) -> str:
    return "{{" + name + "}}"


def load_workflow(path: Path) -> dict[str, Any]:
    """A workflow in the API format of ComfyUI (node id -> class_type, inputs)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "nodes" in data:
        raise SystemExit(
            f"{path}: not in the API format (ComfyUI: Workflow > Export (API))"
        )
    if not all(isinstance(n, dict) and "class_type" in n for n in data.values()):
        raise SystemExit(f"{path}: every node needs a class_type")
    return data


def placeholders_in(workflow: dict[str, Any]) -> set[str]:
    """The placeholders a workflow holds."""
    found = set()
    for node in workflow.values():
        for value in (node.get("inputs") or {}).values():
            for name in PLACEHOLDERS:
                if value == _token(name):
                    found.add(name)
    return found


def fill(workflow: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    """A copy of the workflow, its placeholders replaced by `values`."""
    filled = copy.deepcopy(workflow)
    tokens = {_token(k): v for k, v in values.items()}
    for node in filled.values():
        inputs = node.get("inputs") or {}
        for key, value in inputs.items():
            if isinstance(value, str) and value in tokens:
                inputs[key] = tokens[value]
    return filled


def _is_link(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], str)


def _consumers(workflow: dict[str, Any]) -> dict[str, list[tuple[str, str]]]:
    """For every node, the (node, input) pairs its outputs go to."""
    found: dict[str, list[tuple[str, str]]] = {}
    for node_id, node in workflow.items():
        for key, value in (node.get("inputs") or {}).items():
            if _is_link(value):
                found.setdefault(value[0], []).append((node_id, key))
    return found


def _role(
    node_id: str,
    consumers: dict[str, list[tuple[str, str]]],
    words: dict[str, tuple[str, ...]],
    depth: int = 6,
) -> str | None:
    """The role of a node from the inputs it feeds, looked for downstream."""
    seen, layer = {node_id}, [node_id]
    for _ in range(depth):
        following = []
        for current in layer:
            for target, key in consumers.get(current, []):
                for role, names in words.items():
                    if any(name in key.lower() for name in names):
                        return role
                if target not in seen:
                    seen.add(target)
                    following.append(target)
        layer = following
    return None


def _text_target(
    workflow: dict[str, Any], node_id: str
) -> tuple[dict[str, Any], str] | None:
    """Where the text of a text encoder is written: its input, or the node it reads."""
    inputs = workflow[node_id].get("inputs") or {}
    value = inputs.get("text")
    if isinstance(value, str):
        return inputs, "text"
    if _is_link(value) and value[0] in workflow:
        source = workflow[value[0]].get("inputs") or {}
        for key in ("value", "text", "string", "prompt"):
            if isinstance(source.get(key), str):
                return source, key
    return None


def _order(node_id: str) -> tuple[int, str]:
    return (int(node_id), node_id) if node_id.isdigit() else (1 << 30, node_id)


def make_template(workflow: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Put the placeholders in an exported workflow; also says what it changed."""
    result = copy.deepcopy(workflow)
    consumers = _consumers(result)
    report: list[str] = []

    def put(node_id: str, inputs: dict[str, Any], key: str, name: str) -> None:
        title = (result[node_id].get("_meta") or {}).get("title")
        label = title or result[node_id]["class_type"]
        report.append(
            f"{_token(name)} <- node {node_id} ({label}).{key} = {inputs[key]!r}"
        )
        inputs[key] = _token(name)

    images = sorted(
        (i for i, n in result.items() if n["class_type"] == "LoadImage"), key=_order
    )
    if len(images) != 2:
        raise SystemExit(f"the workflow needs 2 Load Image nodes, it has {len(images)}")
    words = {"FIRST_FRAME": ("first", "start"), "LAST_FRAME": ("last", "end")}
    roles = {i: _role(i, consumers, words) for i in images}
    if sorted(map(str, roles.values())) != ["FIRST_FRAME", "LAST_FRAME"]:
        roles = dict(zip(images, ("FIRST_FRAME", "LAST_FRAME"), strict=True))
        report.append("first and last images told apart by node order: check them")
    for node_id in images:
        put(node_id, result[node_id]["inputs"], "image", str(roles[node_id]))

    texts = {"NEGATIVE": ("negative",), "PROMPT": ("positive",)}
    for node_id in sorted(result, key=_order):
        node = result[node_id]
        if "TextEncode" in node["class_type"]:
            role = _role(node_id, consumers, texts)
            target = _text_target(result, node_id)
            if role and target:
                put(node_id, target[0], target[1], role)
        inputs = node.get("inputs") or {}
        for key, name in (
            ("seed", "SEED"),
            ("noise_seed", "SEED"),
            ("width", "WIDTH"),
            ("height", "HEIGHT"),
            ("length", "LENGTH"),
            ("duration", "DURATION"),
            ("strength", "STRENGTH"),
        ):
            value = inputs.get(key)
            if isinstance(value, int | float) and not isinstance(value, bool):
                put(node_id, inputs, key, name)
        if isinstance(inputs.get("filename_prefix"), str):
            put(node_id, inputs, "filename_prefix", "PREFIX")
    missing = [n for n in REQUIRED if n not in placeholders_in(result)]
    if missing:
        report.append("missing, to put by hand: " + ", ".join(map(_token, missing)))
    return result, report


# ---------------------------------------------------------------------------
# ComfyUI
# ---------------------------------------------------------------------------


def _call(
    url: str, data: bytes | None = None, headers: dict[str, str] | None = None
) -> bytes:
    """One request to ComfyUI; a fatal error when it cannot be reached."""
    request = urllib.request.Request(url, data=data, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return response.read()
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", "replace")[:500]
        raise ComfyError(f"HTTP {err.code} {detail}".strip()) from err
    except (urllib.error.URLError, TimeoutError, ConnectionError) as err:
        raise ComfyError(f"ComfyUI not reachable: {err}", fatal=True) from err


def _json(blob: bytes) -> dict[str, Any]:
    try:
        data = json.loads(blob)
    except ValueError as err:
        raise ComfyError("ComfyUI did not answer JSON") from err
    if not isinstance(data, dict):
        raise ComfyError("ComfyUI did not answer a JSON object")
    return data


def upload(base: str, path: Path, name: str) -> str:
    """Send an image to the input folder of ComfyUI; the name to load it by."""
    boundary = uuid.uuid4().hex
    body = b"".join(
        [
            f'--{boundary}\r\nContent-Disposition: form-data; name="image"; '
            f'filename="{name}"\r\nContent-Type: image/png\r\n\r\n'.encode(),
            path.read_bytes(),
            f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="overwrite"'
            f"\r\n\r\ntrue\r\n--{boundary}--\r\n".encode(),
        ]
    )
    answer = _json(
        _call(
            f"{base}/upload/image",
            body,
            {"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
    )
    folder = answer.get("subfolder") or ""
    return f"{folder}/{answer['name']}" if folder else str(answer["name"])


def queue(base: str, workflow: dict[str, Any]) -> str:
    """Queue a workflow; its prompt id."""
    body = json.dumps({"prompt": workflow, "client_id": uuid.uuid4().hex}).encode()
    answer = _json(_call(f"{base}/prompt", body, {"Content-Type": "application/json"}))
    if not answer.get("prompt_id"):
        raise ComfyError(f"workflow refused: {answer.get('error') or answer}")
    return str(answer["prompt_id"])


def _failure(status: dict[str, Any]) -> str:
    """The error of a failed workflow, without the inputs ComfyUI dumps with it."""
    for message in status.get("messages") or []:
        if (
            isinstance(message, list)
            and len(message) == 2
            and message[0] == "execution_error"
            and isinstance(message[1], dict)
        ):
            error = message[1]
            text = " ".join(
                str(error.get(key, "")).strip()
                for key in ("node_type", "exception_type", "exception_message")
            )
            return " ".join(text.split())[:500]
    return "no error message"


def wait(
    base: str,
    prompt_id: str,
    timeout: float,
    poll: float = 5.0,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """The outputs of a queued workflow, once it is done."""
    waited = 0.0
    while True:
        history = _json(_call(f"{base}/history/{prompt_id}"))
        entry = history.get(prompt_id)
        if isinstance(entry, dict):
            status = entry.get("status") or {}
            if status.get("status_str") == "error":
                raise ComfyError(f"the workflow failed: {_failure(status)}")
            if status.get("completed", True):
                return entry.get("outputs") or {}
        if waited >= timeout:
            raise ComfyError(f"no video after {timeout:g} s")
        sleep(poll)
        waited += poll


def video_of(outputs: dict[str, Any]) -> dict[str, Any]:
    """The file of the video a workflow wrote (Save Video, Video Combine...)."""
    for node in outputs.values():
        for files in node.values() if isinstance(node, dict) else []:
            for item in files if isinstance(files, list) else []:
                name = item.get("filename", "") if isinstance(item, dict) else ""
                if Path(name).suffix.lower() in VIDEO_EXTENSIONS:
                    return item
    raise ComfyError("the workflow wrote no video")


def fetch(base: str, item: dict[str, Any]) -> bytes:
    """The bytes of a file ComfyUI wrote."""
    query = urllib.parse.urlencode(
        {
            "filename": item["filename"],
            "subfolder": item.get("subfolder", ""),
            "type": item.get("type", "output"),
        }
    )
    return _call(f"{base}/view?{query}")


# ---------------------------------------------------------------------------
# Generation and staging
# ---------------------------------------------------------------------------


def clip_file(folder: Path, clip: str) -> Path | None:
    """The video of a clip in a species folder, if made."""
    for path in sorted(folder.glob(f"{clip}.*")):
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            return path
    return None


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", "utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_meta(folder: Path, spec: dict[str, Any]) -> None:
    """`meta.json` of the clips: the source of the animation (other keys kept)."""
    meta = _read_json(folder / "meta.json")
    meta["source"] = {
        "type": "ai-generated",
        "model": str(spec["model"]),
        "date": datetime.now(UTC).date().isoformat(),
    }
    _write_json(folder / "meta.json", meta)


def make_clip(
    base: str,
    workflow: dict[str, Any],
    spec: dict[str, Any],
    folder: Path,
    clip: str,
    species: str,
    args: argparse.Namespace,
    form: str | None = None,
) -> Path:
    """Have ComfyUI make one clip of a species; the video written."""
    entry_id = folder.name
    first = upload(base, folder / "reference.png", f"{entry_id}_reference.png")
    if spec["clips"][clip].get("last") == "mirror":
        last = upload(base, folder / "reference_mirror.png", f"{entry_id}_mirror.png")
    else:
        last = first
    seed = args.seed if args.seed is not None else random.randrange(1 << 48)
    width, height = (int(v) for v in spec["size"])
    background = reference_background(folder)[0]
    prompt = clip_prompt(spec, clip, species, background, form)
    values = {
        "FIRST_FRAME": first,
        "LAST_FRAME": last,
        "PROMPT": prompt,
        "NEGATIVE": str(spec.get("negative", "")),
        "SEED": seed,
        "WIDTH": width,
        "HEIGHT": height,
        "LENGTH": int(spec["clips"][clip]["frames"]),
        "DURATION": duration_of(spec, clip),
        "STRENGTH": clip_strength(spec),
        "PREFIX": f"reeftank/{entry_id}_{clip}",
    }
    prompt_id = queue(base, fill(workflow, values))
    item = video_of(wait(base, prompt_id, args.timeout, args.poll))
    video = folder / f"{clip}{Path(item['filename']).suffix.lower()}"
    video.write_bytes(fetch(base, item))
    log = _read_json(folder / "wan.json")
    log[clip] = {
        "seed": seed,
        "prompt": prompt,
        "frames": values["LENGTH"],
        "strength": values["STRENGTH"],
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    _write_json(folder / "wan.json", log)
    return video


def matte_session(model: str, gpu: bool = False) -> Any:
    """A rembg session (the model is downloaded on its first use).

    On the CPU unless `gpu`: on the GPU, onnxruntime keeps its memory for the
    whole run, and the next clips of ComfyUI then run out of video memory.
    """
    try:
        from rembg import new_session
    except ImportError as err:
        raise SystemExit(
            'cutting the fish out needs rembg: pip install "rembg[cpu]"'
            ' (or "rembg[gpu]")'
        ) from err
    if gpu:
        return new_session(model)
    return new_session(model, providers=["CPUExecutionProvider"])


def remove_background(frame: Image.Image, session: Any) -> Image.Image:
    """A frame cut out by the model: RGBA."""
    from rembg import remove

    return remove(frame, session=session).convert("RGBA")


def unmix(cut: Image.Image, background: tuple[int, int, int]) -> Image.Image:
    """Edge pixels freed from the background colour mixed into them.

    A soft edge pixel is a blend of fish and background: its colour is
    unmixed (as a chroma key does), the faint ones dropped.
    """
    rgba = np.asarray(cut).astype(np.float32)
    rgb, alpha = rgba[..., :3], rgba[..., 3:] / 255.0
    bg = np.asarray(background, dtype=np.float32)
    soft = (alpha > 0) & (alpha < 1)
    unmixed = (rgb - (1.0 - alpha) * bg) / np.maximum(alpha, 0.05)
    rgb = np.where(soft, np.clip(unmixed, 0, 255), rgb)
    alpha = np.where(alpha < MATTE_FLOOR, 0.0, alpha)
    out = np.dstack([rgb, alpha * 255.0])
    return Image.fromarray((out + 0.5).astype(np.uint8), "RGBA")


def matte_clip(video: Path, out: Path, session: Any) -> Path:
    """Every frame of a clip cut out, as `out/NNNN.png`."""
    frames = build_atlas.read_frames(video)
    first = np.asarray(frames[0].convert("RGB"))
    median = np.median(corners(first), axis=0)
    background = (int(median[0]), int(median[1]), int(median[2]))
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for number, frame in enumerate(frames):
        cut = remove_background(frame.convert("RGB"), session)
        unmix(cut, background).save(out / f"{number:04d}.png")
    return out


def _matte_fresh(out: Path, video: Path) -> bool:
    return (
        out.is_dir()
        and any(out.iterdir())
        and (out.stat().st_mtime >= video.stat().st_mtime)
    )


def stage(
    root: Path,
    folder: Path,
    model: str = DEFAULT_MATTE,
    gpu: bool = False,
    section: str = "fish",
) -> Path:
    """Build the atlas of a species folder next to its clips; the staged entry.

    The recipe is written from the clips (or kept, once it reads the cut-out
    frames: a hand edit stays), each clip is cut out into `matte/<clip>/`
    when its video is newer, and the atlas is built from those frames.
    """
    entry_id = folder.name
    recipe = folder / f"{entry_id}.json"
    data = _read_json(recipe)
    if data.get("key") != "none":
        # no recipe yet, or one cutting the videos out by their key colour
        build_atlas.write_auto_recipe(folder, entry_id)
        data = _read_json(recipe)
    session = None
    for name, item in data["clips"].items():
        video = clip_file(folder, name)
        out = folder / "matte" / name
        item["source"] = f"matte/{name}"
        if video is None or _matte_fresh(out, video):
            continue
        if session is None:
            session = matte_session(model, gpu)
        print(f"{entry_id}: cutting {name} out...", flush=True)
        matte_clip(video, out, session)
    data["key"] = "none"
    _write_json(recipe, data)
    out = folder / "build"
    staged = out / entry_id
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir(parents=True)
    current = root / section / entry_id / "species.json"
    if current.exists():
        # the descriptive fields (names, behaviour...) of the catalog are kept
        shutil.copy2(current, staged / "species.json")
    build_atlas.main(
        [KINDS[section], entry_id, "--recipe", str(recipe), "--out", str(out)]
    )
    return staged


def generate(args: argparse.Namespace) -> int:
    """`generate`: clips of the species to animate, then their staged atlas."""
    spec = load_spec(args.prompts)
    workflow = load_workflow(args.workflow)
    missing = [n for n in REQUIRED if n not in placeholders_in(workflow)]
    if missing:
        raise SystemExit(
            f"{args.workflow}: no {', '.join(map(_token, missing))} (see `template`)"
        )
    todo = targets(args.root, args.only, args.section, animated_forms(spec))
    if args.limit:
        todo = todo[: args.limit]
    size = (int(spec["size"][0]), int(spec["size"][1]))
    gemini_key = os.environ.get(stills.KEY_ENV, "") if args.gemini else ""
    if args.gemini and not gemini_key:
        print(f"--gemini: set {stills.KEY_ENV} to the Gemini API key")
        return 2
    done, failed, waiting = 0, [], 0
    for entry in todo:
        folder = args.work / entry.name
        photo = reference_photo(args.photos, entry.name)
        if photo is None and gemini_key and not (folder / "reference.png").exists():
            print(f"{entry.name}: asking Gemini for a photo...", flush=True)
            try:
                photo = ask_photo(
                    entry,
                    args.photos,
                    gemini_key,
                    args.gemini_model,
                    args.attempts,
                    args.delay,
                    args.section,
                )
            except stills.GenerationError as err:
                print(f"stopped on {entry.name}: {err}; start again later")
                return 2
            if photo is None:
                failed.append(entry.name)
                print(f"{entry.name}: failed (no photo fits)")
                continue
        if photo is None and not (folder / "reference.png").exists():
            waiting += 1
            print(f"{entry.name}: no photo in {args.photos} (--gemini to ask one)")
            continue
        if photo is not None and (args.redo or not (folder / "reference.png").exists()):
            prepare_references(photo, folder, size)
        write_meta(folder, spec)
        species = species_name(entry)
        form = entry_form(args.root, args.section, entry.name)
        try:
            for clip in args.clips:
                if clip_file(folder, clip) and not args.redo:
                    continue
                print(f"{entry.name}: {clip}...", flush=True)
                video = make_clip(
                    args.comfy, workflow, spec, folder, clip, species, args, form
                )
                print(f"{entry.name}: {clip} -> {video}")
            if all(clip_file(folder, c) for c in CLIP_NAMES):
                if args.redo:
                    (folder / f"{entry.name}.json").unlink(missing_ok=True)
                stage(
                    args.root,
                    folder,
                    args.matte_model,
                    args.matte_gpu,
                    args.section,
                )
                print(f"{entry.name}: staged, see `preview {entry.name}`")
            done += 1
        except ComfyError as err:
            if err.fatal:
                print(f"stopped on {entry.name}: {err}; start again later")
                return 2
            failed.append(entry.name)
            print(f"{entry.name}: failed ({err})")
    print(f"{done} done, {len(failed)} failed, {waiting} without photo")
    return 1 if failed else 0


def build(args: argparse.Namespace) -> int:
    """`build`: stage again the species whose three clips are made."""
    built = 0
    for folder in work_entries(args.work, args.only, args.root, args.section):
        if all(clip_file(folder, c) for c in CLIP_NAMES):
            stage(args.root, folder, args.matte_model, args.matte_gpu, args.section)
            built += 1
    print(f"{built} staged")
    return 0


def in_section(root: Path, section: str, folder: Path) -> bool:
    """Whether a work folder belongs to a section: its id is in that catalog.

    The clips of every section live side by side in the work folder, so the
    commands that walk it (`build`, `review`, `gallery`) ask the catalog which
    section an id belongs to: a fish is never staged as an invertebrate.
    """
    return (root / section / folder.name).is_dir()


def work_entries(
    work: Path,
    only: list[str] | None = None,
    root: Path | None = None,
    section: str = "fish",
) -> list[Path]:
    """The work folders of a section (every one of them without `root`)."""
    if not work.is_dir():
        return []
    return [
        folder
        for folder in sorted(p for p in work.iterdir() if p.is_dir())
        if (not only or folder.name in only)
        and (root is None or in_section(root, section, folder))
    ]


def staged_entries(
    work: Path,
    only: list[str] | None = None,
    root: Path | None = None,
    section: str = "fish",
) -> list[Path]:
    """The species folders of a section with a staged atlas."""
    return [
        folder
        for folder in work_entries(work, only, root, section)
        if (folder / "build" / folder.name / "species.json").exists()
    ]


def accept(root: Path, folder: Path, section: str = "fish") -> Path:
    """Put the staged atlas of a species in the catalog; the catalog folder."""
    entry_id = folder.name
    staged = folder / "build" / entry_id
    target = root / section / entry_id
    target.mkdir(parents=True, exist_ok=True)
    built = json.loads((staged / "species.json").read_text(encoding="utf-8"))
    # what the catalog says of the species now wins (edited since the staging)
    built.update(build_atlas.described(target / "species.json"))
    for path in staged.iterdir():
        if path.name != "species.json":
            shutil.copy2(path, target / path.name)
    _write_json(target / "species.json", built)
    (folder / "accepted").write_text(
        datetime.now(UTC).isoformat(timespec="seconds") + "\n", encoding="utf-8"
    )
    return target


def redo(folder: Path, clips: list[str]) -> None:
    """Drop clips of a species (and its recipe, staging, preview): made again later."""
    for clip in clips:
        path = clip_file(folder, clip)
        if path is not None:
            path.unlink()
        shutil.rmtree(folder / "matte" / clip, ignore_errors=True)
    (folder / f"{folder.name}.json").unlink(missing_ok=True)
    (folder / "accepted").unlink(missing_ok=True)
    shutil.rmtree(folder / "build", ignore_errors=True)
    for name in ("preview.webp", "preview.html"):
        (folder / name).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------


def _clip_frames(
    frames: list[Image.Image], clip: dict[str, Any] | None
) -> list[Image.Image]:
    if not clip:
        return []
    shown = frames[clip["from"] : clip["to"] + 1]
    if clip.get("pingpong"):
        shown = shown + shown[-2:0:-1]
    return shown


def timeline(folder: Path, fps: int = PREVIEW_FPS) -> list[Image.Image]:
    """The frames of an atlas played at `fps`: swims, idle, turn, back, turn back.

    The atlas an animation replaces is often the still picture of the species,
    which holds none of those clips: whatever single clip it has is played in
    their place, so the two sides of a preview can still be compared.
    """
    frames, clips = frames_of(folder)
    moving = "swim" if "swim" in clips else next(iter(clips), "")
    rate = {name: int(clip.get("fps", fps)) for name, clip in clips.items()}
    pace = rate.get(moving, fps)
    swim = _clip_frames(frames, clips.get(moving))
    turn = _clip_frames(frames, clips.get("turn")) if moving != "turn" else []
    parts: list[tuple[list[Image.Image], int]] = [(swim, pace)] * PREVIEW_SWIMS
    if "idle" in clips and moving != "idle":
        parts.append((_clip_frames(frames, clips["idle"]), rate["idle"]))
    if turn:
        parts.append((turn, rate["turn"]))
    parts += [([ImageOps.mirror(f) for f in swim], pace)] * PREVIEW_SWIMS
    if turn:
        parts.append(([ImageOps.mirror(f) for f in turn], rate["turn"]))
    shown: list[Image.Image] = []
    for part, rate_of in parts:
        ticks = max(1, round(len(part) * fps / max(1, rate_of)))
        shown += [part[min(len(part) - 1, t * rate_of // fps)] for t in range(ticks)]
    return shown


def _on_background(frame: Image.Image, size: tuple[int, int]) -> Image.Image:
    tile = Image.new("RGBA", size, PREVIEW_BACKGROUND)
    big = frame.resize(
        (frame.width * PREVIEW_ZOOM, frame.height * PREVIEW_ZOOM),
        Image.Resampling.LANCZOS,
    )
    tile.alpha_composite(big, ((size[0] - big.width) // 2, (size[1] - big.height) // 2))
    return tile


def render_preview(staged: Path, current: Path | None, out: Path) -> Path:
    """The animated WebP of a staged atlas, the catalog one on its left."""
    sequences = [timeline(staged)]
    if current is not None and (current / "species.json").exists():
        desc = read_descriptor(current) or {}
        if (desc.get("atlas") or {}).get("1x"):
            sequences.insert(0, timeline(current))
    cell = (
        max(f.width for s in sequences for f in s[:1]) * PREVIEW_ZOOM,
        max(f.height for s in sequences for f in s[:1]) * PREVIEW_ZOOM,
    )
    length = max(len(s) for s in sequences)
    shown = []
    for tick in range(length):
        sheet = Image.new("RGBA", (cell[0] * len(sequences), cell[1]))
        for n, sequence in enumerate(sequences):
            frame = sequence[tick % len(sequence)]
            sheet.paste(_on_background(frame, cell), (n * cell[0], 0))
        shown.append(sheet)
    shown[0].save(
        out,
        format="WEBP",
        save_all=True,
        append_images=shown[1:],
        duration=round(1000 / PREVIEW_FPS),
        loop=0,
        quality=85,
        method=4,
    )
    html = out.with_suffix(".html")
    html.write_text(
        "<!doctype html><meta charset='utf-8'><title>"
        f"{staged.name}</title><body style='margin:0;background:#101820;color:#ccc;"
        "font:14px sans-serif;text-align:center'>"
        f"<p>{staged.name}: "
        f"{'catalog (left) / new (right)' if len(sequences) > 1 else 'new'}</p>"
        f"<img src='{out.name}' style='image-rendering:auto;max-width:100%'></body>\n",
        encoding="utf-8",
    )
    return html


def open_preview(path: Path, player: str | None) -> None:
    """Show a preview without waiting for the viewer."""
    command = shlex.split(player) if player else ["xdg-open"]
    subprocess.Popen(
        [*command, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def preview(args: argparse.Namespace, folder: Path) -> Path:
    """Write (and open) the preview of a staged species."""
    staged = folder / "build" / folder.name
    if not (staged / "species.json").exists():
        raise SystemExit(f"{folder.name}: nothing staged (run generate or build)")
    out = render_preview(
        staged, args.root / args.section / folder.name, folder / "preview.webp"
    )
    print(f"{folder.name}: {out}")
    if not args.no_open:
        open_preview(out if not args.player else folder / "preview.webp", args.player)
    return out


def gallery(args: argparse.Namespace) -> Path | None:
    """`gallery`: one page with the new animations of every staged species.

    Only the new atlases (not the catalog ones), not accepted yet unless `--all`;
    each one plays in a loop under its id and scientific name. The page uses
    relative paths: it opens from wherever the work folder is mounted.
    """
    entries = [
        f
        for f in staged_entries(args.work, args.only, args.root, args.section)
        if args.all or not (f / "accepted").exists()
    ]
    if not entries:
        print("nothing staged")
        return None
    cards = []
    for folder in entries:
        staged = folder / "build" / folder.name
        render_preview(staged, None, folder / "gallery.webp")
        (folder / "gallery.html").unlink(missing_ok=True)
        name = (read_descriptor(staged) or {}).get("scientific", "")
        state = " (accepted)" if (folder / "accepted").exists() else ""
        cards.append(
            "<figure style='margin:8px;display:inline-block;vertical-align:top'>"
            f"<img src='{folder.name}/gallery.webp' style='max-width:100%'>"
            f"<figcaption><b>{folder.name}</b>{state}<br><i>{name}</i>"
            "</figcaption></figure>"
        )
        print(f"{folder.name}: ready")
    page = args.work / "gallery.html"
    page.write_text(
        "<!doctype html><meta charset='utf-8'><title>Wan animations</title>"
        "<body style='margin:0;padding:8px;background:#101820;color:#ccc;"
        "font:14px sans-serif;text-align:center'>"
        f"<p>{len(cards)} new animations. Then: <code>wan_clips.py accept ID...</code>"
        " or <code>wan_clips.py redo ID --clips ...</code></p>\n"
        + "\n".join(cards)
        + "\n</body>\n",
        encoding="utf-8",
    )
    print(f"written {page}")
    if not args.no_open:
        open_preview(page, args.player)
    return page


def review(args: argparse.Namespace, ask: Callable[[str], str] = input) -> int:
    """`review`: every staged species not accepted yet, shown then judged."""
    pending = [
        f
        for f in staged_entries(args.work, args.only, args.root, args.section)
        if not (f / "accepted").exists()
    ]
    if not pending:
        print("nothing to review")
    for folder in pending:
        preview(args, folder)
        while True:
            answer = (
                ask(
                    f"{folder.name}: [a]ccept, [r]edo [swim idle turn], [s]kip, [q]uit: "
                )
                .strip()
                .lower()
            )
            word, *rest = answer.split() or [""]
            if word in ("a", "accept"):
                print(
                    f"{folder.name}: accepted -> "
                    f"{accept(args.root, folder, args.section)}"
                )
            elif word in ("r", "redo"):
                clips = [c for c in rest if c in CLIP_NAMES] or list(CLIP_NAMES)
                redo(folder, clips)
                print(f"{folder.name}: {' '.join(clips)} to make again (generate)")
            elif word in ("q", "quit"):
                return 0
            elif word not in ("s", "skip"):
                continue
            break
    return 0


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def todo(args: argparse.Namespace) -> int:
    """`todo`: the fish to animate and where each one stands."""
    entries = targets(
        args.root, args.only, args.section, animated_forms(load_spec(args.prompts))
    )
    for entry in entries:
        folder = args.work / entry.name
        made = [c for c in CLIP_NAMES if clip_file(folder, c)]
        if (folder / "accepted").exists():
            state = "accepted"
        elif (folder / "build" / entry.name / "species.json").exists():
            state = "staged"
        elif made:
            state = "clips " + ",".join(made)
        elif (
            reference_photo(args.photos, entry.name)
            or (folder / "reference.png").exists()
        ):
            state = "ready"
        else:
            state = "no photo"
        print(f"{entry.name:<40} {state}")
    print(f"{len(entries)} {args.section} without a real animation")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=ROOT, help="the catalog")
    parser.add_argument("--work", type=Path, help="clips folder (default: clips/)")
    parser.add_argument("--photos", type=Path, help="photos kept (default: photos/)")
    parser.add_argument(
        "--section",
        choices=SECTIONS,
        default="fish",
        help="fish (default) or invertebrates",
    )
    parser.add_argument(
        "--prompts", type=Path, help="default: the wording of the section"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def with_only(sub: argparse.ArgumentParser) -> argparse.ArgumentParser:
        sub.add_argument("--only", nargs="+", metavar="ID", help="these species only")
        return sub

    with_only(commands.add_parser("todo", help="the species to animate"))
    sub = commands.add_parser("prompt", help="print the prompt of a clip")
    sub.add_argument("id")
    sub.add_argument("clip", choices=CLIP_NAMES)
    sub = commands.add_parser("template", help="put the placeholders in a workflow")
    sub.add_argument("export", type=Path, help="workflow exported in the API format")
    sub.add_argument("--out", type=Path, help="default: clips/wan_workflow.json")
    sub = with_only(commands.add_parser("generate", help="make clips and stage"))
    sub.add_argument("--limit", type=int, help="at most this many species")
    sub.add_argument("--clips", nargs="+", choices=CLIP_NAMES, default=list(CLIP_NAMES))
    sub.add_argument("--redo", action="store_true", help="make the clips again")
    sub.add_argument("--seed", type=int, help="fixed seed (default: random)")
    sub.add_argument("--comfy", default=DEFAULT_COMFY, help="ComfyUI address")
    sub.add_argument("--workflow", type=Path, help="default: clips/wan_workflow.json")
    sub.add_argument("--timeout", type=float, default=3600, help="seconds per clip")
    sub.add_argument("--poll", type=float, default=5, help="seconds between checks")
    sub.add_argument(
        "--gemini", action="store_true", help="ask Gemini the missing photos"
    )
    sub.add_argument("--gemini-model", default=stills.DEFAULT_MODEL)
    sub.add_argument("--attempts", type=int, default=3, help="photo requests a fish")
    sub.add_argument("--delay", type=float, default=3.0, help="seconds between them")
    sub.add_argument("--matte-model", default=DEFAULT_MATTE, help="rembg model")
    sub.add_argument(
        "--matte-gpu", action="store_true", help="cut out on the GPU (ComfyUI idle)"
    )
    sub = with_only(commands.add_parser("build", help="stage again from the clips"))
    sub.add_argument("--matte-model", default=DEFAULT_MATTE, help="rembg model")
    sub.add_argument(
        "--matte-gpu", action="store_true", help="cut out on the GPU (ComfyUI idle)"
    )
    for name in ("preview", "review", "gallery"):
        sub = commands.add_parser(name, help=f"{name} staged atlases")
        if name == "preview":
            sub.add_argument("id")
        else:
            with_only(sub)
        if name == "gallery":
            sub.add_argument("--all", action="store_true", help="accepted ones too")
        sub.add_argument("--player", help="viewer command (default: xdg-open)")
        sub.add_argument("--no-open", action="store_true", help="only write it")
    sub = commands.add_parser("accept", help="put staged atlases in the catalog")
    sub.add_argument("ids", nargs="*", help="none: every staged species not accepted")
    sub = commands.add_parser("redo", help="drop clips to make them again")
    sub.add_argument("id")
    sub.add_argument("--clips", nargs="+", choices=CLIP_NAMES, default=list(CLIP_NAMES))

    args = parser.parse_args(argv)
    args.work = args.work or args.root / "clips"
    args.photos = args.photos or args.root / "photos"
    args.prompts = args.prompts or PROMPT_FILES[args.section]

    if args.command == "todo":
        return todo(args)
    if args.command == "prompt":
        folder = args.root / args.section / args.id
        background = "magenta"
        if (args.work / args.id / "reference.png").exists():
            background = reference_background(args.work / args.id)[0]
        spec = load_spec(args.prompts)
        print(
            clip_prompt(
                spec,
                args.clip,
                species_name(folder),
                background,
                entry_form(args.root, args.section, args.id),
            )
        )
        return 0
    if args.command == "template":
        result, report = make_template(load_workflow(args.export))
        out = args.out or args.work / WORKFLOW_NAME
        out.parent.mkdir(parents=True, exist_ok=True)
        _write_json(out, result)
        print("\n".join(report))
        print(f"written {out}")
        return 0
    if args.command == "generate":
        args.workflow = args.workflow or args.work / WORKFLOW_NAME
        return generate(args)
    if args.command == "build":
        return build(args)
    if args.command == "preview":
        preview(args, args.work / args.id)
        return 0
    if args.command == "review":
        return review(args)
    if args.command == "gallery":
        gallery(args)
        return 0
    if args.command == "accept":
        folders = []
        for entry_id in args.ids:
            folder = args.work / entry_id
            if not (folder / "build" / entry_id / "species.json").exists():
                raise SystemExit(f"{entry_id}: nothing staged")
            folders.append(folder)
        if not args.ids:  # no id: every staged species of the section, once each
            folders = [
                f
                for f in staged_entries(args.work, None, args.root, args.section)
                if not (f / "accepted").exists()
            ]
            if not folders:
                print("nothing staged")
                return 0
        for folder in folders:
            print(
                f"{folder.name}: accepted -> {accept(args.root, folder, args.section)}"
            )
        print("now: python scripts/validate.py, then commit")
        return 0
    redo(args.work / args.id, args.clips)
    print(f"{args.id}: {' '.join(args.clips)} to make again (generate)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
