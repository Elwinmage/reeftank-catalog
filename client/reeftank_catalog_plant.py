"""Degraded atlas of a plant made from a still photo: prompt, checks and atlas.

Canonical copy: reeftank-catalog/client/. `scripts/plant_prompt.py` and
`scripts/plant_atlas.py` (command line) and the catalog editor (which asks an
image model for the photo, then builds the atlas) use the same code, so that an
atlas made in the editor is the one the command would make. Tools get a copy
through `scripts/sync_client.py`; edit it here, never in a copy.

A plant is one specimen on a flat magenta or green background, no roots, no pot
(see `PLANT_PROMPT`). The background is keyed out and the plant is fitted,
standing on the bottom of a square transparent frame: an atlas of a single frame
(clip `still`) with its thumbnail.

A plant is asked on a magenta background; green is only for the species listed
in `species_keys` (red, pink or purple plants, that magenta would eat). A green
plant on a green background is keyed out and turns grey, so `key_advice` warns
when green is chosen for a plant that is not listed.

    prompt = render_plant_prompt("egeria_densa", descriptor)
    problems = check_plant_photo(photo)              # empty when the photo is fit
    still = build_plant_still(photo, "egeria_densa", {"type": "ai-generated", ...})

No Home Assistant dependency; needs Pillow and numpy.
"""

from __future__ import annotations

from typing import Any

from PIL import Image

try:  # inside a package (a tool holding a copy) or as a top-level module (scripts)
    from .reeftank_catalog_still import (
        KEYS,
        OPAQUE,
        THUMB_SIZE,
        Still,
        check_photo,
        encode_webp,
        key_out,
        thumbnail,
    )
except ImportError:  # pragma: no cover - depends on how the module is imported
    from reeftank_catalog_still import (  # type: ignore[no-redef]
        KEYS,
        OPAQUE,
        THUMB_SIZE,
        Still,
        check_photo,
        encode_webp,
        key_out,
        thumbnail,
    )

#: The prompt that asks an image model for the photo of a plant (the same text is
#: prompts/still_plant.json, a test checks it): one template, and a growth form
#: that comes from the species itself (`species_forms`), else from its `zone`.
PLANT_PROMPT: dict[str, Any] = {
    "version": 1,
    "keys": {"magenta": "flat magenta #FF00FF", "green": "flat green #00FF00"},
    "default_key": "magenta",
    "alternate_key": {
        "green": "red, pink or purple plants: a magenta background would eat them"
    },
    "template": "Realistic photograph of [SPECIE], [FORM] Clean cut specimen: no roots, no soil, no substrate, no pot, no rockwork, no driftwood, no cable tie, no other plant, no animal. Natural shape and colours of the species, healthy and saturated, fine detail on the leaf edges and veins, whole specimen visible with all its leaves and tips, nothing cropped. On a [KEY] background (the colour code only tells which colour the background is: never write it, nor any other text, on the image), perfectly flat uniform background without gradient or vignette, no floor line, no shadow, no reflection. Neutral white studio lighting, soft and even, no [KEY_NAME] tint or spill on the edges of the leaves, sharp edges so that the background can be keyed out cleanly. Highest resolution, the specimen fills about 70% of the image height, centred, with a clear margin of empty background on every side (no leaf, tip or stem touches the edge of the picture), clean image: no text, no watermark, no mark in the corners.",
    "default_aspect": "1:1",
    "aspects": {
        "stem": "3:4",
        "ribbon": "3:4",
        "mangrove": "3:4",
        "rosette": "1:1",
        "carpet": "1:1",
        "floating": "1:1",
        "duckweed": "1:1",
        "epiphyte": "1:1",
        "moss": "1:1",
        "alga": "3:4",
        "ball": "1:1",
    },
    "forms": {
        "stem": "a bunch of 5 to 7 stems with their whole foliage, cut cleanly just below the lowest leaves, standing upright, seen from the side at eye level.",
        "rosette": "one plant seen from the side at eye level: a rosette of leaves rising from a short crown, the crown trimmed flat just under the leaf bases.",
        "ribbon": "a tuft of long ribbon-like leaves standing upright, seen from the side at eye level, cut cleanly at the base.",
        "carpet": "a small tuft of about 10 short shoots with their leaves, seen from the side at eye level, cut cleanly at the base (a small clump, not a whole carpet).",
        "floating": "a floating plant seen from slightly above, in three-quarter view, only the leaves or rosette that stay at the water surface, the hanging roots removed.",
        "duckweed": "a loose cluster of 6 to 10 tiny floating fronds seen from slightly above, the hanging rootlets removed.",
        "epiphyte": "a clump of leaves on a short trimmed piece of rhizome, seen from the side at eye level, the roots removed (no wood, no rock holding it).",
        "moss": "a small rounded clump of moss, seen from the side at eye level, cut cleanly at the base.",
        "alga": "the whole alga (its thallus) seen from the side at eye level, the holdfast removed.",
        "ball": "one single round ball of filamentous alga, seen from the side at eye level, nothing else.",
        "mangrove": "a young seedling with its stem and leaves, seen from the side at eye level, the roots removed.",
    },
    "zone_forms": {
        "background": "stem",
        "midground": "rosette",
        "foreground": "carpet",
        "floating": "floating",
        "epiphyte": "epiphyte",
    },
    "species_forms": {
        "aegagropila_linnaei": "ball",
        "caulerpa_prolifera": "alga",
        "chaetomorpha_linum": "alga",
        "cryptocoryne_parva": "rosette",
        "eleocharis_parvula": "ribbon",
        "halimeda_opuntia": "alga",
        "lemna_minor": "duckweed",
        "ludwigia_repens": "stem",
        "rhizophora_mangle": "mangrove",
        "sagittaria_subulata": "ribbon",
        "staurogyne_repens": "stem",
        "thalassia_testudinum": "ribbon",
        "vallisneria_spiralis": "ribbon",
        "vesicularia_dubyana": "moss",
    },
    "species_keys": {"ludwigia_repens": "green", "rotala_rotundifolia": "green"},
}

#: Side of the square frame, px, and the margin kept around the plant
FRAME = 320
PAD = 12
#: A plant fills at least this share of the picture (its larger side)
MIN_SHARE = 0.4
SHARE_MESSAGE = "too small in the picture"
CLIP = {"from": 0, "to": 0, "fps": 1, "loop": False}


def plant_form(
    plant_id: str, desc: dict[str, Any], prompt: dict[str, Any] | None = None
) -> str:
    """The growth form (key of `forms`) of a plant."""
    prompt = prompt or PLANT_PROMPT
    form = prompt["species_forms"].get(plant_id) or prompt["zone_forms"].get(
        desc.get("zone", "")
    )
    if form not in prompt["forms"]:
        raise ValueError(f"no growth form for plant {plant_id}")
    return str(form)


def plant_key(plant_id: str, prompt: dict[str, Any] | None = None) -> str:
    """The background colour advised for a plant."""
    prompt = prompt or PLANT_PROMPT
    return str(prompt["species_keys"].get(plant_id, prompt["default_key"]))


def plant_aspect(
    plant_id: str, desc: dict[str, Any], prompt: dict[str, Any] | None = None
) -> str:
    """The picture ratio to ask for: tall for a stem or a ribbon, else square."""
    prompt = prompt or PLANT_PROMPT
    form = plant_form(plant_id, desc, prompt)
    return str(prompt["aspects"].get(form, prompt["default_aspect"]))


def key_advice(
    plant_id: str, key: str, prompt: dict[str, Any] | None = None
) -> str | None:
    """Why a background is a bad idea for a plant (None when it is fine).

    A green background turns a green plant grey once keyed out: it is only fit
    for the species of `species_keys`.
    """
    prompt = prompt or PLANT_PROMPT
    if key != "green" or prompt["species_keys"].get(plant_id) == "green":
        return None
    return "a green background turns a green plant grey: keep it for red, pink or purple plants"


def render_plant_prompt(
    plant_id: str,
    desc: dict[str, Any],
    key: str | None = None,
    prompt: dict[str, Any] | None = None,
) -> str:
    """The prompt of a plant (`key` forces the background colour)."""
    prompt = prompt or PLANT_PROMPT
    key = key or plant_key(plant_id, prompt)
    if key not in prompt["keys"]:
        raise ValueError(f"key must be one of {', '.join(prompt['keys'])}")
    name = desc.get("scientific") or plant_id.replace("_", " ")
    common = (desc.get("names") or {}).get("en")
    specie = f"{name} ({common})" if common and common != name else name
    text = prompt["template"].replace("[SPECIE]", specie)
    text = text.replace("[FORM]", prompt["forms"][plant_form(plant_id, desc, prompt)])
    return text.replace("[KEY]", prompt["keys"][key]).replace("[KEY_NAME]", key)


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


#: Background colours a plant photo can have
PLANT_KEYS = tuple(KEYS)
