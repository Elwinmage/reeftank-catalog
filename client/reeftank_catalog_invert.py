"""Degraded atlas of an invertebrate made from a still photo: prompt, checks, atlas.

Canonical copy: reeftank-catalog/client/. `scripts/invert_prompt.py` and
`scripts/invert_atlas.py` (command line) and the catalog editor (which asks an
image model for the photo, then builds the atlas) use the same code, so that an
atlas made in the editor is the one the command would make. Tools get a copy
through `scripts/sync_client.py`; edit it here, never in a copy.

An invertebrate is one adult specimen on a flat magenta or green background,
nothing else in the picture (see `INVERT_PROMPT`). The background is keyed out
and the animal is fitted in a square transparent frame: an atlas of a single
frame (clip `still`) with its thumbnail. A single frame is what tells the card
to make the motion itself; a species filmed later ships several and they are
played instead.

What changes from one animal to the next is **where it is seen from**, and that
is what the growth-form sentence of a plant is here: a starfish is photographed
from straight above, a snail in three-quarter view, a shrimp from the side. The
view also says how the animal is fitted in its frame — one seen from the side
stands on the bottom of the frame, one seen from above lies in its middle.

A shrimp, a crab, a snail is asked head to the right, as a fish is: the card
mirrors the sprite to turn it round.

Red, pink and purple species are asked on green (`species_keys`): magenta would
eat them. The others are asked on magenta, which a green animal (an emerald
crab) needs, so `key_advice` warns when green is chosen for one of those.

    prompt = render_invert_prompt("nerita", descriptor)
    problems = check_invert_photo(photo)           # empty when the photo is fit
    still = build_invert_still(photo, "nerita", descriptor, {"type": "photo", ...})

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

#: The prompt that asks an image model for the photo of an invertebrate (the same
#: text is prompts/still_invert.json, a test checks it): one template, and a form
#: that comes from the species itself (`species_forms`), else from its `movement`.
INVERT_PROMPT: dict[str, Any] = {
    "version": 2,
    "keys": {"magenta": "flat magenta #FF00FF", "green": "flat green #00FF00"},
    "default_key": "magenta",
    "alternate_key": {
        "green": "red, pink or purple species: a magenta background would eat them"
    },
    "template": "Realistic photograph of [SPECIE], [FORM] One single adult specimen, alive and healthy: no substrate, no sand, no rock, no coral, no plant, no shell other than its own, no hand, no other animal. Natural shape and colours of the species, saturated, fine detail on the shell, the carapace or the spines, every appendage complete and clear of the body, nothing cropped. On a [KEY] background (the colour code only tells which colour the background is: never write it, nor any other text, on the image), perfectly flat uniform background without gradient or vignette, no floor line, no shadow, no reflection. Neutral white studio lighting, soft and even, no [KEY_NAME] tint or spill on the edges of the animal, sharp edges so that the background can be keyed out cleanly. Highest resolution, the specimen fills about 70% of the image, centred, with a clear margin of empty background on every side (no leg, antenna, spine or arm touches the edge of the picture), clean image: no text, no watermark, no mark in the corners.",
    "default_aspect": "1:1",
    "default_view": "side",
    "aspects": {
        "shrimp": "3:2",
        "crab": "1:1",
        "hermit": "1:1",
        "snail": "1:1",
        "urchin": "1:1",
        "cucumber": "3:2",
        "worm": "3:4",
        "clam": "1:1",
        "star": "1:1",
        "brittle_star": "1:1",
    },
    "forms": {
        "shrimp": "one adult shrimp standing on the ground on its walking legs, seen from the side at eye level, head to the right, caught mid-stride with its legs clearly separated and spread apart under the body, neither tucked nor bunched together, and not swimming or floating. Every leg, claw and antenna complete and clear of the body, the rostrum and the tail fan whole.",
        "crab": "one adult crab standing high on its legs, seen from the side at eye level in three-quarter view, head to the right, caught mid-stride with its legs clearly separated and spread apart, its claws held forward, the carapace and all its legs visible.",
        "hermit": "one adult hermit crab walking, seen from the side at eye level, head to the right, caught mid-stride with its walking legs out of the shell, clearly separated and spread apart, the shell whole with its pattern and the legs, claws and eyestalks clear of it.",
        "snail": "one adult snail seen in three-quarter view from slightly above, moving to the right, the shell whole with its pattern and its spire, the foot out of the shell and the tentacles extended.",
        "urchin": "one adult sea urchin seen from the side at eye level, the whole test with its spines evenly spread all around it, nothing cropped.",
        "cucumber": "one adult sea cucumber seen from the side at eye level, laid out straight, head to the right, the whole body with its tube feet and its papillae.",
        "worm": "one adult tube worm seen from the side at eye level, its feather crown wide open and whole, the tube hidden or trimmed away, nothing cropped.",
        "clam": "one adult clam seen in three-quarter view from slightly above, the shell half open with its mantle fully spread and its pattern visible, the byssus threads and the rock removed.",
        "star": "one adult starfish seen from straight above, flat and symmetrical, all its arms whole and spread apart, nothing cropped.",
        "brittle_star": "one adult brittle star seen from straight above, its central disc flat and its thin arms spread apart and whole, nothing cropped.",
    },
    #: Where the animal is seen from: it fixes the form sentence above and how the
    #: animal is fitted in its frame (`side` stands on the bottom, `above` lies in
    #: the middle).
    "views": {
        "shrimp": "side",
        "crab": "side",
        "hermit": "side",
        "snail": "side",
        "urchin": "side",
        "cucumber": "side",
        "worm": "side",
        "clam": "side",
        "star": "above",
        "brittle_star": "above",
    },
    #: Fallback for a species with no form of its own: the commonest of its
    #: movement. A new species is expected to name its form (a test checks it).
    "movement_forms": {
        "ground": "snail",
        "glass": "snail",
        "decor": "snail",
        "sessile": "worm",
        "swim": "shrimp",
    },
    "species_forms": {
        "archaster_typicus": "star",
        "astraea_tecta": "snail",
        "atyopsis_moluccensis": "shrimp",
        "caridina_cantonensis": "shrimp",
        "caridina_multidentata": "shrimp",
        "cerithium": "snail",
        "clea_helena": "snail",
        "clibanarius_tricolor": "hermit",
        "holothuria": "cucumber",
        "hymenocera_picta": "shrimp",
        "lysmata_amboinensis": "shrimp",
        "lysmata_debelius": "shrimp",
        "lysmata_wurdemanni": "shrimp",
        "melanoides_tuberculata": "snail",
        "mithraculus_sculptus": "crab",
        "nassarius": "snail",
        "neocaridina_davidi": "shrimp",
        "nerita": "snail",
        "neritina_natalensis": "snail",
        "ophiolepis_superba": "brittle_star",
        "paguristes_cadenati": "hermit",
        "planorbella_duryi": "snail",
        "pomacea_bridgesii": "snail",
        "prionocidaris_baculosa": "urchin",
        "sabellastarte": "worm",
        "saron_marmoratus": "shrimp",
        "spirobranchus_giganteus": "worm",
        "stenopus_hispidus": "shrimp",
        "tectus_fenestratus": "snail",
        "tridacna_crocea": "clam",
        "tripneustes_gratilla": "urchin",
        "turbo_fluctuosus": "snail",
    },
    "species_keys": {
        "caridina_cantonensis": "green",
        "hymenocera_picta": "green",
        "lysmata_amboinensis": "green",
        "lysmata_debelius": "green",
        "lysmata_wurdemanni": "green",
        "neocaridina_davidi": "green",
        "paguristes_cadenati": "green",
        "stenopus_hispidus": "green",
    },
}

#: Side of the square frame, px, and the margin kept around the animal
FRAME = 320
PAD = 12
#: An invertebrate fills at least this share of the picture (its larger side)
MIN_SHARE = 0.4
SHARE_MESSAGE = "too small in the picture"
CLIP = {"from": 0, "to": 0, "fps": 1, "loop": False}


def invert_form(
    invert_id: str, desc: dict[str, Any], prompt: dict[str, Any] | None = None
) -> str:
    """The form (key of `forms`) of an invertebrate."""
    prompt = prompt or INVERT_PROMPT
    form = prompt["species_forms"].get(invert_id) or prompt["movement_forms"].get(
        desc.get("movement", "")
    )
    if form not in prompt["forms"]:
        raise ValueError(f"no form for invertebrate {invert_id}")
    return str(form)


def invert_view(
    invert_id: str, desc: dict[str, Any], prompt: dict[str, Any] | None = None
) -> str:
    """Where the animal is seen from: `side` or `above`.

    Unlike the prompt, this one never fails: an entry whose form is not known
    yet is fitted the commonest way rather than refused.
    """
    prompt = prompt or INVERT_PROMPT
    form = prompt["species_forms"].get(invert_id) or prompt["movement_forms"].get(
        desc.get("movement", "")
    )
    return str(prompt["views"].get(form, prompt["default_view"]))


def invert_key(invert_id: str, prompt: dict[str, Any] | None = None) -> str:
    """The background colour advised for an invertebrate."""
    prompt = prompt or INVERT_PROMPT
    return str(prompt["species_keys"].get(invert_id, prompt["default_key"]))


def invert_aspect(
    invert_id: str, desc: dict[str, Any], prompt: dict[str, Any] | None = None
) -> str:
    """The picture ratio to ask for: wide for a shrimp, else square."""
    prompt = prompt or INVERT_PROMPT
    form = invert_form(invert_id, desc, prompt)
    return str(prompt["aspects"].get(form, prompt["default_aspect"]))


def key_advice(
    invert_id: str, key: str, prompt: dict[str, Any] | None = None
) -> str | None:
    """Why a background is a bad idea for a species (None when it is fine).

    A green background turns a green animal grey once keyed out: it is only fit
    for the species of `species_keys`.
    """
    prompt = prompt or INVERT_PROMPT
    if key != "green" or prompt["species_keys"].get(invert_id) == "green":
        return None
    return "a green background turns a green animal grey: keep it for red, pink or purple species"


def render_invert_prompt(
    invert_id: str,
    desc: dict[str, Any],
    key: str | None = None,
    prompt: dict[str, Any] | None = None,
) -> str:
    """The prompt of an invertebrate (`key` forces the background colour)."""
    prompt = prompt or INVERT_PROMPT
    key = key or invert_key(invert_id, prompt)
    if key not in prompt["keys"]:
        raise ValueError(f"key must be one of {', '.join(prompt['keys'])}")
    name = desc.get("scientific") or invert_id.replace("_", " ")
    common = (desc.get("names") or {}).get("en")
    specie = f"{name} ({common})" if common and common != name else name
    text = prompt["template"].replace("[SPECIE]", specie)
    text = text.replace("[FORM]", prompt["forms"][invert_form(invert_id, desc, prompt)])
    return text.replace("[KEY]", prompt["keys"][key]).replace("[KEY_NAME]", key)


def check_invert_photo(photo: Image.Image, key: str | None = None) -> list[str]:
    """Problems that make a photo unfit (empty when fine).

    `check_photo` judges the width of a fish; an invertebrate is judged on its
    larger side, a starfish being as tall as it is wide and a shrimp wider.
    """
    problems = [p.replace("fish", "invertebrate") for p in check_photo(photo, key)]
    kept = [p for p in problems if SHARE_MESSAGE not in p]
    if len(kept) != len(problems):  # the animal was found: measure it our way
        box = key_out(photo, key).getchannel("A").point(OPAQUE).getbbox()
        assert box is not None  # check_photo found it
        width, height = photo.size
        if max((box[2] - box[0]) / width, (box[3] - box[1]) / height) < MIN_SHARE:
            kept.append(f"the invertebrate is {SHARE_MESSAGE} (under {MIN_SHARE:.0%})")
    return kept


def build_invert_still(
    photo: Image.Image,
    name: str,
    desc: dict[str, Any],
    source: dict[str, str],
    key: str | None = None,
    prompt: dict[str, Any] | None = None,
) -> Still:
    """The atlas and the thumbnail of an invertebrate (`name`: its id).

    An animal seen from the side stands on the bottom of its frame, as a plant
    does; one seen from above lies in the middle of it.
    """
    sub = key_out(photo, key)
    box = sub.getchannel("A").point(OPAQUE).getbbox() or (0, 0, *sub.size)
    animal = sub.crop(box)
    room = FRAME - 2 * PAD
    scale = min(room / animal.width, room / animal.height)
    size = (max(1, round(animal.width * scale)), max(1, round(animal.height * scale)))
    small = (
        animal.convert("RGBa").resize(size, Image.Resampling.LANCZOS).convert("RGBA")
    )
    frame = Image.new("RGBA", (FRAME, FRAME), (0, 0, 0, 0))
    top = (
        (FRAME - size[1]) // 2
        if invert_view(name, desc, prompt) == "above"
        else FRAME - PAD - size[1]
    )
    frame.paste(small, ((FRAME - size[0]) // 2, top))
    keys = {
        "kind": "invertebrate",
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


#: Background colours an invertebrate photo can have
INVERT_KEYS = tuple(KEYS)
