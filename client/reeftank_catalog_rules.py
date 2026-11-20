"""Reeftank catalog rules: what a species must follow, beyond its JSON Schema.

Canonical copy: reeftank-catalog/client/. `scripts/validate.py` (CI) and the
tools editing the catalog (catalog editor) agree on the same rules, so that a
draft is accepted by the editor when the CI accepts it (tests/test_client.py
checks that they do). Tools get a copy through `scripts/sync_client.py`; edit
it here, never in a copy.

Same interface as `house_catalog_rules.py` of house-catalog, so that a tool
handling several catalogs drives them the same way. No Home Assistant
dependency; needs `jsonschema` and `referencing`. Everything works on data
already read (dicts, file names and sizes), never on the repository layout:

    validators = load_validators(schema_dir)        # schema/ of the repository,
                                                    # or schema.zip of a release
    ctx = Context(fish=...)
    report = validate_draft("fish", "amphiprion", data, {"species.json": 900},
                            ctx, validators)
    report.errors, report.warnings

`Context` holds the *other* entries a draft is checked against: when a draft
replaces an entry, leave that entry out of the context.

What the pictures of a species are made of (atlas, clips, thumbnail) is checked
by the CI, which has the files: a draft only carries their descriptor keys.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

#: Languages every species is named in
KNOWN_LANGS = ("en", "fr", "de", "es", "it", "nl", "pl", "pt")

#: Files of an entry other than the descriptor are pictures, named by the
#: descriptor: none of them is a fixed name
ASSET_FILES: dict[str, int] = {}
SHIPPED_SUFFIXES = (".webp", ".png")
DESCRIPTOR_MAX = 32_000
MAX_ENTRY_BYTES = 32 * 1024 * 1024
MAX_ENTRY_FILES = 16
FILE_RE = re.compile(r"^[A-Za-z0-9_@.-]{1,120}$")

#: Descriptor file of each kind of entry
DESCRIPTORS = {
    "fish": "species.json",
    "coral": "species.json",
    "plant": "species.json",
    "invertebrate": "species.json",
}
#: Folder of each kind, from the root of the repository
FOLDERS = {
    "fish": "fish",
    "coral": "corals",
    "plant": "plants",
    "invertebrate": "invertebrates",
}
#: Keys of a descriptor that come with the pictures, built by the tools: the
#: editor keeps them as they are and does not show them
PICTURE_KEYS = (
    "atlas",
    "thumbnail",
    "frame",
    "columns",
    "clips",
    "facing",
    "length_frac",
    "degraded",
    "source",
)
#: Keys with a fixed value, that the editor writes itself
FIXED_KEYS = ("kind",)


@dataclass
class Report:
    """Errors (the entry is refused) and warnings (it is accepted, but could be better)."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def error(self, where: str, message: str) -> None:
        self.errors.append(f"{where}: {message}")

    def warn(self, where: str, message: str) -> None:
        self.warnings.append(f"{where}: {message}")


@dataclass
class Context:
    """The entries a draft is checked against, by id (one group per section)."""

    fish: dict[str, dict[str, Any]] = field(default_factory=dict)
    corals: dict[str, dict[str, Any]] = field(default_factory=dict)
    plants: dict[str, dict[str, Any]] = field(default_factory=dict)
    invertebrates: dict[str, dict[str, Any]] = field(default_factory=dict)


def where_of(kind: str, entry_id: str) -> str:
    """Location of an entry in the repository, as shown in the reports."""
    return f"{FOLDERS[kind]}/{entry_id}"


def with_fixed(kind: str, data: dict[str, Any]) -> dict[str, Any]:
    """The descriptor with the keys of fixed value in front (`kind`)."""
    return {"kind": kind, **{k: v for k, v in data.items() if k != "kind"}}


def load_validators(schema_dir: Path) -> dict[str, Draft202012Validator]:
    """One validator per kind of entry, from a folder of `<kind>.schema.json`."""
    resources = []
    schemas = {}
    for path in schema_dir.glob("*.schema.json"):
        schema = json.loads(path.read_text(encoding="utf-8"))
        resources.append((schema["$id"], Resource.from_contents(schema)))
        schemas[path.name.split(".")[0]] = schema
    registry = Registry().with_resources(resources)
    return {
        name: Draft202012Validator(schema, registry=registry)
        for name, schema in schemas.items()
        if name != "common"
    }


# -- checks -----------------------------------------------------------------------


def check_schema(
    report: Report, where: str, data: Any, validator: Draft202012Validator
) -> bool:
    """Report what the JSON Schema refuses; True when it accepts the data."""
    errors = sorted(validator.iter_errors(data), key=lambda e: [str(p) for p in e.path])
    for err in errors:
        location = "/".join(str(p) for p in err.path) or "(root)"
        report.error(where, f"{location}: {err.message}")
    return not errors


def check_files(
    report: Report, where: str, files: Mapping[str, int], descriptor: str
) -> None:
    """The files of an entry (name -> size in bytes): descriptor, pictures, limits."""
    if descriptor not in files:
        report.error(where, f"missing {descriptor}")
    for name, size in files.items():
        if name == descriptor:
            if size > DESCRIPTOR_MAX:
                report.error(where, f"{name} is {size} bytes, limit {DESCRIPTOR_MAX}")
        elif not name.lower().endswith(SHIPPED_SUFFIXES) or not FILE_RE.match(name):
            report.error(where, f"unexpected file {name!r}")
    if len(files) > MAX_ENTRY_FILES:
        report.error(where, f"{len(files)} files (at most {MAX_ENTRY_FILES})")
    if sum(files.values()) > MAX_ENTRY_BYTES:
        report.error(where, f"{sum(files.values())} bytes (at most {MAX_ENTRY_BYTES})")


def _ordered(report: Report, where: str, label: str, pair: Any) -> None:
    """[a, b] must have a <= b (the schema checks the numbers themselves)."""
    if (
        isinstance(pair, list)
        and len(pair) == 2
        and all(isinstance(v, int | float) and not isinstance(v, bool) for v in pair)
        and pair[0] > pair[1]
    ):
        report.error(where, f"{label}: the first value must not exceed the second")


def check_source(report: Report, where: str, data: Mapping[str, Any]) -> None:
    """`degraded` and `source` (where the picture comes from) go together."""
    degraded = data.get("degraded", False)
    source = data.get("source")
    if not isinstance(source, dict):
        if degraded is True:
            report.error(where, "degraded: source is needed (photo or ai-generated)")
        return
    kind = source.get("type")
    if degraded is True and kind == "video":
        report.error(where, "degraded: the source is a photo, not a video")
    if kind == "ai-generated" and not source.get("model"):
        report.error(where, "source.model: the model that made the picture")
    if kind == "photo":
        if not source.get("author"):
            report.error(where, "source.author: who made the photo")
        if not source.get("license"):
            report.error(where, "source.license: the license of the photo")


def validate_entry(
    report: Report,
    kind: str,
    entry_id: str,
    data: Mapping[str, Any],
    ctx: Context,
    validators: Mapping[str, Draft202012Validator],
) -> None:
    """The rules of one descriptor, once its files are checked."""
    where = where_of(kind, entry_id)
    if not check_schema(report, where, data, validators[kind]):
        return
    _ordered(report, where, "size_cm", data.get("size_cm"))
    behavior = data.get("behavior")
    if isinstance(behavior, dict):
        for key in ("band", "depth", "speed_cm_s"):
            _ordered(report, where, f"behavior.{key}", behavior.get(key))
    check_source(report, where, data)
    scientific = data.get("scientific")
    for other_id, other in getattr(ctx, FOLDERS[kind]).items():
        if other_id != entry_id and other.get("scientific") == scientific:
            report.warn(where, f"{scientific} is already the species {other_id}")


def validate_draft(
    kind: str,
    entry_id: str,
    data: dict[str, Any],
    files: Mapping[str, int],
    ctx: Context,
    validators: Mapping[str, Draft202012Validator],
) -> Report:
    """Everything the CI checks about one entry, for a tool that edits the catalog.

    `files` maps the file names of the entry, descriptor included, to their
    sizes in bytes. The context is left as it was given.
    """
    report = Report()
    check_files(report, where_of(kind, entry_id), files, DESCRIPTORS[kind])
    scratch = Context(
        fish=dict(ctx.fish),
        corals=dict(ctx.corals),
        plants=dict(ctx.plants),
        invertebrates=dict(ctx.invertebrates),
    )
    validate_entry(report, kind, entry_id, data, scratch, validators)
    return report
