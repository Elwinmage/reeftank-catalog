#!/usr/bin/env python3
"""Export and import the description of the species (not their pictures).

The description is the part of `species.json` written by hand: names, size,
behaviour, night mode, feeding response, coral palette... The rest (clips,
atlas, frame) comes from the build and is left alone.

    python scripts/species_meta.py export [--out species-meta.tgz]

writes an archive holding `fish/<id>/meta.json` and `corals/<id>/meta.json`,
with every field to fill shown (null when unknown), to edit or to hand over
(translations, behaviour).

    python scripts/species_meta.py import species-meta.tgz [--dry-run]

merges each `meta.json` of the archive (or of a folder) into the matching
`species.json`: the fields it gives replace the current ones (names and
behaviour merged key by key), null fields and missing fields leave the
current value alone. Then check with
`scripts/validate.py`. A later rebuild keeps the imported description: the
catalog wins over the meta files of the clips.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_atlas import GENERATED_KEYS
from validate import ID_RE, KINDS, LANGUAGES

META_FILE = "meta.json"

#: Fields shown in an export, so that nothing is forgotten
TEMPLATES: dict[str, dict[str, Any]] = {
    "fish": {
        "names": {},
        "scientific": None,
        "size_cm": None,
        "behavior": {
            "profile": None,
            "band": None,
            "depth": None,
            "speed_cm_s": None,
            "turn_rate": None,
            "territory_cm": None,
        },
        "night": None,
        "feeding_response": None,
    },
    "corals": {"names": {}, "scientific": None, "size_cm": None, "palette": None},
}


def description(species: dict[str, Any], kind: str) -> dict[str, Any]:
    """The hand written part of a descriptor, with the template's gaps."""
    meta = {k: v for k, v in species.items() if k not in GENERATED_KEYS}
    for key, value in TEMPLATES[kind].items():
        if key not in meta:
            meta[key] = json.loads(json.dumps(value))
        elif isinstance(value, dict) and isinstance(meta[key], dict):
            for sub, empty in value.items():
                meta[key].setdefault(sub, empty)
    names = meta.get("names")
    if isinstance(names, dict):
        for lang in LANGUAGES:
            names.setdefault(lang, None)
    return meta


def _prune(value: Any) -> Any:
    """Drop the null fields (left empty in an export), and the objects left
    empty by it."""
    if isinstance(value, dict):
        pruned = {k: _prune(v) for k, v in value.items() if v is not None}
        return {k: v for k, v in pruned.items() if v != {}}
    return value


def export(root: Path, out: Path) -> int:
    """Write the descriptions of every species into a tgz. Returns the count."""
    count = 0
    with tarfile.open(out, "w:gz") as archive:
        for kind in KINDS:
            folder = root / kind
            if not folder.is_dir():
                continue
            for entry in sorted(p for p in folder.iterdir() if p.is_dir()):
                file = entry / "species.json"
                if not file.is_file():
                    continue
                species = json.loads(file.read_text(encoding="utf-8"))
                blob = (
                    json.dumps(description(species, kind), indent=2, ensure_ascii=False)
                    + "\n"
                ).encode("utf-8")
                info = tarfile.TarInfo(f"{kind}/{entry.name}/{META_FILE}")
                info.size = len(blob)
                info.mode = 0o644
                archive.addfile(info, io.BytesIO(blob))
                count += 1
    return count


def _read_source(source: Path) -> dict[str, dict[str, Any]]:
    """meta.json files of an archive or a folder, by `kind/id`."""
    found: dict[str, Any] = {}
    if source.is_dir():
        for kind in KINDS:
            for file in sorted((source / kind).glob(f"*/{META_FILE}")):
                found[f"{kind}/{file.parent.name}"] = file.read_text(encoding="utf-8")
    else:
        with tarfile.open(source, "r:*") as archive:
            for member in archive.getmembers():
                parts = PurePosixPath(member.name.removeprefix("./")).parts
                if (
                    not member.isfile()
                    or len(parts) != 3
                    or parts[0] not in KINDS
                    or parts[2] != META_FILE
                ):
                    continue
                handle = archive.extractfile(member)
                if handle is not None:
                    found[f"{parts[0]}/{parts[1]}"] = handle.read().decode("utf-8")
    metas: dict[str, dict[str, Any]] = {}
    for key, text in found.items():
        try:
            data = json.loads(text)
        except ValueError as err:
            raise SystemExit(f"{key}/{META_FILE}: {err}") from err
        if not isinstance(data, dict):
            raise SystemExit(f"{key}/{META_FILE}: not an object")
        metas[key] = data
    return metas


def _merge(base: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    """`base` updated with `changes`, objects merged key by key."""
    out = dict(base)
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def import_meta(root: Path, source: Path, dry_run: bool = False) -> list[str]:
    """Merge the descriptions into the species. Returns what was done."""
    report: list[str] = []
    for key, meta in _read_source(source).items():
        kind, entry_id = key.split("/", 1)
        file = root / kind / entry_id / "species.json"
        if not ID_RE.match(entry_id) or not file.is_file():
            report.append(f"{key}: no such species, skipped")
            continue
        species = json.loads(file.read_text(encoding="utf-8"))
        changes = _prune({k: v for k, v in meta.items() if k not in GENERATED_KEYS})
        merged = _merge(species, changes)
        changed = sorted(
            k for k in set(merged) | set(species) if merged.get(k) != species.get(k)
        )
        if not changed:
            report.append(f"{key}: unchanged")
            continue
        report.append(f"{key}: {', '.join(changed)}")
        if not dry_run:
            file.write_text(
                json.dumps(merged, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
    return report


def main(argv: list[str] | None = None) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="catalog root (default: this repository)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    out = commands.add_parser("export", help="descriptions -> tgz")
    out.add_argument("--out", type=Path, default=Path("species-meta.tgz"))
    inp = commands.add_parser("import", help="tgz (or folder) -> species.json")
    inp.add_argument("source", type=Path)
    inp.add_argument("--dry-run", action="store_true", help="show, change nothing")
    args = parser.parse_args(argv)

    if args.command == "export":
        count = export(args.root, args.out)
        print(f"{count} species exported to {args.out}")
        return 0
    for line in import_meta(args.root, args.source, args.dry_run):
        print(line)
    if args.dry_run:
        print("dry run: nothing written")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
