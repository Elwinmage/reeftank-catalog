#!/usr/bin/env python3
"""Build the files of a catalog release: one archive per entry + manifest.

    python scripts/make_release.py 2026.10.0 --out dist \\
        [--previous prev/manifest.json] [--min-integration 0.1.0]

writes into `dist/`:

- `<kind>-<id>.zip` for every entry: species.json and its pictures, flat.
  Archives are reproducible (sorted names, fixed dates), so an unchanged
  entry keeps its sha256 and the integration does not download it again;
- `manifest.json` (read by the integration, see pack.py there);
- `notes.md`: entries added, updated and removed since `--previous`.

An entry whose archive did not change keeps its version from the previous
manifest; the others take the new version.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate import KINDS, check_catalog, shipped_files

MANIFEST_FORMAT = 1
DEFAULT_MIN_INTEGRATION = "0.1.0"
_FIXED_DATE = (2020, 1, 1, 0, 0, 0)


def entry_archive(folder: Path) -> bytes:
    """Reproducible zip of the shipped files of an entry."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in shipped_files(folder):
            info = zipfile.ZipInfo(path.name, date_time=_FIXED_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    return buffer.getvalue()


def entry_name(folder: Path) -> str:
    """Display name of an entry (English, else its id)."""
    try:
        names = json.loads((folder / "species.json").read_text(encoding="utf-8")).get(
            "names", {}
        )
    except (OSError, ValueError, AttributeError):
        names = {}
    name = names.get("en") if isinstance(names, dict) else None
    return f"{name} (`{folder.name}`)" if name else f"`{folder.name}`"


def build(
    root: Path,
    version: str,
    out: Path,
    previous: dict[str, Any] | None = None,
    min_integration: str = DEFAULT_MIN_INTEGRATION,
) -> dict[str, Any]:
    """Write the archives, the manifest and the notes; return the manifest."""
    out.mkdir(parents=True, exist_ok=True)
    before: dict[str, Any] = (previous or {}).get("entries", {})
    entries: dict[str, Any] = {}
    added: list[str] = []
    updated: list[str] = []
    for kind in KINDS:
        folder = root / kind
        if not folder.is_dir():
            continue
        for entry in sorted(p for p in folder.iterdir() if p.is_dir()):
            key = f"{kind}/{entry.name}"
            blob = entry_archive(entry)
            sha = hashlib.sha256(blob).hexdigest()
            file = f"{kind}-{entry.name}.zip"
            (out / file).write_bytes(blob)
            old = before.get(key)
            if old is None:
                added.append(entry_name(entry))
            elif old.get("sha256") != sha:
                updated.append(entry_name(entry))
            entries[key] = {
                "version": old["version"]
                if old is not None and old.get("sha256") == sha
                else version,
                "sha256": sha,
                "size": len(blob),
                "file": file,
            }
    removed = [
        f"`{key.split('/', 1)[1]}`" for key in sorted(before) if key not in entries
    ]

    lines = [f"## {version}", ""]
    for title, items in (("Added", added), ("Updated", updated), ("Removed", removed)):
        if items:
            lines.append(f"**{title}**: " + ", ".join(items))
            lines.append("")
    if len(lines) == 2:
        lines.append("No species change.")
        lines.append("")
    notes = "\n".join(lines)
    (out / "notes.md").write_text(notes, encoding="utf-8")

    manifest = {
        "format": MANIFEST_FORMAT,
        "version": version,
        "min_integration": min_integration,
        "notes": notes.strip(),
        "entries": entries,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("version", help="release version, ex: 2026.10.0")
    parser.add_argument("--out", type=Path, default=Path("dist"))
    parser.add_argument(
        "--previous", type=Path, help="manifest.json of the previous release"
    )
    parser.add_argument("--min-integration", default=DEFAULT_MIN_INTEGRATION)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="catalog root (default: this repository)",
    )
    args = parser.parse_args(argv)

    problems = check_catalog(args.root)
    if problems:
        for key, errors in problems.items():
            for error in errors:
                print(f"{key}: {error}")
        return 1
    previous = None
    if args.previous and args.previous.is_file():
        previous = json.loads(args.previous.read_text(encoding="utf-8"))
    manifest = build(args.root, args.version, args.out, previous, args.min_integration)
    print(f"{len(manifest['entries'])} entries written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
