#!/usr/bin/env python3
"""Copy the client modules of this catalog into a tool that uses them.

`client/` holds what the tools editing the catalog share with its CI: the
sections it declares and the rules its entries follow. A tool gets a copy of
each module, marked as generated:

    python scripts/sync_client.py <folder of the tool's package>
    python scripts/sync_client.py <folder> --check     # nothing written, exit 1 if stale
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CLIENT = Path(__file__).resolve().parent.parent / "client"
HEADER = (
    "# Generated from reeftank-catalog/client/{name} by scripts/sync_client.py: "
    "do not edit this copy.\n"
)


def copies(client: Path = CLIENT) -> dict[str, str]:
    """The text of each module of the client, as a tool holds it."""
    return {
        path.name: HEADER.format(name=path.name) + path.read_text(encoding="utf-8")
        for path in sorted(client.glob("*.py"))
    }


def sync(dest: Path, check: bool = False, client: Path = CLIENT) -> list[str]:
    """Write the copies into `dest`; returns the names of the ones that differed."""
    changed = []
    for name, text in copies(client).items():
        target = dest / name
        current = target.read_text(encoding="utf-8") if target.is_file() else None
        if current != text:
            changed.append(name)
            if not check:
                target.write_text(text, encoding="utf-8")
    return changed


def main(argv: list[str] | None = None) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("dest", type=Path, help="package folder of the tool")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    changed = sync(args.dest, args.check)
    for name in changed:
        print(f"{'stale' if args.check else 'updated'}: {name}")
    return 1 if args.check and changed else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
