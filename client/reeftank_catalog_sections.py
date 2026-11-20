"""What the reeftank catalog publishes: its repository and the layout of its sections.

Canonical copy: reeftank-catalog/client/. The tools editing the catalog (the
catalog editor) get a copy through `scripts/sync_client.py`; edit it here,
never in a copy.

The catalog client (`catalog_pack.py`, from reeftech-catalog) only knows the
sections of its own repository: a tool declares the ones of this catalog before
using it, when its package is imported:

    from . import reeftank_catalog_sections as reeftank
    from .catalog_pack import register_section

    for section, kinds in reeftank.SECTIONS.items():
        register_section(section, kinds, reeftank.REPO)

The releases of this catalog are not published per section: one release holds
every species (`manifest.json`, `<folder>-<id>.zip`, `schema.zip`), and a tool
cuts that manifest into sections by the folder of each entry. No dependency:
plain data.
"""

from __future__ import annotations

#: `owner/name` of the GitHub repository publishing the releases
REPO = "Elwinmage/reeftank-catalog"

#: section -> {kind of entry: (folder, descriptor file)}; an entry is the folder
#: `<folder>/<id>/` holding its descriptor and its pictures. A section is named
#: after its folder; its kind is the `kind` written in the descriptors.
SECTIONS: dict[str, dict[str, tuple[str, str]]] = {
    "fish": {"fish": ("fish", "species.json")},
    "corals": {"coral": ("corals", "species.json")},
    "plants": {"plant": ("plants", "species.json")},
    "invertebrates": {"invertebrate": ("invertebrates", "species.json")},
}

#: Name of the schema archive in a release, and of its record in the manifest
SCHEMA_ARCHIVE = "schema.zip"
