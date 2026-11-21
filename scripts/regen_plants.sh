#!/usr/bin/env bash
# Ask Gemini again for the photos of some plants, then rebuild what shows them.
#
# Why: a green plant photographed on a green background is keyed out with its
# background (grey leaves). The plants of this list had a refused magenta photo
# (plant cut by the edge of the picture), then an accepted green one. A plant now
# keeps its own background for every attempt, and the prompt asks for a margin.
#
#   export GEMINI_API_KEY=...
#   scripts/regen_plants.sh                       # the plants listed below
#   scripts/regen_plants.sh vallisneria_spiralis  # only these ones
#   ATTEMPTS=10 scripts/regen_plants.sh           # more requests per plant
#   BACKGROUND=green scripts/regen_plants.sh ludwigia_repens   # force green, for a
#                                                              # red plant only
#   DRY_RUN=1 scripts/regen_plants.sh             # list the plants, no request
#
# Exit codes of generate_stills.py: 1 some plant failed (start again, only those
# are asked), 2 key refused or quota reached (start again later).

set -euo pipefail

cd "$(dirname "$0")/.."

# Plants whose picture came out grey or cut (see above)
DEFAULT_IDS=(
  aegagropila_linnaei
  cabomba_caroliniana
  limnobium_laevigatum
  rhizophora_mangle
  rotala_rotundifolia
  vallisneria_spiralis
  vesicularia_dubyana
)
ATTEMPTS="${ATTEMPTS:-6}"
# Always this background (the plant list above is green-leaved: magenta)
BACKGROUND="${BACKGROUND:-magenta}"
PHOTOS="${PHOTOS:-photos-plants}"

if [ "$#" -gt 0 ]; then
  IDS=("$@")
else
  IDS=("${DEFAULT_IDS[@]}")
fi

for id in "${IDS[@]}"; do
  if [ ! -f "plants/${id}/species.json" ]; then
    echo "Erreur : plants/${id}/species.json est introuvable." >&2
    exit 1
  fi
done

if [ -n "${DRY_RUN:-}" ]; then
  exec python3 scripts/generate_stills.py --section plants --redo --dry-run \
    --only "${IDS[@]}"
fi

if [ -z "${GEMINI_API_KEY:-}" ]; then
  echo "Erreur : définir GEMINI_API_KEY (clé de l'API Gemini)." >&2
  exit 1
fi

# Keep the old photos aside: a later --from-photos run must not read them again
mkdir -p "${PHOTOS}/old"
for id in "${IDS[@]}"; do
  for photo in "${PHOTOS}/${id}"-*.png; do
    [ -e "$photo" ] && mv -f "$photo" "${PHOTOS}/old/"
  done
done

echo "--- Génération de ${#IDS[@]} plante(s), fond ${BACKGROUND}, ${ATTEMPTS} essais max ---"
status=0
python3 scripts/generate_stills.py --section plants --redo \
  --only "${IDS[@]}" --background "${BACKGROUND}" \
  --attempts "${ATTEMPTS}" --photos "${PHOTOS}" || status=$?

if [ "$status" -eq 2 ]; then
  echo "Clé refusée ou quota atteint : relancer plus tard." >&2
  exit 2
fi

# The previews and the tables of the READMEs show the new atlases
python3 scripts/make_demo.py
python3 scripts/make_readme.py
python3 scripts/validate.py

echo
echo "À regarder : ${PHOTOS}/<id>-<n>.png (les refusées portent -refused)"
exit "$status"
