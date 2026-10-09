# reeftank-catalog

Fish and coral species for the aquarium view of
[ha-reef-card](https://github.com/Elwinmage/ha-reef-card), served by the
[ReefTank integration](https://github.com/Elwinmage/ha-reeftank-component).

The integration downloads the releases of this repository into
`<config>/reeftank/pack/` and keeps them up to date
(`update.reeftank_catalog`, automatic by default): nothing to do on the Home
Assistant side.

| Species | Kind |
|---|---|
| Foxface rabbitfish (`siganus_vulpinus`) | fish |

Species assets are under **CC BY 4.0**, scripts under MIT: see
[LICENSE.md](LICENSE.md).

---

## Contents

```
fish/<id>/
  species.json        # descriptor: clips, size, behaviour, names
  <id>.webp           # atlas: every frame of every clip
  thumb.webp          # 96x64 picture for the editor's lists
corals/<id>/
  species.json
  <id>.shade.webp     # brightness
  <id>.mask.png       # palette weights
  thumb.webp          # in the default colours
scripts/
  build_atlas.py      # clips -> atlas + descriptor + thumbnail
  validate.py         # checks every entry
  make_release.py     # archives + manifest of a release
```

Only `species.json`, `.webp` and `.png` files are shipped: a recipe or notes
kept in a species folder stay in the repository.

## Adding a species

### 1. Generate the clips

With an image-to-video model (Kling, Wan, Veo…), from **one reference
picture** of the fish: strict profile, head to the right, flat magenta
background (`#FF00FF`; chroma green `#00FF00` for pink or purple fish),
neutral white light. For each clip, impose the first and last frames:

| Clip | First frame | Last frame | Duration |
|---|---|---|---|
| `swim` | reference | reference | ~4 s |
| `idle` (optional) | reference | reference | 3–4 s |
| `turn` (optional) | reference | reference mirrored left-right | 4–5 s |

Turn off any "loop" option of the tool: it often makes forward-then-backward
clips. Mirror the reference yourself (any picture editor), do not ask the model
for a left-facing fish.

<details>
<summary>Prompt: swim</summary>

```
Side view of a single [SPECIES] swimming in place, as if on a treadmill,
for a looping game sprite.

Subject
- Exactly one fish, adult, natural colors and patterns, anatomically accurate
  fins and tail.
- Strict lateral (profile) view, head pointing to the RIGHT, body parallel to
  the image plane. No rotation, no turning toward the camera.
- Active cruising swim: clear side-to-side sweeps of the caudal fin
  (about 2 full beats per second), the rear half of the body flexing slightly
  with each beat, pectoral fins sculling. The fish keeps the same position
  and size in the frame, as if swimming against a current.
- The whole fish (including the tail at full swing and every fin) stays inside
  the frame, with a margin of at least 15% on every side.

Background and lighting
- Perfectly flat, uniform magenta (#FF00FF) background, filling the whole
  frame. No gradient, no texture, no vignette.
- No water, no bubbles, no particles, no plants, no rocks, no sand.
- Neutral white studio lighting, soft and even, as for a product photo.
  True colors: no blue or green color cast, no underwater haze,
  no caustics, no light rays.
- No shadow and no reflection on the background. The fish casts nothing.
- No magenta (#FF00FF) tint on the fish edges.

Camera
- Fixed camera: no zoom, no pan, no shake, no depth of field change.
- Orthographic-like framing, the fish in sharp focus from head to tail.
- No motion blur.

Animation
- The clip must loop seamlessly: the last frame matches the first one.
- Duration: 4 seconds.

Output
- Highest available resolution, landscape framing with the fish filling about
  70% of the width.
- No text, no watermark, no logo, no frame border.
```
</details>

<details>
<summary>Prompt: idle</summary>

Same as `swim`, with:

```
Side view of a single [SPECIES] hovering in place, almost motionless,
for a looping game sprite (idle animation).

- Idle hovering: the fish stays still in the water, holding its position.
  The pectoral fins scull gently and continuously, the dorsal and anal fins
  ripple slightly, the tail barely moves (small, slow corrections only).
  Subtle breathing: gill covers and mouth open and close slowly.
- No forward swimming motion, no strong tail beat, no drifting.
  The body stays in the same place, same size, same height, centered.

Animation
- Calm, slow, continuous motion, always moving forward in time: never played
  in reverse, no back-and-forth playback.
- The clip must loop seamlessly: the last frame matches the first one.
- Duration: 3 to 4 seconds, two or three slow fin-sculling cycles.
```
</details>

<details>
<summary>Prompt: turn</summary>

Same as `swim`, with:

```
Animation
- Starts exactly on the first frame: the fish in strict profile, head to the
  RIGHT, swimming in place.
- After about one second, the fish makes one smooth 180° turn in place:
  it bends its body and swings its head toward the camera, passes through a
  front three-quarter view, and ends in strict profile with the head to the
  LEFT.
- The turn itself lasts about one second. The fish stays at the same
  position, same size and same height in the frame during the whole turn.
- After the turn, the fish keeps swimming in place, head to the LEFT, and the
  clip ends exactly on the last frame.
- Play once. No loop, no reverse playback.
```
</details>

Negative prompt, when the tool takes one:

```
multiple fish, turning, head toward camera, camera movement, zoom,
underwater scene, blue tint, haze, caustics, bubbles, particles, shadow,
reflection, gradient background, motion blur, depth of field, cropped fins,
cropped tail, text, watermark
```

(remove `turning, head toward camera` for the turn clip).

### 2. Describe the species

Next to the clips, a `meta.json` merged into the descriptor:

```json
{
  "size_cm": [15, 22],
  "behavior": {
    "profile": "cruiser",
    "band": [0.15, 0.75],
    "depth": [0.2, 0.8],
    "speed_cm_s": [4, 10],
    "turn_rate": 2.0
  },
  "night": "rest_bottom",
  "feeding_response": 1.0,
  "names": {"en": "Foxface rabbitfish", "fr": "Poisson-lapin à tête de renard"}
}
```

| Field | Meaning |
|---|---|
| `size_cm` | adult length range, cm (an inventory line may give its own) |
| `behavior.profile` | `shoal` (school), `cruiser` (open water, long paths), `benthic` (bottom and decor, stops), `hover` (around a spot) |
| `behavior.band` | height in the tank, 0 surface … 1 bottom |
| `behavior.depth` | 0 front glass … 1 back |
| `behavior.speed_cm_s` | cruising speed range |
| `behavior.turn_rate` | how fast it turns, radians per second |
| `behavior.shoal` | `cohesion` (1), `alignment` (0.8), `separation` (1.2) for a school |
| `night` | `hide` (behind the decor), `hover`, `rest_bottom` |
| `feeding_response` | 0 ignores the food … 1 rushes to it |
| `names` | display names by language |

### 3. Build

Tools, once, in a virtual environment (so that numpy and scipy match):

```bash
python3 -m venv ~/.venvs/atlas
~/.venvs/atlas/bin/pip install -r requirements.txt
# and ffmpeg (with ffprobe) from your system
```

Clips stay on your computer, one folder per species named after its id:

```
~/clips/
  siganus_vulpinus/  swim.mp4  idle.mp4  turn.mp4  meta.json
  pseudanthias_squamipinnis/  swim.mp4  turn.mp4  meta.json
```

```bash
~/.venvs/atlas/bin/python scripts/build_atlas.py batch ~/clips --out fish
```

For each folder the clips are analysed:

- **swim**: the best seamless loop of 1–3 s, leaving out the pauses of the
  tail;
- **idle**: a forward-then-backward clip is cut in half and played as a
  ping-pong, otherwise a 2–4 s loop; kept at 12 fps, at most 72 frames;
- **turn**: the frames where the fish actually turns (from the reference to
  its mirror), sped up to about 1.2 s;
- clips that do not start or end on the reference picture are reported.

The choices are written to `<id>.json` next to the clips (a recipe), then the
atlas is built: the fish is cut out of the key colour (soft edges, the key
seen through translucent fins turned into transparency, specks dropped), every
clip is brought to one scale (192 px for the fish length), and a thumbnail is
written. A recipe is reused on the next run, so hand edits stay
(`--reanalyse` rewrites them). One species alone:

```bash
python scripts/build_atlas.py fish siganus_vulpinus --auto ~/clips/siganus_vulpinus --out fish
```

Then **look at the atlas** (`fish/<id>/<id>.webp`): a clip gone wrong (a fin
melting, a second fish) is not detected.

<details>
<summary>By hand</summary>

`loops` lists the seamless loops of a clip and detects ping-pong clips:

```bash
python scripts/build_atlas.py loops ~/clips/siganus_vulpinus/swim.mp4
```

A recipe:

```json
{
  "length": 192,
  "clips": {
    "swim": {"source": "swim.mp4", "fps": 24, "mode": "loop", "range": [118, 153]},
    "idle": {"source": "idle.mp4", "fps": 12, "mode": "pingpong", "range": [0, 146], "step": 2},
    "turn": {"source": "turn.mp4", "fps": 24, "mode": "once", "range": [85, 202], "step": 4}
  },
  "meta": {"size_cm": [15, 22]}
}
```

```bash
python scripts/build_atlas.py fish siganus_vulpinus --recipe ~/clips/siganus_vulpinus/siganus_vulpinus.json --out fish
```

or `--clip name=source[:fps[:mode[:from-to[:step]]]]` on the command line
(mode `loop`, `once` or `pingpong`). Options: `--key '#00ff00'` (key colour;
found on the frame borders by default), `--length 160` (lighter atlas),
`--hd` (double resolution atlas too).
</details>

Corals: a coral clip is painted with its zones in **pure red, green and
blue** (any shading), grey parts taking the fourth colour; the script splits
it into the shade and mask atlases, and the card recolours them with the
palette of each coral (`palette` in `meta.json`):

```bash
python scripts/build_atlas.py coral euphyllia_glabrescens \
    --clip day=~/clips/euphyllia/day.mp4 --clip close=~/clips/euphyllia/close.mp4 \
    --meta ~/clips/euphyllia/meta.json --out corals
```

### 4. Check and publish

```bash
python scripts/validate.py
git add fish/<id> && git commit -m "Add <id>" && git push
```

Then run the **Release** workflow (*Actions → Release → Run workflow*) with
the new version (`2026.10.0`, `2026.10.1`…). It checks the catalog, builds
one archive per species and the manifest, and publishes the release: every
Home Assistant with automatic updates installs it within 12 hours.

## Release format

Read by the integration (`custom_components/reeftank/pack.py`):

```json
{
  "format": 1,
  "version": "2026.10.0",
  "min_integration": "0.1.0",
  "notes": "## 2026.10.0\n\n**Added**: Foxface rabbitfish (`siganus_vulpinus`)",
  "entries": {
    "fish/siganus_vulpinus": {
      "version": "2026.10.0",
      "sha256": "…",
      "size": 830000,
      "file": "fish-siganus_vulpinus.zip"
    }
  }
}
```

- One zip per entry, flat (species.json and its pictures). Archives are
  reproducible: an unchanged species keeps its sha256 and its version, and
  is not downloaded again.
- `min_integration`: older integrations do not install the release (raise it
  when the descriptors use something new).
- The integration reads `…/releases/latest/download/manifest.json`: publish
  releases as *latest*, not as pre-releases.

## Development

```bash
pip install -r requirements.test.txt
pre-commit install          # checks before every commit
pre-commit run --all-files  # the same, on demand
```

Before each commit, pre-commit:

- refuses video files (source clips stay on your computer) and files over
  4 MB;
- checks JSON and YAML, line endings and trailing spaces;
- lints and formats the scripts (ruff);
- checks every species (`scripts/validate.py`);
- runs the tests, with 100 % coverage of the scripts.

Its hooks install their own environment (Pillow, numpy, scipy): no conflict
with the Python packages of the system. The same checks run on GitHub for
every push.
