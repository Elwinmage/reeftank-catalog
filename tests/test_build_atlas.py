"""Atlas building: the coral mask encoding the card relies on."""

from __future__ import annotations

import importlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
build_atlas = importlib.import_module("build_atlas")


def test_split_coral_weights() -> None:
    sheet = Image.new("RGBA", (5, 1))
    sheet.putpixel((0, 0), (255, 0, 0, 255))  # pure colour 1
    sheet.putpixel((1, 0), (0, 128, 0, 255))  # colour 2, darker
    sheet.putpixel((2, 0), (128, 128, 128, 255))  # grey: colour 4
    sheet.putpixel((3, 0), (255, 255, 0, 255))  # half 1, half 2
    sheet.putpixel((4, 0), (0, 0, 0, 0))  # transparent
    shade, mask = build_atlas.split_coral(sheet)
    assert shade.getpixel((1, 0))[:3] == (128, 128, 128)
    assert mask.getpixel((0, 0)) == (255, 0, 0, 255)
    assert mask.getpixel((1, 0)) == (0, 255, 0, 255)
    assert mask.getpixel((2, 0)) == (0, 0, 0, 255)
    r, g, b, _ = mask.getpixel((3, 0))
    assert abs(r - 128) <= 4 and abs(g - 128) <= 4 and b == 0
    assert mask.getpixel((4, 0))[3] == 0


def test_build_atlas_layout() -> None:
    frames = [Image.new("RGBA", (10, 5), (255, 0, 0, 255)) for _ in range(5)]
    clips = [
        build_atlas.Clip("swim", frames[:3], 24, True),
        build_atlas.Clip("turn", frames[3:], 12, False),
    ]
    atlas = build_atlas.build_atlas(clips, (10, 5), columns=2)
    assert atlas.sheet.size == (20, 15)
    assert atlas.clips == {
        "swim": {"from": 0, "to": 2, "fps": 24, "loop": True},
        "turn": {"from": 3, "to": 4, "fps": 12, "loop": False},
    }


def test_parse_clip() -> None:
    spec = build_atlas.parse_clip("swim=a.png")
    assert (spec.name, spec.source, spec.fps, spec.mode) == (
        "swim",
        Path("a.png"),
        24,
        None,
    )
    assert spec.resolved_mode == "loop"
    assert build_atlas.parse_clip("turn=b").resolved_mode == "once"
    spec = build_atlas.parse_clip("idle=c.mp4:12:pingpong:0-146:2")
    assert (spec.fps, spec.mode, spec.range, spec.step) == (12, "pingpong", (0, 146), 2)
    assert build_atlas.parse_clip("turn=b:12:true").mode == "loop"
    assert build_atlas.parse_clip("turn=b:12:no").mode == "once"
    for bad in ("nope", "a=b:x", "a=b:12:sideways", "a=b:12:loop:4"):
        with pytest.raises(SystemExit):
            build_atlas.parse_clip(bad)


def test_cli_writes_fish_entry(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    for n in range(4):
        image = Image.new("RGBA", (40, 20), (0, 0, 0, 0))
        image.paste((0, 200, 180, 255), (5 + n, 5, 30 + n, 15))
        image.save(frames / f"{n:03d}.png")
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"size_cm": [5, 8]}), encoding="utf-8")
    out = tmp_path / "out"
    assert (
        build_atlas.main(
            [
                "fish",
                "test_fish",
                "--clip",
                f"swim={frames}",
                "--meta",
                str(meta),
                "--out",
                str(out),
                "--frame",
                "32x16",
            ]
        )
        == 0
    )
    desc = json.loads((out / "test_fish" / "species.json").read_text(encoding="utf-8"))
    assert desc["atlas"] == {"1x": "test_fish.webp"}
    assert desc["clips"]["swim"] == {"from": 0, "to": 3, "fps": 24, "loop": True}
    assert desc["size_cm"] == [5, 8]
    assert desc["length_frac"] == round(192 / 32, 4)
    assert Image.open(out / "test_fish" / "test_fish.webp").size == (64, 32)


# ---------------------------------------------------------------------------
# Clips generated on a key colour
# ---------------------------------------------------------------------------

MAGENTA = (249, 5, 244)


def fish_frame(
    width: int = 200,
    height: int = 100,
    length: int = 120,
    tail: int = 0,
    dy: int = 0,
    speck: bool = False,
) -> Image.Image:
    """A yellow ellipse 'fish' on magenta, tail growing with `tail`."""
    image = Image.new("RGB", (width, height), MAGENTA)
    draw = ImageDraw.Draw(image)
    cx, cy = width // 2, height // 2 + dy
    draw.ellipse(
        (cx - length // 2, cy - length // 5, cx + length // 2, cy + length // 5),
        fill=(250, 210, 0),
    )
    draw.polygon(
        [
            (cx - length // 2, cy),
            (cx - length // 2 - 10 - tail, cy - 12),
            (cx - length // 2 - 10 - tail, cy + 12),
        ],
        fill=(250, 210, 0),
    )
    if speck:
        draw.rectangle((5, 5, 7, 7), fill=(255, 255, 255))
    return image.convert("RGBA")


def test_detect_key() -> None:
    assert build_atlas.detect_key(fish_frame()) == MAGENTA
    # transparency already there: nothing to key
    clear = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
    assert build_atlas.detect_key(clear) is None
    # a dull or busy border is no key
    assert (
        build_atlas.detect_key(Image.new("RGBA", (20, 20), (120, 120, 120, 255)))
        is None
    )
    noisy = Image.effect_noise((40, 40), 80).convert("RGBA")
    assert build_atlas.detect_key(noisy) is None


def test_parse_key() -> None:
    assert build_atlas.parse_key("auto") == "auto"
    assert build_atlas.parse_key("none") is None
    assert build_atlas.parse_key("#00FF00") == (0, 255, 0)
    for bad in ("green", "#12345", "#gggggg"):
        with pytest.raises(SystemExit):
            build_atlas.parse_key(bad)


def test_chroma_key_cuts_unmixes_and_despills() -> None:
    frame = fish_frame(speck=True)
    # a translucent fin: a third of magenta seen through a pale yellow
    fin = tuple(round(0.67 * c + 0.33 * k) for c, k in zip((240, 230, 180), MAGENTA))
    frame.putpixel((150, 50), (*fin, 255))
    cut = build_atlas.chroma_key(frame, MAGENTA)
    assert cut.getpixel((0, 0))[3] == 0
    assert cut.getpixel((100, 50)) == (250, 210, 0, 255)
    if build_atlas.ndimage is not None:
        # the speck is dropped (scipy)
        assert cut.getpixel((6, 6))[3] == 0
    r, g, b, a = cut.getpixel((150, 50))
    assert 120 < a < 255, "the key seen through becomes transparency"
    assert r <= g + 1 and b <= g + 1, "no pink left"


def test_drop_specks_keeps_single_blob() -> None:
    alpha = np.zeros((10, 10), dtype=np.float32)
    alpha[2:5, 2:5] = 1.0
    assert np.array_equal(build_atlas.drop_specks(alpha), alpha)


def test_clips_share_one_scale(tmp_path: Path) -> None:
    """Two clips of different sizes give the same fish length."""
    swim = tmp_path / "swim"
    turn = tmp_path / "turn"
    swim.mkdir()
    turn.mkdir()
    for n in range(6):
        fish_frame(length=120, tail=n * 2).save(swim / f"{n:03d}.png")
        # the turn clip is drawn bigger, and starts after 3 profile frames
        length = 180 if n < 5 else 60
        fish_frame(300, 150, length, dy=n).save(turn / f"{n:03d}.png")
    specs = [
        build_atlas.parse_clip(f"swim={swim}"),
        build_atlas.parse_clip(f"turn={turn}:24:once:3-5"),
    ]
    clips, places = build_atlas.prepare_fish(specs, "auto", length=64)
    assert [len(c.frames) for c in clips] == [6, 3]
    assert clips[1].loop is False
    size = build_atlas.frame_size(clips, places)
    assert size[0] % 4 == 0 and size[1] % 4 == 0
    placed_swim = build_atlas.place(clips[0].frames[0], places[0], size)
    placed_turn = build_atlas.place(clips[1].frames[0], places[1], size)
    box_swim = build_atlas.alpha_box(placed_swim)
    box_turn = build_atlas.alpha_box(placed_turn)
    assert box_swim and box_turn
    body = lambda box: box[2] - box[0]
    # lengths (tail included) are medians over the clip: 130..140 -> 135
    assert abs(body(box_swim) - 130 * 64 / 135) <= 2
    # the turn's profile frames (before the range) set its scale
    assert abs(body(box_turn) - 64) <= 2
    # double resolution: twice the size
    big = build_atlas.place(
        clips[0].frames[0], places[0], (size[0] * 2, size[1] * 2), 2
    )
    box_big = build_atlas.alpha_box(big)
    assert box_big and abs(body(box_big) - 2 * body(box_swim)) <= 3
    # no key: frames kept as they are
    clear, _ = build_atlas.prepare_fish(specs[:1], None, length=64)
    assert clear[0].frames[0].getpixel((0, 0))[3] == 255


def test_measure_without_subject() -> None:
    blank = [Image.new("RGBA", (40, 20), (0, 0, 0, 0))]
    assert build_atlas.measure(blank, once=False) == (40.0, 20.0, 10.0)


def test_build_atlas_square_and_pingpong() -> None:
    frames = [Image.new("RGBA", (20, 10)) for _ in range(8)]
    atlas = build_atlas.build_atlas(
        [build_atlas.Clip("idle", frames, 12, True, True)], (20, 10), None
    )
    assert atlas.columns == 2
    assert atlas.clips["idle"]["pingpong"] is True


def _wave(n: int, period: int) -> list[Image.Image]:
    """Frames of a square moving on a circle: a clean loop of `period`."""
    out = []
    for i in range(n):
        angle = 2 * math.pi * i / period
        image = Image.new("RGB", (64, 64), (0, 0, 0))
        x = 28 + round(18 * math.cos(angle))
        y = 28 + round(18 * math.sin(angle))
        ImageDraw.Draw(image).rectangle((x, y, x + 8, y + 8), fill=(255, 255, 255))
        out.append(image.convert("RGBA"))
    return out


def test_find_loops() -> None:
    result = build_atlas.find_loops(_wave(80, 30), min_len=24, max_len=40)
    assert result["frames"] == 80
    best = result["loops"][0]
    assert best["to"] - best["from"] + 1 == 30
    assert best["score"] < 0.5
    assert result["pingpong"] is None
    # forward then backward
    frames = _wave(40, 90)
    pingpong = build_atlas.find_loops(frames + frames[-2::-1], min_len=24)
    assert pingpong["pingpong"] is not None
    assert abs(pingpong["pingpong"]["pivot"] - 39) <= 1


def test_loops_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    folder = tmp_path / "wave"
    folder.mkdir()
    frames = _wave(40, 90)
    for n, image in enumerate(frames + frames[-2::-1]):
        image.save(folder / f"{n:03d}.png")
    assert build_atlas.main(["loops", str(folder), "--max", "40"]) == 0
    out = capsys.readouterr().out
    assert "79 frames" in out
    assert "ping-pong" in out
    assert "keep 0-39" in out


def test_recipe(tmp_path: Path) -> None:
    clips = tmp_path / "clips"
    for name in ("swim", "idle"):
        (clips / name).mkdir(parents=True)
        for n in range(5):
            fish_frame(tail=n).save(clips / name / f"{n:03d}.png")
    recipe = clips / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "length": 48,
                "columns": 3,
                "clips": {
                    "swim": {"source": "swim", "range": [1, 4]},
                    "idle": {
                        "source": "idle",
                        "fps": 12,
                        "mode": "pingpong",
                        "step": 2,
                    },
                },
                "meta": {"size_cm": [15, 22]},
            }
        ),
        encoding="utf-8",
    )
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"night": "hide"}), encoding="utf-8")
    out = tmp_path / "out"
    args = ["fish", "fox", "--recipe", str(recipe), "--out", str(out), "--hd"]
    # a --clip replaces the recipe's clip of that name
    args += ["--clip", f"swim={clips / 'swim'}:20:loop:0-1", "--meta", str(meta)]
    assert build_atlas.main(args) == 0
    desc = json.loads((out / "fox" / "species.json").read_text(encoding="utf-8"))
    assert desc["clips"]["idle"] == {
        "from": 0,
        "to": 2,
        "fps": 12,
        "loop": True,
        "pingpong": True,
    }
    assert desc["clips"]["swim"] == {"from": 3, "to": 4, "fps": 20, "loop": True}
    assert desc["columns"] == 3
    assert desc["size_cm"] == [15, 22] and desc["night"] == "hide"
    assert desc["atlas"]["2x"] == "fox@2x.webp"
    assert 0 < desc["length_frac"] < 1


def test_cli_needs_a_clip(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        build_atlas.main(["fish", "x", "--out", str(tmp_path)])


def test_coral_with_range(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    for n in range(4):
        Image.new("RGBA", (8, 8), (255, 0, 0, 255)).save(frames / f"{n:03d}.png")
    out = tmp_path / "out"
    args = ["coral", "c", "--clip", f"day={frames}:12:loop:1-2", "--out", str(out)]
    assert build_atlas.main(args + ["--frame", "8x8"]) == 0
    desc = json.loads((out / "c" / "species.json").read_text(encoding="utf-8"))
    assert desc["clips"]["day"] == {"from": 0, "to": 1, "fps": 12, "loop": True}


def test_read_frames_errors(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit):
        build_atlas.read_frames(empty)
    (empty / "a.png").write_bytes(b"")
    Image.new("RGBA", (4, 4)).save(empty / "a.png")
    with pytest.raises(SystemExit):
        build_atlas.read_frames(empty, (3, 5))


# ---------------------------------------------------------------------------
# Automatic recipe
# ---------------------------------------------------------------------------


def _save(frames: list[Image.Image], folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for n, image in enumerate(frames):
        image.save(folder / f"{n:03d}.png")
    return folder


def _turn(ref: Image.Image, hold: int = 10, blend: int = 10) -> list[Image.Image]:
    """Reference, a blend to its mirror, then the mirror."""
    mirror = ref.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    frames = [ref] * hold
    frames += [Image.blend(ref, mirror, (i + 1) / (blend + 1)) for i in range(blend)]
    return frames + [mirror] * hold


def test_turn_window() -> None:
    ref = _wave(1, 30)[0]
    small = build_atlas._small(_turn(ref))
    start, end = build_atlas.turn_window(small)
    assert 8 <= start <= 12
    assert 14 <= end <= 21
    # nothing turns: the whole clip
    still = build_atlas._small([ref] * 6)
    assert build_atlas.turn_window(still) == (0, 5)


def test_loop_choice() -> None:
    # a clean loop inside a longer clip
    small = build_atlas._small(_wave(80, 30))
    (first, last), mode = build_atlas._loop_choice(small, 24, (1.0, 1.5))
    assert mode == "loop" and last - first + 1 == 30
    # forward then backward
    frames = _wave(30, 90)
    (first, last), mode = build_atlas._loop_choice(
        build_atlas._small(frames + frames[-2::-1]), 24, (1.0, 2.0)
    )
    assert (first, last, mode) == (0, 29, "pingpong")
    # no loop of that length: the whole clip, without its repeated last frame
    small = build_atlas._small(_wave(31, 30))
    assert build_atlas._loop_choice(small, 24, (2.0, 3.0)) == ((0, 29), "loop")
    small = build_atlas._small(_wave(20, 50))
    assert build_atlas._loop_choice(small, 24, (2.0, 3.0)) == ((0, 19), "loop")


def test_auto_recipe(tmp_path: Path) -> None:
    folder = tmp_path / "fox"
    swim = _wave(61, 30)
    _save(swim, folder / "swim")
    idle = _wave(30, 90)
    _save(idle + idle[-2::-1], folder / "idle")
    _save(_turn(swim[0]), folder / "turn")
    (folder / "meta.json").write_text(json.dumps({"size_cm": [15, 22]}))
    (folder / "notes.txt").write_text("not a clip")
    lines: list[str] = []
    recipe = build_atlas.auto_recipe(folder, lines.append)
    clips = recipe["clips"]
    assert clips["swim"]["mode"] == "loop" and clips["swim"]["step"] == 1
    assert clips["swim"]["range"][1] - clips["swim"]["range"][0] + 1 == 30
    assert clips["idle"] == {
        "source": "idle",
        "fps": 12,
        "mode": "pingpong",
        "range": [0, 29],
        "step": 2,
    }
    assert clips["turn"]["mode"] == "once" and clips["turn"]["fps"] == 24
    assert recipe["meta"] == {"size_cm": [15, 22]}
    assert not any("warning" in line for line in lines)

    # clips not following the brief are reported
    odd = tmp_path / "odd"
    white = Image.new("RGBA", (64, 64), (255, 255, 255, 255))
    _save([*_wave(40, 90), white], odd / "swim")
    _save(_turn(_wave(20, 90)[10]), odd / "turn")
    lines = []
    recipe = build_atlas.auto_recipe(odd, lines.append)
    assert "meta" not in recipe
    warnings = [line for line in lines if "warning" in line]
    assert any("swim: does not end" in w for w in warnings)
    assert any("turn: does not start" in w for w in warnings)

    with pytest.raises(SystemExit):
        build_atlas.auto_recipe(tmp_path / "fox" / "idle")


def test_turn_end_warning(tmp_path: Path) -> None:
    folder = tmp_path / "x"
    swim = _wave(31, 30)
    _save(swim, folder / "swim")
    _save([swim[0]] * 5 + [swim[15]] * 5, folder / "turn")
    lines: list[str] = []
    build_atlas.auto_recipe(folder, lines.append)
    assert any("turn: does not end on the mirrored" in line for line in lines)


def test_auto_and_batch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "clips"
    fox = root / "fox"
    frames = [fish_frame(tail=(n % 6) * 2) for n in range(13)]
    _save(frames, fox / "swim")
    _save(_turn(frames[0], 4, 4), fox / "turn")
    (root / "empty").mkdir(parents=True)
    broken = root / "broken"
    (broken / "swim").mkdir(parents=True)  # no frame in it
    out = tmp_path / "out"

    assert build_atlas.main(["fish", "fox", "--auto", str(fox), "--out", str(out)]) == 0
    recipe = json.loads((fox / "fox.json").read_text(encoding="utf-8"))
    assert set(recipe["clips"]) == {"swim", "turn"}
    assert (out / "fox" / "species.json").exists()
    with pytest.raises(SystemExit):
        build_atlas.main(["coral", "c", "--auto", str(fox), "--out", str(out)])

    # batch: the recipe kept (hand edits stay), the broken folder reported
    recipe["clips"]["swim"]["fps"] = 15
    (fox / "fox.json").write_text(json.dumps(recipe), encoding="utf-8")
    assert build_atlas.main(["batch", str(root), "--out", str(out), "--hd"]) == 1
    desc = json.loads((out / "fox" / "species.json").read_text(encoding="utf-8"))
    assert desc["clips"]["swim"]["fps"] == 15
    assert desc["atlas"]["2x"] == "fox@2x.webp"
    printed = capsys.readouterr().out
    assert "broken: failed" in printed and "1 built, 1 failed" in printed
    # --reanalyse rewrites the recipes
    (broken / "swim").rmdir()
    assert build_atlas.main(["batch", str(root), "--out", str(out), "--reanalyse"]) == 0
    desc = json.loads((out / "fox" / "species.json").read_text(encoding="utf-8"))
    assert desc["clips"]["swim"]["fps"] == 24


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_video_sources(tmp_path: Path) -> None:
    frames = _save(_wave(24, 12), tmp_path / "frames")
    video = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-framerate",
            "12",
            "-i",
            str(frames / "%03d.png"),
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )
    assert build_atlas.probe_fps(video) == 12
    assert build_atlas.probe_fps(frames) == 24
    assert build_atlas.probe_fps(tmp_path / "missing.mp4") == 24
    small, fps = build_atlas.read_small(video, width=32)
    assert fps == 12 and small.shape == (24, 32, 32)
    small, fps = build_atlas.read_small(frames, width=32)
    assert fps == 24 and len(small) == 24
    apng = tmp_path / "clip.png"
    images = _wave(4, 4)
    images[0].save(apng, save_all=True, append_images=images[1:])
    assert len(build_atlas.read_small(apng, width=16)[0]) == 4
    assert len(build_atlas.read_frames(video, (2, 5))) == 4


def test_long_clip_without_loop_is_capped(tmp_path: Path) -> None:
    folder = tmp_path / "x"
    swim = _wave(31, 30)
    _save(swim, folder / "swim")
    # 200 frames slowly drifting: no 2-4 s loop, the whole clip is too long
    _save([*_wave(199, 400), swim[0]], folder / "idle")
    lines: list[str] = []
    recipe = build_atlas.auto_recipe(folder, lines.append)
    idle = recipe["clips"]["idle"]
    assert idle["step"] == 3 and idle["fps"] == 8
    assert any("no 2-4 s loop" in line for line in lines)


# ---------------------------------------------------------------------------
# Thumbnails
# ---------------------------------------------------------------------------


def test_fish_thumbnail(tmp_path: Path) -> None:
    clips = tmp_path / "clips" / "fox"
    _save([fish_frame(tail=n) for n in range(5)], clips / "swim")
    out = tmp_path / "out"
    assert (
        build_atlas.main(["fish", "fox", "--auto", str(clips), "--out", str(out)]) == 0
    )
    desc = json.loads((out / "fox" / "species.json").read_text(encoding="utf-8"))
    assert desc["thumbnail"] == "thumb.webp"
    thumb = Image.open(out / "fox" / "thumb.webp").convert("RGBA")
    assert thumb.size == build_atlas.THUMB_SIZE
    # the fish fills the width, yellow in the middle, clear corners
    assert thumb.getchannel("A").getbbox()[2] - thumb.getchannel("A").getbbox()[0] >= 90
    r, _g, b, a = thumb.getpixel((48, 32))
    assert a > 200 and r > 200 and b < 80
    assert thumb.getpixel((0, 0))[3] == 0


def test_coral_thumbnail_in_default_colours(tmp_path: Path) -> None:
    frames = tmp_path / "day"
    frames.mkdir()
    image = Image.new("RGBA", (40, 20), (0, 0, 0, 0))
    ImageDraw.Draw(image).rectangle((0, 0, 19, 19), fill=(255, 0, 0, 255))
    ImageDraw.Draw(image).rectangle((20, 0, 39, 19), fill=(128, 128, 128, 255))
    image.save(frames / "000.png")
    meta = tmp_path / "meta.json"
    meta.write_text(
        json.dumps(
            {"palette": [{"default": "#00ff00"}, "#000000", {}, {"default": "zz"}]}
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    args = ["coral", "c", "--clip", f"day={frames}", "--meta", str(meta)]
    assert build_atlas.main([*args, "--out", str(out), "--frame", "40x20"]) == 0
    thumb = Image.open(out / "c" / "thumb.webp").convert("RGBA")
    left = thumb.getpixel((10, thumb.height // 2))
    right = thumb.getpixel((thumb.width - 10, thumb.height // 2))
    assert left[1] > 200 and left[0] < 60, "red zone in palette colour 1 (green)"
    assert abs(right[0] - right[1]) < 20, "grey zone in colour 4 (grey fallback)"


def test_palette_fallbacks() -> None:
    assert build_atlas._palette_rgb(None) == [(0.6, 0.6, 0.6)] * 4
    colours = build_atlas._palette_rgb(["#ff0000", 3])
    assert colours[0] == (1.0, 0.0, 0.0) and colours[1] == (0.6, 0.6, 0.6)


def test_thumbnail_of_empty_frame() -> None:
    thumb = build_atlas.thumbnail(Image.new("RGBA", (30, 10)))
    assert thumb.size == build_atlas.THUMB_SIZE
