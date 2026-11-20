"""The animated pictures of the README, built from the atlases of the catalog."""

from __future__ import annotations

import importlib
import io
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
make_demo = importlib.import_module("make_demo")
make_readme = importlib.import_module("make_readme")

ROOT = Path(__file__).parent.parent
#: A picture of the README stays light (GitHub loads it with the page)
MAX_BYTES = 400 * 1024


def test_the_demo_species_exist_and_have_the_clips_they_play() -> None:
    for species in make_demo.DEMOS.values():
        frames, clips = make_demo.frames_of(ROOT / "fish" / species)
        assert frames and {"swim", "turn"} <= set(clips)


def test_the_pictures_of_the_readme_are_the_ones_built_here() -> None:
    wanted = {f"docs/img/{name}.webp" for name in make_demo.DEMOS}
    assert wanted == {src for src, _ in make_readme.DEMO_IMAGES}


def test_a_picture_is_an_animation_that_swims_turns_and_comes_back() -> None:
    for name, species in make_demo.DEMOS.items():
        shown, delay = make_demo.sequence(ROOT / "fish" / species)
        _, clips = make_demo.frames_of(ROOT / "fish" / species)
        swim = clips["swim"]["to"] - clips["swim"]["from"] + 1
        turn = clips["turn"]["to"] - clips["turn"]["from"] + 1
        expected = 2 * (swim * make_demo.SWIM_CYCLES + turn)
        assert abs(len(shown) - expected / make_demo.STEP) <= 2, name
        assert delay == round(1000 * make_demo.STEP / clips["swim"]["fps"])
        assert shown[0].width == make_demo.WIDTH


def test_the_committed_pictures_play_and_stay_light() -> None:
    for name in make_demo.DEMOS:
        path = ROOT / "docs" / "img" / f"{name}.webp"
        assert 0 < path.stat().st_size < MAX_BYTES, name
        with Image.open(path) as picture:
            assert getattr(picture, "n_frames", 1) > 10, name


def test_render_makes_an_animated_webp() -> None:
    name, species = next(iter(make_demo.DEMOS.items()))
    data = make_demo.render(name, species)
    with Image.open(io.BytesIO(data)) as picture:
        assert picture.format == "WEBP"
        assert getattr(picture, "n_frames", 1) > 10


def test_main_writes_the_pictures(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(make_demo, "ROOT", tmp_path)
    monkeypatch.setattr(make_demo, "OUT", tmp_path / "img")
    monkeypatch.setattr(make_demo, "render", lambda name, species: b"webp")
    assert make_demo.main([]) == 0
    assert {p.name for p in (tmp_path / "img").iterdir()} == {
        f"{n}.webp" for n in make_demo.DEMOS
    }
    assert "KB" in capsys.readouterr().out


def test_check_fails_on_a_missing_picture_and_passes_when_present(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(make_demo, "ROOT", tmp_path)
    monkeypatch.setattr(make_demo, "OUT", tmp_path / "img")
    monkeypatch.setattr(make_demo, "render", lambda name, species: b"webp")
    assert make_demo.main(["--check"]) == 1
    assert "missing: " in capsys.readouterr().err
    assert not list((tmp_path / "img").iterdir())  # --check writes nothing
    make_demo.main([])
    assert make_demo.main(["--check"]) == 0
