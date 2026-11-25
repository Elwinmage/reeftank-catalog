"""Photos of Gemini turned into degraded atlases, species by species."""

from __future__ import annotations

import argparse
import base64
import importlib
import io
import json
import sys
import urllib.error
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
gen = importlib.import_module("generate_stills")

MAGENTA = (255, 0, 255)
GREEN = (0, 255, 0)


def png(
    key: tuple[int, int, int] = MAGENTA, size: tuple[int, int] = (640, 400)
) -> bytes:
    """A flat background with a fish (an ellipse and a tail)."""
    image = Image.new("RGB", size, key)
    draw = ImageDraw.Draw(image)
    w, h = size
    draw.ellipse((160, h * 0.3, w - 100, h * 0.7), fill=(230, 120, 20))
    draw.polygon([(180, h / 2), (100, h * 0.3), (100, h * 0.7)], fill=(230, 120, 20))
    out = io.BytesIO()
    image.save(out, "PNG")
    return out.getvalue()


def answer(blob: bytes) -> dict[str, Any]:
    data = base64.b64encode(blob).decode()
    return {"candidates": [{"content": {"parts": [{"inlineData": {"data": data}}]}}]}


def make_root(tmp_path: Path, entries: dict[str, Any]) -> Path:
    for name, descriptor in entries.items():
        folder = tmp_path / "fish" / name
        folder.mkdir(parents=True)
        if descriptor is not None:
            (folder / "species.json").write_text(json.dumps(descriptor))
    return tmp_path


def namespace(**changes: Any) -> argparse.Namespace:
    values: dict[str, Any] = {
        "section": "fish",
        "model": "gemini-x",
        "only": None,
        "limit": None,
        "redo": False,
        "attempts": 3,
        "background": None,
        "delay": 0,
        "photos": None,
        "from_photos": None,
        "dry_run": False,
    }
    return argparse.Namespace(**{**values, **changes})


@pytest.fixture(autouse=True)
def no_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gen.time, "sleep", lambda _s: None)


class Reply:
    def __init__(self, payload: Any) -> None:
        self.payload = payload

    def __enter__(self) -> Reply:
        return self

    def __exit__(self, *_a: Any) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def http_error(code: int, body: bytes) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("u", code, "m", {}, io.BytesIO(body))  # type: ignore[arg-type]


def test_post(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Any] = []

    def urlopen(request: Any, timeout: int) -> Reply:
        seen.append(request)
        return Reply({"ok": 1})

    monkeypatch.setattr(gen.urllib.request, "urlopen", urlopen)
    assert gen.post("https://x", "k", {"a": 1}) == {"ok": 1}
    assert seen[0].get_header("X-goog-api-key") == "k"
    monkeypatch.setattr(gen.urllib.request, "urlopen", lambda *a, **k: Reply([1]))
    with pytest.raises(gen.GenerationError, match="JSON object"):
        gen.post("https://x", "k", {})


@pytest.mark.parametrize(
    ("code", "body", "fatal"),
    [
        (429, b'{"error": {"message": "quota"}}', True),
        (403, b"not json", True),
        (400, b'{"error": {"message": "API key not valid"}}', True),
        (400, b'{"error": {"message": "bad request"}}', False),
        (500, b"", False),
    ],
)
def test_post_http_errors(
    monkeypatch: pytest.MonkeyPatch, code: int, body: bytes, fatal: bool
) -> None:
    def urlopen(*_a: Any, **_k: Any) -> Any:
        raise http_error(code, body)

    monkeypatch.setattr(gen.urllib.request, "urlopen", urlopen)
    with pytest.raises(gen.GenerationError) as caught:
        gen.post("https://x", "k", {})
    assert caught.value.fatal is fatal
    assert f"HTTP {code}" in str(caught.value)


def test_post_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    def urlopen(*_a: Any, **_k: Any) -> Any:
        raise urllib.error.URLError("down")

    monkeypatch.setattr(gen.urllib.request, "urlopen", urlopen)
    with pytest.raises(gen.GenerationError, match="not reachable") as caught:
        gen.post("https://x", "k", {})
    assert not caught.value.fatal


def test_generate(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[Any] = []

    def fake(url: str, key: str, body: dict[str, Any]) -> dict[str, Any]:
        sent.append((url, key, body))
        return answer(b"picture")

    monkeypatch.setattr(gen, "post", fake)
    assert gen.generate("a fish", "gemini-x", "k") == b"picture"
    url, key, body = sent[0]
    assert url.endswith("/gemini-x:generateContent") and key == "k"
    assert body["generationConfig"]["imageConfig"] == {"aspectRatio": "3:2"}
    assert body["contents"][0]["parts"][0]["text"] == "a fish"
    for payload, message in (
        ({"candidates": [{"content": {"parts": [{"text": "words"}]}}]}, "no picture"),
        ({"promptFeedback": {"blockReason": "SAFETY"}}, "blocked: SAFETY"),
        (
            {"candidates": [{"content": {"parts": [{"inlineData": {"data": "@@"}}]}}]},
            "not readable",
        ),
    ):
        monkeypatch.setattr(gen, "post", lambda *_a, p=payload: p)
        with pytest.raises(gen.GenerationError, match=message):
            gen.generate("a fish", "m", "k")


def test_targets(tmp_path: Path) -> None:
    root = make_root(
        tmp_path,
        {
            "bare": {"kind": "fish", "scientific": "Bare bare"},
            "real": {"kind": "fish", "atlas": {"1x": "real.webp"}},
            "soft": {"kind": "fish", "atlas": {"1x": "soft.webp"}, "degraded": True},
            "broken": None,
            "list": [],
        },
    )
    (root / "fish" / "list" / "species.json").write_text("[]")
    assert [f.name for f in gen.targets(root, [], False)] == ["bare"]
    assert [f.name for f in gen.targets(root, [], True)] == ["bare", "soft"]
    assert [f.name for f in gen.targets(root, ["soft"], True)] == ["soft"]
    assert gen.species_name(root / "fish" / "bare") == "Bare bare"
    assert gen.species_name(root / "fish" / "broken") == "Broken"
    (root / "fish" / "real" / "species.json").write_text("{}")
    assert gen.species_name(root / "fish" / "real") == "Real"


def test_a_species_gets_its_atlas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    root = make_root(
        tmp_path, {"betta_splendens": {"kind": "fish", "scientific": "Betta splendens"}}
    )
    prompts: list[str] = []

    def fake(prompt: str, model: str, key: str, aspect: str) -> bytes:
        prompts.append(prompt)
        return png()

    monkeypatch.setattr(gen, "generate", fake)
    monkeypatch.setenv(gen.KEY_ENV, "k")
    photos = tmp_path / "photos"
    assert gen.run(namespace(photos=photos), root) == 0
    assert "Betta splendens" in prompts[0] and "magenta" in prompts[0]
    folder = root / "fish" / "betta_splendens"
    data = json.loads((folder / "species.json").read_text())
    assert data["degraded"] is True and data["scientific"] == "Betta splendens"
    assert (
        data["source"]["type"] == "ai-generated"
        and data["source"]["model"] == "gemini-x"
    )
    assert (folder / "betta_splendens.webp").exists() and (
        photos / "betta_splendens-1.png"
    ).exists()
    assert "1 done, 0 failed" in capsys.readouterr().out
    # started again: nothing left to do
    assert gen.run(namespace(), root) == 0
    assert "0 done, 0 failed" in capsys.readouterr().out
    assert len(prompts) == 1


def test_a_refused_photo_is_asked_again_on_the_other_background(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_root(tmp_path, {"pink": {"kind": "fish", "scientific": "Pink fish"}})
    answers = iter(
        [b"not an image", Image.new("RGB", (600, 600), (90, 90, 90)), png(MAGENTA)]
    )
    prompts: list[str] = []

    def fake(prompt: str, model: str, key: str, aspect: str) -> bytes:
        prompts.append(prompt)
        item = next(answers)
        if isinstance(item, Image.Image):
            out = io.BytesIO()
            item.save(out, "PNG")
            return out.getvalue()
        return item

    monkeypatch.setattr(gen, "generate", fake)
    monkeypatch.setenv(gen.KEY_ENV, "k")
    photos = tmp_path / "photos"
    assert gen.run(namespace(photos=photos), root) == 0
    assert ["magenta" in p for p in prompts] == [
        True,
        False,
        True,
    ] or "green" in prompts[1]
    assert (photos / "pink-2-refused.png").exists() and (photos / "pink-3.png").exists()
    assert json.loads((root / "fish" / "pink" / "species.json").read_text())["degraded"]


def test_failures_are_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    root = make_root(tmp_path, {"a": {"kind": "fish"}, "b": {"kind": "fish"}})

    def fake(prompt: str, model: str, key: str, aspect: str) -> bytes:
        raise gen.GenerationError("HTTP 500")

    monkeypatch.setattr(gen, "generate", fake)
    monkeypatch.setenv(gen.KEY_ENV, "k")
    assert gen.run(namespace(attempts=2, background="green"), root) == 1
    out = capsys.readouterr().out
    assert "fish/a: failed (HTTP 500)" in out and "0 done, 2 failed" in out
    assert (
        gen.make_atlas(root / "fish" / "a", namespace(attempts=0), "k") == "no attempt"
    )


def test_the_run_stops_when_the_service_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    root = make_root(tmp_path, {"a": {"kind": "fish"}, "b": {"kind": "fish"}})
    calls = []

    def fake(prompt: str, model: str, key: str, aspect: str) -> bytes:
        calls.append(prompt)
        raise gen.GenerationError("HTTP 429 quota", fatal=True)

    monkeypatch.setattr(gen, "generate", fake)
    monkeypatch.setenv(gen.KEY_ENV, "k")
    assert gen.run(namespace(), root) == 2
    assert len(calls) == 1 and "stopped on a" in capsys.readouterr().out


def test_dry_run_and_missing_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    root = make_root(
        tmp_path, {n: {"kind": "fish", "scientific": f"Sp {n}"} for n in "abc"}
    )
    assert gen.run(namespace(dry_run=True, limit=2), root) == 0
    out = capsys.readouterr().out
    assert (
        "fish/a: Sp a" in out and "fish/c" not in out and "2 species, 6 requests" in out
    )
    monkeypatch.delenv(gen.KEY_ENV, raising=False)
    assert gen.run(namespace(), root) == 2
    assert gen.KEY_ENV in capsys.readouterr().out


def test_command(tmp_path: Path, capsys: Any) -> None:
    root = make_root(tmp_path, {"a": {"kind": "fish"}})
    assert gen.main(["--root", str(root), "--dry-run", "--only", "a"]) == 0
    assert "1 species" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        gen.main(["--attempts", "0"])


def test_atlases_from_the_photos_of_an_earlier_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    root = make_root(
        tmp_path, {n: {"kind": "fish", "scientific": f"Sp {n}"} for n in "abcdef"}
    )
    photos = tmp_path / "photos"
    photos.mkdir()
    gray = io.BytesIO()
    Image.new("RGB", (600, 600), (90, 90, 90)).save(gray, "PNG")
    # a: the refused photo comes after the accepted one, which is on the green
    (photos / "a-1-refused.png").write_bytes(png())
    (photos / "a-2.png").write_bytes(png(GREEN))
    # b: only a refused photo, that the checks accept now, on the other background
    (photos / "b-2-refused.png").write_bytes(png())
    # c: nothing kept; d: only unfit photos; e: not an image; f: a name that is not ours
    (photos / "d-1-refused.png").write_bytes(gray.getvalue())
    (photos / "e-1.png").write_bytes(b"not an image")
    (photos / "f-x.png").write_bytes(png())
    monkeypatch.delenv(gen.KEY_ENV, raising=False)  # no key, no request
    monkeypatch.setattr(gen, "generate", lambda *_a: pytest.fail("a request was made"))
    assert gen.run(namespace(from_photos=photos), root) == 1
    out = capsys.readouterr().out
    assert "fish/a: done" in out and "fish/b: done" in out
    assert "fish/d: failed (d-1-refused.png: " in out
    assert "fish/e: failed (e-1.png is not an image)" in out
    assert "2 done, 2 failed, 2 without photo" in out
    assert [p for _, p in gen.saved_photos(photos, "a")] == [
        photos / "a-2.png",
        photos / "a-1-refused.png",
    ]
    for name in "ab":
        data = json.loads((root / "fish" / name / "species.json").read_text())
        assert data["degraded"] is True and data["source"]["type"] == "ai-generated"
    # started again, the finished ones are left; a fixed background is the only one tried
    assert (
        gen.run(namespace(from_photos=photos, background="green", only=["b"]), root)
        == 0
    )
    assert (
        gen.run(namespace(from_photos=photos, background="green", only=["d"]), root)
        == 1
    )
    assert (
        gen.main(["--root", str(root), "--from-photos", str(photos), "--only", "c"])
        == 0
    )


# -- the plants ----------------------------------------------------------------------


def plant_png(
    key: tuple[int, int, int] = GREEN, size: tuple[int, int] = (600, 800)
) -> bytes:
    """A tall plant (a stem with leaves) on a flat background."""
    image = Image.new("RGB", size, key)
    draw = ImageDraw.Draw(image)
    w, h = size
    draw.rectangle((w / 2 - 8, h * 0.1, w / 2 + 8, h * 0.9), fill=(40, 120, 30))
    draw.ellipse((w * 0.3, h * 0.15, w * 0.7, h * 0.35), fill=(60, 160, 40))
    draw.ellipse((w * 0.3, h * 0.5, w * 0.7, h * 0.7), fill=(60, 160, 40))
    out = io.BytesIO()
    image.save(out, "PNG")
    return out.getvalue()


def make_plants(tmp_path: Path, entries: dict[str, Any]) -> Path:
    for name, descriptor in entries.items():
        folder = tmp_path / "plants" / name
        folder.mkdir(parents=True)
        (folder / "species.json").write_text(json.dumps(descriptor))
    return tmp_path


PLANTS = {
    "egeria_densa": {
        "kind": "plant",
        "scientific": "Egeria densa",
        "zone": "background",
    },
    "ludwigia_repens": {
        "kind": "plant",
        "scientific": "Ludwigia repens",
        "zone": "midground",
    },
}


def test_a_plant_gets_its_atlas_with_the_prompt_of_its_form(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    root = make_plants(tmp_path, PLANTS)
    asked: list[tuple[str, str]] = []

    def fake(prompt: str, model: str, key: str, aspect: str) -> bytes:
        asked.append((prompt, aspect))
        return plant_png(GREEN if "green" in prompt else MAGENTA)

    monkeypatch.setattr(gen, "generate", fake)
    monkeypatch.setenv(gen.KEY_ENV, "k")
    assert gen.run(namespace(section="plants", photos=tmp_path / "photos"), root) == 0
    assert "plants/egeria_densa: done" in capsys.readouterr().out
    prompts = {p.split(",")[0]: (p, a) for p, a in asked}
    egeria, aspect = prompts["Realistic photograph of Egeria densa"]
    assert "stems" in egeria and "magenta" in egeria and aspect == "3:4"
    ludwigia, _ = next(v for k, v in prompts.items() if "Ludwigia" in k)
    assert "flat green #00FF00" in ludwigia  # a red plant starts on green
    folder = root / "plants" / "egeria_densa"
    data = json.loads((folder / "species.json").read_text())
    assert data["degraded"] is True and data["kind"] == "plant"
    assert data["scientific"] == "Egeria densa" and data["columns"] == 1
    assert (folder / "egeria_densa.webp").exists() and (folder / "thumb.webp").exists()
    assert (tmp_path / "photos" / "egeria_densa-1.png").exists()


def test_a_refused_plant_is_asked_again_on_its_own_background_and_from_photos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_plants(tmp_path, {"egeria_densa": PLANTS["egeria_densa"]})
    answers = iter([plant_png(MAGENTA)])  # asked again on magenta, never on green
    prompts: list[str] = []

    def fake(prompt: str, model: str, key: str, aspect: str) -> bytes:
        prompts.append(prompt)
        return next(answers) if len(prompts) > 1 else b"not an image"

    monkeypatch.setattr(gen, "generate", fake)
    monkeypatch.setenv(gen.KEY_ENV, "k")
    photos = tmp_path / "photos"
    assert gen.run(namespace(section="plants", photos=photos), root) == 0
    assert all("flat magenta" in p for p in prompts)
    assert (photos / "egeria_densa-2.png").exists()
    # from the photos kept: the same entry again, with --redo (a degraded one)
    monkeypatch.setattr(gen, "generate", lambda *_a: pytest.fail("a request was made"))
    redo = namespace(section="plants", from_photos=photos, redo=True)
    assert gen.run(redo, root) == 0


def test_a_plant_never_goes_to_the_green_background_by_itself() -> None:
    plants = Path("plants")
    # a green plant: magenta only; a red one: green only, as `species_keys` says
    assert gen.backgrounds_for(plants / "egeria_densa", "plants") == ["magenta"]
    assert gen.backgrounds_for(plants / "ludwigia_repens", "plants") == ["green"]
    # a fish still goes from one to the other
    assert gen.backgrounds_for(Path("fish/x"), "fish") == ["magenta", "green"]


def test_the_plant_commands_list_the_plants(tmp_path: Path, capsys: Any) -> None:
    root = make_plants(tmp_path, PLANTS)
    assert gen.main(["--root", str(root), "--section", "plants", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "plants/egeria_densa: Egeria densa" in out and "2 species" in out
    assert [f.name for f in gen.targets(root, [], False, "plants")] == [
        "egeria_densa",
        "ludwigia_repens",
    ]


# -- the invertebrates ---------------------------------------------------------------


def invert_png(
    key: tuple[int, int, int] = MAGENTA, size: tuple[int, int] = (900, 600)
) -> bytes:
    """A wide animal (a shell with a foot) on a flat background."""
    image = Image.new("RGB", size, key)
    draw = ImageDraw.Draw(image)
    w, h = size
    draw.ellipse((w * 0.15, h * 0.25, w * 0.85, h * 0.75), fill=(190, 120, 60))
    draw.rectangle((w * 0.2, h * 0.6, w * 0.8, h * 0.72), fill=(210, 190, 170))
    out = io.BytesIO()
    image.save(out, "PNG")
    return out.getvalue()


def make_inverts(tmp_path: Path, entries: dict[str, Any]) -> Path:
    for name, descriptor in entries.items():
        folder = tmp_path / "invertebrates" / name
        folder.mkdir(parents=True)
        (folder / "species.json").write_text(json.dumps(descriptor))
    return tmp_path


INVERTS = {
    "nerita": {
        "kind": "invertebrate",
        "scientific": "Nerita sp.",
        "movement": "glass",
        "role": ["algae_eater"],
    },
    "neocaridina_davidi": {
        "kind": "invertebrate",
        "scientific": "Neocaridina davidi",
        "movement": "ground",
        "role": ["detritivore"],
    },
}


def test_an_invertebrate_gets_its_atlas_with_the_prompt_of_its_form(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    root = make_inverts(tmp_path, INVERTS)
    asked: list[tuple[str, str]] = []

    def fake(prompt: str, model: str, key: str, aspect: str) -> bytes:
        asked.append((prompt, aspect))
        return invert_png(GREEN if "green" in prompt else MAGENTA)

    monkeypatch.setattr(gen, "generate", fake)
    monkeypatch.setenv(gen.KEY_ENV, "k")
    assert (
        gen.run(namespace(section="invertebrates", photos=tmp_path / "photos"), root)
        == 0
    )
    assert "invertebrates/nerita: done" in capsys.readouterr().out
    prompts = {p.split(",")[0]: (p, a) for p, a in asked}
    snail, aspect = prompts["Realistic photograph of Nerita sp."]
    # A snail: three-quarter view, square picture, on magenta
    assert "one adult snail" in snail and "magenta" in snail and aspect == "1:1"
    shrimp, shrimp_aspect = next(v for k, v in prompts.items() if "Neocaridina" in k)
    # A shrimp: side view, head to the right, wide picture, and green because
    # a cherry shrimp would be eaten by magenta
    assert "one adult shrimp" in shrimp and "head to the right" in shrimp
    assert shrimp_aspect == "3:2" and "flat green #00FF00" in shrimp
    folder = root / "invertebrates" / "nerita"
    data = json.loads((folder / "species.json").read_text())
    assert data["degraded"] is True and data["kind"] == "invertebrate"
    assert data["scientific"] == "Nerita sp." and data["columns"] == 1
    assert data["role"] == ["algae_eater"] and data["movement"] == "glass"
    assert (folder / "nerita.webp").exists() and (folder / "thumb.webp").exists()
    assert (tmp_path / "photos" / "nerita-1.png").exists()


def test_an_invertebrate_keeps_the_background_of_its_species() -> None:
    inverts = Path("invertebrates")
    # a brown snail: magenta; a cherry shrimp: green, as `species_keys` says
    assert gen.backgrounds_for(inverts / "nerita", "invertebrates") == ["magenta"]
    assert gen.backgrounds_for(inverts / "neocaridina_davidi", "invertebrates") == [
        "green"
    ]


def test_an_invertebrate_photo_is_checked_as_an_invertebrate() -> None:
    tiny = Image.new("RGB", (900, 600), MAGENTA)
    ImageDraw.Draw(tiny).rectangle((430, 280, 470, 320), fill=(190, 120, 60))
    problems = gen.problems_of(tiny, "magenta", "invertebrates")
    assert any("invertebrate" in p for p in problems)


def test_the_invertebrate_commands_list_them(tmp_path: Path, capsys: Any) -> None:
    root = make_inverts(tmp_path, INVERTS)
    assert (
        gen.main(["--root", str(root), "--section", "invertebrates", "--dry-run"]) == 0
    )
    out = capsys.readouterr().out
    assert "invertebrates/nerita: Nerita sp." in out and "2 species" in out
