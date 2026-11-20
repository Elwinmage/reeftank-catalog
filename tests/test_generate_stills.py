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

    def fake(prompt: str, model: str, key: str) -> bytes:
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

    def fake(prompt: str, model: str, key: str) -> bytes:
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

    def fake(prompt: str, model: str, key: str) -> bytes:
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

    def fake(prompt: str, model: str, key: str) -> bytes:
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
