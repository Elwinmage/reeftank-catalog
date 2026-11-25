"""Clips of Wan (run by ComfyUI) turned into staged atlases, looked at, accepted."""

from __future__ import annotations

import argparse
import importlib
import io
import json
import sys
import types
import urllib.error
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
wan = importlib.import_module("wan_clips")
validate = importlib.import_module("validate")

MAGENTA = (255, 0, 255)


def fake_remove(image: Image.Image, session: Any) -> Image.Image:
    """rembg stand-in: opaque where the colour is far from the corners'."""
    rgb = np.asarray(image.convert("RGB")).astype(np.float32)
    corner = rgb[0, 0]
    far = np.sqrt(((rgb - corner) ** 2).sum(axis=-1)) > 60
    alpha = np.where(far, 255, 0).astype(np.uint8)
    alpha[:, :1] = 128  # a soft column, to unmix
    return Image.fromarray(np.dstack([rgb.astype(np.uint8), alpha]), "RGBA")


@pytest.fixture(autouse=True)
def rembg(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """A fake rembg module; the models asked for."""
    models: list[str] = []
    module = types.ModuleType("rembg")

    def new_session(model: str, **kwargs: Any) -> dict[str, Any]:
        models.append(model + ("@cpu" if kwargs.get("providers") else "@gpu"))
        return {"model": model}

    module.new_session = new_session  # type: ignore[attr-defined]
    module.remove = fake_remove  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rembg", module)
    return models


def fish(
    tail: int = 0,
    size: tuple[int, int] = (200, 100),
    key: tuple[int, int, int] = MAGENTA,
) -> Image.Image:
    """A yellow fish (ellipse and tail) on magenta, head to the right."""
    image = Image.new("RGB", size, key)
    draw = ImageDraw.Draw(image)
    w, h = size
    draw.ellipse((w * 0.3, h * 0.35, w * 0.8, h * 0.65), fill=(250, 210, 0))
    draw.polygon(
        [(w * 0.3, h / 2), (w * 0.15 - tail, h * 0.3), (w * 0.15 - tail, h * 0.7)],
        fill=(250, 210, 0),
    )
    return image


def gif(frames: list[Image.Image]) -> bytes:
    out = io.BytesIO()
    frames[0].save(
        out, "GIF", save_all=True, append_images=frames[1:], duration=62, loop=0
    )
    return out.getvalue()


def swim_clip(key: tuple[int, int, int] = MAGENTA) -> bytes:
    """A loop: starts and ends on the reference picture."""
    return gif([fish(tail=(n % 6) * 3, key=key) for n in range(13)] + [fish(key=key)])


def turn_clip(key: tuple[int, int, int] = MAGENTA) -> bytes:
    """The reference, a blend to its mirror, the mirror."""
    ref = fish(key=key)
    mirror = ref.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    frames = [ref] * 4
    frames += [Image.blend(ref, mirror, (i + 1) / 7) for i in range(6)]
    return gif(frames + [mirror] * 4)


SPEC = {
    "version": 1,
    "model": "wan-test",
    "fps": 16,
    "size": [200, 100],
    "clips": {
        "swim": {"frames": 33, "last": "same", "prompt": "swim [SPECIE]"},
        "idle": {
            "frames": 50,
            "last": "same",
            "prompt": "idle [SPECIE] on [BACKGROUND]",
        },
        "turn": {"frames": 49, "last": "mirror", "prompt": "turn [SPECIE]"},
    },
    "negative": "blurry",
}

#: An exported workflow: two images, two text encoders, a sampler, a save
EXPORT: dict[str, Any] = {
    "1": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
    "2": {"class_type": "LoadImage", "inputs": {"image": "b.png"}},
    "3": {
        "class_type": "ImageScale",
        "inputs": {"image": ["2", 0], "width": 832, "height": 480},
    },
    "4": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "a fish", "clip": ["9", 0]},
        "_meta": {"title": "Positive"},
    },
    "5": {"class_type": "PrimitiveString", "inputs": {"value": "ugly"}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": ["5", 0]}},
    "7": {
        "class_type": "WanFirstLastFrameToVideo",
        "inputs": {
            "positive": ["4", 0],
            "negative": ["6", 0],
            "end_image": ["3", 0],
            "start_image": ["1", 0],
            "length": 81,
        },
    },
    "8": {
        "class_type": "KSampler",
        "inputs": {"positive": ["7", 0], "seed": 1, "cfg": 1.0, "flag": True},
    },
    "9": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5.safetensors"}},
    "10": {
        "class_type": "SaveVideo",
        "inputs": {"video": ["8", 0], "filename_prefix": "video/x"},
    },
    "x": {"class_type": "Note", "inputs": {}},
}


@pytest.fixture
def catalog(tmp_path: Path) -> Path:
    """A catalog: a degraded fish, a real one, one without atlas, a broken folder."""
    root = tmp_path / "catalog"
    sheet = Image.new("RGBA", (40 * 8, 20), (0, 0, 0, 0))
    for n in range(8):
        sheet.paste(fish(n).resize((40, 20)).convert("RGBA"), (n * 40, 0))
    degraded = {
        "kind": "fish",
        "frame": [40, 20],
        "columns": 8,
        "clips": {
            "swim": {"from": 0, "to": 3, "fps": 24, "loop": True},
            "idle": {"from": 4, "to": 5, "fps": 12, "loop": True, "pingpong": True},
            "turn": {"from": 6, "to": 7, "fps": 24, "loop": False},
        },
        "atlas": {"1x": "betta.webp"},
        "degraded": True,
        "source": {"type": "ai-generated", "model": "gemini"},
        "scientific": "Betta splendens",
        "names": {"en": "Betta"},
    }
    for entry_id, desc in (
        ("betta", degraded),
        ("real", {**degraded, "degraded": False, "atlas": {"1x": "real.webp"}}),
        ("bare", {"kind": "fish", "scientific": "Bare fish"}),
    ):
        folder = root / "fish" / entry_id
        folder.mkdir(parents=True)
        (folder / "species.json").write_text(json.dumps(desc), encoding="utf-8")
    sheet.save(root / "fish" / "betta" / "betta.webp")
    (root / "fish" / "broken").mkdir()
    photos = root / "photos"
    photos.mkdir()
    fish(size=(150, 100)).save(photos / "betta-1-refused.png")
    fish(size=(150, 100)).save(photos / "betta-2.png")
    (root / "prompts.json").write_text(json.dumps(SPEC), encoding="utf-8")
    return root


class FakeComfy:
    """ComfyUI: uploads, queue, history, files; the clip follows the prefix asked."""

    def __init__(self) -> None:
        self.queued: list[dict[str, Any]] = []
        self.uploads: list[str] = []
        self.pending = 0  # polls answering "not done" before the outputs
        self.fail: str | None = None
        self.key = MAGENTA  # background of the clips made
        self.clip: bytes | None = None  # a clip answered whatever is asked

    def __call__(
        self, url: str, data: bytes | None = None, headers: dict | None = None
    ) -> bytes:
        if url.endswith("/upload/image"):
            name = data.split(b'filename="')[1].split(b'"')[0].decode()  # type: ignore[union-attr]
            self.uploads.append(name)
            return json.dumps({"name": name, "subfolder": "in"}).encode()
        if url.endswith("/prompt"):
            self.queued.append(json.loads(data or b"{}")["prompt"])
            return json.dumps({"prompt_id": f"p{len(self.queued)}"}).encode()
        if "/history/" in url:
            prompt_id = url.rsplit("/", 1)[1]
            if self.pending:
                self.pending -= 1
                return b"{}"
            if self.fail:
                error = {
                    "node_type": "KSampler",
                    "exception_type": "torch.OutOfMemoryError",
                    "exception_message": self.fail,
                    "current_inputs": {"latent_image": ["tensor([[0., 0.]])"]},
                }
                messages = [["execution_start", {}], ["execution_error", error]]
                status = {"status_str": "error", "messages": messages}
                return json.dumps({prompt_id: {"status": status}}).encode()
            prefix = self.queued[-1]["10"]["inputs"]["filename_prefix"]
            item = {"filename": prefix.split("/")[-1] + ".gif", "subfolder": "reeftank"}
            outputs = {"10": {"images": [item], "animated": [True]}}
            return json.dumps(
                {prompt_id: {"status": {"completed": True}, "outputs": outputs}}
            ).encode()
        assert "/view?" in url
        if self.clip is not None:
            return self.clip
        return turn_clip(self.key) if "_turn" in url else swim_clip(self.key)


def run(catalog: Path, *argv: str) -> int:
    return wan.main(
        ["--root", str(catalog), "--prompts", str(catalog / "prompts.json"), *argv]
    )


def template_file(catalog: Path) -> Path:
    export = catalog / "export.json"
    export.write_text(json.dumps(EXPORT), encoding="utf-8")
    assert run(catalog, "template", str(export)) == 0
    return catalog / "clips" / "wan_workflow.json"


# ---------------------------------------------------------------------------


def test_real_prompts_file() -> None:
    spec = wan.load_spec()
    assert set(spec["clips"]) == set(wan.CLIP_NAMES)
    assert spec["clips"]["turn"]["last"] == "mirror"
    for clip in wan.CLIP_NAMES:
        assert (spec["clips"][clip]["frames"] - 1) % 4 == 0  # Wan: 4n + 1 frames
        assert "Betta" in wan.clip_prompt(spec, clip, "Betta")
    assert wan.duration_of(spec, "swim") == 2
    assert wan.duration_of(SPEC, "idle") == pytest.approx(3.062)


def test_targets_photo_and_reference(catalog: Path, tmp_path: Path) -> None:
    assert [f.name for f in wan.targets(catalog)] == ["bare", "betta"]
    assert [f.name for f in wan.targets(catalog, ["betta"])] == ["betta"]
    photos = catalog / "photos"
    assert wan.reference_photo(photos, "betta") == photos / "betta-2.png"
    assert wan.reference_photo(photos, "bare") is None
    assert wan.reference_photo(tmp_path / "none", "betta") is None
    (photos / "betta-2.png").unlink()
    assert wan.reference_photo(photos, "betta") is None  # only a refused one
    # padded to the ratio of the video, sideways or above and below
    wide = wan.fit_reference(fish(size=(150, 100)), (200, 100))
    assert wide.size == (200, 100) and wide.getpixel((2, 50)) == MAGENTA
    tall = wan.fit_reference(fish(size=(300, 100)), (200, 100))
    assert tall.size == (200, 100) and tall.getpixel((100, 2)) == MAGENTA


def test_template(catalog: Path, capsys: pytest.CaptureFixture[str]) -> None:
    workflow = wan.load_workflow(template_file(catalog))
    printed = capsys.readouterr().out
    inputs = {k: n["inputs"] for k, n in workflow.items()}
    assert inputs["1"]["image"] == "{{FIRST_FRAME}}"
    assert inputs["2"]["image"] == "{{LAST_FRAME}}"
    assert inputs["3"]["width"] == "{{WIDTH}}" and inputs["3"]["height"] == "{{HEIGHT}}"
    assert inputs["4"]["text"] == "{{PROMPT}}"
    assert inputs["5"]["value"] == "{{NEGATIVE}}"  # the text the encoder reads
    assert inputs["7"]["length"] == "{{LENGTH}}"
    assert inputs["8"]["seed"] == "{{SEED}}" and inputs["8"]["cfg"] == 1.0
    assert inputs["8"]["flag"] is True
    assert inputs["10"]["filename_prefix"] == "{{PREFIX}}"
    assert "node 4 (Positive).text" in printed and "node order" not in printed
    assert wan.placeholders_in(workflow) >= set(wan.REQUIRED)


def test_template_by_order_and_missing() -> None:
    export = {
        "b": {"class_type": "LoadImage", "inputs": {"image": "1.png"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "2.png"}},
        "3": {"class_type": "Mix", "inputs": {"a": ["2", 0], "b": ["b", 0]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": ["9", 0]}},
        "5": {"class_type": "Sampler", "inputs": {"negative": ["4", 0]}},
        "9": {"class_type": "Number", "inputs": {"value": 3}},
        "6": {"class_type": "Video", "inputs": {"duration": 5, "noise_seed": 2}},
    }
    result, report = wan.make_template(export)
    assert result["2"]["inputs"]["image"] == "{{FIRST_FRAME}}"
    assert result["b"]["inputs"]["image"] == "{{LAST_FRAME}}"
    assert result["6"]["inputs"] == {
        "duration": "{{DURATION}}",
        "noise_seed": "{{SEED}}",
    }
    assert any("node order" in line for line in report)
    assert report[-1] == "missing, to put by hand: {{PROMPT}}"
    with pytest.raises(SystemExit, match="2 Load Image"):
        wan.make_template({"1": {"class_type": "LoadImage", "inputs": {}}})


def test_load_workflow_errors(tmp_path: Path) -> None:
    path = tmp_path / "w.json"
    path.write_text(json.dumps({"nodes": []}), encoding="utf-8")
    with pytest.raises(SystemExit, match="API format"):
        wan.load_workflow(path)
    path.write_text(json.dumps({"1": {"inputs": {}}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="class_type"):
        wan.load_workflow(path)


def test_fill() -> None:
    workflow = {
        "1": {"class_type": "A", "inputs": {"seed": "{{SEED}}", "x": "{{OTHER}}"}},
        "2": {"class_type": "B"},
    }
    filled = wan.fill(workflow, {"SEED": 7})
    assert filled["1"]["inputs"] == {"seed": 7, "x": "{{OTHER}}"}
    assert workflow["1"]["inputs"]["seed"] == "{{SEED}}"  # a copy


def test_call(monkeypatch: pytest.MonkeyPatch) -> None:
    class Answer(io.BytesIO):
        def __enter__(self) -> Answer:
            return self

        def __exit__(self, *_: object) -> None:
            return None

    def ok(request: Any, timeout: float) -> Answer:
        assert request.full_url == "http://c/x" and timeout == wan.TIMEOUT
        return Answer(b"done")

    monkeypatch.setattr(wan.urllib.request, "urlopen", ok)
    assert wan._call("http://c/x") == b"done"

    def http(request: Any, timeout: float) -> Answer:
        raise urllib.error.HTTPError(
            "http://c/x", 400, "bad", {}, io.BytesIO(b"bad node")
        )

    monkeypatch.setattr(wan.urllib.request, "urlopen", http)
    with pytest.raises(wan.ComfyError, match="HTTP 400 bad node") as err:
        wan._call("http://c/x")
    assert not err.value.fatal

    def down(request: Any, timeout: float) -> Answer:
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(wan.urllib.request, "urlopen", down)
    with pytest.raises(wan.ComfyError, match="not reachable") as err:
        wan._call("http://c/x")
    assert err.value.fatal


def test_comfy_answers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with pytest.raises(wan.ComfyError, match="JSON"):
        wan._json(b"<html>")
    with pytest.raises(wan.ComfyError, match="object"):
        wan._json(b"[1]")
    monkeypatch.setattr(wan, "_call", lambda *a, **k: b'{"error": "no model"}')
    with pytest.raises(wan.ComfyError, match="refused: no model"):
        wan.queue("http://c", {})
    monkeypatch.setattr(wan, "_call", lambda *a, **k: b'{"name": "r.png"}')
    image = tmp_path / "r.png"
    image.write_bytes(b"png")
    assert wan.upload("http://c", image, "r.png") == "r.png"
    # waiting: not done, then done; or a timeout
    comfy = FakeComfy()
    monkeypatch.setattr(wan, "_call", comfy)
    comfy.queued.append({"10": {"inputs": {"filename_prefix": "reeftank/b_swim"}}})
    comfy.pending = 1
    slept: list[float] = []
    outputs = wan.wait("http://c", "p1", timeout=10, poll=2, sleep=slept.append)
    assert slept == [2] and wan.video_of(outputs)["filename"] == "b_swim.gif"
    comfy.pending = 5
    with pytest.raises(wan.ComfyError, match="no video after 4 s"):
        wan.wait("http://c", "p1", timeout=4, poll=2, sleep=lambda _: None)
    comfy.pending, comfy.fail = 0, "out of memory"
    with pytest.raises(wan.ComfyError, match="out of memory"):
        wan.wait("http://c", "p1", timeout=4)
    with pytest.raises(wan.ComfyError, match="no video"):
        wan.video_of(
            {"1": {"images": [{"filename": "a.png"}, "x"]}, "2": [], "3": {"t": 1}}
        )


def test_generate_preview_review_accept(
    catalog: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    template_file(catalog)
    comfy = FakeComfy()
    monkeypatch.setattr(wan, "_call", comfy)
    opened: list[list[str]] = []

    assert run(catalog, "todo") == 0
    assert "betta" in capsys.readouterr().out

    assert run(catalog, "generate", "--only", "betta", "--clips", "swim") == 0
    clips = catalog / "clips" / "betta"
    assert (clips / "swim.gif").exists() and not (clips / "build").exists()
    assert run(catalog, "todo", "--only", "betta") == 0
    assert "clips swim" in capsys.readouterr().out

    assert run(catalog, "generate", "--poll", "0") == 0
    printed = capsys.readouterr().out
    assert (
        "bare: no photo" in printed and "1 done, 0 failed, 1 without photo" in printed
    )
    assert len(comfy.queued) == 3  # the swim already made is kept
    turn = comfy.queued[-1]
    assert turn["1"]["inputs"]["image"] == "in/betta_reference.png"
    assert turn["2"]["inputs"]["image"] == "in/betta_mirror.png"
    assert turn["4"]["inputs"]["text"] == "turn Betta splendens"
    assert turn["7"]["inputs"]["length"] == 49
    idle = comfy.queued[1]
    assert idle["2"]["inputs"]["image"] == "in/betta_reference.png"
    meta = json.loads((clips / "meta.json").read_text(encoding="utf-8"))
    assert meta["source"]["model"] == "wan-test"
    log = json.loads((clips / "wan.json").read_text(encoding="utf-8"))
    assert set(log) == {"swim", "idle", "turn"}
    staged = clips / "build" / "betta" / "species.json"
    desc = json.loads(staged.read_text(encoding="utf-8"))
    assert desc["names"] == {"en": "Betta"} and "degraded" not in desc
    assert desc["source"]["model"] == "wan-test"
    assert run(catalog, "todo", "--only", "betta") == 0
    assert "staged" in capsys.readouterr().out

    # the preview: catalog (left) and new (right), opened in the browser
    with monkeypatch.context() as patch:
        patch.setattr(wan.subprocess, "Popen", lambda cmd, **_: opened.append(cmd))
        assert run(catalog, "preview", "betta") == 0
        assert run(catalog, "preview", "betta", "--player", "gwenview -f") == 0
    html = clips / "preview.html"
    assert "catalog (left)" in html.read_text(encoding="utf-8")
    assert opened[0] == ["xdg-open", str(html)]
    preview = Image.open(clips / "preview.webp")
    assert preview.n_frames > 10 and preview.width > preview.height
    assert opened[1] == ["gwenview", "-f", str(clips / "preview.webp")]
    assert run(catalog, "preview", "betta", "--no-open") == 0
    assert len(opened) == 2

    # review: an unknown answer, then redo the turn
    answers = iter(["what", "r turn"])
    args = argparse.Namespace(
        work=catalog / "clips",
        root=catalog,
        only=None,
        player=None,
        no_open=True,
        section="fish",
    )
    assert wan.review(args, ask=lambda _: next(answers)) == 0
    assert not (clips / "turn.gif").exists() and not (clips / "build").exists()
    assert (clips / "swim.gif").exists()
    assert run(catalog, "review") == 0
    assert "nothing to review" in capsys.readouterr().out

    # made again with a fixed seed, then skipped, quit, accepted
    assert run(catalog, "generate", "--only", "betta", "--seed", "5") == 0
    assert json.loads((clips / "wan.json").read_text())["turn"]["seed"] == 5
    for answer in ("s", "q"):
        assert wan.review(args, ask=lambda _, a=answer: a) == 0
    names = json.loads((catalog / "fish" / "betta" / "species.json").read_text())
    names["names"] = {"en": "Siamese fighting fish"}  # edited since the staging
    (catalog / "fish" / "betta" / "species.json").write_text(json.dumps(names))
    assert wan.review(args, ask=lambda _: "a") == 0
    desc = json.loads((catalog / "fish" / "betta" / "species.json").read_text())
    assert desc["names"] == {"en": "Siamese fighting fish"}
    assert "degraded" not in desc and desc["source"]["model"] == "wan-test"
    assert (catalog / "fish" / "betta" / "thumb.webp").exists()
    assert (clips / "accepted").exists()
    capsys.readouterr()
    assert run(catalog, "todo") == 0  # a real animation now: no longer to do
    assert "betta" not in capsys.readouterr().out


def test_generate_redo_build_accept_commands(
    catalog: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    template_file(catalog)
    comfy = FakeComfy()
    monkeypatch.setattr(wan, "_call", comfy)
    assert run(catalog, "generate", "--limit", "2", "--poll", "0") == 0
    clips = catalog / "clips" / "betta"
    (clips / "accepted").write_text("x")
    # --redo: photo fitted again, every clip made again, recipe rewritten
    assert run(catalog, "generate", "--only", "betta", "--redo") == 0
    assert len(comfy.queued) == 6
    assert run(catalog, "todo", "--only", "betta") == 0
    assert "accepted" in capsys.readouterr().out
    # build again after an edit of the recipe
    (catalog / "clips" / "empty").mkdir()
    assert run(catalog, "build") == 0
    assert run(catalog, "build", "--only", "other") == 0
    assert "0 staged" in capsys.readouterr().out
    # a reference already made is enough without the photo
    (catalog / "photos" / "betta-2.png").unlink()
    assert run(catalog, "generate", "--only", "betta") == 0
    assert run(catalog, "todo", "--only", "betta", "bare") == 0
    printed = capsys.readouterr().out
    assert "bare" in printed and "no photo" in printed
    assert run(catalog, "redo", "betta", "--clips", "idle") == 0
    assert run(catalog, "todo", "--only", "betta") == 0
    assert "clips swim,turn" in capsys.readouterr().out
    assert wan.staged_entries(catalog / "nowhere") == []
    with pytest.raises(SystemExit, match="nothing staged"):
        run(catalog, "preview", "betta")
    with pytest.raises(SystemExit, match="nothing staged"):
        run(catalog, "accept", "betta")
    assert run(catalog, "generate", "--only", "betta") == 0
    assert run(catalog, "accept", "betta") == 0
    assert "validate.py" in capsys.readouterr().out


def test_accept_without_ids(
    catalog: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`accept` with no id takes every staged species, and none of them twice."""
    template_file(catalog)
    monkeypatch.setattr(wan, "_call", FakeComfy())
    assert run(catalog, "accept") == 0
    assert "nothing staged" in capsys.readouterr().out
    assert run(catalog, "generate", "--limit", "2", "--poll", "0") == 0
    assert run(catalog, "accept") == 0
    printed = capsys.readouterr().out
    assert "accepted ->" in printed and "validate.py" in printed
    assert run(catalog, "accept") == 0
    assert "nothing staged" in capsys.readouterr().out


def test_todo_ready_and_prompt(
    catalog: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(catalog, "todo", "--only", "betta") == 0
    assert "ready" in capsys.readouterr().out
    assert run(catalog, "prompt", "betta", "idle") == 0
    assert capsys.readouterr().out.strip() == "idle Betta splendens on magenta"
    folder = catalog / "clips" / "betta"
    folder.mkdir(parents=True)
    fish(key=(0, 255, 0)).save(folder / "reference.png")
    assert run(catalog, "prompt", "betta", "idle") == 0
    assert capsys.readouterr().out.strip() == "idle Betta splendens on green"


def test_generate_stops_and_fails(
    catalog: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workflow = template_file(catalog)
    comfy = FakeComfy()
    comfy.fail = "out of memory"
    monkeypatch.setattr(wan, "_call", comfy)
    assert run(catalog, "generate", "--only", "betta") == 1
    printed = capsys.readouterr().out
    assert (
        "betta: failed (the workflow failed: KSampler torch.OutOfMemoryError "
        "out of memory)" in printed
    )
    assert "tensor" not in printed
    assert wan._failure({"messages": [["execution_error", "x"], "y"]}) == (
        "no error message"
    )

    def down(*_: Any, **__: Any) -> bytes:
        raise wan.ComfyError("ComfyUI not reachable", fatal=True)

    monkeypatch.setattr(wan, "_call", down)
    assert run(catalog, "generate") == 2
    assert "stopped on betta" in capsys.readouterr().out
    # a workflow without its placeholders is refused before any request
    workflow.write_text(json.dumps({"1": {"class_type": "A", "inputs": {}}}))
    with pytest.raises(SystemExit, match="FIRST_FRAME"):
        run(catalog, "generate")


def test_preview_without_catalog_atlas(catalog: Path, tmp_path: Path) -> None:
    staged = catalog / "fish" / "betta"
    out = wan.render_preview(staged, catalog / "fish" / "bare", tmp_path / "p.webp")
    assert "new" in out.read_text(encoding="utf-8")
    assert "left" not in out.read_text(encoding="utf-8")
    assert wan.render_preview(staged, None, tmp_path / "q.webp").exists()
    # an atlas without idle nor turn still plays
    desc = json.loads((staged / "species.json").read_text(encoding="utf-8"))
    desc["clips"] = {"swim": desc["clips"]["swim"]}
    (staged / "species.json").write_text(json.dumps(desc), encoding="utf-8")
    assert len(wan.timeline(staged)) == 2 * wan.PREVIEW_SWIMS * 4
    assert wan._clip_frames([], None) == []


def png(image: Image.Image) -> bytes:
    out = io.BytesIO()
    image.save(out, "PNG")
    return out.getvalue()


def test_generate_asks_gemini_the_missing_photo(
    catalog: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    template_file(catalog)
    comfy = FakeComfy()
    comfy.key = (0, 255, 0)  # the photo Gemini gives is on green
    monkeypatch.setattr(wan, "_call", comfy)
    monkeypatch.setattr(wan.time, "sleep", lambda _: None)
    for path in (catalog / "photos").iterdir():
        path.unlink()
    # no key: nothing asked
    monkeypatch.delenv(wan.stills.KEY_ENV, raising=False)
    assert run(catalog, "generate", "--gemini") == 2
    assert "GEMINI_API_KEY" in capsys.readouterr().out
    monkeypatch.setenv(wan.stills.KEY_ENV, "key")

    good = png(fish(size=(600, 400), key=(0, 255, 0)))  # 4th attempt: green
    blank = png(Image.new("RGB", (600, 400), MAGENTA))  # no fish: refused
    answers: list[Any] = [
        wan.stills.GenerationError("busy"),
        b"not an image",
        blank,
        good,
    ]
    asked: list[str] = []

    def gemini(prompt: str, model: str, key: str, aspect: str) -> bytes:
        asked.append(prompt)
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(wan.stills, "generate", gemini)
    assert (
        run(catalog, "generate", "--only", "betta", "--gemini", "--attempts", "4") == 0
    )
    printed = capsys.readouterr().out
    assert "no photo (busy)" in printed and "not an image" in printed
    assert "photo refused" in printed and "betta: staged" in printed
    assert "Betta splendens" in asked[0] and "magenta" in asked[0]
    assert "green" in asked[1]
    assert comfy.queued[1]["4"]["inputs"]["text"] == "idle Betta splendens on green"
    names = sorted(p.name for p in (catalog / "photos").iterdir())
    assert names == ["betta-3-refused.png", "betta-4.png"]
    # a photo kept is not asked again; the catalog atlas stays degraded
    assert wan.reference_photo(catalog / "photos", "betta").name == "betta-4.png"
    assert json.loads((catalog / "fish" / "betta" / "species.json").read_text())[
        "degraded"
    ]

    # every attempt refused: the fish failed; a refused key stops the run
    answers[:] = [blank, blank]
    assert (
        run(catalog, "generate", "--only", "bare", "--gemini", "--attempts", "2") == 1
    )
    assert "bare: failed (no photo fits)" in capsys.readouterr().out
    answers[:] = [wan.stills.GenerationError("HTTP 403", fatal=True)]
    assert run(catalog, "generate", "--only", "bare", "--gemini") == 2
    assert "stopped on bare: HTTP 403" in capsys.readouterr().out


def test_gallery(
    catalog: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert run(catalog, "gallery") == 0
    assert "nothing staged" in capsys.readouterr().out
    template_file(catalog)
    monkeypatch.setattr(wan, "_call", FakeComfy())
    assert run(catalog, "generate", "--only", "betta") == 0
    opened: list[list[str]] = []
    with monkeypatch.context() as patch:
        patch.setattr(wan.subprocess, "Popen", lambda cmd, **_: opened.append(cmd))
        assert run(catalog, "gallery") == 0
    page = catalog / "clips" / "gallery.html"
    html = page.read_text(encoding="utf-8")
    assert "betta/gallery.webp" in html and "Betta splendens" in html
    assert "accepted" not in html and opened == [["xdg-open", str(page)]]
    clips = catalog / "clips" / "betta"
    assert not (clips / "gallery.html").exists()
    # only the new atlas: a single column, not the catalog one beside it
    frame = json.loads((clips / "build" / "betta" / "species.json").read_text())[
        "frame"
    ]
    assert Image.open(clips / "gallery.webp").width == frame[0] * wan.PREVIEW_ZOOM
    assert run(catalog, "accept", "betta") == 0
    assert run(catalog, "gallery", "--no-open") == 0
    assert "nothing staged" in capsys.readouterr().out
    assert run(catalog, "gallery", "--all", "--no-open") == 0
    assert "(accepted)" in page.read_text(encoding="utf-8")


def test_cut_out_by_the_model(
    catalog: Path,
    monkeypatch: pytest.MonkeyPatch,
    rembg: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    template_file(catalog)
    monkeypatch.setattr(wan, "_call", FakeComfy())
    assert run(catalog, "generate", "--only", "betta", "--matte-model", "u2net") == 0
    clips = catalog / "clips" / "betta"
    assert rembg == ["u2net@cpu"]  # one session for the three clips, on the CPU
    recipe = json.loads((clips / "betta.json").read_text(encoding="utf-8"))
    assert recipe["key"] == "none"
    assert {c["source"] for c in recipe["clips"].values()} == {
        "matte/swim",
        "matte/idle",
        "matte/turn",
    }
    frame = Image.open(clips / "matte" / "swim" / "0000.png")
    assert frame.mode == "RGBA" and frame.getpixel((100, 2))[3] == 0
    assert frame.getpixel((120, 50))[3] == 255
    # building again: fresh cut-outs and a hand-edited recipe are kept
    recipe["length"] = 150
    (clips / "betta.json").write_text(json.dumps(recipe), encoding="utf-8")
    assert run(catalog, "build", "--only", "betta") == 0
    assert rembg == ["u2net@cpu"]
    assert json.loads((clips / "betta.json").read_text())["length"] == 150
    # an older recipe cutting the videos by colour is written again
    recipe["key"] = "auto"
    (clips / "betta.json").write_text(json.dumps(recipe), encoding="utf-8")
    assert run(catalog, "build", "--only", "betta") == 0
    assert json.loads((clips / "betta.json").read_text())["length"] != 150
    # a clip made again is cut out again
    assert run(catalog, "redo", "betta", "--clips", "turn") == 0
    assert not (clips / "matte" / "turn").exists()
    assert (clips / "matte" / "swim").exists()
    capsys.readouterr()
    assert run(catalog, "generate", "--only", "betta", "--matte-gpu") == 0
    assert "cutting turn out" in capsys.readouterr().out
    assert rembg == ["u2net@cpu", wan.DEFAULT_MATTE + "@gpu"]


def test_unmix_and_missing_rembg(monkeypatch: pytest.MonkeyPatch) -> None:
    # half magenta, half white, at half alpha: white again once unmixed
    rgba = np.zeros((1, 3, 4), dtype=np.uint8)
    rgba[0, 0] = (255, 128, 255, 128)
    rgba[0, 1] = (10, 20, 30, 255)
    rgba[0, 2] = (9, 9, 9, 3)
    out = np.asarray(wan.unmix(Image.fromarray(rgba, "RGBA"), MAGENTA))
    assert tuple(out[0, 0, :3]) == (255, 255, 255) and out[0, 0, 3] == 128
    assert tuple(out[0, 1]) == (10, 20, 30, 255)
    assert out[0, 2, 3] == 0  # too faint: dropped
    monkeypatch.setitem(sys.modules, "rembg", None)
    with pytest.raises(SystemExit, match="rembg"):
        wan.matte_session("isnet-general-use")


# -- the invertebrates ---------------------------------------------------------------

#: The wording of a section that words its clips after the form of the animal
INVERT_SPEC = {
    **SPEC,
    "clips": {
        "swim": {"frames": 33, "last": "same", "prompt": "walk [SPECIE]: [MOTION]"},
        "idle": {"frames": 50, "last": "same", "prompt": "still [SPECIE]: [MOTION]"},
        "turn": {"frames": 49, "last": "mirror", "prompt": "turn [SPECIE]: [MOTION]"},
    },
    "motions": {
        "swim": {"shrimp": "it walks", "crab": "it sidles", "hermit": "it drags"},
        "idle": {"shrimp": "it grooms", "crab": "it picks", "hermit": "it rests"},
        "turn": {"shrimp": "it turns", "crab": "it pivots", "hermit": "it wheels"},
    },
}


@pytest.fixture
def inverts(catalog: Path) -> Path:
    """The catalog, with a shrimp and a crab to animate, and a snail that is not."""
    entries = {
        "lysmata_debelius": {"movement": "decor", "scientific": "Lysmata debelius"},
        "mithraculus_sculptus": {"movement": "decor", "scientific": "Mithraculus"},
        "nerita": {"movement": "glass", "scientific": "Nerita sp."},
    }
    for entry_id, extra in entries.items():
        folder = catalog / "invertebrates" / entry_id
        folder.mkdir(parents=True)
        desc = {
            "kind": "invertebrate",
            "role": ["cleaner"],
            "names": dict.fromkeys(validate.LANGUAGES, entry_id),
            "water_types": ["marine"],
            "size_cm": [2, 5],
            **extra,
        }
        (folder / "species.json").write_text(json.dumps(desc), encoding="utf-8")
    (catalog / "invert_prompts.json").write_text(
        json.dumps(INVERT_SPEC), encoding="utf-8"
    )
    photos = catalog / "photos"
    fish(size=(150, 100)).save(photos / "lysmata_debelius-1.png")
    return catalog


def run_invert(catalog: Path, *argv: str) -> int:
    return wan.main(
        [
            "--root",
            str(catalog),
            "--section",
            "invertebrates",
            "--prompts",
            str(catalog / "invert_prompts.json"),
            *argv,
        ]
    )


def test_the_real_invertebrate_prompts_file() -> None:
    spec = wan.load_spec(wan.PROMPT_FILES["invertebrates"])
    assert set(spec["clips"]) == set(wan.CLIP_NAMES)
    assert wan.animated_forms(spec) == {"shrimp", "crab", "hermit"}
    for clip in wan.CLIP_NAMES:
        text = wan.clip_prompt(spec, clip, "Lysmata debelius", "green", "shrimp")
        assert "[" not in text and "Lysmata debelius" in text
        assert "bright green background" in text
    # The fish keep their own wording, with no form to name
    fish_spec = wan.load_spec()
    assert wan.animated_forms(fish_spec) is None
    assert "[" not in wan.clip_prompt(fish_spec, "swim", "Betta", "magenta")


def test_the_motion_of_the_clip_follows_the_form(inverts: Path) -> None:
    prompt = wan.clip_prompt(INVERT_SPEC, "swim", "Lysmata", "magenta", "shrimp")
    assert prompt == "walk Lysmata: it walks"
    assert "it sidles" in wan.clip_prompt(
        INVERT_SPEC, "swim", "Mithraculus", "magenta", "crab"
    )
    assert "it wheels" in wan.clip_prompt(
        INVERT_SPEC, "turn", "Clibanarius", "magenta", "hermit"
    )
    # An animal nobody films is refused, with the ones that are
    with pytest.raises(SystemExit, match="no swim for a snail"):
        wan.clip_prompt(INVERT_SPEC, "swim", "Nerita sp.", "magenta", "snail")
    with pytest.raises(SystemExit, match="without a form"):
        wan.clip_prompt(INVERT_SPEC, "swim", "Nerita sp.", "magenta")


def test_the_form_of_an_entry(inverts: Path) -> None:
    assert wan.entry_form(inverts, "invertebrates", "lysmata_debelius") == "shrimp"
    assert wan.entry_form(inverts, "invertebrates", "mithraculus_sculptus") == "crab"
    assert wan.entry_form(inverts, "invertebrates", "nerita") == "snail"
    # A fish has no form to name
    assert wan.entry_form(inverts, "fish", "betta") is None


def test_only_the_filmed_forms_are_to_animate(inverts: Path) -> None:
    forms = wan.animated_forms(INVERT_SPEC)
    found = wan.targets(inverts, None, "invertebrates", forms)
    assert [f.name for f in found] == ["lysmata_debelius", "mithraculus_sculptus"]
    # Without the filter every entry is there, the snail included
    assert len(wan.targets(inverts, None, "invertebrates")) == 3
    # A snail is never asked for, even by name
    assert wan.targets(inverts, ["nerita"], "invertebrates", forms) == []


def test_the_commands_list_and_print_the_invertebrates(
    inverts: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_invert(inverts, "todo") == 0
    out = capsys.readouterr().out
    assert "lysmata_debelius" in out and "nerita" not in out
    assert "2 invertebrates without a real animation" in out
    assert run_invert(inverts, "prompt", "mithraculus_sculptus", "idle") == 0
    assert "it picks" in capsys.readouterr().out


def test_the_section_chooses_its_wording(inverts: Path) -> None:
    # No --prompts: the file of the section, and its motions
    assert wan.main(["--root", str(inverts), "--section", "invertebrates", "todo"]) == 0
    assert wan.main(["--root", str(inverts), "todo"]) == 0


def test_a_shrimp_goes_through_to_the_catalog(
    inverts: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    comfy = FakeComfy()
    monkeypatch.setattr(wan, "_call", comfy)
    workflow = template_file(inverts)
    capsys.readouterr()
    assert (
        run_invert(
            inverts,
            "generate",
            "--only",
            "lysmata_debelius",
            "--workflow",
            str(workflow),
            "--seed",
            "1",
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "lysmata_debelius: staged" in out
    # The clips were asked with the walk of a shrimp
    asked = [q["4"]["inputs"]["text"] for q in comfy.queued]
    assert any("it walks" in text for text in asked)
    assert all("[" not in text for text in asked)
    # The staged entry is an invertebrate, and it keeps what the catalog says
    staged = inverts / "clips" / "lysmata_debelius" / "build" / "lysmata_debelius"
    built = json.loads((staged / "species.json").read_text())
    assert built["kind"] == "invertebrate" and built["role"] == ["cleaner"]
    assert built["movement"] == "decor" and "swim" in built["clips"]
    assert built["atlas"] == {"1x": "lysmata_debelius.webp"}
    assert run_invert(inverts, "accept", "lysmata_debelius") == 0
    target = inverts / "invertebrates" / "lysmata_debelius"
    kept = json.loads((target / "species.json").read_text())
    assert kept["kind"] == "invertebrate" and kept["scientific"] == "Lysmata debelius"
    assert (target / "lysmata_debelius.webp").exists()
    assert validate.check_entry(target, "invertebrates") == []


def test_a_fish_is_never_staged_as_an_invertebrate(
    inverts: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The clips of both sections live side by side: the catalog sorts them out."""
    comfy = FakeComfy()
    monkeypatch.setattr(wan, "_call", comfy)
    workflow = template_file(inverts)
    # a fish and a shrimp, both with their clips made and staged
    assert run(inverts, "generate", "--only", "betta", "--workflow", str(workflow)) == 0
    assert (
        run_invert(
            inverts,
            "generate",
            "--only",
            "lysmata_debelius",
            "--workflow",
            str(workflow),
        )
        == 0
    )
    capsys.readouterr()

    # Each section sees only its own, and `gallery` with it
    assert [f.name for f in wan.work_entries(inverts / "clips", None, inverts)] == [
        "betta"
    ]
    staged = wan.staged_entries(inverts / "clips", None, inverts, "invertebrates")
    assert [f.name for f in staged] == ["lysmata_debelius"]
    assert run_invert(inverts, "gallery", "--no-open") == 0
    out = capsys.readouterr().out
    assert "lysmata_debelius: ready" in out and "betta" not in out

    # `build` of one section leaves the other alone: the fish keeps its kind
    assert run_invert(inverts, "build") == 0
    assert "1 staged" in capsys.readouterr().out
    fish_built = inverts / "clips" / "betta" / "build" / "betta" / "species.json"
    assert json.loads(fish_built.read_text())["kind"] == "fish"
    # Without a catalog to ask, every work folder is there
    assert len(wan.work_entries(inverts / "clips")) == 2


def test_preview_against_the_still_picture_being_replaced(
    inverts: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The catalog holds a still atlas until the animation lands: it is shown too."""
    invert_atlas = importlib.import_module("invert_atlas")
    entry = inverts / "invertebrates" / "lysmata_debelius"
    still = invert_atlas.write_invert_atlas(
        fish(size=(600, 400)), entry, {"type": "ai-generated", "date": "2026-11-24"}
    )
    assert list(still["clips"]) == ["still"]

    # The still alone plays: a single frame, and its mirror
    alone = wan.timeline(entry)
    assert alone and len({f.tobytes() for f in alone}) == 2

    comfy = FakeComfy()
    monkeypatch.setattr(wan, "_call", comfy)
    workflow = template_file(inverts)
    assert (
        run_invert(
            inverts,
            "generate",
            "--only",
            "lysmata_debelius",
            "--workflow",
            str(workflow),
        )
        == 0
    )
    capsys.readouterr()
    assert run_invert(inverts, "preview", "lysmata_debelius", "--no-open") == 0
    assert "preview.html" in capsys.readouterr().out
    assert (inverts / "clips" / "lysmata_debelius" / "preview.webp").exists()


def test_the_clip_is_told_the_colour_the_picture_really_has(tmp_path: Path) -> None:
    """An image model gives a washed out key: the prompt must still name it.

    Telling a video model the background is magenta while it is shown a green
    one leaves it reconciling the two instead of animating the animal.
    """
    folder = tmp_path / "clips" / "x"
    folder.mkdir(parents=True)
    # a sage green, as a model answers when asked for #00FF00
    fish(key=(148, 204, 141)).save(folder / "reference.png")
    name, colour = wan.reference_background(folder)
    assert name == "green" and colour == (148, 204, 141)
    # a dusty pink, for magenta
    fish(key=(225, 115, 165)).save(folder / "reference.png")
    assert wan.reference_background(folder)[0] == "magenta"
    # and a clean key is named as before
    fish(key=(0, 255, 0)).save(folder / "reference.png")
    assert wan.reference_background(folder)[0] == "green"
