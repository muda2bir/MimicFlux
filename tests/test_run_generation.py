"""Tests for the shared generation orchestrator and new helpers."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

import run_pipeline as rp


# --------------------------------------------------------------------------- #
# parse_prompts_text
# --------------------------------------------------------------------------- #
def test_parse_prompts_text_skips_blanks_and_comments():
    text = "# comment\n\n  a lion  \n   \nbike\n# trailing\n"
    assert rp.parse_prompts_text(text) == ["a lion", "bike"]


def test_parse_prompts_text_empty():
    assert rp.parse_prompts_text("") == []
    assert rp.parse_prompts_text("# only comments\n\n") == []


# --------------------------------------------------------------------------- #
# save_api_key_to_env
# --------------------------------------------------------------------------- #
def test_save_api_key_creates_env_file(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "load_dotenv", lambda *a, **k: None)
    env = tmp_path / ".env"
    rp.save_api_key_to_env("sk-test-123", env_path=env)
    assert env.exists()
    assert "OPENROUTER_API_KEY=sk-test-123" in env.read_text(encoding="utf-8")


def test_save_api_key_replaces_existing(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "load_dotenv", lambda *a, **k: None)
    env = tmp_path / ".env"
    env.write_text("OPENROUTER_API_KEY=old-key\nOTHER_VAR=keep\n", encoding="utf-8")
    rp.save_api_key_to_env("new-key", env_path=env)
    content = env.read_text(encoding="utf-8")
    assert "OPENROUTER_API_KEY=new-key" in content
    assert "old-key" not in content
    assert "OTHER_VAR=keep" in content


def test_save_api_key_appends_if_no_key_line(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "load_dotenv", lambda *a, **k: None)
    env = tmp_path / ".env"
    env.write_text("OTHER_VAR=keep\n", encoding="utf-8")
    rp.save_api_key_to_env("sk-test", env_path=env)
    content = env.read_text(encoding="utf-8")
    assert "OPENROUTER_API_KEY=sk-test" in content
    assert "OTHER_VAR=keep" in content


# --------------------------------------------------------------------------- #
# run_generation helpers
# --------------------------------------------------------------------------- #
def _seed_image(tmp_path: Path) -> Path:
    img = tmp_path / "my_style.jpg"
    img.write_bytes(b"style-bytes")
    return img


def _collect_events(gen) -> list[rp.GenEvent]:
    return list(gen)


def _options(**overrides) -> rp.GenerationOptions:
    defaults = {"cooldown": 0}
    defaults.update(overrides)
    return rp.GenerationOptions(**defaults)


# --------------------------------------------------------------------------- #
# run_generation: normal generate
# --------------------------------------------------------------------------- #
def test_run_generation_normal(monkeypatch, tmp_path):
    img = _seed_image(tmp_path)
    out = tmp_path / "outputs"
    monkeypatch.setattr(rp, "_vision_call", lambda *a, **k: "dark style")
    monkeypatch.setattr(
        rp,
        "generate_image",
        lambda *a, **k: ({"b64_json": base64.b64encode(b"png-bytes").decode()}, "b64"),
    )

    events = _collect_events(
        rp.run_generation(
            "key", [img], ["a lion", "a bike"], out, _options(),
        )
    )

    kinds = [e.kind for e in events]
    assert kinds == [
        "style_start", "style_done", "queue",
        "gen_start", "gen_done",
        "gen_start", "gen_done",
        "done",
    ]
    assert events[1].payload["style"] == "dark style"
    assert events[2].payload["total"] == 2
    assert events[4].payload["filename"] == "01_lion.png"
    assert events[6].payload["filename"] == "02_bike.png"
    assert (out / "01_lion.png").read_bytes() == b"png-bytes"
    assert (out / "02_bike.png").read_bytes() == b"png-bytes"

    manifest = rp.load_manifest(out)
    assert manifest[1].subject == "a lion"
    assert manifest[2].subject == "a bike"


# --------------------------------------------------------------------------- #
# run_generation: skip
# --------------------------------------------------------------------------- #
def test_run_generation_skip(monkeypatch, tmp_path):
    img = _seed_image(tmp_path)
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "01_lion.png").write_bytes(b"old")
    rp.write_manifest(
        out,
        {1: rp.ManifestEntry(1, "a lion", "x", "p", "m", "t", "01_lion.png", "b64")},
    )

    monkeypatch.setattr(rp, "_vision_call", lambda *a, **k: "dark style")
    monkeypatch.setattr(
        rp,
        "generate_image",
        lambda *a, **k: ({"b64_json": base64.b64encode(b"new").decode()}, "b64"),
    )

    events = _collect_events(
        rp.run_generation("key", [img], ["a lion", "a bike"], out, _options())
    )

    kinds = [e.kind for e in events]
    assert kinds == [
        "style_start", "style_done", "queue",
        "gen_start", "skip",
        "gen_start", "gen_done",
        "done",
    ]
    assert (out / "01_lion.png").read_bytes() == b"old"
    assert (out / "02_bike.png").read_bytes() == b"new"

    manifest = rp.load_manifest(out)
    assert manifest[1].skipped is True
    assert manifest[2].skipped is False


# --------------------------------------------------------------------------- #
# run_generation: error
# --------------------------------------------------------------------------- #
def test_run_generation_error(monkeypatch, tmp_path):
    img = _seed_image(tmp_path)
    out = tmp_path / "outputs"

    monkeypatch.setattr(rp, "_vision_call", lambda *a, **k: "dark style")
    monkeypatch.setattr(
        rp,
        "generate_image",
        lambda *a, **k: (_ for _ in ()).throw(rp.PermanentAPIError("boom")),
    )

    events = _collect_events(
        rp.run_generation("key", [img], ["a lion"], out, _options())
    )

    kinds = [e.kind for e in events]
    assert kinds == [
        "style_start", "style_done", "queue",
        "gen_start", "gen_error",
        "done",
    ]
    assert events[4].payload["error"] == "boom"

    manifest = rp.load_manifest(out)
    assert manifest[1].source == "error"
    assert manifest[1].error == "boom"


# --------------------------------------------------------------------------- #
# run_generation: style error
# --------------------------------------------------------------------------- #
def test_run_generation_style_error(monkeypatch, tmp_path):
    img = _seed_image(tmp_path)
    out = tmp_path / "outputs"

    def fail_vision(*a, **k):
        raise RuntimeError("vision down")

    monkeypatch.setattr(rp, "_vision_call", fail_vision)

    events = _collect_events(
        rp.run_generation("key", [img], ["a lion"], out, _options())
    )

    kinds = [e.kind for e in events]
    assert kinds == ["style_start", "style_error"]
    assert "vision down" in events[1].payload["error"]


# --------------------------------------------------------------------------- #
# run_generation: dry-run
# --------------------------------------------------------------------------- #
def test_run_generation_dry_run(monkeypatch, tmp_path):
    img = _seed_image(tmp_path)
    out = tmp_path / "outputs"

    monkeypatch.setattr(rp, "_vision_call", lambda *a, **k: "dark style")

    def _fail_if_called(*a, **k):
        pytest.fail("image API must not be called in dry-run")

    monkeypatch.setattr(rp, "generate_image", _fail_if_called)

    events = _collect_events(
        rp.run_generation(
            "key", [img], ["a lion", "a bike"], out, _options(dry_run=True),
        )
    )

    kinds = [e.kind for e in events]
    assert kinds == [
        "style_start", "style_done", "queue",
        "dry_run", "dry_run",
        "done",
    ]
    assert events[3].payload["merged_prompt"] == "a lion. dark style. Highly detailed, sharp focus, professional quality."
    assert events[4].payload["merged_prompt"] == "a bike. dark style. Highly detailed, sharp focus, professional quality."
    assert not any(out.iterdir())


# --------------------------------------------------------------------------- #
# run_generation: force regenerates
# --------------------------------------------------------------------------- #
def test_run_generation_force(monkeypatch, tmp_path):
    img = _seed_image(tmp_path)
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "01_lion.png").write_bytes(b"old")
    rp.write_manifest(
        out,
        {1: rp.ManifestEntry(1, "a lion", "x", "p", "m", "t", "01_lion.png", "b64")},
    )

    monkeypatch.setattr(rp, "_vision_call", lambda *a, **k: "dark style")
    monkeypatch.setattr(
        rp,
        "generate_image",
        lambda *a, **k: ({"b64_json": base64.b64encode(b"new").decode()}, "b64"),
    )

    events = _collect_events(
        rp.run_generation("key", [img], ["a lion"], out, _options(force=True))
    )

    kinds = [e.kind for e in events]
    assert "skip" not in kinds
    assert "gen_done" in kinds
    assert (out / "01_lion.png").read_bytes() == b"new"


# --------------------------------------------------------------------------- #
# run_generation: legacy names
# --------------------------------------------------------------------------- #
def test_run_generation_legacy_names(monkeypatch, tmp_path):
    img = _seed_image(tmp_path)
    out = tmp_path / "outputs"

    monkeypatch.setattr(rp, "_vision_call", lambda *a, **k: "dark style")
    monkeypatch.setattr(
        rp,
        "generate_image",
        lambda *a, **k: ({"b64_json": base64.b64encode(b"x").decode()}, "b64"),
    )

    events = _collect_events(
        rp.run_generation("key", [img], ["a lion"], out, _options(legacy_names=True))
    )

    assert events[4].payload["filename"] == "generation_1.png"
    assert (out / "generation_1.png").exists()


# --------------------------------------------------------------------------- #
# run_generation: img2img suppression and ref_images passthrough
# --------------------------------------------------------------------------- #
def test_run_generation_no_img2img_suppresses_ref_images(monkeypatch, tmp_path):
    img = _seed_image(tmp_path)
    out = tmp_path / "outputs"

    monkeypatch.setattr(rp, "_vision_call", lambda *a, **k: "dark style")

    captured = {}

    def fake_generate(api_key, model, prompt, fmt, **k):
        captured["ref_images"] = k.get("ref_images")
        return ({"b64_json": base64.b64encode(b"x").decode()}, "b64")

    monkeypatch.setattr(rp, "generate_image", fake_generate)

    list(
        rp.run_generation(
            "key", [img], ["a lion"], out, _options(no_img2img=True)
        )
    )
    assert captured["ref_images"] is None


def test_run_generation_img2img_passes_ref_images(monkeypatch, tmp_path):
    img = _seed_image(tmp_path)
    out = tmp_path / "outputs"

    monkeypatch.setattr(rp, "_vision_call", lambda *a, **k: "dark style")

    captured = {}

    def fake_generate(api_key, model, prompt, fmt, **k):
        captured["ref_images"] = k.get("ref_images")
        return ({"b64_json": base64.b64encode(b"x").decode()}, "b64")

    monkeypatch.setattr(rp, "generate_image", fake_generate)

    list(
        rp.run_generation(
            "key", [img], ["a lion"], out, _options(no_img2img=False)
        )
    )
    assert captured["ref_images"] is not None
    assert len(captured["ref_images"]) == 1


def test_run_generation_passes_steps_guidance_seed(monkeypatch, tmp_path):
    img = _seed_image(tmp_path)
    out = tmp_path / "outputs"

    monkeypatch.setattr(rp, "_vision_call", lambda *a, **k: "dark style")

    captured = {}

    def fake_generate(api_key, model, prompt, fmt, **k):
        captured.update(k)
        return ({"b64_json": base64.b64encode(b"x").decode()}, "b64")

    monkeypatch.setattr(rp, "generate_image", fake_generate)

    list(
        rp.run_generation(
            "key", [img], ["a lion"], out,
            _options(steps=50, guidance=4.0, seed=123),
        )
    )
    assert captured["steps"] == 50
    assert captured["guidance"] == 4.0
    assert captured["seed"] == 123
