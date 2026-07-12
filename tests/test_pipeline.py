"""Unit tests for run_pipeline.

These tests never touch the network: HTTP calls are mocked and the vision
step is stubbed out so ``main()`` can be exercised end-to-end without an API key.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

import run_pipeline as rp


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
class FakeResponse:
    def __init__(self, json_data=None, status_code=200, text="", raw_bytes=b""):
        self._json = json_data
        self.status_code = status_code
        self.text = text
        self.content = raw_bytes

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


@pytest.fixture(autouse=True)
def _isolate_time_sleep(monkeypatch):
    """Avoid real waits during retries/cooldowns."""
    monkeypatch.setattr(rp.time, "sleep", lambda *_args, **_kw: None)


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def test_slugify_basic():
    assert rp.slugify("A Majestic Lion!!!") == "a-majestic-lion"


def test_slugify_truncates():
    long = "a" * 120
    assert len(rp.slugify(long)) == 50


def test_slugify_empty_fallback():
    assert rp.slugify("!!!") == "prompt"


def test_build_prompt_merges_subject_and_style():
    assert (
        rp.build_prompt("a lion", "dark mood")
        == "a lion. dark mood. Highly detailed, sharp focus, professional quality."
    )


def test_extract_subject_slug_single_word():
    # Leading descriptors stripped, subject noun captured, "with the..." stops the run.
    s = "A photorealistic, perfectly set faceted Amethyst with the ring clearly visible."
    assert rp.extract_subject_slug(s) == "amethyst"


def test_extract_subject_slug_two_word_subject():
    s = "A photorealistic, perfectly set faceted Lapis lazuli with the ring clearly visible."
    assert rp.extract_subject_slug(s) == "lapis-lazuli"


def test_extract_subject_slug_stops_at_filler():
    # "on" ends the leading noun phrase after "lion".
    assert rp.extract_subject_slug("A majestic lion on a neon throne") == "lion"


def test_extract_subject_slug_all_filler_falls_back():
    assert rp.extract_subject_slug("the a of in photorealistic") == "prompt"


def test_extract_subject_slug_caps_max_words():
    s = "A vintage brass ornate engraved pocket watch on a chain"
    # Caps at 4 tokens of the leading run.
    assert rp.extract_subject_slug(s, max_words=4).split("-") == ["vintage", "brass", "ornate", "engraved"]


def test_output_filename_default_descriptive():
    assert rp.output_filename(1, "A photorealistic faceted Amethyst with a ring") == "01_amethyst.png"


def test_output_filename_zero_padding_widens():
    # pad_width drives zero-fill alignment.
    assert rp.output_filename(5, "Lapis lazuli on a chain", pad_width=3) == "005_lapis-lazuli.png"


def test_output_filename_legacy():
    assert rp.output_filename(3, "Amethyst", legacy=True) == "generation_3.png"


def test_encode_image_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        rp.encode_image(tmp_path / "nope.png")


def test_encode_image_returns_base64(tmp_path):
    img = tmp_path / "img.bin"
    img.write_bytes(b"hello")
    encoded = rp.encode_image(img)
    assert base64.b64decode(encoded) == b"hello"


# --------------------------------------------------------------------------- #
# Prompts file
# --------------------------------------------------------------------------- #
def test_load_prompts_missing_creates_file_and_returns_none(tmp_path):
    path = tmp_path / "prompts.txt"
    result = rp.load_prompts(path)
    assert result is None
    assert path.exists()
    assert "landscape" in path.read_text(encoding="utf-8")


def test_load_prompts_skips_blanks_and_comments(tmp_path):
    path = tmp_path / "prompts.txt"
    path.write_text(
        "# a comment\n\n  a lion  \n   \nbike\n# trailing comment\n",
        encoding="utf-8",
    )
    assert rp.load_prompts(path) == ["a lion", "bike"]


# --------------------------------------------------------------------------- #
# Style cache
# --------------------------------------------------------------------------- #
def test_style_cache_roundtrip(tmp_path):
    img = tmp_path / "ref.jpeg"
    img.write_bytes(b"imgdata")
    cache = rp.cache_path_for(img)

    assert rp.style_cache_valid(cache, [img], "model-a") is False
    rp.save_cached_style(cache, [img], "model-a", "dark moody palette")
    assert rp.style_cache_valid(cache, [img], "model-a") is True
    assert rp.load_cached_style(cache) == "dark moody palette"


def test_style_cache_invalidates_on_model_change(tmp_path):
    img = tmp_path / "ref.jpeg"
    img.write_bytes(b"imgdata")
    cache = rp.cache_path_for(img)
    rp.save_cached_style(cache, [img], "model-a", "style")
    assert rp.style_cache_valid(cache, [img], "model-b") is False


def test_style_cache_invalidates_on_content_change(tmp_path):
    img = tmp_path / "ref.jpeg"
    img.write_bytes(b"imgdata")
    cache = rp.cache_path_for(img)
    rp.save_cached_style(cache, [img], "model-a", "style")
    # Change the file content (size + mtime change).
    img.write_bytes(b"different")
    assert rp.style_cache_valid(cache, [img], "model-a") is False


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
def test_manifest_roundtrip(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    manifest = {
        1: rp.ManifestEntry(1, "lion", "dark", "lion, dark", "m", "t1", "generation_1.png", "b64"),
        2: rp.ManifestEntry(2, "bike", "dark", "bike, dark", "m", "t2", "generation_2.png", "url", error="boom"),
    }
    rp.write_manifest(out, manifest)

    loaded = rp.load_manifest(out)
    assert set(loaded.keys()) == {1, 2}
    assert loaded[1].subject == "lion"
    assert loaded[1].source == "b64"
    assert loaded[2].error == "boom"


def test_load_manifest_missing_returns_empty(tmp_path):
    assert rp.load_manifest(tmp_path / "nope") == {}


def test_load_manifest_tolerates_garbage(tmp_path):
    path = tmp_path / rp.MANIFEST_NAME
    path.write_text("not json at all", encoding="utf-8")
    assert rp.load_manifest(tmp_path) == {}


def test_load_manifest_accepts_list_shape(tmp_path):
    path = tmp_path / rp.MANIFEST_NAME
    path.write_text(
        json.dumps(
            [
                {"index": 1, "subject": "lion", "source": "b64"},
                {"subject": "bike", "source": "url"},
            ]
        ),
        encoding="utf-8",
    )
    loaded = rp.load_manifest(tmp_path)
    assert loaded[1].subject == "lion"
    assert loaded[2].subject == "bike"


# --------------------------------------------------------------------------- #
# should_skip
# --------------------------------------------------------------------------- #
def test_should_skip_logic(tmp_path):
    target = tmp_path / "generation_1.png"
    target.write_bytes(b"x")
    entry = rp.ManifestEntry(1, "lion", "s", "p", "m", "t", target.name, "b64")
    assert rp.should_skip(target, "lion", entry, force=False) is True
    assert rp.should_skip(target, "lion", entry, force=True) is False
    assert rp.should_skip(target, "bike", entry, force=False) is False
    assert rp.should_skip(target, "lion", None, force=False) is False


# --------------------------------------------------------------------------- #
# _generate_image_once (HTTP handling, mocked)
# --------------------------------------------------------------------------- #
def test_generate_image_once_b64(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return FakeResponse(json_data={"data": [{"b64_json": "aGVsbG8="}]}, status_code=200)

    monkeypatch.setattr(rp.requests, "post", fake_post)
    img_item, source = rp._generate_image_once("key", "model", "prompt", "png")
    assert source == "b64"
    assert img_item == {"b64_json": "aGVsbG8="}
    assert captured["url"] == rp.OPENROUTER_IMAGES_URL
    assert captured["json"]["prompt"] == "prompt"


def test_generate_image_once_url_source(monkeypatch):
    monkeypatch.setattr(
        rp.requests,
        "post",
        lambda *a, **k: FakeResponse(json_data={"data": [{"url": "http://x/y.png"}]}),
    )
    _, source = rp._generate_image_once("key", "model", "prompt", "png")
    assert source == "url"


@pytest.mark.parametrize("status", [429, 500, 502, 503])
def test_generate_image_once_transient_errors(monkeypatch, status):
    monkeypatch.setattr(
        rp.requests,
        "post",
        lambda *a, **k: FakeResponse(json_data={}, status_code=status, text="err"),
    )
    with pytest.raises(rp.TransientAPIError):
        rp._generate_image_once("key", "model", "prompt", "png")


def test_generate_image_once_permanent_error(monkeypatch):
    monkeypatch.setattr(
        rp.requests,
        "post",
        lambda *a, **k: FakeResponse(json_data={}, status_code=400, text="bad request"),
    )
    with pytest.raises(rp.PermanentAPIError):
        rp._generate_image_once("key", "model", "prompt", "png")


def test_generate_image_once_non_json(monkeypatch):
    monkeypatch.setattr(
        rp.requests,
        "post",
        lambda *a, **k: FakeResponse(json_data=None, status_code=200, text="<html>"),
    )
    with pytest.raises(rp.PermanentAPIError):
        rp._generate_image_once("key", "model", "prompt", "png")


def test_generate_image_once_empty_data(monkeypatch):
    monkeypatch.setattr(
        rp.requests,
        "post",
        lambda *a, **k: FakeResponse(json_data={"data": []}, status_code=200),
    )
    with pytest.raises(rp.PermanentAPIError):
        rp._generate_image_once("key", "model", "prompt", "png")


def test_generate_image_once_with_input_references(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["json"] = json
        return FakeResponse(json_data={"data": [{"b64_json": "aGVsbG8="}]})

    monkeypatch.setattr(rp.requests, "post", fake_post)
    ref_images = [("dGVzdA==", "image/jpeg")]
    rp._generate_image_once(
        "key", "model", "prompt", "png", ref_images=ref_images,
    )
    payload = captured["json"]
    assert "input_references" in payload
    assert len(payload["input_references"]) == 1
    assert payload["input_references"][0]["type"] == "image_url"
    assert "data:image/jpeg;base64,dGVzdA==" in payload["input_references"][0]["image_url"]["url"]


def test_generate_image_once_with_steps_and_guidance(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["json"] = json
        return FakeResponse(json_data={"data": [{"b64_json": "aGVsbG8="}]})

    monkeypatch.setattr(rp.requests, "post", fake_post)
    rp._generate_image_once(
        "key", "model", "prompt", "png", steps=40, guidance=3.5,
    )
    payload = captured["json"]
    assert payload["provider"]["options"]["black-forest-labs"]["steps"] == 40
    assert payload["provider"]["options"]["black-forest-labs"]["guidance"] == 3.5


def test_generate_image_once_with_seed(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["json"] = json
        return FakeResponse(json_data={"data": [{"b64_json": "aGVsbG8="}]})

    monkeypatch.setattr(rp.requests, "post", fake_post)
    rp._generate_image_once("key", "model", "prompt", "png", seed=42)
    assert captured["json"]["seed"] == 42


def test_generate_image_once_without_optional_params(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["json"] = json
        return FakeResponse(json_data={"data": [{"b64_json": "aGVsbG8="}]})

    monkeypatch.setattr(rp.requests, "post", fake_post)
    rp._generate_image_once("key", "model", "prompt", "png")
    payload = captured["json"]
    assert "input_references" not in payload
    assert "provider" not in payload
    assert "seed" not in payload


def test_generation_options_validates_steps():
    with pytest.raises(ValueError, match="steps"):
        rp.GenerationOptions(steps=0)
    with pytest.raises(ValueError, match="steps"):
        rp.GenerationOptions(steps=101)


def test_generation_options_validates_guidance():
    with pytest.raises(ValueError, match="guidance"):
        rp.GenerationOptions(guidance=-1)
    with pytest.raises(ValueError, match="guidance"):
        rp.GenerationOptions(guidance=21)


def test_generation_options_validates_seed():
    with pytest.raises(ValueError, match="seed"):
        rp.GenerationOptions(seed=-1)


# --------------------------------------------------------------------------- #
# ModelCapabilities defaults
# --------------------------------------------------------------------------- #
def test_model_capabilities_defaults_are_optimistic():
    caps = rp.ModelCapabilities()
    assert caps.supports_img2img is True
    assert caps.provider_slug == "black-forest-labs"
    assert caps.allowed_passthrough is None


# --------------------------------------------------------------------------- #
# fetch_model_capabilities
# --------------------------------------------------------------------------- #
def test_fetch_model_capabilities_supported(monkeypatch):
    endpoint_data = {
        "id": "black-forest-labs/flux.2-pro",
        "endpoints": [
            {
                "provider_name": "Black Forest Labs",
                "provider_slug": "black-forest-labs",
                "supported_parameters": {
                    "output_format": {"type": "enum", "values": ["png", "jpeg"]},
                    "input_references": {"type": "range", "min": 0, "max": 8},
                    "seed": {"type": "boolean"},
                },
                "allowed_passthrough_parameters": ["steps", "guidance", "safety_tolerance"],
            }
        ],
    }
    monkeypatch.setattr(
        rp.requests,
        "get",
        lambda *a, **k: FakeResponse(json_data=endpoint_data, status_code=200),
    )
    caps = rp.fetch_model_capabilities("key", "black-forest-labs/flux.2-pro")
    assert caps.supports_img2img is True
    assert caps.provider_slug == "black-forest-labs"
    assert "steps" in caps.allowed_passthrough
    assert "guidance" in caps.allowed_passthrough


def test_fetch_model_capabilities_unsupported_img2img(monkeypatch):
    endpoint_data = {
        "id": "some/model",
        "endpoints": [
            {
                "provider_name": "Some Provider",
                "provider_slug": "some-provider",
                "supported_parameters": {
                    "output_format": {"type": "enum", "values": ["png", "jpeg"]},
                    "seed": {"type": "boolean"},
                },
                "allowed_passthrough_parameters": [],
            }
        ],
    }
    monkeypatch.setattr(
        rp.requests,
        "get",
        lambda *a, **k: FakeResponse(json_data=endpoint_data, status_code=200),
    )
    caps = rp.fetch_model_capabilities("key", "some/model")
    assert caps.supports_img2img is False
    assert caps.provider_slug == "some-provider"
    assert caps.allowed_passthrough == []


def test_fetch_model_capabilities_http_error_returns_defaults(monkeypatch):
    monkeypatch.setattr(
        rp.requests,
        "get",
        lambda *a, **k: FakeResponse(status_code=404, text="not found"),
    )
    caps = rp.fetch_model_capabilities("key", "nonexistent/model")
    assert caps.supports_img2img is True
    assert caps.provider_slug == "black-forest-labs"
    assert caps.allowed_passthrough is None


def test_fetch_model_capabilities_network_error_returns_defaults(monkeypatch):
    def raise_error(*a, **k):
        raise rp.requests.RequestException("network down")

    monkeypatch.setattr(rp.requests, "get", raise_error)
    caps = rp.fetch_model_capabilities("key", "any/model")
    assert caps.supports_img2img is True
    assert caps.provider_slug == "black-forest-labs"
    assert caps.allowed_passthrough is None


def test_fetch_model_capabilities_empty_endpoints_returns_defaults(monkeypatch):
    monkeypatch.setattr(
        rp.requests,
        "get",
        lambda *a, **k: FakeResponse(json_data={"endpoints": []}, status_code=200),
    )
    caps = rp.fetch_model_capabilities("key", "some/model")
    assert caps.supports_img2img is True
    assert caps.provider_slug == "black-forest-labs"


# --------------------------------------------------------------------------- #
# _build_image_payload with caps
# --------------------------------------------------------------------------- #
def test_build_image_payload_caps_suppresses_input_references():
    caps = rp.ModelCapabilities(
        supports_img2img=False,
        provider_slug="some-provider",
        allowed_passthrough=[],
    )
    payload = rp._build_image_payload(
        "model", "prompt", "png",
        ref_images=[("dGVzdA==", "image/jpeg")],
        caps=caps,
    )
    assert "input_references" not in payload


def test_build_image_payload_caps_allows_input_references():
    caps = rp.ModelCapabilities(
        supports_img2img=True,
        provider_slug="black-forest-labs",
        allowed_passthrough=["steps", "guidance"],
    )
    payload = rp._build_image_payload(
        "model", "prompt", "png",
        ref_images=[("dGVzdA==", "image/jpeg")],
        caps=caps,
    )
    assert "input_references" in payload
    assert len(payload["input_references"]) == 1


def test_build_image_payload_caps_uses_provider_slug():
    caps = rp.ModelCapabilities(
        supports_img2img=True,
        provider_slug="openai",
        allowed_passthrough=["steps"],
    )
    payload = rp._build_image_payload(
        "model", "prompt", "png",
        steps=30,
        caps=caps,
    )
    assert "openai" in payload["provider"]["options"]
    assert "black-forest-labs" not in payload["provider"]["options"]
    assert payload["provider"]["options"]["openai"]["steps"] == 30


def test_build_image_payload_caps_filters_unsupported_passthrough():
    caps = rp.ModelCapabilities(
        supports_img2img=True,
        provider_slug="openai",
        allowed_passthrough=["moderation"],
    )
    payload = rp._build_image_payload(
        "model", "prompt", "png",
        steps=40,
        guidance=3.5,
        caps=caps,
    )
    # steps and guidance are NOT in allowed_passthrough, so neither should appear
    assert "provider" not in payload


def test_build_image_payload_caps_none_preserves_legacy_behavior():
    payload = rp._build_image_payload(
        "model", "prompt", "png",
        ref_images=[("dGVzdA==", "image/jpeg")],
        steps=40,
        guidance=3.5,
        seed=42,
    )
    assert "input_references" in payload
    assert payload["provider"]["options"]["black-forest-labs"]["steps"] == 40
    assert payload["provider"]["options"]["black-forest-labs"]["guidance"] == 3.5
    assert payload["seed"] == 42


def test_build_image_payload_caps_allowed_passthrough_none_keeps_steps():
    caps = rp.ModelCapabilities(
        supports_img2img=True,
        provider_slug="black-forest-labs",
        allowed_passthrough=None,
    )
    payload = rp._build_image_payload(
        "model", "prompt", "png",
        steps=40,
        guidance=3.5,
        caps=caps,
    )
    assert payload["provider"]["options"]["black-forest-labs"]["steps"] == 40
    assert payload["provider"]["options"]["black-forest-labs"]["guidance"] == 3.5


# --------------------------------------------------------------------------- #
# save_image
# --------------------------------------------------------------------------- #
def test_save_image_b64(tmp_path):
    out = tmp_path / "out.png"
    rp.save_image({"b64_json": base64.b64encode(b"pixels").decode()}, out)
    assert out.read_bytes() == b"pixels"


def test_save_image_url(monkeypatch, tmp_path):
    out = tmp_path / "out.png"
    monkeypatch.setattr(
        rp.requests,
        "get",
        lambda *a, **k: FakeResponse(raw_bytes=b"from-url", status_code=200),
    )
    rp.save_image({"url": "http://x/y.png"}, out)
    assert out.read_bytes() == b"from-url"


def test_save_image_neither_raises(tmp_path):
    with pytest.raises(rp.PermanentAPIError):
        rp.save_image({"foo": "bar"}, tmp_path / "x.png")


# --------------------------------------------------------------------------- #
# main() end-to-end (no network)
# --------------------------------------------------------------------------- #
def _seed_inputs(tmp_path: Path) -> tuple[Path, Path]:
    img = tmp_path / "my_style.jpeg"
    img.write_bytes(b"style-bytes")
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("a lion\na bike\n", encoding="utf-8")
    return img, prompts


def test_main_dry_run(monkeypatch, tmp_path, capsys):
    img, prompts = _seed_inputs(tmp_path)
    out = tmp_path / "outputs"

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    # Stub the vision call so no SDK/network is used.
    monkeypatch.setattr(rp, "_vision_call", lambda *a, **k: "dark moody palette")
    # Safety: ensure generate_image is never called.
    def _fail_if_called(*a, **k):
        pytest.fail("image API must not be called in dry-run")
    monkeypatch.setattr(rp, "generate_image", _fail_if_called)

    rc = rp.main(
        [
            "--image", str(img),
            "--prompts", str(prompts),
            "--output", str(out),
            "--dry-run",
            "--no-cache",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert "Dry-run mode" in captured.out
    assert "a lion. dark moody palette" in captured.out
    assert "a bike. dark moody palette" in captured.out
    assert not out.exists() or not any(out.iterdir())


def test_main_missing_api_key_returns_2(monkeypatch, tmp_path):
    # load_dotenv() searches from the script's directory and would reload the
    # project .env; neutralize it so the "no key" branch is actually reached.
    monkeypatch.setattr(rp, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    rc = rp.main(["--dry-run"])
    assert rc == 2


def test_main_creates_starter_prompts_when_missing(monkeypatch, tmp_path):
    img = tmp_path / "my_style.jpeg"
    img.write_bytes(b"x")
    out = tmp_path / "outputs"
    prompts = tmp_path / "prompts.txt"

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(rp, "_vision_call", lambda *a, **k: "style")

    rc = rp.main(
        [
            "--image", str(img),
            "--prompts", str(prompts),
            "--output", str(out),
            "--no-cache",
        ]
    )

    assert rc == 0
    assert prompts.exists()
    assert "landscape" in prompts.read_text(encoding="utf-8")


def test_main_generates_and_writes_manifest(monkeypatch, tmp_path):
    img, prompts = _seed_inputs(tmp_path)
    out = tmp_path / "outputs"

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(rp, "_vision_call", lambda *a, **k: "dark style")

    calls = {"generate": 0}

    def fake_generate_image(api_key, model, prompt, fmt, **k):
        calls["generate"] += 1
        return ({"b64_json": base64.b64encode(b"png-bytes").decode()}, "b64")

    monkeypatch.setattr(rp, "generate_image", fake_generate_image)

    rc = rp.main(
        [
            "--image", str(img),
            "--prompts", str(prompts),
            "--output", str(out),
            "--no-cache",
            "--cooldown", "0",
        ]
    )

    assert rc == 0
    assert calls["generate"] == 2
    assert (out / "01_lion.png").read_bytes() == b"png-bytes"
    assert (out / "02_bike.png").read_bytes() == b"png-bytes"

    manifest = rp.load_manifest(out)
    assert manifest[1].subject == "a lion"
    assert manifest[1].filename == "01_lion.png"
    assert manifest[2].subject == "a bike"
    assert manifest[1].source == "b64"
    assert manifest[1].model == rp.DEFAULT_IMAGE_MODEL


def test_main_skips_existing_matching(monkeypatch, tmp_path):
    img, prompts = _seed_inputs(tmp_path)
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "01_lion.png").write_bytes(b"old")
    # Seed manifest so skip logic matches subject on index 1.
    rp.write_manifest(
        out,
        {1: rp.ManifestEntry(1, "a lion", "x", "p", "m", "t", "01_lion.png", "b64")},
    )

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(rp, "_vision_call", lambda *a, **k: "dark style")

    def fake_generate_image(*a, **k):
        return ({"b64_json": base64.b64encode(b"new").decode()}, "b64")

    monkeypatch.setattr(rp, "generate_image", fake_generate_image)

    rc = rp.main(
        [
            "--image", str(img),
            "--prompts", str(prompts),
            "--output", str(out),
            "--no-cache",
            "--cooldown", "0",
        ]
    )

    assert rc == 0
    # Index 1 skipped (old bytes preserved), index 2 generated.
    assert (out / "01_lion.png").read_bytes() == b"old"
    assert (out / "02_bike.png").read_bytes() == b"new"

    manifest = rp.load_manifest(out)
    assert manifest[1].skipped is True
    assert manifest[2].skipped is False


def test_main_force_regenerates(monkeypatch, tmp_path):
    img, prompts = _seed_inputs(tmp_path)
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "01_lion.png").write_bytes(b"old")
    rp.write_manifest(
        out,
        {1: rp.ManifestEntry(1, "a lion", "x", "p", "m", "t", "01_lion.png", "b64")},
    )

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(rp, "_vision_call", lambda *a, **k: "dark style")
    monkeypatch.setattr(
        rp,
        "generate_image",
        lambda *a, **k: ({"b64_json": base64.b64encode(b"new").decode()}, "b64"),
    )

    rc = rp.main(
        [
            "--image", str(img),
            "--prompts", str(prompts),
            "--output", str(out),
            "--no-cache",
            "--force",
            "--cooldown", "0",
        ]
    )

    assert rc == 0
    assert (out / "01_lion.png").read_bytes() == b"new"


def test_main_legacy_names(monkeypatch, tmp_path):
    img, prompts = _seed_inputs(tmp_path)
    out = tmp_path / "outputs"

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(rp, "_vision_call", lambda *a, **k: "dark style")
    monkeypatch.setattr(
        rp,
        "generate_image",
        lambda *a, **k: ({"b64_json": base64.b64encode(b"new").decode()}, "b64"),
    )

    rc = rp.main(
        [
            "--image", str(img),
            "--prompts", str(prompts),
            "--output", str(out),
            "--no-cache",
            "--legacy-names",
            "--cooldown", "0",
        ]
    )

    assert rc == 0
    assert (out / "generation_1.png").read_bytes() == b"new"
    assert (out / "generation_2.png").read_bytes() == b"new"


def test_main_uses_cached_style(monkeypatch, tmp_path):
    img, prompts = _seed_inputs(tmp_path)
    out = tmp_path / "outputs"

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    # Pre-seed a valid cache.
    cache = rp.cache_path_for(img)
    rp.save_cached_style(cache, [img], rp.DEFAULT_VISION_MODEL, "cached style")

    vision_calls = {"n": 0}

    def fail_if_called(*a, **k):
        vision_calls["n"] += 1
        raise AssertionError("vision must not be called when cache is valid")

    monkeypatch.setattr(rp, "_vision_call", fail_if_called)
    monkeypatch.setattr(
        rp,
        "generate_image",
        lambda *a, **k: ({"b64_json": base64.b64encode(b"new").decode()}, "b64"),
    )

    rc = rp.main(
        [
            "--image", str(img),
            "--prompts", str(prompts),
            "--output", str(out),
            "--cooldown", "0",
        ]
    )

    assert rc == 0
    assert vision_calls["n"] == 0


def test_main_handles_generation_error(monkeypatch, tmp_path, capsys):
    img, prompts = _seed_inputs(tmp_path)
    out = tmp_path / "outputs"

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(rp, "_vision_call", lambda *a, **k: "dark style")
    monkeypatch.setattr(rp, "generate_image", lambda *a, **k: (_ for _ in ()).throw(rp.PermanentAPIError("boom")))

    rc = rp.main(
        [
            "--image", str(img),
            "--prompts", str(prompts),
            "--output", str(out),
            "--no-cache",
            "--cooldown", "0",
        ]
    )

    assert rc == 0  # pipeline continues after per-item failure
    manifest = rp.load_manifest(out)
    assert manifest[1].error == "boom"
    assert manifest[1].source == "error"


# --------------------------------------------------------------------------- #
# main() capability detection — interactive prompt flow
# --------------------------------------------------------------------------- #
def test_main_unsupported_img2img_user_continues(monkeypatch, tmp_path):
    img, prompts = _seed_inputs(tmp_path)
    out = tmp_path / "outputs"

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(rp, "_vision_call", lambda *a, **k: "dark style")
    monkeypatch.setattr(
        rp,
        "fetch_model_capabilities",
        lambda *a, **k: rp.ModelCapabilities(
            supports_img2img=False, provider_slug="openai", allowed_passthrough=[],
        ),
    )
    monkeypatch.setattr("builtins.input", lambda *a: "y")

    gen_calls = []

    def fake_generate(api_key, model, prompt, fmt, **k):
        gen_calls.append(k)
        return ({"b64_json": base64.b64encode(b"png").decode()}, "b64")

    monkeypatch.setattr(rp, "generate_image", fake_generate)

    rc = rp.main(
        [
            "--image", str(img),
            "--prompts", str(prompts),
            "--output", str(out),
            "--no-cache",
            "--cooldown", "0",
        ]
    )
    assert rc == 0
    assert (out / "01_lion.png").exists()


def test_main_unsupported_img2img_user_aborts(monkeypatch, tmp_path):
    img, prompts = _seed_inputs(tmp_path)
    out = tmp_path / "outputs"

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(rp, "_vision_call", lambda *a, **k: "dark style")
    monkeypatch.setattr(
        rp,
        "fetch_model_capabilities",
        lambda *a, **k: rp.ModelCapabilities(
            supports_img2img=False, provider_slug="openai", allowed_passthrough=[],
        ),
    )
    monkeypatch.setattr("builtins.input", lambda *a: "n")

    def fail_if_called(*a, **k):
        pytest.fail("generate_image must not be called when user aborts")

    monkeypatch.setattr(rp, "generate_image", fail_if_called)

    rc = rp.main(
        [
            "--image", str(img),
            "--prompts", str(prompts),
            "--output", str(out),
            "--no-cache",
            "--cooldown", "0",
        ]
    )
    assert rc == 1


def test_main_unsupported_img2img_dry_run_auto_suppresses(monkeypatch, tmp_path, capsys):
    img, prompts = _seed_inputs(tmp_path)
    out = tmp_path / "outputs"

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(rp, "_vision_call", lambda *a, **k: "dark style")
    monkeypatch.setattr(
        rp,
        "fetch_model_capabilities",
        lambda *a, **k: rp.ModelCapabilities(
            supports_img2img=False, provider_slug="openai", allowed_passthrough=[],
        ),
    )

    input_called = {"n": 0}
    monkeypatch.setattr("builtins.input", lambda *a: input_called.__setitem__("n", input_called["n"] + 1) or "y")

    def fail_if_called(*a, **k):
        pytest.fail("generate_image must not be called in dry-run")

    monkeypatch.setattr(rp, "generate_image", fail_if_called)

    rc = rp.main(
        [
            "--image", str(img),
            "--prompts", str(prompts),
            "--output", str(out),
            "--dry-run",
            "--no-cache",
        ]
    )
    assert rc == 0
    assert input_called["n"] == 0  # no interactive prompt in dry-run
    captured = capsys.readouterr()
    assert "Dry-run" in captured.out


def test_main_unsupported_steps_silently_dropped(monkeypatch, tmp_path):
    img, prompts = _seed_inputs(tmp_path)
    out = tmp_path / "outputs"

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(rp, "_vision_call", lambda *a, **k: "dark style")
    monkeypatch.setattr(
        rp,
        "fetch_model_capabilities",
        lambda *a, **k: rp.ModelCapabilities(
            supports_img2img=True, provider_slug="openai", allowed_passthrough=[],
        ),
    )

    captured_kwargs = {}

    def fake_generate(api_key, model, prompt, fmt, **k):
        captured_kwargs.update(k)
        return ({"b64_json": base64.b64encode(b"png").decode()}, "b64")

    monkeypatch.setattr(rp, "generate_image", fake_generate)

    rc = rp.main(
        [
            "--image", str(img),
            "--prompts", str(prompts),
            "--output", str(out),
            "--no-cache",
            "--cooldown", "0",
            "--steps", "40",
            "--guidance", "3.5",
        ]
    )
    assert rc == 0
    # steps and guidance should have been silently dropped
    assert captured_kwargs.get("steps") is None
    assert captured_kwargs.get("guidance") is None
