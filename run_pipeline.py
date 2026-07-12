"""Automated image generation pipeline using OpenRouter.

Reads a local reference image, extracts its aesthetic style via a vision model,
then loops through a prompts file generating styled images with an image model.

Run with no flags to use the documented defaults::

    python3 run_pipeline.py
"""

from __future__ import annotations

import sys

if sys.version_info < (3, 11):  # noqa: UP036 - guard helps direct `python3 run_pipeline.py` users on old interpreters
    sys.exit(
        "MimicFlux requires Python 3.11 or newer. You have "
        f"{sys.version.split()[0]}. See README.md for install help."
    )

import argparse
import base64
import hashlib
import json
import logging
import os
import re
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import openai
import requests
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

# ==========================================
# DEFAULT CONFIGURATION
# ==========================================
DEFAULT_SAMPLE_IMAGE = "my_style.jpg"
DEFAULT_PROMPTS_FILE = "prompts.txt"
DEFAULT_OUTPUT_DIR = "automated_outputs"
DEFAULT_VISION_MODEL = "openai/gpt-4o-mini"
DEFAULT_IMAGE_MODEL = "black-forest-labs/flux.2-pro"
DEFAULT_COOLDOWN_SECONDS = 4
DEFAULT_OUTPUT_FORMAT = "png"
MAX_REFERENCE_IMAGES = 5
STYLE_CACHE_NAME = ".style_cache.txt"
MANIFEST_NAME = "manifest.json"
IMAGE_DOWNLOAD_TIMEOUT = 30
IMAGE_API_TIMEOUT = 60

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_IMAGES_URL = f"{OPENROUTER_BASE_URL}/images"

# Tokens stripped from a subject before deriving its filename slug, so e.g.
# "A photorealistic, perfectly set faceted Amethyst with the ring clearly
# visible." -> "amethyst" (the leading noun phrase, minus generic descriptors).
TAIL_FILLER_WORDS = frozenset(
    {
        "a", "an", "the", "of", "in", "on", "at", "with", "and", "or", "to",
        "for", "by", "from", "into", "onto", "over", "under", "is", "are",
        "was", "were", "photorealistic", "realistic", "perfectly", "set",
        "faceted", "stunning", "majestic", "beautiful", "gorgeous", "pretty",
        "image", "photo", "picture", "shot", "render", "rendered",
    }
)
SUBJECT_SLUG_MAX_WORDS = 4
SUBJECT_SLUG_MAX_LEN = 60

VISION_PROMPT = (
    "Analyze the reference image(s) and write a dense, comma-separated style "
    "description (5-8 sentences) covering each of the following aspects, in order:\n"
    "1. Composition and framing of the central object.\n"
    "2. Color palette and saturation level.\n"
    "3. Lighting — direction, quality, and contrast.\n"
    "4. Material textures and surface finishes.\n"
    "5. Physical form and proportions of the central object.\n"
    "6. Background and surrounding environment.\n"
    "7. Camera characteristics — lens, depth of field, and angle.\n"
    "8. Overall mood and aesthetic.\n"
    "If multiple images are provided, synthesize their shared visual style "
    "into one unified description. Focus strictly on verifiable visual "
    "details you can observe; do not invent attributes that are not visible. "
    "Do not describe the subject matter itself — only the visual style, "
    "lighting, colors, textures, and camera work. Use concrete, specific "
    "language (e.g. 'warm amber rim light from the left' rather than 'nice "
    "lighting'). The output will be used as a style and texture prompt "
    "modifier for an image generation model."
)

logger = logging.getLogger("mimicflux")

# Transient OpenAI/SDK errors worth retrying on the vision step.
_VISION_RETRYABLE: tuple[type[BaseException], ...] = tuple(
    exc
    for exc in (
        getattr(openai, "APIConnectionError", None),
        getattr(openai, "RateLimitError", None),
        getattr(openai, "InternalServerError", None),
    )
    if isinstance(exc, type)
) or (openai.APIError,)


# ==========================================
# ERRORS
# ==========================================
class TransientAPIError(Exception):
    """Raised for transient HTTP failures (429 / 5xx) that should be retried."""


class PermanentAPIError(Exception):
    """Raised for non-retryable HTTP failures (4xx other than 429)."""


# ==========================================
# DATA MODEL
# ==========================================
@dataclass
class ManifestEntry:
    index: int
    subject: str
    style: str
    merged_prompt: str
    model: str
    timestamp: str
    filename: str
    source: str  # "b64" | "url" | "skip" | "error"
    skipped: bool = False
    error: str | None = None


@dataclass
class GenEvent:
    """A single event emitted by :func:`run_generation` for the CLI or web UI."""

    kind: str
    payload: dict = field(default_factory=dict)


@dataclass
class GenerationOptions:
    """Tweakable options shared by the CLI ``main()`` and the web UI."""

    vision_model: str = DEFAULT_VISION_MODEL
    image_model: str = DEFAULT_IMAGE_MODEL
    cooldown: float = DEFAULT_COOLDOWN_SECONDS
    no_cache: bool = False
    force: bool = False
    legacy_names: bool = False
    dry_run: bool = False
    output_format: str = DEFAULT_OUTPUT_FORMAT
    no_img2img: bool = False
    steps: int | None = None
    guidance: float | None = None
    seed: int | None = None

    VALID_FORMATS: tuple[str, ...] = field(default=("png", "jpeg", "webp"), init=False, repr=False)

    def __post_init__(self) -> None:
        if self.output_format not in self.VALID_FORMATS:
            raise ValueError(
                f"output_format must be one of {self.VALID_FORMATS}, "
                f"got {self.output_format!r}"
            )
        if self.steps is not None and not (1 <= self.steps <= 100):
            raise ValueError(f"steps must be between 1 and 100, got {self.steps}")
        if self.guidance is not None and not (0 <= self.guidance <= 20):
            raise ValueError(f"guidance must be between 0 and 20, got {self.guidance}")
        if self.seed is not None and self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")


@dataclass
class ModelCapabilities:
    """Capabilities of an image model, queried from OpenRouter's Image API.

    Defaults are optimistic so existing behavior is preserved when the
    capability endpoint is unreachable or returns an unexpected response.
    ``allowed_passthrough`` is ``None`` when unknown (don't filter) and a
    list (possibly empty) when the API gave a definitive answer (do filter).
    """

    supports_img2img: bool = True
    provider_slug: str = "black-forest-labs"
    allowed_passthrough: list[str] | None = None


# ==========================================
# LOGGING
# ==========================================
class TqdmHandler(logging.StreamHandler):
    """Stream handler that emits through ``tqdm.write`` so progress bars stay intact."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=sys.stdout)
            self.flush()
        except Exception:  # noqa: BLE001 - mirror stdlib handler contract
            self.handleError(record)


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s %(levelname)s %(name)s %(message)s" if verbose else "%(message)s"
    logging.basicConfig(level=logging.WARNING, format=fmt, handlers=[TqdmHandler()], force=True)
    logger.setLevel(level)
    for name in ("urllib3", "openai", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


# ==========================================
# SMALL HELPERS
# ==========================================
def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def slugify(text: str, max_len: int = 50) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    slug = slug[:max_len].strip("-")
    return slug or "prompt"


def build_prompt(subject: str, style: str) -> str:
    return f"{subject}. {style}. Highly detailed, sharp focus, professional quality."


def encode_image(image_path: Path) -> str:
    if not image_path.exists():
        raise FileNotFoundError(
            f"Missing local style image at: {image_path}. "
            "Place a file named my_style.jpg in the project folder, "
            "or pass --image <path> to point to a different one. "
            "See README.md for details."
        )
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def encode_images(image_paths: list[Path]) -> list[tuple[str, str]]:
    """Encode multiple reference images, returning ``(b64, mime)`` pairs.

    Validates that at least one image is provided and the count does not
    exceed :data:`MAX_REFERENCE_IMAGES`.
    """
    if not image_paths:
        raise FileNotFoundError("At least one reference image is required.")
    if len(image_paths) > MAX_REFERENCE_IMAGES:
        raise ValueError(
            f"Too many reference images: {len(image_paths)}. "
            f"Maximum is {MAX_REFERENCE_IMAGES}."
        )
    return [
        (encode_image(p), detect_image_mime(p)) for p in image_paths
    ]


def extract_subject_slug(
    subject: str, max_words: int = SUBJECT_SLUG_MAX_WORDS
) -> str:
    """Derive a short slug from the leading noun phrase of a subject.

    Leading filler words (articles, descriptors like "photorealistic") are
    skipped, then the contiguous non-filler tokens are collected until the
    next filler word (e.g. "with"/"on") ends the phrase. This picks the actual
    subject, e.g. "...faceted Amethyst with the ring..." -> "amethyst".
    Falls back to ``"prompt"`` when nothing meaningful remains.
    """
    tokens = re.split(r"[^a-zA-Z0-9]+", subject.lower())
    tokens = [t for t in tokens if t]
    run: list[str] = []
    started = False
    for t in tokens:
        if t in TAIL_FILLER_WORDS:
            if started:
                break
            continue
        started = True
        run.append(t)
        if len(run) >= max_words:
            break
    slug = slugify(" ".join(run), max_len=SUBJECT_SLUG_MAX_LEN)
    return slug or "prompt"


def output_filename(
    index: int, subject: str, pad_width: int = 2, legacy: bool = False,
    ext: str = DEFAULT_OUTPUT_FORMAT,
) -> str:
    """Return the output filename for a generated image.

    Default: zero-padded index + subject slug, e.g. ``01_amethyst.png``.
    With ``legacy=True``: the original ``generation_{index}.png`` naming.
    ``ext`` controls the file extension (defaults to ``png``).
    """
    if legacy:
        return f"generation_{index}.{ext}"
    slug = extract_subject_slug(subject)
    return f"{index:0{pad_width}d}_{slug}.{ext}"


# ==========================================
# PROMPTS FILE
# ==========================================
def parse_prompts_text(text: str) -> list[str]:
    """Split raw text into non-empty, non-comment prompt lines."""
    return [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]


def load_prompts(path: Path) -> list[str] | None:
    """Load non-empty, non-comment prompt lines.

    If ``path`` does not exist, a starter file is written and ``None`` is
    returned so the caller can inform the user and exit cleanly.
    """
    if not path.exists():
        path.write_text(
            "A stunning landscape painting\nA high-tech cybernetic robot\n",
            encoding="utf-8",
        )
        return None
    return parse_prompts_text(path.read_text(encoding="utf-8"))


# ==========================================
# STYLE CACHE
# ==========================================
def cache_path_for(image_path: Path) -> Path:
    return image_path.parent / STYLE_CACHE_NAME


def _vision_prompt_hash() -> str:
    """Short hash of the current VISION_PROMPT, used to invalidate stale caches."""
    return hashlib.sha256(VISION_PROMPT.encode("utf-8")).hexdigest()[:16]


def style_cache_valid(
    cache_path: Path, image_paths: list[Path], vision_model: str
) -> bool:
    if not cache_path.exists():
        return False
    if not all(p.exists() for p in image_paths):
        return False
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(data, dict) or not data.get("style"):
        return False
    if data.get("model") != vision_model:
        return False
    if data.get("prompt_hash") != _vision_prompt_hash():
        return False
    cached_images = data.get("images")
    if not isinstance(cached_images, list):
        return False
    if len(cached_images) != len(image_paths):
        return False
    for ci, ip in zip(cached_images, image_paths, strict=True):
        if not isinstance(ci, dict):
            return False
        st = ip.stat()
        if ci.get("size") != st.st_size or ci.get("mtime") != int(st.st_mtime):
            return False
    return True


def load_cached_style(cache_path: Path) -> str | None:
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data.get("style")
    except (json.JSONDecodeError, OSError):
        return None
    return None


def save_cached_style(
    cache_path: Path, image_paths: list[Path], vision_model: str, style: str
) -> None:
    images_meta = []
    for ip in image_paths:
        st = ip.stat()
        images_meta.append(
            {"path": str(ip), "size": st.st_size, "mtime": int(st.st_mtime)}
        )
    payload = {
        "images": images_meta,
        "model": vision_model,
        "prompt_hash": _vision_prompt_hash(),
        "style": style,
    }
    cache_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    try:
        os.chmod(cache_path, 0o600)
    except OSError:
        pass


# ==========================================
# VISION STEP
# ==========================================
_IMAGE_MIME_SIGNATURES: dict[bytes, str] = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG": "image/png",
    b"RIFF": "image/webp",
    b"BM": "image/bmp",
    b"GIF8": "image/gif",
}


def detect_image_mime(image_path: Path) -> str:
    """Detect image MIME type from file header bytes, with extension fallback."""
    try:
        head = image_path.read_bytes()[:16]
    except OSError:
        return "image/jpeg"
    for sig, mime in _IMAGE_MIME_SIGNATURES.items():
        if head.startswith(sig):
            return mime
    return "image/jpeg"


@retry(
    retry=retry_if_exception_type(_VISION_RETRYABLE),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _vision_call(
    client: OpenAI, model: str, images: list[tuple[str, str]], prompt: str,
) -> str:
    content: list[dict] = [{"type": "text", "text": prompt}]
    for b64, mime in images:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            }
        )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
    )
    return response.choices[0].message.content or ""


def resolve_style(
    api_key: str, image_paths: list[Path], vision_model: str, no_cache: bool
) -> str:
    cache = cache_path_for(image_paths[0])
    if not no_cache and style_cache_valid(cache, image_paths, vision_model):
        logger.info("♻️ Reusing cached style from '%s'.", cache.name)
        return load_cached_style(cache) or ""

    logger.info("✨ Step 1: Analyzing aesthetic style using %s...", vision_model)
    images = encode_images(image_paths)
    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    style = _vision_call(client, vision_model, images, VISION_PROMPT)
    save_cached_style(cache, image_paths, vision_model, style)
    return style


# ==========================================
# IMAGE GENERATION STEP
# ==========================================
def fetch_model_capabilities(api_key: str, model: str) -> ModelCapabilities:
    """Query OpenRouter's Image API for a model's capabilities.

    Returns optimistic defaults (:class:`ModelCapabilities`) on any error
    so the pipeline degrades to the pre-existing behavior instead of
    blocking the user when the metadata endpoint is unreachable.
    """
    url = f"{OPENROUTER_BASE_URL}/images/models/{model}/endpoints"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code >= 400:
            logger.debug(
                "Model capabilities lookup failed (HTTP %d) for '%s'; "
                "using optimistic defaults.",
                response.status_code, model,
            )
            return ModelCapabilities()
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        logger.debug(
            "Model capabilities lookup error for '%s': %s; "
            "using optimistic defaults.",
            model, exc,
        )
        return ModelCapabilities()

    endpoints = data.get("endpoints", [])
    if not endpoints:
        return ModelCapabilities()
    ep = endpoints[0]
    supported = ep.get("supported_parameters", {})
    return ModelCapabilities(
        supports_img2img="input_references" in supported,
        provider_slug=ep.get("provider_slug", "black-forest-labs"),
        allowed_passthrough=list(ep.get("allowed_passthrough_parameters", [])),
    )


def _build_image_payload(
    model: str,
    prompt: str,
    output_format: str,
    ref_images: list[tuple[str, str]] | None = None,
    steps: int | None = None,
    guidance: float | None = None,
    seed: int | None = None,
    caps: ModelCapabilities | None = None,
) -> dict:
    payload: dict = {
        "model": model,
        "prompt": prompt,
        "output_format": output_format,
    }
    if ref_images and (caps is None or caps.supports_img2img):
        payload["input_references"] = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            }
            for b64, mime in ref_images
        ]
    provider_slug = caps.provider_slug if caps else "black-forest-labs"
    allowed = caps.allowed_passthrough if caps else None
    provider_opts: dict = {}
    if steps is not None and (allowed is None or "steps" in allowed):
        provider_opts["steps"] = steps
    if guidance is not None and (allowed is None or "guidance" in allowed):
        provider_opts["guidance"] = guidance
    if provider_opts:
        payload["provider"] = {"options": {provider_slug: provider_opts}}
    if seed is not None:
        payload["seed"] = seed
    return payload


def _generate_image_once(
    api_key: str,
    model: str,
    prompt: str,
    output_format: str,
    ref_images: list[tuple[str, str]] | None = None,
    steps: int | None = None,
    guidance: float | None = None,
    seed: int | None = None,
    caps: ModelCapabilities | None = None,
) -> tuple[dict, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = _build_image_payload(
        model, prompt, output_format,
        ref_images=ref_images, steps=steps, guidance=guidance, seed=seed,
        caps=caps,
    )
    response = requests.post(
        OPENROUTER_IMAGES_URL,
        headers=headers,
        json=payload,
        timeout=IMAGE_API_TIMEOUT,
    )
    if response.status_code == 429 or response.status_code >= 500:
        raise TransientAPIError(
            f"HTTP {response.status_code} from image API"
        )
    if response.status_code >= 400:
        raise PermanentAPIError(
            f"HTTP {response.status_code} from image API"
        )
    try:
        data = response.json()
    except ValueError as exc:
        raise PermanentAPIError(f"Non-JSON response from image API: {exc}") from exc
    if "data" not in data or len(data["data"]) == 0:
        raise PermanentAPIError(f"No 'data' block in image API response: {data}")
    img_item = data["data"][0]
    source = "b64" if "b64_json" in img_item else "url"
    return img_item, source


@retry(
    retry=retry_if_exception_type((TransientAPIError, requests.RequestException)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def generate_image(
    api_key: str,
    model: str,
    prompt: str,
    output_format: str,
    ref_images: list[tuple[str, str]] | None = None,
    steps: int | None = None,
    guidance: float | None = None,
    seed: int | None = None,
    caps: ModelCapabilities | None = None,
) -> tuple[dict, str]:
    return _generate_image_once(
        api_key, model, prompt, output_format,
        ref_images=ref_images, steps=steps, guidance=guidance, seed=seed,
        caps=caps,
    )


def save_image(img_item: dict, out_path: Path) -> None:
    if "b64_json" in img_item:
        out_path.write_bytes(base64.b64decode(img_item["b64_json"]))
        return
    if "url" in img_item:
        resp = requests.get(img_item["url"], timeout=IMAGE_DOWNLOAD_TIMEOUT)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
        return
    raise PermanentAPIError(f"Image item has neither 'b64_json' nor 'url': {img_item}")


# ==========================================
# MANIFEST
# ==========================================
def load_manifest(output_dir: Path) -> dict[int, ManifestEntry]:
    path = output_dir / MANIFEST_NAME
    result: dict[int, ManifestEntry] = {}
    if not path.exists():
        return result
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return result

    items: list[dict] = []
    if isinstance(raw, dict):
        items = [v for v in raw.values() if isinstance(v, dict)]
    elif isinstance(raw, list):
        items = [v for v in raw if isinstance(v, dict)]

    for i, v in enumerate(items, start=1):
        try:
            idx = int(v.get("index", i))
        except (TypeError, ValueError):
            continue
        result[idx] = ManifestEntry(
            index=idx,
            subject=v.get("subject", ""),
            style=v.get("style", ""),
            merged_prompt=v.get("merged_prompt", ""),
            model=v.get("model", ""),
            timestamp=v.get("timestamp", ""),
            filename=v.get("filename", ""),
            source=v.get("source", ""),
            skipped=bool(v.get("skipped", False)),
            error=v.get("error"),
        )
    return result


def write_manifest(output_dir: Path, manifest: dict[int, ManifestEntry]) -> None:
    path = output_dir / MANIFEST_NAME
    serializable = {str(k): asdict(v) for k, v in sorted(manifest.items())}
    path.write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def should_skip(
    out_path: Path, subject: str, entry: ManifestEntry | None, force: bool
) -> bool:
    if force or entry is None:
        return False
    if entry.subject != subject:
        return False
    return out_path.exists()


# ==========================================
# API KEY HELPERS
# ==========================================
def save_api_key_to_env(api_key: str, env_path: Path | None = None) -> None:
    """Persist ``OPENROUTER_API_KEY`` to a ``.env`` file and reload it.

    Creates the file if missing, replaces the key line if present, or appends.
    Used by the web UI's "enter key" flow.
    """
    env_path = env_path or Path(".env")
    line = f"OPENROUTER_API_KEY={api_key.strip()}"
    if env_path.exists():
        content = env_path.read_text(encoding="utf-8")
        if re.search(r"^OPENROUTER_API_KEY=", content, flags=re.MULTILINE):
            new_content = re.sub(
                r"^OPENROUTER_API_KEY=.*$", line, content, flags=re.MULTILINE
            )
            env_path.write_text(new_content, encoding="utf-8")
        else:
            env_path.write_text(
                content.rstrip("\n") + "\n" + line + "\n", encoding="utf-8"
            )
    else:
        env_path.write_text(line + "\n", encoding="utf-8")
    try:
        os.chmod(env_path, 0o600)
    except OSError:
        pass
    load_dotenv(override=True)


# ==========================================
# GENERATION ORCHESTRATOR (shared by CLI + web UI)
# ==========================================
def run_generation(
    api_key: str,
    image_paths: list[Path],
    subjects: list[str],
    output_dir: Path,
    options: GenerationOptions,
    caps: ModelCapabilities | None = None,
) -> Iterator[GenEvent]:
    """Run the full pipeline, yielding :class:`GenEvent`s for live UI updates.

    Yields events in this order:
    ``style_start`` → ``style_done`` (or ``style_error``) → ``queue`` →
    per subject: ``gen_start`` → ``gen_done`` | ``skip`` | ``gen_error`` →
    ``done``.

    In ``dry_run`` mode, yields ``style_start`` → ``style_done`` →
    per subject: ``dry_run`` → ``done`` (no image API calls).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: style extraction ---
    yield GenEvent("style_start", {"model": options.vision_model})
    try:
        extracted_style = resolve_style(
            api_key, image_paths, options.vision_model, options.no_cache
        )
    except Exception as e:  # noqa: BLE001
        yield GenEvent("style_error", {"error": str(e)})
        return
    yield GenEvent("style_done", {"style": extracted_style})

    total = len(subjects)
    yield GenEvent("queue", {"total": total})

    # --- Dry-run: print merged prompts only ---
    if options.dry_run:
        for i, subject in enumerate(subjects, start=1):
            merged = build_prompt(subject, extracted_style)
            yield GenEvent(
                "dry_run", {"index": i, "total": total, "merged_prompt": merged}
            )
        yield GenEvent("done", {"output_dir": str(output_dir)})
        return

    # --- Encode reference images once for img2img (if enabled) ---
    ref_images: list[tuple[str, str]] | None = None
    if not options.no_img2img:
        ref_images = encode_images(image_paths)

    # --- Step 2: generation loop ---
    manifest = load_manifest(output_dir)
    pad_width = max(2, len(str(total)))

    for index, subject in enumerate(subjects, start=1):
        merged_prompt = build_prompt(subject, extracted_style)
        file_path = output_dir / output_filename(
            index, subject, pad_width=pad_width, legacy=options.legacy_names,
            ext=options.output_format,
        )
        entry = manifest.get(index)

        yield GenEvent(
            "gen_start",
            {"index": index, "total": total, "subject": subject},
        )

        if should_skip(file_path, subject, entry, options.force):
            manifest[index] = ManifestEntry(
                index=index,
                subject=subject,
                style=extracted_style,
                merged_prompt=merged_prompt,
                model=options.image_model,
                timestamp=now_iso(),
                filename=file_path.name,
                source="skip",
                skipped=True,
            )
            yield GenEvent(
                "skip",
                {
                    "index": index,
                    "total": total,
                    "subject": subject,
                    "filename": file_path.name,
                },
            )
            continue

        try:
            img_item, source = generate_image(
                api_key, options.image_model, merged_prompt, options.output_format,
                ref_images=ref_images,
                steps=options.steps,
                guidance=options.guidance,
                seed=options.seed,
                caps=caps,
            )
            save_image(img_item, file_path)
        except Exception as e:  # noqa: BLE001
            manifest[index] = ManifestEntry(
                index=index,
                subject=subject,
                style=extracted_style,
                merged_prompt=merged_prompt,
                model=options.image_model,
                timestamp=now_iso(),
                filename=file_path.name,
                source="error",
                error=str(e),
            )
            yield GenEvent(
                "gen_error",
                {
                    "index": index,
                    "total": total,
                    "subject": subject,
                    "filename": file_path.name,
                    "error": str(e),
                },
            )
            time.sleep(options.cooldown)
            continue

        manifest[index] = ManifestEntry(
            index=index,
            subject=subject,
            style=extracted_style,
            merged_prompt=merged_prompt,
            model=options.image_model,
            timestamp=now_iso(),
            filename=file_path.name,
            source=source,
        )
        yield GenEvent(
            "gen_done",
            {
                "index": index,
                "total": total,
                "subject": subject,
                "filename": file_path.name,
                "file_path": str(file_path),
                "source": source,
            },
        )
        time.sleep(options.cooldown)

    write_manifest(output_dir, manifest)
    yield GenEvent("done", {"output_dir": str(output_dir)})


# ==========================================
# CLI
# ==========================================
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automated image generation pipeline using OpenRouter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image",
        nargs="+",
        default=[DEFAULT_SAMPLE_IMAGE],
        help=f"Reference style image path(s), up to {MAX_REFERENCE_IMAGES}.",
    )
    parser.add_argument("--prompts", default=DEFAULT_PROMPTS_FILE, help="Prompts file path.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--vision-model", default=DEFAULT_VISION_MODEL, help="Vision model.")
    parser.add_argument(
        "--image-model", default=DEFAULT_IMAGE_MODEL, help="Image generation model."
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=DEFAULT_COOLDOWN_SECONDS,
        help="Seconds between generations.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print prompts, skip image API.")
    parser.add_argument("--no-cache", action="store_true", help="Ignore cached style.")
    parser.add_argument(
        "--legacy-names",
        action="store_true",
        help="Use generation_N.png filenames instead of descriptive ones.",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate existing outputs.")
    parser.add_argument(
        "--output-format",
        choices=["png", "jpeg", "webp"],
        default=DEFAULT_OUTPUT_FORMAT,
        help="Image format for generated outputs.",
    )
    parser.add_argument(
        "--no-img2img",
        action="store_true",
        help="Disable image-to-image (don't pass reference images to the image model).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="FLUX inference steps (1-100, higher = more refined). Default uses model's own.",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=None,
        help="FLUX guidance scale (0-20, higher = follows prompt more closely).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Fixed seed for reproducible generation. Omit for random.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


# ==========================================
# MAIN
# ==========================================
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)

    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error(
            "API Key missing! Copy .env.example to .env and paste your "
            "OpenRouter key into it. See README.md for how to get a key."
        )
        return 2

    image_paths = [Path(p) for p in args.image]
    if len(image_paths) > MAX_REFERENCE_IMAGES:
        logger.error(
            "Too many reference images: %d. Maximum is %d.",
            len(image_paths), MAX_REFERENCE_IMAGES,
        )
        return 2
    prompts_path = Path(args.prompts)
    output_dir = Path(args.output)

    subjects = load_prompts(prompts_path)
    if subjects is None:
        logger.info("ℹ️ Created a starter prompts file: '%s'. Populate it and rerun.", prompts_path)
        return 0

    options = GenerationOptions(
        vision_model=args.vision_model,
        image_model=args.image_model,
        cooldown=args.cooldown,
        no_cache=args.no_cache,
        force=args.force,
        legacy_names=args.legacy_names,
        dry_run=args.dry_run,
        output_format=args.output_format,
        no_img2img=args.no_img2img,
        steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
    )

    # --- Check model capabilities and warn/fallback for unsupported features ---
    caps = fetch_model_capabilities(api_key, options.image_model)

    if not caps.supports_img2img and not options.no_img2img and image_paths:
        logger.warning(
            "⚠️  Model '%s' doesn't support image references (img2img). "
            "Consider using '%s' for best results.",
            options.image_model, DEFAULT_IMAGE_MODEL,
        )
        if options.dry_run:
            logger.info("ℹ️ Dry-run: automatically continuing without image references.")
            options.no_img2img = True
        else:
            response = input("Continue without image references? [Y/n] ").strip().lower()
            if response == "n":
                logger.info(
                    "Aborting. Try --image-model %s or pass --no-img2img.",
                    DEFAULT_IMAGE_MODEL,
                )
                return 1
            options.no_img2img = True

    if (
        options.steps is not None
        and caps.allowed_passthrough is not None
        and "steps" not in caps.allowed_passthrough
    ):
        logger.info(
            "ℹ️ Model '%s' doesn't support 'steps'; ignoring.",
            options.image_model,
        )
        options.steps = None

    if (
        options.guidance is not None
        and caps.allowed_passthrough is not None
        and "guidance" not in caps.allowed_passthrough
    ):
        logger.info(
            "ℹ️ Model '%s' doesn't support 'guidance'; ignoring.",
            options.image_model,
        )
        options.guidance = None

    total = len(subjects)
    pbar: tqdm | None = None

    for event in run_generation(
        api_key, image_paths, subjects, output_dir, options, caps=caps
    ):
        kind = event.kind
        p = event.payload

        if kind == "style_done":
            logger.info("🎨 Extracted Style Anchor:\n> %s\n", p["style"])
            logger.info("📦 Successfully loaded %d items from your queue.\n", total)
        elif kind == "style_error":
            logger.error("❌ Failed to parse style image: %s", p["error"])
            return 1
        elif kind == "queue":
            if options.dry_run:
                logger.info("🧪 Dry-run mode: no images will be generated.")
            else:
                logger.info("🚀 Initializing generation queue on model: %s...", args.image_model)
                pbar = tqdm(total=total, desc="Generating", unit="img")
        elif kind == "dry_run":
            logger.info(
                "[%d/%d] (dry-run) merged prompt: '%s'",
                p["index"],
                p["total"],
                p["merged_prompt"],
            )
        elif kind == "gen_start":
            if pbar is not None:
                pbar.set_postfix_str(p["subject"][:30])
            logger.info(
                "[%d/%d] Generating target: '%s'",
                p["index"], p["total"], p["subject"][:40],
            )
        elif kind == "skip":
            if pbar is not None:
                pbar.update(1)
            logger.info(
                "[%d/%d] Skipped (already generated): '%s'",
                p["index"],
                p["total"],
                p["subject"][:40],
            )
        elif kind == "gen_done":
            if pbar is not None:
                pbar.update(1)
            logger.info("💾 File Saved -> %s", p["filename"])
        elif kind == "gen_error":
            if pbar is not None:
                pbar.update(1)
            logger.error("❌ Error occurred generating index %d: %s", p["index"], p["error"])
        elif kind == "done":
            if pbar is not None:
                pbar.close()
            logger.info("🎉 Process completed! Outputs are in '%s'.", p["output_dir"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
