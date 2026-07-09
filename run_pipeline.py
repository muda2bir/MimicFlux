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
from dataclasses import asdict, dataclass
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
    "Analyze this image and write a detailed, structured style description "
    "(5-8 sentences) covering each of the following aspects, in order:\n"
    "1. Composition and framing of the central object.\n"
    "2. Color palette and saturation level.\n"
    "3. Lighting — direction, quality, and contrast.\n"
    "4. Material textures and surface finishes.\n"
    "5. Physical form and proportions of the central object.\n"
    "6. Background and surrounding environment.\n"
    "7. Camera characteristics — lens, depth of field, and angle.\n"
    "8. Overall mood and aesthetic.\n"
    "Focus strictly on verifiable visual details you can observe; do not "
    "invent attributes that are not visible. The output will be used as a "
    "style and texture prompt modifier for an image generation model."
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
    fmt = "%(asctime)s %(levelname)s %(message)s" if verbose else "%(message)s"
    logging.basicConfig(level=level, format=fmt, handlers=[TqdmHandler()], force=True)


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
    return f"{subject}, {style}"


def encode_image(image_path: Path) -> str:
    if not image_path.exists():
        raise FileNotFoundError(
            f"Missing local style image at: {image_path}. "
            "Place a file named my_style.jpg in the project folder, "
            "or pass --image <path> to point to a different one. "
            "See README.md for details."
        )
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


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
    index: int, subject: str, pad_width: int = 2, legacy: bool = False
) -> str:
    """Return the output filename for a generated image.

    Default: zero-padded index + subject slug, e.g. ``01_amethyst.png``.
    With ``legacy=True``: the original ``generation_{index}.png`` naming.
    """
    if legacy:
        return f"generation_{index}.png"
    slug = extract_subject_slug(subject)
    return f"{index:0{pad_width}d}_{slug}.png"


# ==========================================
# PROMPTS FILE
# ==========================================
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
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


# ==========================================
# STYLE CACHE
# ==========================================
def cache_path_for(image_path: Path) -> Path:
    return image_path.parent / STYLE_CACHE_NAME


def _vision_prompt_hash() -> str:
    """Short hash of the current VISION_PROMPT, used to invalidate stale caches."""
    return hashlib.sha256(VISION_PROMPT.encode("utf-8")).hexdigest()[:16]


def style_cache_valid(cache_path: Path, image_path: Path, vision_model: str) -> bool:
    if not cache_path.exists() or not image_path.exists():
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
    st = image_path.stat()
    return data.get("size") == st.st_size and data.get("mtime") == int(st.st_mtime)


def load_cached_style(cache_path: Path) -> str | None:
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data.get("style")
    except (json.JSONDecodeError, OSError):
        return None
    return None


def save_cached_style(
    cache_path: Path, image_path: Path, vision_model: str, style: str
) -> None:
    st = image_path.stat()
    payload = {
        "image": str(image_path),
        "model": vision_model,
        "prompt_hash": _vision_prompt_hash(),
        "size": st.st_size,
        "mtime": int(st.st_mtime),
        "style": style,
    }
    cache_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ==========================================
# VISION STEP
# ==========================================
@retry(
    retry=retry_if_exception_type(_VISION_RETRYABLE),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _vision_call(
    client: OpenAI, model: str, image_b64: str, prompt: str
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content or ""


def resolve_style(
    api_key: str, image_path: Path, vision_model: str, no_cache: bool
) -> str:
    cache = cache_path_for(image_path)
    if not no_cache and style_cache_valid(cache, image_path, vision_model):
        logger.info("♻️ Reusing cached style from '%s'.", cache.name)
        return load_cached_style(cache) or ""

    logger.info("✨ Step 1: Analyzing aesthetic style using %s...", vision_model)
    image_b64 = encode_image(image_path)
    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    style = _vision_call(client, vision_model, image_b64, VISION_PROMPT)
    save_cached_style(cache, image_path, vision_model, style)
    return style


# ==========================================
# IMAGE GENERATION STEP
# ==========================================
def _generate_image_once(
    api_key: str, model: str, prompt: str, output_format: str
) -> tuple[dict, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "prompt": prompt,
        "output_format": output_format,
    }
    response = requests.post(
        OPENROUTER_IMAGES_URL,
        headers=headers,
        json=payload,
        timeout=IMAGE_API_TIMEOUT,
    )
    if response.status_code == 429 or response.status_code >= 500:
        raise TransientAPIError(
            f"HTTP {response.status_code} from image API: {response.text[:200]}"
        )
    if response.status_code >= 400:
        raise PermanentAPIError(
            f"HTTP {response.status_code} from image API: {response.text[:300]}"
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
    api_key: str, model: str, prompt: str, output_format: str
) -> tuple[dict, str]:
    return _generate_image_once(api_key, model, prompt, output_format)


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


def should_skip(
    out_path: Path, subject: str, entry: ManifestEntry | None, force: bool
) -> bool:
    if force or entry is None:
        return False
    if entry.subject != subject:
        return False
    return out_path.exists()


# ==========================================
# CLI
# ==========================================
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automated image generation pipeline using OpenRouter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", default=DEFAULT_SAMPLE_IMAGE, help="Reference style image.")
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

    image_path = Path(args.image)
    prompts_path = Path(args.prompts)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: style extraction (cached when possible) ---
    try:
        extracted_style = resolve_style(
            api_key, image_path, args.vision_model, args.no_cache
        )
    except Exception as e:  # noqa: BLE001
        logger.error("❌ Failed to parse style image: %s", e)
        return 1
    logger.info("🎨 Extracted Style Anchor:\n> %s\n", extracted_style)

    # --- Step 2: load prompts ---
    subjects = load_prompts(prompts_path)
    if subjects is None:
        logger.info("ℹ️ Created a starter prompts file: '%s'. Populate it and rerun.", prompts_path)
        return 0
    logger.info("📦 Successfully loaded %d items from your queue.\n", len(subjects))

    if args.dry_run:
        logger.info("🧪 Dry-run mode: no images will be generated.")
        for i, subject in enumerate(subjects, start=1):
            logger.info(
                "[%d/%d] (dry-run) merged prompt: '%s'",
                i,
                len(subjects),
                build_prompt(subject, extracted_style),
            )
        return 0

    # --- Step 3: generation loop ---
    logger.info("🚀 Initializing generation queue on model: %s...", args.image_model)
    manifest = load_manifest(output_dir)
    total = len(subjects)
    pad_width = max(2, len(str(total)))

    pbar = tqdm(subjects, desc="Generating", unit="img")
    for index, subject in enumerate(pbar, start=1):
        pbar.set_postfix_str(subject[:30])
        merged_prompt = build_prompt(subject, extracted_style)
        file_path = output_dir / output_filename(
            index, subject, pad_width=pad_width, legacy=args.legacy_names
        )
        entry = manifest.get(index)

        if should_skip(file_path, subject, entry, args.force):
            logger.info("[%d/%d] Skipped (already generated): '%s'", index, total, subject[:40])
            manifest[index] = ManifestEntry(
                index=index,
                subject=subject,
                style=extracted_style,
                merged_prompt=merged_prompt,
                model=args.image_model,
                timestamp=now_iso(),
                filename=file_path.name,
                source="skip",
                skipped=True,
            )
            continue

        logger.info("[%d/%d] Generating target: '%s'", index, total, subject[:40])
        try:
            img_item, source = generate_image(
                api_key, args.image_model, merged_prompt, DEFAULT_OUTPUT_FORMAT
            )
            save_image(img_item, file_path)
        except Exception as e:  # noqa: BLE001
            logger.error("❌ Error occurred generating index %d: %s", index, e)
            manifest[index] = ManifestEntry(
                index=index,
                subject=subject,
                style=extracted_style,
                merged_prompt=merged_prompt,
                model=args.image_model,
                timestamp=now_iso(),
                filename=file_path.name,
                source="error",
                error=str(e),
            )
            time.sleep(args.cooldown)
            continue

        logger.info("💾 File Saved -> %s", file_path)
        manifest[index] = ManifestEntry(
            index=index,
            subject=subject,
            style=extracted_style,
            merged_prompt=merged_prompt,
            model=args.image_model,
            timestamp=now_iso(),
            filename=file_path.name,
            source=source,
        )
        time.sleep(args.cooldown)

    write_manifest(output_dir, manifest)
    logger.info("🎉 Process completed! Outputs are in '%s'.", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
