"""MimicFlux browser UI powered by Streamlit.

Launch with::

    streamlit run web_app.py

Then open http://localhost:8501 in your browser.
"""

from __future__ import annotations

import io
import os
import tempfile
import zipfile
from pathlib import Path

import streamlit as st

import run_pipeline as rp

st.set_page_config(
    page_title="MimicFlux",
    page_icon=":material/auto_awesome:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# CONSTANTS
# ==========================================
MIME_TYPES: dict[str, str] = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}

SANDBOX_ROOT = Path(__file__).resolve().parent


# ==========================================
# OUTPUT DIRECTORY VALIDATION
# ==========================================
def validate_output_dir(dir_name: str) -> Path | None:
    """Resolve and validate the output directory stays within the project sandbox.

    Returns the resolved Path if safe, or None if the path escapes the sandbox
    or is otherwise invalid.
    """
    if not dir_name.strip():
        return None
    candidate = (SANDBOX_ROOT / dir_name).resolve()
    try:
        candidate.relative_to(SANDBOX_ROOT)
    except ValueError:
        return None
    return candidate


# ==========================================
# API KEY MANAGEMENT
# ==========================================
def get_api_key() -> str | None:
    rp.load_dotenv()
    return os.getenv("OPENROUTER_API_KEY")


def render_api_key_panel() -> str | None:
    existing = get_api_key()

    with st.sidebar:
        st.subheader("API key")

        if existing:
            st.success("Loaded from `.env`")
            with st.expander("Change key", icon=":material/key:"):
                new_key = st.text_input(
                    "OpenRouter API key", type="password", key="key_change"
                )
                if st.button("Save key", key="save_key_btn", icon=":material/save:"):
                    if new_key.strip():
                        rp.save_api_key_to_env(new_key.strip())
                        st.success("Saved! Reloading…")
                        st.rerun()
            return existing

        st.warning("No API key found. Enter your OpenRouter key below.")
        new_key = st.text_input(
            "OpenRouter API key", type="password", key="key_new"
        )
        if st.button("Save key", key="save_key_btn_new", icon=":material/save:"):
            if new_key.strip():
                rp.save_api_key_to_env(new_key.strip())
                st.success("Saved! Reloading…")
                st.rerun()
        return new_key.strip() or None


def render_options() -> dict:
    with st.sidebar:
        with st.expander("Advanced options", expanded=False, icon=":material/tune:"):
            vision_model = st.text_input("Vision model", value=rp.DEFAULT_VISION_MODEL)
            image_model = st.text_input("Image model", value=rp.DEFAULT_IMAGE_MODEL)
            cooldown = st.number_input(
                "Cooldown (seconds)",
                min_value=0.0,
                value=float(rp.DEFAULT_COOLDOWN_SECONDS),
                step=1.0,
            )
            col_flags1, col_flags2 = st.columns(2)
            with col_flags1:
                force = st.checkbox(
                    "Force regenerate", help="Recreate even if they exist."
                )
                no_cache = st.checkbox(
                    "Ignore style cache", help="Re-analyze the image."
                )
            with col_flags2:
                legacy_names = st.checkbox(
                    "Legacy filenames", help="Use generation_N.png naming."
                )
                no_img2img = st.checkbox(
                    "Disable img2img",
                    help="Don't pass reference images to the image model "
                    "(text-only style transfer).",
                )
            output_dir_name = st.text_input(
                "Output folder", value=rp.DEFAULT_OUTPUT_DIR
            )
            output_format = st.selectbox(
                "Output format",
                options=["png", "jpeg", "webp"],
                index=0,
            )

            st.markdown("**FLUX quality controls**")
            col_q1, col_q2 = st.columns(2)
            with col_q1:
                steps = st.slider(
                    "Steps",
                    min_value=0,
                    max_value=50,
                    value=0,
                    step=1,
                    help="0 = model default. Higher = more refined (slower). "
                    "FLUX-specific; ignored by other models.",
                )
            with col_q2:
                guidance = st.slider(
                    "Guidance",
                    min_value=0.0,
                    max_value=20.0,
                    value=0.0,
                    step=0.5,
                    help="0 = model default. Higher = follows prompt more "
                    "literally. FLUX-specific; ignored by other models.",
                )
            seed = st.number_input(
                "Seed",
                min_value=0,
                value=0,
                step=1,
                help="0 = random each time. Set a fixed value for "
                "reproducible generation.",
            )

    return {
        "vision_model": vision_model,
        "image_model": image_model,
        "cooldown": cooldown,
        "force": force,
        "no_cache": no_cache,
        "legacy_names": legacy_names,
        "output_dir_name": output_dir_name,
        "output_format": output_format,
        "no_img2img": no_img2img,
        "steps": steps or None,
        "guidance": guidance or None,
        "seed": seed or None,
    }


# ==========================================
# MAIN
# ==========================================
def main() -> None:
    st.title(":material/auto_awesome: MimicFlux")
    st.caption(
        "Upload reference image(s), list what you want drawn, and generate "
        "new pictures in that style — all in your browser."
    )

    api_key = render_api_key_panel()
    opts = render_options()

    col_img, col_prompts = st.columns(2)

    with col_img:
        with st.container(border=True):
            st.subheader(":material/image: Reference style image")
            uploaded_files = st.file_uploader(
                f"Upload up to {rp.MAX_REFERENCE_IMAGES} photos whose style you want to copy",
                type=["jpg", "jpeg", "png", "webp"],
                accept_multiple_files=True,
            )
            if uploaded_files:
                if len(uploaded_files) > rp.MAX_REFERENCE_IMAGES:
                    st.warning(
                        f"Too many images ({len(uploaded_files)}). "
                        f"Only the first {rp.MAX_REFERENCE_IMAGES} will be used."
                    )
                    uploaded_files = uploaded_files[: rp.MAX_REFERENCE_IMAGES]
                img_cols = st.columns(min(3, len(uploaded_files)))
                for i, uf in enumerate(uploaded_files):
                    with img_cols[i % len(img_cols)]:
                        st.image(uf, use_container_width=True, caption=uf.name)

    with col_prompts:
        with st.container(border=True):
            st.subheader(":material/edit_note: Prompts")
            default_prompts = ""
            prompts_file = Path(rp.DEFAULT_PROMPTS_FILE)
            if prompts_file.exists():
                default_prompts = prompts_file.read_text(encoding="utf-8")
            if not default_prompts.strip():
                default_prompts = (
                    "A photorealistic faceted Amethyst\n"
                    "A vintage brass pocket watch on a wooden desk\n"
                )
            prompts_text = st.text_area(
                "One subject per line. Lines starting with # are ignored.",
                value=default_prompts,
                height=180,
            )

    has_images = bool(uploaded_files)
    can_generate = bool(api_key) and has_images and bool(prompts_text.strip())

    if not can_generate:
        if not api_key:
            st.caption(
                ":material/key: Enter your OpenRouter API key in the sidebar to begin."
            )
        if not has_images:
            st.caption(":material/image: Upload at least one reference style image above.")
        if not prompts_text.strip():
            st.caption(":material/edit_note: Add at least one prompt above.")

    if st.button(
        "Generate",
        type="primary",
        disabled=not can_generate,
        icon=":material/rocket_launch:",
        width="stretch",
    ):
        run_pipeline_in_browser(
            api_key,
            uploaded_files,
            prompts_text,
            opts["vision_model"],
            opts["image_model"],
            opts["cooldown"],
            opts["force"],
            opts["no_cache"],
            opts["legacy_names"],
            opts["output_dir_name"],
            opts["output_format"],
            opts["no_img2img"],
            opts["steps"],
            opts["guidance"],
            opts["seed"],
        )

    render_results()


def run_pipeline_in_browser(
    api_key: str,
    uploaded_files: list,
    prompts_text: str,
    vision_model: str,
    image_model: str,
    cooldown: float,
    force: bool,
    no_cache: bool,
    legacy_names: bool,
    output_dir_name: str,
    output_format: str,
    no_img2img: bool,
    steps: int | None,
    guidance: float | None,
    seed: int | None,
) -> None:
    subjects = rp.parse_prompts_text(prompts_text)
    if not subjects:
        st.warning(
            "No valid prompts found. Add at least one non-empty, non-comment line."
        )
        return

    output_dir = validate_output_dir(output_dir_name)
    if output_dir is None:
        st.error(
            f"Unsafe output directory: {output_dir_name!r}. "
            "The output folder must stay within the project directory."
        )
        return

    image_paths: list[Path] = []
    for uf in uploaded_files:
        suffix = Path(uf.name).suffix or ".jpg"
        tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_img.write(uf.getvalue())
        tmp_img.close()
        image_paths.append(Path(tmp_img.name))

    try:
        options = rp.GenerationOptions(
            vision_model=vision_model,
            image_model=image_model,
            cooldown=cooldown,
            no_cache=no_cache,
            force=force,
            legacy_names=legacy_names,
            dry_run=False,
            output_format=output_format,
            no_img2img=no_img2img,
            steps=steps,
            guidance=guidance,
            seed=seed,
        )

        # --- Check model capabilities and warn/fallback for unsupported features ---
        caps = rp.fetch_model_capabilities(api_key, options.image_model)

        if not caps.supports_img2img and not options.no_img2img and uploaded_files:
            st.warning(
                f"Model '{image_model}' doesn't support image references (img2img). "
                f"Consider using '{rp.DEFAULT_IMAGE_MODEL}'. "
                "Continuing without image references."
            )
            options.no_img2img = True

        if (
            options.steps is not None
            and caps.allowed_passthrough is not None
            and "steps" not in caps.allowed_passthrough
        ):
            st.info(
                f"Model '{image_model}' doesn't support 'steps'; ignoring."
            )
            options.steps = None

        if (
            options.guidance is not None
            and caps.allowed_passthrough is not None
            and "guidance" not in caps.allowed_passthrough
        ):
            st.info(
                f"Model '{image_model}' doesn't support 'guidance'; ignoring."
            )
            options.guidance = None

        generated_files: list[Path] = []
        total = len(subjects)
        completed = 0

        with st.status("Running pipeline…", expanded=True) as status:
            progress_bar = st.progress(0.0, text="Starting…")

            for event in rp.run_generation(
                api_key, image_paths, subjects, output_dir, options, caps=caps
            ):
                kind = event.kind
                p = event.payload

                if kind == "style_start":
                    progress_bar.progress(0.0, text="Analyzing style…")
                    st.write(f"Analyzing style using {p['model']}…")
                elif kind == "style_done":
                    st.write("Style extracted.")
                    st.markdown(f"> {p['style']}")
                    progress_bar.progress(
                        0.05, text="Style extracted. Starting generation…"
                    )
                elif kind == "style_error":
                    st.error(f"Failed to analyze style image: {p['error']}")
                    progress_bar.empty()
                    status.update(state="error", label="Style analysis failed")
                    return
                elif kind == "queue":
                    total = p["total"]
                    progress_bar.progress(0.05, text=f"Generating {total} images…")
                elif kind == "gen_start":
                    st.write(f"[{p['index']}/{p['total']}] Generating: {p['subject'][:50]}…")
                elif kind == "skip":
                    completed += 1
                    frac = 0.05 + 0.95 * completed / total
                    progress_bar.progress(
                        frac, text=f"[{p['index']}/{p['total']}] Skipped (exists)"
                    )
                    st.write(f"[{p['index']}/{p['total']}] Skipped (already exists)")
                elif kind == "gen_done":
                    completed += 1
                    frac = 0.05 + 0.95 * completed / total
                    progress_bar.progress(
                        frac, text=f"[{p['index']}/{p['total']}] Saved: {p['filename']}"
                    )
                    st.write(f"[{p['index']}/{p['total']}] Saved: {p['filename']}")
                    generated_files.append(Path(p["file_path"]))
                elif kind == "gen_error":
                    completed += 1
                    frac = 0.05 + 0.95 * completed / total
                    progress_bar.progress(
                        frac, text=f"[{p['index']}/{p['total']}] Error"
                    )
                    st.warning(f"[{p['index']}/{p['total']}] Error: {p['error']}")
                elif kind == "done":
                    progress_bar.progress(1.0, text="Done!")
                    status.update(
                        state="complete",
                        label=f"Generated {len(generated_files)} image(s) in '{p['output_dir']}'.",
                    )

        st.session_state["last_results"] = generated_files
        st.session_state["last_output_format"] = output_format
    except Exception as exc:
        st.error(f"Pipeline failed unexpectedly: {exc}")
    finally:
        for ip in image_paths:
            try:
                ip.unlink(missing_ok=True)
            except OSError:
                pass


# ==========================================
# RESULTS (rendered from session state — survives reruns)
# ==========================================
def render_results() -> None:
    results: list[Path] | None = st.session_state.get("last_results")
    if not results:
        return

    output_format: str = st.session_state.get("last_output_format", "png")
    mime = MIME_TYPES.get(output_format, "image/png")

    st.subheader(":material/photo_library: Results")

    cols = st.columns(min(3, len(results)))
    for i, file_path in enumerate(results):
        with cols[i % len(cols)]:
            with st.container(border=True):
                st.image(str(file_path), caption=file_path.name, use_container_width=True)
                with open(file_path, "rb") as f:
                    st.download_button(
                        label=f"Download {file_path.name}",
                        data=f.read(),
                        file_name=file_path.name,
                        mime=mime,
                        key=f"dl_{i}",
                        icon=":material/download:",
                    )

    if len(results) > 1:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for fp in results:
                zf.write(fp, fp.name)
        zip_buffer.seek(0)
        st.download_button(
            label="Download all (.zip)",
            data=zip_buffer,
            file_name="mimicflux_outputs.zip",
            mime="application/zip",
            key="dl_zip",
            icon=":material/download:",
        )


if __name__ == "__main__":
    main()
