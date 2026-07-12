# MimicFlux — Automated Image Generation Loop

Give it one or more photos whose look you like, and a list of things to draw.
MimicFlux figures out the **style** of those photos, then makes a new picture
for each item on your list — in that same style.

It uses **OpenAI GPT-4o-mini** to read the style of your photo and
**Black Forest Labs FLUX 2 Pro** to draw the new images, both through
**OpenRouter**.

> ⚠️ **This tool costs money to run.** It calls paid AI services on
> [OpenRouter](https://openrouter.ai). You'll need an OpenRouter account and a
> few cents of credit. Generating a small batch of images typically costs well
> under $1, but you are billed by OpenRouter, not by this tool.

---

## Quick Start (if you already have Python and an OpenRouter key)

Already set up? Run these commands from the project folder.

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Then choose how you want to use MimicFlux:

**🖥️ Browser (easiest):**

```bash
streamlit run web_app.py
```

Your browser opens at `http://localhost:8501` — upload reference image(s),
type prompts, paste your API key in the sidebar, and click **Generate**.

**⌨️ Terminal:**

```bash
cp .env.example .env          # then open .env and paste your OpenRouter key
python3 run_pipeline.py
```

Generated images appear in the `automated_outputs/` folder.

New to all this? Read the **Prerequisites** and **Step-by-Step Setup** below.

---

## Prerequisites

Before you start, you need two things: **Python 3.11 or newer**, and an
**OpenRouter API key**.

### 1. Install Python 3.11 or newer

MimicFlux needs Python version 3.11, 3.12, or newer.

- **Check if you have it** — open a terminal (Terminal on macOS, Command Prompt
  or PowerShell on Windows) and type:

  ```bash
  python3 --version
  ```

  You should see something like `Python 3.11.5` or `Python 3.12.0`. If the
  number starts with `3.11`, `3.12`, or higher, you're ready.

- **If you don't have it, or your version is too old** — download the latest
  Python from [python.org/downloads](https://www.python.org/downloads/) and
  run the installer. Use the **default options** (in particular, on Windows
  leave the box **"Add python.exe to PATH"** checked). Then close and reopen
  your terminal, and check again with `python3 --version`.

- **macOS extra step** — if `python3` says "command not found", install Apple's
  Command Line Tools first:

  ```bash
  xcode-select --install
  ```

  Then run `python3 --version` again.

### 2. Get an OpenRouter API key

MimicFlux talks to AI models through OpenRouter, which is a paid service.

1. Go to [openrouter.ai](https://openrouter.ai) and sign up for a free account.
2. Add some credits so your account can make image requests. Open the
   **Credits** page and add a small amount (a few dollars is plenty to start;
   see OpenRouter's pricing for details).
3. Go to the **Keys** page: [openrouter.ai/keys](https://openrouter.ai/keys).
4. Click **Create Key**, give it any name, and **copy** the key it shows you.
   It looks like `sk-or-v1-...long string of characters...`.

> 🔒 **Keep your key private.** Treat it like a password — never paste it into
> a chat, an email, or a public file. MimicFlux stores it in a local `.env`
> file that the project's `.gitignore` already keeps out of version control.

---

## Step-by-Step Setup

### 1. Open the project folder in a terminal

If you downloaded or cloned this project, open a terminal and move into the
project folder (the one that contains `run_pipeline.py`):

```bash
cd path/to/your/project-folder
```

### 2. Create a virtual environment and install the dependencies

A virtual environment keeps MimicFlux's packages separate from the rest of
your computer. Create it once and install everything it needs:

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

> 💡 **You must re-run the `activate` line every time you reopen the terminal.**
> When the environment is active, your prompt shows `(.venv)` at the start. If
> you close the terminal and come back later, just `cd` into the project folder
> and run the `activate` line again before running the script.

---

## Running in the Browser (Easiest)

If you followed Steps 1–2 above, you can launch the browser UI — no need to
edit files or use the terminal for anything else. The browser lets you upload
your reference image(s), type your prompts, paste your API key, and download
results, all with buttons and forms.

**macOS / Linux:**

```bash
streamlit run web_app.py
```

**Windows:**

```powershell
streamlit run web_app.py
```

Your browser opens automatically at `http://localhost:8501`. If it doesn't,
open that address manually.

### Using the browser app

1. **API key** — If you already have a `.env` file with your key, it loads
   automatically. Otherwise, paste your OpenRouter key into the sidebar field
   and click **Save key** (it's saved to `.env` so you only do this once).
2. **Upload image(s)** — Click the upload area and pick one or more reference
   photos (up to 5). The style of all images is analyzed together in a single
   vision call.
3. **Prompts** — Type your subjects in the text box, one per line. Lines
   starting with `#` are ignored.
4. **Options** (sidebar) — Adjust models, cooldown, or check "Force
   regenerate" / "Ignore style cache" if needed. You can also fine-tune
   generation quality with **Steps** and **Guidance** sliders, set a fixed
   **Seed** for reproducibility, or uncheck **Image-to-image** to rely on text
   style only. Defaults work fine for most people.
5. **Generate** — Click the **🚀 Generate** button. A progress bar shows each
   image as it's created.
6. **Download** — Finished images appear below with individual download
   buttons, plus a **Download all (.zip)** button for the whole batch.

> 💡 The browser app and the terminal script share the same `automated_outputs/`
> folder and `manifest.json`, so images generated in either mode are visible to
> both.

---

## Using the Terminal (Alternative)

Prefer the command line? The steps below set up the files the terminal script
needs. Skip this section if you only use the browser app.

### 3. Add your OpenRouter key

Copy the provided sample file to create your own private settings file:

**macOS / Linux:**

```bash
cp .env.example .env
```

**Windows:**

```powershell
copy .env.example .env
```

Then open the `.env` file in any text editor and replace the placeholder with
your real key, so the line looks like:

```env
OPENROUTER_API_KEY=sk-or-v1-your-actual-key-here
```

> ⚠️ **Important:** Do not put spaces around the `=` sign, and do not wrap
> your key in quotes. Just `OPENROUTER_API_KEY=` followed immediately by the
> key.

### 4. Add your prompts

Open `prompts.txt` in a text editor and list the things you want drawn, one per
line. For example:

```text
A photorealistic faceted Amethyst
A vintage brass pocket watch on a wooden desk
A misty mountain at sunrise
```

Lines starting with `#` are ignored, so you can leave yourself notes. Don't
worry if `prompts.txt` doesn't exist yet — on the first run MimicFlux creates
a starter file for you to edit.

### 5. Add your reference style image(s)

Place one or more photos whose look you like into the project folder. By
default, MimicFlux looks for a single file named `my_style.jpg`, analyzes it,
and copies its style onto every generated picture.

You can also pass **multiple** reference images (up to 5) to blend their
styles together:

```bash
python3 run_pipeline.py --image photo1.jpg photo2.jpg photo3.jpg
```

> Prefer a different image or filename? You can point to any file with
> `--image your-photo.png` when you run the script.

Your folder should now contain at least these files:

```text
project-folder/
├── .env                 # Your private API key (created in step 3, or via the browser)
├── requirements.txt     # List of packages (used in step 2)
├── prompts.txt          # Your list of things to draw (step 4, or typed in the browser)
├── my_style.jpg         # The reference photo whose style you want (step 5, or uploaded in the browser)
├── run_pipeline.py      # The terminal script
└── web_app.py           # The browser app
```

---

## Running in the Terminal

With the virtual environment active (you see `(.venv)` in your prompt), run:

**macOS / Linux:**

```bash
python3 run_pipeline.py
```

**Windows:**

```powershell
python run_pipeline.py
```

A progress bar appears as each image is generated. When it finishes, your new
pictures are in the `automated_outputs/` folder, named after each prompt
(for example, `01_amethyst.png`, `02_lapis-lazuli.png`).

### What happens behind the scenes

1. **Style extraction** — MimicFlux reads your reference image(s), sends them
   to the vision model in a single call to describe their shared colors,
   lighting, and texture, and saves that description so later runs skip this
   step (use `--no-cache` to redo it). For noticeably better style reading,
   switch the vision model to `openai/gpt-4o` with `--vision-model`.
2. **Prompt building** — each subject from `prompts.txt` is combined with the
   extracted style into a single, detailed prompt ending with
   *"Highly detailed, sharp focus, professional quality."* Use `--dry-run` to
   preview these without generating anything.
3. **Image-to-image reference** (img2img) — your reference photos are also sent
   directly to FLUX 2 Pro as `input_references`, so the model can match their
   composition and texture, not just the text description. Pass `--no-img2img`
   to disable this and rely on the text style alone.
4. **Generation** — FLUX 2 Pro draws each image. You can fine-tune quality and
   fidelity with `--steps` (more refinement), `--guidance` (prompt adherence),
   and `--seed` (reproducibility). If a request fails temporarily (busy
   servers, rate limits), it automatically retries a few times with pauses in
   between.
5. **Model capability detection** — before generating, MimicFlux queries
   OpenRouter for the image model's supported features. If the model doesn't
   support image references (img2img), you're warned and prompted to continue
   text-only (the web app auto-falls back). Unsupported `--steps`/`--guidance`
   are silently dropped. If the capability query fails, MimicFlux assumes all
   features are available (optimistic defaults).
6. **Saving** — each image is saved to `automated_outputs/`, with a record of
   what was made stored in `automated_outputs/manifest.json`.
7. **Resume** — run the script again and it skips images you already have
   (matching the same prompt). Pass `--force` to regenerate everything.
8. **Rate-limit safety** — a small pause (4 seconds by default) between
   generations keeps you under OpenRouter's request limits.

---

## Troubleshooting

| Problem | What it means | How to fix |
| --- | --- | --- |
| `python3: command not found` (macOS) | Python isn't installed, or the Command Line Tools are missing. | Run `xcode-select --install`, then try again. If that doesn't help, install Python from [python.org](https://www.python.org/downloads). |
| `python: command not found` (Windows) | Python isn't on your PATH. | Reinstall Python from [python.org](https://www.python.org/downloads) and tick **"Add python.exe to PATH"** during setup. Close and reopen the terminal. |
| `pip: command not found` | `pip` isn't available directly. | Use `python3 -m pip ...` (macOS/Linux) or `python -m pip ...` (Windows) instead of just `pip`. |
| `ModuleNotFoundError: No module named 'tenacity'` (or `tqdm`, `openai`) | Dependencies aren't installed, or the virtual environment isn't active. | Run the `activate` line from Step 2, then `python3 -m pip install -r requirements.txt` again. |
| `ModuleNotFoundError: No module named 'streamlit'` | Streamlit isn't installed. | Run `python3 -m pip install -r requirements.txt` (it includes Streamlit). |
| `MimicFlux requires Python 3.11 or newer` | Your Python is too old. | Install Python 3.11+ from [python.org](https://www.python.org/downloads) and use that version. |
| `API Key missing! Copy .env.example to .env...` | The `.env` file is missing or empty. | Run `cp .env.example .env` (macOS/Linux) or `copy .env.example .env` (Windows), then paste your key into `.env`. |
| `Missing local style image at: my_style.jpg` | The reference photo is missing. | Put a file named `my_style.jpg` in the project folder, or pass `--image your-photo.png` (you can pass multiple: `--image a.jpg b.jpg`). |
| `HTTP 401 Unauthorized` | Your API key is wrong or expired. | Re-create the key at [openrouter.ai/keys](https://openrouter.ai/keys) and update `.env`. |
| `HTTP 402 Payment Required` | Your OpenRouter account is out of credits. | Add credits on OpenRouter's **Credits** page. |
| `HTTP 429 Too Many Requests` | You're sending requests too fast. | Re-run (it auto-retries), or increase the pause: `python3 run_pipeline.py --cooldown 8`. |
| `Model doesn't support image references` | You're using an image model without img2img capability. | Switch to `--image-model black-forest-labs/flux.2-pro` (supports img2img), or continue text-only with `--no-img2img`. |

Still stuck? Run with `--verbose` for detailed logs that can help identify the
problem:

```bash
python3 run_pipeline.py --verbose
```

---

## Command-Line Options

All flags are optional. Running `python3 run_pipeline.py` with no arguments
uses the defaults below.

| Flag | Default | Description |
| --- | --- | --- |
| `--image` | `my_style.jpg` | Reference style image path(s). Accepts up to 5 images; styles are blended in a single vision call. |
| `--prompts` | `prompts.txt` | Prompts file (one subject per line; `#` lines ignored). |
| `--output` | `automated_outputs` | Output directory for generated images. |
| `--vision-model` | `openai/gpt-4o-mini` | OpenRouter model for style extraction. For best results, upgrade to `openai/gpt-4o` — it reads visual style with far greater accuracy. |
| `--image-model` | `black-forest-labs/flux.2-pro` | OpenRouter model for image generation. |
| `--cooldown` | `4` | Seconds to wait between generations. |
| `--steps` | model default | FLUX inference steps (1–100). Higher = more refined but slower and costlier. Omit to let the model choose. |
| `--guidance` | model default | FLUX guidance scale (0–20). Higher = follows the prompt more literally; lower = more creative freedom. |
| `--seed` | random | Fixed seed for reproducible generation. Omit for a random result each time. |
| `--no-img2img` | off | Skip sending reference images to the image model. Only the extracted text style is used. |
| `--dry-run` | off | Print the merged prompts and exit without calling the image API. |
| `--no-cache` | off | Ignore the cached style and re-run the vision step. |
| `--legacy-names` | off | Use `generation_N.png` filenames instead of descriptive ones. |
| `--force` | off | Regenerate even if a matching output already exists. |
| `--verbose` | off | Enable debug logging with timestamps. |

**Examples:**

```bash
# Preview merged prompts without spending image-generation credits
python3 run_pipeline.py --dry-run

# Use a faster, cheaper image model with a 6-second pause between images
python3 run_pipeline.py --image-model black-forest-labs/flux.1-schnell --cooldown 6

# Regenerate everything, using the original generation_N.png filenames
python3 run_pipeline.py --force --legacy-names

# Blend styles from multiple reference images (up to 5)
python3 run_pipeline.py --image portrait.jpg landscape.jpg

# High-quality run with more steps, tighter guidance, and a fixed seed
python3 run_pipeline.py --steps 50 --guidance 3.5 --seed 42

# Use the stronger vision model for better style extraction
python3 run_pipeline.py --vision-model openai/gpt-4o

# Text-style only — skip img2img, use just the extracted style description
python3 run_pipeline.py --no-img2img
```

---

## Development

Lint and tests use `ruff` and `pytest` (configured in `pyproject.toml`):

```bash
python3 -m pip install ruff pytest   # one-time, inside the virtual environment
ruff check .
pytest -q
```
