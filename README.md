# MimicFlux — Automated Image Generation Loop

Give it a photo whose look you like, and a list of things to draw. MimicFlux
figures out the **style** of that photo, then makes a new picture for each
item on your list — in that same style.

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
cp .env.example .env          # then open .env and paste your OpenRouter key
python3 run_pipeline.py
```

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
copy .env.example .env        # then open .env and paste your OpenRouter key
python run_pipeline.py
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

### 5. Add your reference style image

Place one photo whose look you like into the project folder and name it
`my_style.jpg`. MimicFlux analyzes this image and copies its style onto every
generated picture.

> Prefer a different image or filename? You can point to any file with
> `--image your-photo.png` when you run the script.

Your folder should now contain at least these files:

```text
project-folder/
├── .env                 # Your private API key (created in step 3)
├── requirements.txt     # List of packages (used in step 2)
├── prompts.txt          # Your list of things to draw (step 4)
├── my_style.jpg         # The reference photo whose style you want (step 5)
└── run_pipeline.py      # The script itself
```

---

## Running the Script

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

1. **Style extraction** — MimicFlux reads your `my_style.jpg`, sends it to the
   vision model to describe its colors, lighting, and texture, and saves that
   description so later runs skip this step (use `--no-cache` to redo it).
2. **Generation** — it goes through `prompts.txt`, combines each line with the
   style description, and asks the image model to draw it. If a request fails
   temporarily (busy servers, rate limits), it automatically retries a few
   times with pauses in between.
3. **Saving** — each image is saved to `automated_outputs/`, with a record of
   what was made stored in `automated_outputs/manifest.json`.
4. **Resume** — run the script again and it skips images you already have
   (matching the same prompt). Pass `--force` to regenerate everything.
5. **Rate-limit safety** — a small pause (4 seconds by default) between
   generations keeps you under OpenRouter's request limits.

---

## Troubleshooting

| Problem | What it means | How to fix |
| --- | --- | --- |
| `python3: command not found` (macOS) | Python isn't installed, or the Command Line Tools are missing. | Run `xcode-select --install`, then try again. If that doesn't help, install Python from [python.org](https://www.python.org/downloads). |
| `python: command not found` (Windows) | Python isn't on your PATH. | Reinstall Python from [python.org](https://www.python.org/downloads) and tick **"Add python.exe to PATH"** during setup. Close and reopen the terminal. |
| `pip: command not found` | `pip` isn't available directly. | Use `python3 -m pip ...` (macOS/Linux) or `python -m pip ...` (Windows) instead of just `pip`. |
| `ModuleNotFoundError: No module named 'tenacity'` (or `tqdm`, `openai`) | Dependencies aren't installed, or the virtual environment isn't active. | Run the `activate` line from Step 2, then `python3 -m pip install -r requirements.txt` again. |
| `MimicFlux requires Python 3.11 or newer` | Your Python is too old. | Install Python 3.11+ from [python.org](https://www.python.org/downloads) and use that version. |
| `API Key missing! Copy .env.example to .env...` | The `.env` file is missing or empty. | Run `cp .env.example .env` (macOS/Linux) or `copy .env.example .env` (Windows), then paste your key into `.env`. |
| `Missing local style image at: my_style.jpg` | The reference photo is missing. | Put a file named `my_style.jpg` in the project folder, or pass `--image your-photo.png`. |
| `HTTP 401 Unauthorized` | Your API key is wrong or expired. | Re-create the key at [openrouter.ai/keys](https://openrouter.ai/keys) and update `.env`. |
| `HTTP 402 Payment Required` | Your OpenRouter account is out of credits. | Add credits on OpenRouter's **Credits** page. |
| `HTTP 429 Too Many Requests` | You're sending requests too fast. | Re-run (it auto-retries), or increase the pause: `python3 run_pipeline.py --cooldown 8`. |

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
| `--image` | `my_style.jpg` | Reference style image path. |
| `--prompts` | `prompts.txt` | Prompts file (one subject per line; `#` lines ignored). |
| `--output` | `automated_outputs` | Output directory for generated images. |
| `--vision-model` | `openai/gpt-4o-mini` | OpenRouter model for style extraction. |
| `--image-model` | `black-forest-labs/flux.2-pro` | OpenRouter model for image generation. |
| `--cooldown` | `4` | Seconds to wait between generations. |
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
```

---

## Development

Lint and tests use `ruff` and `pytest` (configured in `pyproject.toml`):

```bash
python3 -m pip install ruff pytest   # one-time, inside the virtual environment
ruff check .
pytest -q
```
