# Automated Image Generation Loop

An automated Python pipeline that uses OpenRouter to analyze the visual style of a local sample reference image, extracts its aesthetic baseline, and automatically generates a batch of custom targets in a loop.

This implementation uses **Google Gemini 2.5 Flash** for rapid, low-cost visual style analysis and **Black Forest Labs FLUX 2 Pro** (or similar OpenRouter image engines) to build high-fidelity custom libraries.

---

## Project Workspace Setup

Before launching the pipeline, your directory structure **must** look exactly like this:

```text
project-folder/
├── .env                 # Stores your private API Key safely
├── prompts.txt          # File containing targets (one per line)
├── my_style.jpg         # Your local baseline style image
└── run_pipeline.py      # The primary execution script

```

---

## Step-by-Step Installation

### 1. Clone or Open Your Project Folder

Open your system terminal (Command Prompt/PowerShell on Windows, or Terminal on macOS/Linux) and navigate to your project directory:

```bash
cd path/to/your/project-folder

```

### 2. Install Required Packages

Install the official OpenRouter/OpenAI compatibility wrapper alongside environment management tools:

**Windows:**

```bash
python -m pip install openai requests python-dotenv

```

**macOS / Linux:**

```bash
python3 -m pip install openai requests python-dotenv

```

---

## Configuration Files Setup

### 1. Configure `.env`

Create a file named exactly `.env` in your root folder and add your OpenRouter credential key.

> ⚠️ **Important:** Do *not* use spaces around the `=` sign, and do *not* warp your key in quotes.

```env
OPENROUTER_API_KEY=sk-or-v1-your-actual-key-here

```

### 2. Prepare `prompts.txt`

Create a `prompts.txt` file and write your image subjects. Put each prompt idea on a fresh new line:

```text
A majestic lion sitting on a neon throne
A futuristic motorcycle parked on a rainy cyberpunk street
A mystical wizard towering over an open glowing spellbook
An old vintage typewriter sitting on a rustic wooden desk

```

### 3. Place Your Sample Image

Place your target aesthetic reference image inside the folder and ensure its name matches the `SAMPLE_IMAGE_PATH` string inside your script (Default configuration searches for: `my_style.jpg`).

---

## 💻 Running the Script

Execute the pipeline using your runtime command. The process is completely autonomous:

**Windows:**

```bash
python run_pipeline.py

```

**macOS / Linux:**

```bash
python3 run_pipeline.py

```

### What Happens Behind the Scenes:

1. **Style Extraction:** The script reads your local `my_style.jpg`, encodes it to base64, and prompts the vision model to analyze the layout, mood, lighting, and palette.
2. **Dynamic Queue Processing:** The script loops through `prompts.txt`, merges each subject line smoothly with the extracted anchor style instructions, and pings the image generator.
3. **Automatic Downloads:** Outputs are decoded or extracted and safely stored in a local folder called `automated_outputs/`.
4. **Rate Limit Prevention:** A built-in `time.sleep(4)` (4-second) countdown prevents your pipeline from executing requests too quickly and hitting server blockages.

---

## 🔧 Model Selection & Tweaks

If you need to change models due to changing generation parameters, open `run_pipeline.py` and modify lines **24-25**:

```python
# Recommended choices on OpenRouter
VISION_MODEL = "google/gemini-2.5-flash"       # Super low-cost visual parsing
IMAGE_MODEL = "black-forest-labs/flux.2-pro"   # photorealism generation engine

```
