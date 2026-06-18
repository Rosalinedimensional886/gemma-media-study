# Gemma Media Study

> Turn any YouTube link, local video, or audio file into a study pack — verbatim
> subtitles, an AI summary, a frame-by-frame visual breakdown, and a built-in chat
> tutor — all running **100% locally** on [Ollama](https://ollama.com) + Gemma 4.

A [Tauri](https://tauri.app) desktop app (Rust + React) with a Python media pipeline.
Nothing leaves your machine: the model runs in Ollama, transcription runs in
[faster-whisper](https://github.com/SYSTRAN/faster-whisper), and downloads use
[yt-dlp](https://github.com/yt-dlp/yt-dlp).

## Demo

A real run of the app:

https://github.com/johunsang/gemma-media-study/raw/main/docs/demo.mp4

> Prefer to download it? Grab [`docs/demo.mp4`](docs/demo.mp4) directly.

---

## Features

- 🎬 **Any source** — YouTube URLs, local video files, or local audio files. Drag &
  drop them in, or paste one per line.
- 📝 **Verbatim subtitles** — uses the video's own captions when present; otherwise
  transcribes the speech word-for-word with Whisper `large-v3` (any language,
  auto-detected). Tiny auto-caption fragments are merged into scene-sized lines.
- 🧠 **AI analysis** — Gemma 4 summarizes the topic, key points, and study insights,
  using both the transcript and sampled video frames as visual context.
- 🖼️ **Frame breakdown** — each sampled frame is summarized by *content* (slide titles,
  bullets, code, equations) rather than described as an image.
- 🌐 **Output language** — write the analysis in the content's own language (auto) or
  force a target language. Subtitles always stay verbatim in the original.
- 🔁 **Loop practice** — click any subtitle line to loop that sentence; toggle
  loop/once and 0.75×/1×/1.25× speed for listening practice.
- 📚 **Study library** — every result is saved and organized into groups you can drag
  between, rename, and delete.
- ⏩ **Batch queue** — feed it a whole list at once and keep adding links while it runs.
- 💬 **Gemma chat** — a chat tab backed by any OpenAI-compatible local endpoint, with
  the current analysis and a subtitle excerpt injected as context.
- 💾 **Markdown out** — analysis, study notes, and the frame breakdown are written as
  `.md` files next to the media, ready to drop into your notes.

## Why it's lightweight

The heavy model runs in **Ollama**, not in this app's Python environment. Ollama handles
GPU offload and quantization automatically, so `gemma4:12b` (a ~7.6 GB Q4 build) runs
across modest GPUs without any `torch` / `transformers` / `bitsandbytes` setup.

- **`ffmpeg` is not required** — downloads use already-merged streams, and frame/audio
  extraction uses [PyAV](https://github.com/PyAV-Org/PyAV)'s bundled libav.
- **No PyTorch** — the Python venv only does media work; ASR uses CTranslate2
  (faster-whisper), and the LLM is reached over HTTP.

## How it works

```
                ┌──────────────────────────────────────────────┐
  YouTube URL   │  Tauri (Rust)                                 │
  Local file ──►│   • spawns the Python analyzer                │
                │   • stores results in downloads/library.json  │
                │   • proxies chat to a local LLM endpoint       │
                └───────────────┬──────────────────────────────┘
                                │ CLI args  ▲ summary JSON
                                ▼           │
                ┌──────────────────────────────────────────────┐
                │  analyze_youtube_gemma.py                      │
                │   yt-dlp ─► download                           │
                │   PyAV   ─► sample frames + audio              │
                │   subtitles: captions ─► Whisper ─► Gemma audio│
                │   Ollama /api/chat ─► analysis + frame notes   │
                └───────────────┬──────────────────────────────┘
                                ▼  HTTP
                       ┌─────────────────┐
                       │ Ollama gemma4:12b│
                       └─────────────────┘
```

The frontend (`src/main.tsx`) talks to Rust commands in `src-tauri/src/main.rs`, which
shell out to `analyze_youtube_gemma.py`. The Python script writes a summary JSON that
Rust folds into the study library.

## Requirements

- **[Ollama](https://ollama.com)** installed and running, with a model pulled:
  ```powershell
  ollama pull gemma4:12b
  ```
  `gemma4:12b` is text + image (used for analysis and frame summaries). You can pick any
  installed tag from the app's model dropdown.
- **Python 3.10+**, **Node.js 18+**, and **Rust / Cargo** (to build the Tauri app).
- An **NVIDIA GPU** is recommended for Whisper (it falls back to CPU automatically).
- Platform: **Windows** is the primary target (the “Open Folder” action uses Explorer).

## Getting started

```powershell
npm install
npm run tauri:dev
```

On first launch, click **First-Time Setup**. It creates `.venv`, installs the light
Python dependencies (`yt-dlp`, `PyAV`, `pillow`, `requests`, `faster-whisper`), and runs
`ollama pull` for the selected model.

Then paste one or more YouTube URLs / local file paths (one per line) — or drag & drop
files — choose your model and options, and click **Start Analysis**. Each item is
processed in turn and added to the Study Library on the left.

### Production build

```powershell
npm run tauri:build
```

## Command-line usage

The pipeline also runs standalone, without the desktop app:

```powershell
# Analyze a YouTube video (default model: gemma4:12b)
python .\analyze_youtube_gemma.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Use a different Ollama model or endpoint
python .\analyze_youtube_gemma.py "URL" --model gemma4:latest --ollama-endpoint http://127.0.0.1:11434

# Analyze a local video or audio file
python .\analyze_youtube_gemma.py --local-media "C:\Videos\lecture.mp4"
python .\analyze_youtube_gemma.py --local-media "C:\Audio\lesson.mp3"

# Force the analysis language and control the frame breakdown (0 disables frames)
python .\analyze_youtube_gemma.py "URL" --output-language Korean --frame-count 16
```

Run `python .\analyze_youtube_gemma.py --help` for the full option list.

### Key options

| Option | Default | Description |
|---|---|---|
| `--model` | `gemma4:12b` | Ollama model tag used for analysis and frame notes. |
| `--ollama-endpoint` | `http://127.0.0.1:11434` | Local Ollama server. |
| `--output-language` | `auto` | `auto` matches the content; or force e.g. `Korean`. |
| `--frame-count` | `32` | Frames sampled for the breakdown (`0` to disable). |
| `--analysis-frames` | `6` | Frames attached as images to the main analysis call. |
| `--max-height` | `480` | Max video height to download. |
| `--max-new-tokens` | `1024` | Generation length for the analysis. |
| `--local-media` | — | Analyze a local file instead of downloading. |

## Output files

Each analyzed video gets its own folder under `downloads/`, containing:

| File | Contents |
|---|---|
| `<name>.analysis.md` | The AI analysis. |
| `<name>.study.md` | Analysis + full subtitle list. |
| `<name>.frames.md` | Frame-by-frame content breakdown. |
| `<name>_frames/` | The extracted JPEG frames. |
| `downloads/library.json` | The combined study-library index used by the app. |

## Gemma chat

The **Gemma Chat** tab connects to any OpenAI-compatible local endpoint. Default:

```text
http://127.0.0.1:11434/v1/chat/completions
```

Use **Check Ollama Models** to populate the dropdown from your local Ollama. When a media
analysis is loaded, your questions automatically include the current analysis and a
subtitle excerpt as context.

## Troubleshooting

- **Gemma E2B / E4B crashes with `GGML_ASSERT(n_inputs < GGML_SCHED_MAX_SPLIT_INPUTS)`** —
  the encoder-based E-series models can crash on multi-GPU setups
  ([Ollama #16506](https://github.com/ollama/ollama/issues/16506)). Use the unified
  **`gemma4:12b`** instead.
- **Whisper runs out of VRAM next to Gemma** — the app unloads the Ollama model
  (`keep_alive: 0`) before loading Whisper, then reloads it. On a small GPU the two
  together can still overflow; lower the Whisper model size or use captions when possible.
- **“event.listen not allowed” / drag-drop does nothing** — drag-drop needs
  `src-tauri/capabilities/default.json` to include `core:default`.
- **Analysis fails with a Python / module error** — run **First-Time Setup** so the
  `.venv` and its dependencies exist.

## Releasing

Versioning is kept in sync across `package.json`, `src-tauri/tauri.conf.json`, and
`src-tauri/Cargo.toml` by a single script:

```powershell
npm run release patch      # 0.1.0 -> 0.1.1  (also: minor, major, or an exact x.y.z)
npm run release 0.2.0 --dry-run   # preview the changes without touching git
```

The script bumps all three files, commits, tags `v<version>`, and pushes the tag. That
triggers **`.github/workflows/build.yml`**, which builds the unsigned Windows installers
(`.msi` + NSIS `.exe`) and publishes them to a GitHub Release. You can also start a build
manually from the repo's **Actions** tab.

> Installers are **unsigned** (no code-signing certificate / notarization), so Windows
> SmartScreen warns on first run — choose *More info → Run anyway*.

### Auto-update

The app has a built-in updater. On launch it checks the latest GitHub Release; when a newer
version is found, a banner offers **Update & Restart**, which downloads, verifies, and
installs it in place. Updates are verified with a minisign key (the public key is baked
into the app; the private key lives only in GitHub Actions secrets). This update signature
is separate from OS code-signing — the installers themselves remain unsigned.

> Auto-update only works for builds that already ship the updater (v0.1.1 and later).

## Tech stack

- **Desktop shell:** Tauri 2 (Rust)
- **Frontend:** React 19 + Vite + TypeScript, [lucide-react](https://lucide.dev) icons
- **Pipeline:** Python — yt-dlp, PyAV, Pillow, requests, faster-whisper
- **Model runtime:** Ollama (Gemma 4)

## Legal & licensing

- Only download or analyze content you have the right to use.
- **Gemma 4** is distributed under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms)
  (Apache-2.0-style, not gated).
- This project does not yet ship a `LICENSE` file — add one before redistributing.
