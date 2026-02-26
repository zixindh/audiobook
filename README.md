# Audiobook Reader

Turn any book or document into a streaming audiobook using Google Gemini TTS. Audio begins playing within seconds — no waiting for the full chapter to process.

## Features

- **Streaming playback** — audio starts after the first 100-word chunk; remaining chunks generate in the background via parallel API calls.
- **Live transcript** — text grows in real time as each chunk is processed.
- **30 Gemini TTS voices** with style presets (Narrator, Storyteller, Podcast, News, Whisper, or custom prompts).
- **6 file formats** — `.epub`, `.pdf`, `.docx`, `.txt`, `.md`, `.mobi`.
- **Chapter detection** — automatically splits books on chapter/part/section headings.
- **Mobile-optimized** — compact layout, controls stay in one row, works on phone screens.
- **Screen-lock safe** — silent audio keep-alive + 24h WebSocket timeout prevents disconnects during long listening sessions.
- **Session persistence** — parsed book is cached to disk; refreshing the page restores your last upload instantly.
- **AI Clean** — optional one-click chapter cleanup via Gemini (removes formatting artifacts, headers, page numbers).
- **Player controls** — play/pause, rewind 15s, forward 15s, chunk navigation slider.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

### API Key

This app uses **Vertex AI Express**. Set your key in one of:

| Method | Location |
|---|---|
| Streamlit secrets | `.streamlit/secrets.toml` → `VERTEX_API_KEY = "your-key"` |
| Environment variable | `export VERTEX_API_KEY=your-key` |
| Streamlit Cloud | Dashboard → Secrets |

A template is provided in `.streamlit/secrets.toml.example`.

## Usage

1. **Upload** a file or paste text in the sidebar.
2. **Pick a chapter** and adjust the start chunk if needed.
3. **Tap Play** — the first chunk generates in ~2-3 seconds, then audio starts immediately. The rest streams in the background.
4. Use **⏪ 15s** / **15s ⏩** to seek, or tap **Play** again to pause/resume.

### Sidebar Settings

| Setting | Description |
|---|---|
| **Voice** | Choose from 30 Gemini TTS voices (each with a tonal style). |
| **Style** | Preset prompts that shape delivery (or write your own with Custom). |
| **Words / chunk** | How many words per TTS call (50–200). Lower = faster first audio, higher = fewer API calls. |

## Architecture

```
Upload/Paste → parse chapters → chunk text (N words each)
                                        ↓
                            ThreadPoolExecutor (3 workers)
                            parallel Gemini TTS API calls
                                        ↓
                            base64 PCM → components.html()
                                        ↓
                            Web Audio API (AudioContext)
                            gapless scheduled playback
```

- **AudioContext streaming** — each PCM chunk is decoded to Float32, wrapped in an AudioBuffer, and scheduled to start right when the previous one ends. No gaps, no restarts.
- **Parallel generation** — 3 concurrent TTS workers overlap API latency. Results are consumed in order so audio stays sequential.
- **Iframe recycling** — a single `st.empty()` placeholder is overwritten on each chunk instead of stacking iframes.
- **Keep-alive** — a silent oscillator (gain=0) prevents mobile browsers from suspending the tab. A `visibilitychange` listener auto-reloads if the WebSocket dropped.
- **Disk cache** — parsed chapters are saved to `.cache/last_book.json` and auto-restored on page refresh.

## File Format Notes

| Format | Notes |
|---|---|
| `.epub` | Best support. ToC-aware chapter splitting. |
| `.pdf` | Extracted via pdfplumber. Quality depends on PDF structure. |
| `.docx` | Splits on Heading styles. |
| `.txt` / `.md` | Splits on `Chapter`, `Part`, `Section`, or `#` headings. |
| `.mobi` | Extracts via the `mobi` package. If it fails, convert to `.epub` with [Calibre](https://calibre-ebook.com/). |

## Requirements

- Python 3.10+
- A Vertex AI Express API key with access to `gemini-2.5-flash-preview-tts`
- Dependencies listed in `requirements.txt`
