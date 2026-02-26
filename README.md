# Audiobook Reader

Gemini TTS-powered text-to-speech reader for books and documents.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## API Key

Set **one** of these:

| Key | Mode |
|-----|------|
| `VERTEX_API_KEY` | Vertex AI Express |
| `GEMINI_API_KEY` | Gemini Developer API |

- **Local**: `.streamlit/secrets.toml` or env var
- **Cloud**: Streamlit Cloud → Settings → Secrets

## Supported Formats

`epub` · `pdf` · `docx` · `txt` · `md` · `mobi`

> mobi needs the `mobi` pip package. If extraction fails, convert to epub via [Calibre](https://calibre-ebook.com).

## How It Works

1. **Upload** file or **paste** text → chapters auto-detected
2. **Select** chapter → text split into ~100-word chunks
3. **Play** → Gemini TTS rolls from the current chunk to chapter end (100-word chunks by default)
4. **Autoplay** starts the generated audio immediately (single click flow)
5. **Control playback** with Apple-style transport buttons: **⏪ 15s / ▶ Play-Pause / 15s ⏩**
6. **Navigate** start position with the chunk slider

## Token Efficiency

- Rolling playback still sends ~100-word chunks to the API (chunk-by-chunk)
- Chapter selection prevents processing the whole book
- AI cleanup is on-demand only (button per chapter)

## AI Cleanup

Badly formatted chapters (broken headings, HTML artifacts, page numbers) can be fixed with the **🧹 AI Clean** button. Uses `gemini-2.5-flash-preview-04-17`, limited to first 20k chars to save cost.

## Voices

30 built-in voices — Kore (Firm), Puck (Upbeat), Enceladus (Breathy), Charon (Informative), etc. Full list in the sidebar dropdown.

## Style Presets

Narrator · Storyteller · Podcast · News · Whisper · Custom

## Stack

| Layer | Tech |
|-------|------|
| UI | Streamlit |
| TTS | `gemini-2.5-flash-preview-tts` (24kHz PCM → WAV) |
| Text cleanup | `gemini-2.5-flash-preview-04-17` (on-demand) |
| epub | ebooklib + BeautifulSoup4 |
| pdf | pdfplumber |
| docx | python-docx |
| mobi | mobi (KindleUnpack) |
