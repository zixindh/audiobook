# Audiobook Reader (Snap Note)

One-click Gemini TTS for long documents.

## Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Set `VERTEX_API_KEY` in env vars or `.streamlit/secrets.toml` (Vertex AI Express).

## Use

1. Upload a file or paste text.
2. Pick chapter + start chunk (default 100 words/chunk).
3. Tap **▶ Play / Pause** once to generate rolling audio to chapter end.
4. Use **⏪ 15s** and **15s ⏩** for quick seek.

## Notes

- Rolling playback is chunked (cost-safe for long books).
- **🧹 AI Clean** is optional, on-demand, and only cleans first ~20k chars for cost control.
- Formats: `epub`, `pdf`, `docx`, `txt`, `md`, `mobi`.
- If `.mobi` extraction fails, convert to `.epub` (Calibre works well).
