"""Audiobook Reader — Gemini TTS-powered text-to-speech for books & documents."""

import streamlit as st
import os, io, wave
from google import genai
from google.genai import types

st.set_page_config(page_title="Audiobook Reader", page_icon="📖", layout="centered")

# ── Voices (all 30 Gemini TTS voices) ───────────────────
VOICES = [
    ("Enceladus", "Breathy"), ("Kore", "Firm"), ("Charon", "Informative"),
    ("Puck", "Upbeat"), ("Zephyr", "Bright"), ("Fenrir", "Excitable"),
    ("Aoede", "Breezy"), ("Leda", "Youthful"), ("Orus", "Firm"),
    ("Callirrhoe", "Easy-going"), ("Umbriel", "Easy-going"),
    ("Algieba", "Smooth"), ("Iapetus", "Clear"), ("Autonoe", "Bright"),
    ("Despina", "Smooth"), ("Erinome", "Clear"), ("Algenib", "Gravelly"),
    ("Rasalgethi", "Informative"), ("Laomedeia", "Upbeat"),
    ("Achernar", "Soft"), ("Alnilam", "Firm"), ("Schedar", "Even"),
    ("Gacrux", "Mature"), ("Pulcherrima", "Forward"),
    ("Achird", "Friendly"), ("Zubenelgenubi", "Casual"),
    ("Vindemiatrix", "Gentle"), ("Sadachbia", "Lively"),
    ("Sadaltager", "Knowledgeable"), ("Sulafat", "Warm"),
]
VOICE_LABELS = [f"{n} — {s}" for n, s in VOICES]
VOICE_NAMES = [n for n, _ in VOICES]

STYLE_PRESETS = {
    "Narrator": "Read clearly in a calm, steady audiobook narrator voice",
    "Storyteller": "Read expressively like an engaging storyteller with natural emotion",
    "Podcast": "Read conversationally like a friendly podcast host",
    "News": "Read in a crisp, formal news anchor style",
    "Whisper": "Read in a soft, intimate whisper",
    "Custom": "",
}


# ── API client ──────────────────────────────────────────
@st.cache_resource
def _init_client(key: str):
    """Create genai client. Tries Vertex AI Express first, then Gemini API."""
    try:
        return genai.Client(vertexai=True, api_key=key)
    except Exception:
        return genai.Client(api_key=key)


def get_client():
    key = None
    for getter in [
        lambda: st.secrets["VERTEX_API_KEY"],
        lambda: st.secrets["GEMINI_API_KEY"],
        lambda: os.environ.get("VERTEX_API_KEY"),
        lambda: os.environ.get("GEMINI_API_KEY"),
    ]:
        try:
            k = getter()
            if k:
                key = k
                break
        except (KeyError, FileNotFoundError):
            continue
    if not key:
        return None
    return _init_client(key)


# ── Audio helpers ───────────────────────────────────────
def pcm_to_wav(pcm: bytes, rate=24000, ch=1, width=2) -> bytes:
    """Wrap raw PCM in a WAV container for st.audio()."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def tts(client, text: str, voice: str, style: str = "") -> bytes:
    """Call Gemini TTS and return PCM audio bytes."""
    prompt = f"{style}:\n\n{text}" if style else text
    r = client.models.generate_content(
        model="gemini-2.5-flash-preview-tts",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice,
                    )
                )
            ),
        ),
    )
    return r.candidates[0].content.parts[0].inline_data.data


def chunk_text(text: str, n: int = 100) -> list[str]:
    """Split text into ~n-word chunks."""
    words = text.split()
    return [" ".join(words[i : i + n]) for i in range(0, len(words), n)] or [""]


# ── Session state defaults ──────────────────────────────
for k, v in {"ch_idx": 0, "ck_idx": 0}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Sidebar ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    voice_i = st.selectbox("Voice", range(len(VOICES)),
                           format_func=lambda i: VOICE_LABELS[i], index=1)
    voice = VOICE_NAMES[voice_i]

    preset = st.selectbox("Style", list(STYLE_PRESETS.keys()))
    style = st.text_input("Style prompt", STYLE_PRESETS[preset]) if preset == "Custom" else STYLE_PRESETS[preset]

    wpc = st.slider("Words / chunk", 50, 200, 100, 10)

    st.divider()
    st.header("📄 Input")
    mode = st.radio("Source", ["Upload File", "Paste Text"], horizontal=True)

    if mode == "Upload File":
        f = st.file_uploader("Upload", type=["epub", "pdf", "docx", "txt", "md", "mobi"])
        if f:
            fkey = f"{f.name}_{f.size}"
            if fkey != st.session_state.get("_fkey"):
                from parsers import parse_file
                try:
                    st.session_state.chapters = parse_file(f, f.name)
                    st.session_state.ch_idx = 0
                    st.session_state.ck_idx = 0
                    st.session_state.pop("audio", None)
                    st.session_state._fkey = fkey
                    st.success(f"✓ {len(st.session_state.chapters)} chapter(s)")
                except Exception as e:
                    st.error(str(e))
    else:
        pasted = st.text_area("Paste text", height=200)
        if pasted and st.button("Load"):
            from parsers import parse_pasted_text
            st.session_state.chapters = parse_pasted_text(pasted)
            st.session_state.ch_idx = 0
            st.session_state.ck_idx = 0
            st.session_state.pop("audio", None)


# ── Main area ───────────────────────────────────────────
st.title("📖 Audiobook Reader")

client = get_client()
if not client:
    st.warning("Set `VERTEX_API_KEY` or `GEMINI_API_KEY` in Streamlit secrets or env vars.")
    st.stop()

if "chapters" not in st.session_state:
    st.info("Upload a file or paste text in the sidebar to begin.")
    st.stop()

chs = st.session_state.chapters

# ── Chapter selector ────────────────────────────────────
def _on_ch_change():
    st.session_state.ck_idx = 0
    st.session_state.pop("audio", None)

st.selectbox("Chapter", range(len(chs)),
             format_func=lambda i: chs[i]["title"],
             key="ch_idx", on_change=_on_ch_change)
ch_i = st.session_state.ch_idx

# AI chapter cleanup (optional, on-demand)
c_clean, c_info = st.columns([1, 3])
with c_clean:
    if st.button("🧹 AI Clean"):
        with st.spinner("Cleaning with Gemini…"):
            raw = chs[ch_i]["text"]
            # limit input to save tokens; process first ~20k chars
            cleaned = client.models.generate_content(
                model="gemini-2.5-flash-preview-04-17",
                contents=(
                    "Clean this book chapter text. Fix formatting, remove "
                    "HTML artifacts, headers, footers, page numbers. "
                    "Return ONLY the cleaned readable text:\n\n" + raw[:20000]
                ),
            ).text
            if len(raw) > 20000:
                cleaned += raw[20000:]
            st.session_state.chapters[ch_i]["text"] = cleaned
            st.session_state.ck_idx = 0
            st.session_state.pop("audio", None)
            st.rerun()
with c_info:
    st.caption("Fix badly formatted text with AI (uses tokens)")

# ── Chunk navigation ────────────────────────────────────
chunks = chunk_text(chs[ch_i]["text"], wpc)
total = len(chunks)
ck_i = min(st.session_state.ck_idx, total - 1)

c1, c2, c3 = st.columns([1, 4, 1])
with c1:
    if st.button("⏮ Prev", disabled=ck_i == 0, use_container_width=True):
        st.session_state.ck_idx = ck_i - 1
        st.session_state.pop("audio", None)
        st.rerun()
with c2:
    new_ck = st.slider("pos", 0, max(total - 1, 0), ck_i, label_visibility="collapsed")
    if new_ck != ck_i:
        st.session_state.ck_idx = new_ck
        st.session_state.pop("audio", None)
        st.rerun()
with c3:
    if st.button("Next ⏭", disabled=ck_i >= total - 1, use_container_width=True):
        st.session_state.ck_idx = ck_i + 1
        st.session_state.pop("audio", None)
        st.rerun()

st.caption(f"Chunk {ck_i + 1} / {total}  ·  {len(chunks[ck_i].split())} words")
st.text_area("chunk_text", chunks[ck_i], height=150, disabled=True,
             label_visibility="collapsed")

# ── Generate & play audio ──────────────────────────────
if st.button("▶  Play", type="primary", use_container_width=True):
    with st.spinner("Generating speech…"):
        try:
            pcm = tts(client, chunks[ck_i], voice, style)
            st.session_state.audio = pcm_to_wav(pcm)
        except Exception as e:
            st.error(f"TTS error: {e}")

if "audio" in st.session_state:
    st.audio(st.session_state.audio, format="audio/wav", autoplay=True)
