"""Audiobook Reader — Gemini TTS with pipeline chunk processing."""

import streamlit as st
import streamlit.components.v1 as components
import os, json, base64, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from google import genai
from google.genai import types

st.set_page_config(page_title="Audiobook Reader", page_icon="📖", layout="centered")

st.markdown("""<style>
.block-container { padding-top: 1rem !important; padding-bottom: 0 !important; }
</style>""", unsafe_allow_html=True)

# ── Book cache (survives refresh) ───────────────────────
_CACHE = Path(".cache/last_book.json")


def _save_cache(name, chapters):
    _CACHE.parent.mkdir(exist_ok=True)
    _CACHE.write_text(json.dumps({"name": name, "chapters": chapters}))


def _load_cache():
    try:
        d = json.loads(_CACHE.read_text())
        return d if d.get("chapters") else None
    except Exception:
        return None


# ── Voices & styles (Puck first = default) ──────────────
VOICES = [
    ("Puck", "Upbeat"), ("Kore", "Firm"), ("Charon", "Informative"),
    ("Enceladus", "Breathy"), ("Zephyr", "Bright"), ("Fenrir", "Excitable"),
    ("Aoede", "Breezy"), ("Leda", "Youthful"), ("Achernar", "Soft"),
    ("Sulafat", "Warm"), ("Gacrux", "Mature"), ("Schedar", "Even"),
]

STYLES = {
    "Storyteller": "Read expressively like an engaging storyteller with natural emotion",
    "Narrator": "Read clearly in a calm, steady audiobook narrator voice",
    "Podcast": "Read conversationally like a friendly podcast host",
}


# ── API client ──────────────────────────────────────────
@st.cache_resource
def _make_client(key):
    try:
        return genai.Client(vertexai=True, api_key=key)
    except Exception:
        return genai.Client(api_key=key)


def get_client():
    for name in ("VERTEX_API_KEY", "GEMINI_API_KEY"):
        k = None
        try:
            k = st.secrets[name]
        except (KeyError, FileNotFoundError):
            pass
        if not k:
            k = os.environ.get(name)
        if k:
            return _make_client(k)
    return None


# ── TTS ─────────────────────────────────────────────────
def tts(client, text, voice, style):
    """Gemini TTS → raw PCM bytes. Retries up to 3x with backoff."""
    prompt = f"{style}:\n\n{text}" if style else text
    for attempt in range(3):
        try:
            r = client.models.generate_content(
                model="gemini-2.5-flash-preview-tts",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice
                            )
                        )
                    ),
                ),
            )
            # Collect ALL audio parts (API may split across multiple parts)
            audio_parts = []
            for cand in getattr(r, "candidates", []):
                for part in getattr(getattr(cand, "content", None), "parts", []):
                    data = getattr(getattr(part, "inline_data", None), "data", None)
                    if data:
                        audio_parts.append(data)
            if audio_parts:
                return b"".join(audio_parts)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)
    raise RuntimeError("No audio returned after 3 attempts.")


def chunk_text(text, n=100):
    words = text.split()
    return [" ".join(words[i : i + n]) for i in range(0, len(words), n)] or [""]


# ── JS audio player (queue-based, gapless) ──────────────
def init_player():
    components.html("""<script>
    (function() {
        var p = window.parent;
        if (p._player) try { p._player.stop(); } catch(e) {}
        var ctx = new AudioContext({sampleRate: 24000});
        if (ctx.state === 'suspended') ctx.resume();
        p._player = {
            ctx: ctx, nextTime: 0, sources: [],
            addChunk: function(b64) {
                if (this.ctx.state === 'closed') return;
                if (this.ctx.state === 'suspended') this.ctx.resume();
                var bin = atob(b64);
                var u8 = new Uint8Array(bin.length);
                for (var i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
                var i16 = new Int16Array(u8.buffer);
                var f32 = new Float32Array(i16.length);
                for (var j = 0; j < i16.length; j++) f32[j] = i16[j] / 32768.0;
                var buf = this.ctx.createBuffer(1, f32.length, 24000);
                buf.copyToChannel(f32, 0);
                var src = this.ctx.createBufferSource();
                src.buffer = buf;
                src.connect(this.ctx.destination);
                var t = Math.max(this.ctx.currentTime + 0.02, this.nextTime);
                src.start(t);
                this.nextTime = t + buf.duration;
                this.sources.push(src);
            },
            togglePause: function() {
                if (this.ctx.state === 'running') this.ctx.suspend();
                else if (this.ctx.state === 'suspended') this.ctx.resume();
            },
            stop: function() {
                this.sources.forEach(function(s) { try { s.stop(); } catch(e) {} });
                this.sources = [];
                this.nextTime = 0;
                try { this.ctx.close(); } catch(e) {}
            }
        };
    })();
    </script>""", height=0)


_MAX_SEG = 1_200_000  # stay under Streamlit's WebSocket frame limit


def send_audio(pcm, container):
    """Send PCM to JS player. Each segment gets its own iframe in the container
    so nothing is replaced/lost (unlike st.empty which overwrites on each call)."""
    if len(pcm) % 2:
        pcm += b"\x00"
    for off in range(0, len(pcm), _MAX_SEG):
        seg = pcm[off : off + _MAX_SEG]
        b64 = base64.b64encode(seg).decode("ascii")
        with container:
            components.html(
                f'<script>(function(){{ var p=window.parent._player; if(p) p.addChunk("{b64}"); }})();</script>',
                height=0,
            )


def player_action(action):
    js = {
        "pause": "p.togglePause();",
        "stop": "p.stop();",
    }
    components.html(
        f'<script>(function(){{ var p=window.parent._player; if(p) {{ {js[action]} }} }})();</script>',
        height=0,
    )


# ── Session defaults ────────────────────────────────────
if "ch_idx" not in st.session_state:
    st.session_state.ch_idx = 0

# ── Sidebar ─────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    voice_idx = st.selectbox(
        "Voice",
        range(len(VOICES)),
        format_func=lambda i: f"{VOICES[i][0]} — {VOICES[i][1]}",
    )
    voice = VOICES[voice_idx][0]

    style_name = st.selectbox("Style", list(STYLES.keys()))
    style = STYLES[style_name]

    wpc = st.slider("Words / chunk", 50, 200, 100, 10)

    st.divider()
    mode = st.radio("Input", ["Upload", "Paste"], horizontal=True)

    if mode == "Upload":
        f = st.file_uploader(
            "File", type=["epub", "pdf", "docx", "txt", "md", "mobi"]
        )
        if f:
            fkey = f"{f.name}_{f.size}"
            if fkey != st.session_state.get("_fkey"):
                from parsers import parse_file

                try:
                    parsed = parse_file(f, f.name)
                    st.session_state.chapters = parsed
                    st.session_state.ch_idx = 0
                    st.session_state._fkey = fkey
                    _save_cache(f.name, parsed)
                    st.success(f"{len(parsed)} chapter(s)")
                except Exception as e:
                    st.error(str(e))
    else:
        pasted = st.text_area("Paste text", height=150)
        if pasted and st.button("Load"):
            from parsers import parse_pasted_text

            st.session_state.chapters = parse_pasted_text(pasted)
            st.session_state.ch_idx = 0
            _save_cache("Pasted", st.session_state.chapters)

# ── Main ────────────────────────────────────────────────
st.title("📖 Audiobook Reader")

client = get_client()
if not client:
    st.warning("Set `VERTEX_API_KEY` or `GEMINI_API_KEY` environment variable.")
    st.stop()

# Restore cached book on fresh session
if "chapters" not in st.session_state:
    cached = _load_cache()
    if cached:
        st.session_state.chapters = cached["chapters"]
        st.toast(f"Restored: {cached.get('name', 'book')}")
    else:
        st.info("Upload a file or paste text in the sidebar.")
        st.stop()

chs = st.session_state.chapters

st.selectbox(
    "Chapter",
    range(len(chs)),
    format_func=lambda i: chs[i]["title"],
    key="ch_idx",
)

ch = chs[st.session_state.ch_idx]
chunks = chunk_text(ch["text"], wpc)
total = len(chunks)

start = st.slider("Start from chunk", 1, max(total, 1), 1) - 1
st.caption(f"{total} chunks · {len(ch['text'].split())} words")

# ── Player controls ─────────────────────────────────────
c1, c2, c3 = st.columns(3)
with c1:
    play = st.button("▶ Play", type="primary", use_container_width=True)
with c2:
    pause = st.button("⏸ Pause", use_container_width=True)
with c3:
    stop = st.button("⏹ Stop", use_container_width=True)

if pause:
    player_action("pause")
if stop:
    player_action("stop")

progress_bar = st.empty()
transcript = st.empty()

# ── Pipeline: transcribe one chunk ahead, play current ──
if play:
    pending = [c for c in chunks[start:] if c.strip()]
    if not pending:
        st.warning("No text to read.")
    else:
        init_player()
        time.sleep(0.3)  # let browser set up AudioContext
        audio_box = st.container()  # iframes accumulate here (not replaced)
        lines = []
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            fut = executor.submit(tts, client, pending[0], voice, style)
            for i in range(len(pending)):
                try:
                    pcm = fut.result(timeout=90)
                except Exception as e:
                    progress_bar.error(f"Chunk {start + i + 1} failed: {e}")
                    break

                # prefetch next chunk while current one plays
                if i + 1 < len(pending):
                    fut = executor.submit(tts, client, pending[i + 1], voice, style)

                send_audio(pcm, audio_box)

                lines.append(pending[i])
                progress_bar.progress(
                    (i + 1) / len(pending),
                    f"Chunk {start + i + 1} / {total}",
                )
                transcript.text_area(
                    "Transcript",
                    "\n\n".join(lines),
                    height=200,
                    disabled=True,
                )
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        if lines:
            progress_bar.success(f"Done — {len(lines)} chunks queued")

# Keep mobile browser tab alive with silent oscillator
components.html("""<script>
(function() {
    var p = window.parent;
    if (p._keepAlive) return;
    p._keepAlive = true;
    try {
        var ka = new AudioContext();
        var osc = ka.createOscillator();
        var gain = ka.createGain();
        gain.gain.value = 0.0;
        osc.connect(gain);
        gain.connect(ka.destination);
        osc.start();
    } catch(e) {}
})();
</script>""", height=0)
