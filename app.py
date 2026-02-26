"""Audiobook Reader — Gemini TTS-powered text-to-speech for books & documents."""

import streamlit as st
import streamlit.components.v1 as components
import os, base64
from concurrent.futures import ThreadPoolExecutor
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
def _extract_audio_from_response(resp) -> bytes | None:
    """Safely extract inline PCM from a Gemini response."""
    for cand in getattr(resp, "candidates", None) or []:
        content = getattr(cand, "content", None)
        for part in getattr(content, "parts", None) or []:
            inline = getattr(part, "inline_data", None)
            data = getattr(inline, "data", None)
            if data:
                return data
    return None


def _tts_response_error(resp) -> str:
    """Build a useful error when TTS returns no audio payload."""
    details = []
    feedback = getattr(resp, "prompt_feedback", None)
    if feedback:
        reason = getattr(feedback, "block_reason", None)
        reason_msg = getattr(feedback, "block_reason_message", None)
        if reason:
            details.append(f"block_reason={reason}")
        if reason_msg:
            details.append(str(reason_msg))
    for cand in (getattr(resp, "candidates", None) or [])[:1]:
        finish = getattr(cand, "finish_reason", None)
        if finish:
            details.append(f"finish_reason={finish}")
    text = getattr(resp, "text", None)
    if text:
        details.append(f"text={str(text)[:180]}")
    suffix = f" ({'; '.join(details)})" if details else ""
    return "No audio returned by TTS model. Try again or lower words/chunk." + suffix


def tts(client, text: str, voice: str, style: str = "") -> bytes:
    """Call Gemini TTS and return PCM audio bytes."""
    prompt = f"{style}:\n\n{text}" if style else text
    last_error = "No audio returned by TTS model."
    # Retry once because the TTS endpoint can intermittently return empty candidates.
    for _ in range(2):
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
        pcm = _extract_audio_from_response(r)
        if pcm:
            return pcm
        last_error = _tts_response_error(r)
    raise RuntimeError(last_error)


def chunk_text(text: str, n: int = 100) -> list[str]:
    """Split text into ~n-word chunks."""
    words = text.split()
    return [" ".join(words[i : i + n]) for i in range(0, len(words), n)] or [""]


def clear_playback():
    """Clear cached audio and streaming state."""
    for key in ("_streaming_active", "_transcript"):
        st.session_state.pop(key, None)
    st.session_state._stop_streamer = True


def init_audio_streamer():
    """Set up a Web Audio API streamer on the parent window for gapless chunk playback."""
    components.html("""
    <script>
    (function() {
        const p = window.parent;
        if (p._streamer) { try { p._streamer.stop(); } catch(e) {} }

        const ctx = new AudioContext({sampleRate: 24000});
        if (ctx.state === 'suspended') ctx.resume();

        p._streamer = {
            ctx: ctx,
            nextTime: 0,
            allPcm: [],
            totalSamples: 0,
            activeSources: [],
            playStartCtx: 0,
            playStartOffset: 0,
            _pausedSample: 0,

            addChunk: function(b64) {
                if (ctx.state === 'suspended') ctx.resume();
                var bin = atob(b64);
                var u8 = new Uint8Array(bin.length);
                for (var i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
                var i16 = new Int16Array(u8.buffer);
                var f32 = new Float32Array(i16.length);
                for (var i = 0; i < i16.length; i++) f32[i] = i16[i] / 32768.0;

                this.allPcm.push(f32);
                this.totalSamples += f32.length;

                var buf = ctx.createBuffer(1, f32.length, 24000);
                buf.copyToChannel(f32, 0);
                var src = ctx.createBufferSource();
                src.buffer = buf;
                src.connect(ctx.destination);
                var t = Math.max(ctx.currentTime + 0.02, this.nextTime);
                src.start(t);
                this.nextTime = t + buf.duration;
                this.activeSources.push(src);
                if (this.allPcm.length === 1) {
                    this.playStartCtx = t;
                    this.playStartOffset = 0;
                }
            },

            currentSample: function() {
                if (ctx.state === 'suspended') return this._pausedSample || 0;
                var elapsed = ctx.currentTime - this.playStartCtx;
                return Math.min(
                    this.playStartOffset + Math.floor(elapsed * 24000),
                    this.totalSamples
                );
            },

            /* Stop current sources and replay from a specific sample offset. */
            _rebuildAndPlay: function(fromSample) {
                var self = this;
                this.activeSources.forEach(function(s) { try { s.stop(); } catch(e) {} });
                this.activeSources = [];
                fromSample = Math.max(0, Math.min(fromSample, this.totalSamples - 1));
                var remaining = this.totalSamples - fromSample;
                if (remaining <= 0) return;

                var allData = new Float32Array(remaining);
                var writePos = 0, soFar = 0;
                for (var ci = 0; ci < this.allPcm.length; ci++) {
                    var chunk = this.allPcm[ci];
                    var chunkEnd = soFar + chunk.length;
                    if (chunkEnd > fromSample) {
                        var start = Math.max(0, fromSample - soFar);
                        var sub = chunk.subarray(start);
                        allData.set(sub, writePos);
                        writePos += sub.length;
                    }
                    soFar = chunkEnd;
                }
                var buf = ctx.createBuffer(1, remaining, 24000);
                buf.copyToChannel(allData, 0);
                var src = ctx.createBufferSource();
                src.buffer = buf;
                src.connect(ctx.destination);
                if (ctx.state === 'suspended') {
                    ctx.resume().then(function() {
                        src.start();
                        self.playStartCtx = ctx.currentTime;
                        self.playStartOffset = fromSample;
                        self.nextTime = ctx.currentTime + buf.duration;
                    });
                } else {
                    src.start();
                    self.playStartCtx = ctx.currentTime;
                    self.playStartOffset = fromSample;
                    self.nextTime = ctx.currentTime + buf.duration;
                }
                this.activeSources = [src];
            },

            togglePause: function() {
                if (ctx.state === 'running') {
                    this._pausedSample = this.currentSample();
                    ctx.suspend();
                } else if (ctx.state === 'suspended') {
                    ctx.resume();
                    this.playStartCtx = ctx.currentTime;
                    this.playStartOffset = this._pausedSample || 0;
                }
            },

            seekRelative: function(seconds) {
                var cur = this.currentSample();
                this._rebuildAndPlay(cur + Math.floor(seconds * 24000));
            },

            stop: function() {
                this.activeSources.forEach(function(s) { try { s.stop(); } catch(e) {} });
                this.activeSources = [];
                this.allPcm = [];
                this.totalSamples = 0;
                try { ctx.close(); } catch(e) {}
            }
        };
    })();
    </script>
    """, height=0)


_MAX_PCM_PER_MSG = 500 * 1024  # keep each WebSocket message well under Streamlit's buffer


def send_audio_chunk(pcm: bytes):
    """Send PCM to the JavaScript streamer, splitting large payloads to avoid WS errors."""
    for off in range(0, len(pcm), _MAX_PCM_PER_MSG):
        seg = pcm[off : off + _MAX_PCM_PER_MSG]
        if len(seg) % 2:
            seg += b"\x00"  # Int16 alignment
        b64 = base64.b64encode(seg).decode("ascii")
        components.html(f"""
        <script>
        (function() {{
            var s = window.parent._streamer;
            if (s) s.addChunk("{b64}");
        }})();
        </script>
        """, height=0)


def streamer_action(action: str):
    """Send a control command (toggle/rewind15/forward15/stop) to the audio streamer."""
    components.html(f"""
    <script>
    (function() {{
        var s = window.parent._streamer;
        if (!s) return;
        if ("{action}" === "toggle") s.togglePause();
        else if ("{action}" === "rewind15") s.seekRelative(-15);
        else if ("{action}" === "forward15") s.seekRelative(15);
        else if ("{action}" === "stop") s.stop();
    }})();
    </script>
    """, height=0)


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
                    clear_playback()
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
            clear_playback()


# ── Main area ───────────────────────────────────────────
st.title("📖 Audiobook Reader")

# Keep the browser tab alive when the phone screen locks.
# A silent AudioContext loop prevents mobile browsers from suspending the tab,
# and visibilitychange handles reconnection if the WS dropped anyway.
components.html("""
<script>
(function() {
    const p = window.parent;
    if (p._keepAlive) return;
    p._keepAlive = true;

    /* Inaudible ~1 Hz oscillator keeps mobile browsers from suspending the tab. */
    try {
        const ka = new AudioContext();
        const osc = ka.createOscillator();
        const gain = ka.createGain();
        gain.gain.value = 0.0;
        osc.connect(gain);
        gain.connect(ka.destination);
        osc.start();
        document.addEventListener("visibilitychange", function() {
            if (!document.hidden && ka.state === "suspended") ka.resume();
        });
    } catch(e) {}

    /* When the user unlocks the screen, reload if Streamlit's WS is gone. */
    document.addEventListener("visibilitychange", function() {
        if (document.hidden) return;
        var ws = p.document.querySelector(
            "iframe[title='streamlitHealthCheck']"
        );
        /* Streamlit shows a modal overlay when disconnected. */
        var modal = p.document.querySelector("[data-testid='stStatusWidget']");
        if (modal && modal.textContent.toLowerCase().includes("reconnect")) {
            p.location.reload();
        }
    });
})();
</script>
""", height=0)

client = get_client()
if not client:
    st.warning("Set `VERTEX_API_KEY` in Streamlit secrets or env vars (Vertex AI Express).")
    st.stop()

if "chapters" not in st.session_state:
    st.info("Upload a file or paste text in the sidebar to begin.")
    st.stop()

chs = st.session_state.chapters

# ── Chapter selector ────────────────────────────────────
def _on_ch_change():
    st.session_state.ck_idx = 0
    clear_playback()

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
            clear_playback()
            st.rerun()
with c_info:
    st.caption("Fix badly formatted text with AI (uses tokens)")

# ── Chunk navigation ────────────────────────────────────
chunks = chunk_text(chs[ch_i]["text"], wpc)
total = len(chunks)
ck_i = min(st.session_state.ck_idx, total - 1)

new_ck = st.slider("Start chunk", 0, max(total - 1, 0), ck_i)
if new_ck != ck_i:
    st.session_state.ck_idx = new_ck
    clear_playback()
    st.rerun()

st.caption(f"Chunk {ck_i + 1} / {total}  ·  {len(chunks[ck_i].split())} words")

# Placeholder for chunk text / live transcript (filled below)
chunk_display = st.empty()

# ── Player controls (streaming) ────────────────────────
is_active = st.session_state.get("_streaming_active", False)
left, middle, right = st.columns([1, 2, 1])

with left:
    rewind_clicked = st.button("⏪ 15s", use_container_width=True, disabled=not is_active)
with middle:
    play_clicked = st.button("▶ Play / Pause", type="primary", use_container_width=True)
with right:
    forward_clicked = st.button("15s ⏩", use_container_width=True, disabled=not is_active)

progress_display = st.empty()

# Handle seek buttons
if rewind_clicked:
    streamer_action("rewind15")
if forward_clicked:
    streamer_action("forward15")

# Handle play / pause
did_stream = False
if play_clicked:
    if is_active:
        streamer_action("toggle")
    else:
        pending = [c for c in chunks[ck_i:] if c.strip()]
        if not pending:
            st.warning("No readable text in this chunk range.")
        else:
            did_stream = True
            init_audio_streamer()
            accumulated = []

            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(tts, client, ck, voice, style)
                           for ck in pending]
                for i, future in enumerate(futures):
                    try:
                        pcm = future.result(timeout=120)
                    except Exception as e:
                        st.error(f"TTS error (chunk {i + 1}/{len(pending)}): {e}")
                        for f in futures[i + 1:]:
                            f.cancel()
                        break
                    send_audio_chunk(pcm)
                    accumulated.append(pending[i])
                    chunk_display.text_area(
                        "📝 Live Transcript",
                        " ".join(accumulated),
                        height=200,
                        disabled=True,
                    )
                    progress_display.progress(
                        (i + 1) / len(pending),
                        f"Streaming: chunk {i + 1} / {len(pending)}",
                    )

            progress_display.empty()
            if accumulated:
                st.session_state._streaming_active = True
                st.session_state._transcript = " ".join(accumulated)
                st.session_state.ck_idx = ck_i + len(accumulated) - 1

# Fill the text display when not actively streaming
if not did_stream:
    if st.session_state.get("_streaming_active"):
        transcript = st.session_state.get("_transcript", "")
        if transcript:
            chunk_display.text_area(
                "📝 Transcript", transcript, height=200, disabled=True
            )
    else:
        chunk_display.text_area(
            "chunk_text", chunks[ck_i], height=150,
            disabled=True, label_visibility="collapsed",
        )

# Stop streamer if flagged (e.g. chapter change, chunk slider move)
if st.session_state.pop("_stop_streamer", False):
    streamer_action("stop")
