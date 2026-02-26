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


# ── JS audio player ─────────────────────────────────────
# Uses <audio> elements instead of Web Audio API so the browser's
# built-in WSOLA time-stretcher preserves pitch at any speed
# (same mechanism YouTube uses).
def init_player():
    js = """
    (function() {
        var p = window.parent;
        if (p._player) try { p._player.stop(); } catch(e) {}
        p._player = {
            queue: [], current: null, speed: p._desiredSpeed || 1.0, paused: false,

            addChunk: function(b64) {
                var url = URL.createObjectURL(this._wav(b64));
                this.queue.push(url);
                if (!this.current && !this.paused) this._next();
            },

            _next: function() {
                if (this.queue.length === 0) { this.current = null; return; }
                var url = this.queue.shift();
                var a = new Audio(url);
                a.playbackRate = this.speed;
                var self = this;
                a.onended = function() {
                    URL.revokeObjectURL(url);
                    a.onended = null;
                    self._next();
                };
                this.current = a;
                a.play().catch(function(){});
            },

            setSpeed: function(s) {
                this.speed = s;
                if (this.current) this.current.playbackRate = s;
            },

            togglePause: function() {
                if (!this.current) return;
                if (this.current.paused) { this.current.play(); this.paused = false; }
                else { this.current.pause(); this.paused = true; }
            },

            stop: function() {
                if (this.current) {
                    this.current.pause();
                    this.current.onended = null;
                    this.current = null;
                }
                this.queue.forEach(function(u) { URL.revokeObjectURL(u); });
                this.queue = [];
                this.paused = false;
            },

            /* Convert raw 24 kHz 16-bit mono PCM (base64) → WAV Blob */
            _wav: function(b64) {
                var bin = atob(b64);
                var pcm = new Uint8Array(bin.length);
                for (var i = 0; i < bin.length; i++) pcm[i] = bin.charCodeAt(i);
                var h = new ArrayBuffer(44), d = new DataView(h);
                d.setUint32(0,  0x52494646, false);
                d.setUint32(4,  36 + pcm.length, true);
                d.setUint32(8,  0x57415645, false);
                d.setUint32(12, 0x666d7420, false);
                d.setUint32(16, 16, true);
                d.setUint16(20, 1, true);
                d.setUint16(22, 1, true);
                d.setUint32(24, 24000, true);
                d.setUint32(28, 48000, true);
                d.setUint16(32, 2, true);
                d.setUint16(34, 16, true);
                d.setUint32(36, 0x64617461, false);
                d.setUint32(40, pcm.length, true);
                return new Blob([h, pcm], {type: 'audio/wav'});
            }
        };
    })();
    """
    components.html(f"<script>{js}</script>", height=0)


_MAX_SEG = 1_200_000


def send_audio(pcm, container):
    """Send PCM to JS player. Each segment gets its own iframe in the container."""
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

# Stop player when chapter changes
if st.session_state.get("_prev_ch") is not None and st.session_state._prev_ch != st.session_state.ch_idx:
    player_action("stop")
st.session_state._prev_ch = st.session_state.ch_idx

ch = chs[st.session_state.ch_idx]
ch_chunks = chunk_text(ch["text"], wpc)

start = st.slider("Start from chunk", 1, max(len(ch_chunks), 1), 1) - 1
st.caption(
    f"Ch {st.session_state.ch_idx + 1}/{len(chs)} · "
    f"{len(ch_chunks)} chunks · {len(ch['text'].split())} words"
)

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

# Speed buttons — pure JS, no Streamlit rerun, no playback interruption
components.html("""
<style>
.sb{border:1px solid #ddd;background:#fafafa;padding:3px 11px;border-radius:4px;
    cursor:pointer;font-size:13px;font-family:sans-serif}
.sb:hover{background:#eee}
.sb.on{background:#ff4b4b;color:#fff;border-color:#ff4b4b}
</style>
<div style="display:flex;gap:6px;align-items:center">
 <span style="font-size:13px;color:#888">Speed</span>
 <button class="sb" onclick="ss(1)">1×</button>
 <button class="sb" onclick="ss(1.25)">1.25×</button>
 <button class="sb" onclick="ss(1.5)">1.5×</button>
 <button class="sb" onclick="ss(2)">2×</button>
</div>
<script>
function ss(s){
    window.parent._desiredSpeed=s;
    var p=window.parent._player; if(p) p.setSpeed(s);
    document.querySelectorAll('.sb').forEach(function(b){
        b.classList.toggle('on',parseFloat(b.textContent)===s);
    });
}
var c=window.parent._desiredSpeed||1;
document.querySelectorAll('.sb').forEach(function(b){
    b.classList.toggle('on',parseFloat(b.textContent)===c);
});
</script>
""", height=36)

progress_bar = st.empty()
transcript = st.empty()

# ── Pipeline: cross-chapter, 1-ahead prefetch ───────────
if play:
    # Build flat playlist from current chapter+chunk to end of book
    playlist = []  # (ch_idx, ch_title, ck_1based, ch_total, text)
    for ci in range(st.session_state.ch_idx, len(chs)):
        cks = chunk_text(chs[ci]["text"], wpc)
        first = start if ci == st.session_state.ch_idx else 0
        for ki, ck in enumerate(cks[first:], first):
            if ck.strip():
                playlist.append((ci, chs[ci]["title"], ki + 1, len(cks), ck))

    if not playlist:
        st.warning("No text to read.")
    else:
        init_player()
        time.sleep(0.3)
        audio_box = st.container()
        lines = []
        prev_ch = -1
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            fut = executor.submit(tts, client, playlist[0][4], voice, style)
            for i, (ci, title, ck_num, ch_tot, chunk) in enumerate(playlist):
                try:
                    pcm = fut.result(timeout=90)
                except Exception as e:
                    progress_bar.error(f"[{title}] chunk {ck_num} failed: {e}")
                    break

                # prefetch next chunk (may be from the next chapter)
                if i + 1 < len(playlist):
                    fut = executor.submit(
                        tts, client, playlist[i + 1][4], voice, style
                    )

                send_audio(pcm, audio_box)

                if ci != prev_ch:
                    lines.append(f"── {title} ──")
                    prev_ch = ci
                lines.append(chunk)

                progress_bar.progress(
                    (i + 1) / len(playlist),
                    f"{title} · {ck_num}/{ch_tot}",
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
            n_chs = prev_ch - st.session_state.ch_idx + 1
            progress_bar.success(
                f"Done — {len(playlist)} chunks, {n_chs} chapter(s)"
            )

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
