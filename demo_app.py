"""
GPT Live Transcribe Demo
A Streamlit demo built around the three things this model actually exposes:

1. Live captions: watch delta events arrive and finalize, with a measured
   time to first partial.
2. Context hints: run the same clip through prompt, keywords, and languages
   to see what each field changes.
3. Delay benchmark: stream the same clip through all five delay levels and
   compare time to first partial.

Every tab opens a real Realtime transcription session over WebSocket and
streams audio at real-time pace, the same way a microphone would. Sending
the whole clip at once would only front-load the buffer and make the delay
comparison meaningless.

Requirements:
    pip install streamlit websocket-client python-dotenv numpy pandas

Usage:
    streamlit run demo_app.py

Deployment note:
    For a public app, put OPENAI_API_KEY in Streamlit Secrets.
    Do not hardcode your API key in this file.
"""

import base64
import io
import json
import os
import re
import threading
import time
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import websocket
from dotenv import load_dotenv

# ---------------------------------------------------------------------
# Basic setup
# ---------------------------------------------------------------------

load_dotenv()

SAMPLE_RATE = 24000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2
CHUNK_MS = 100

MODEL = "gpt-live-transcribe"
WS_URL = "wss://api.openai.com/v1/realtime?intent=transcription"
DELAY_LEVELS = ["minimal", "low", "medium", "high", "xhigh"]

DELTA_EVENT = "conversation.item.input_audio_transcription.delta"
COMPLETED_EVENT = "conversation.item.input_audio_transcription.completed"

ASSETS = Path(__file__).parent / "assets"
LOGO = ASSETS / "datacamp-logo.png"
MARK = ASSETS / "datacamp-mark.svg"


@st.cache_data
def logo_uri(path: str) -> str:
    """Inline the wordmark so the app needs no static file server."""
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


@st.cache_data
def svg_uri(path: str) -> str:
    """Streamlit strips inline <svg> from markdown, so serve it as a data URI."""
    svg = Path(path).read_text(encoding="utf-8")
    svg = re.sub(r"<metadata>.*?</metadata>", "", svg, flags=re.S)
    svg = re.sub(r"<\?xml.*?\?>", "", svg, flags=re.S).strip()
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode()


st.set_page_config(
    page_title="GPT Live Transcribe Demo",
    page_icon="🎧",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------

APP_CSS = """
<style>
:root {
    --bg-main: #F6FBFA;
    --text-main: #142430;
    --text-muted: #5F7686;
    --accent: #199A8E;
    --accent-soft: #E4F6F3;
    --accent-warm: #F2994A;
    --border: rgba(25, 154, 142, 0.18);
    --shadow: 0 18px 55px rgba(20, 36, 48, 0.10);
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(25, 154, 142, 0.14), transparent 34%),
        radial-gradient(circle at top right, rgba(242, 153, 74, 0.12), transparent 30%),
        linear-gradient(180deg, #F6FBFA 0%, #FFFFFF 70%);
    color: var(--text-main);
}

.block-container {
    max-width: 1060px;
    padding-top: 2.3rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(240,250,248,0.96));
    border-right: 1px solid var(--border);
}

.logo-link {
    display: block;
    margin: 0 0 2px 0;
    text-decoration: none !important;
    border-bottom: none !important;
}

.logo-link img {
    width: 100%;
    display: block;
    transition: opacity .15s ease;
}

.logo-link:hover img { opacity: .82; }

.logo-sub {
    color: var(--text-muted);
    font-size: .78rem;
    margin: 2px 0 10px 2px;
}

/* The sidebar already carries the wordmark, so repeating the mark in the main
   column is noise while the sidebar is open. Show it only once the sidebar is
   collapsed, which is when the branding would otherwise disappear. */
.brand { display: none; }

[data-testid="stSidebar"][aria-expanded="false"] ~ div .brand,
[data-testid="stSidebar"][aria-expanded="false"] ~ section .brand {
    display: flex;
    align-items: center;
    width: max-content;
    margin: 0 0 12px 0;
    text-decoration: none !important;
    border-bottom: none !important;
}

.brand img { height: 34px; width: auto; display: block; }
.brand:hover { opacity: .8; }

.hero {
    padding: 2rem 2rem 1.7rem 2rem;
    border-radius: 30px;
    border: 1px solid var(--border);
    background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(236,249,246,0.9));
    box-shadow: var(--shadow);
    margin-bottom: 1.4rem;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: .45rem;
    font-size: .82rem;
    font-weight: 700;
    letter-spacing: .02em;
    color: #0E6A61;
    background: var(--accent-soft);
    border: 1px solid rgba(25, 154, 142, 0.22);
    padding: .38rem .72rem;
    border-radius: 999px;
    margin-bottom: .85rem;
}

.hero h1 {
    font-size: 2.4rem;
    line-height: 1.1;
    margin: 0 0 .75rem 0;
    color: var(--text-main);
}

.hero p {
    color: var(--text-muted);
    font-size: 1.02rem;
    line-height: 1.7;
    max-width: 760px;
    margin: 0;
}

.soft-card {
    padding: 1.1rem 1.15rem;
    border-radius: 24px;
    border: 1px solid var(--border);
    background: rgba(255, 255, 255, 0.82);
    box-shadow: 0 12px 35px rgba(20, 36, 48, 0.065);
    margin-bottom: 1rem;
}

.soft-card h3 {
    margin-top: 0;
    margin-bottom: .4rem;
    color: var(--text-main);
}

.soft-card p {
    color: var(--text-muted);
    margin-bottom: 0;
    line-height: 1.6;
}

.caption-stage {
    min-height: 128px;
    padding: 1.15rem 1.25rem;
    border-radius: 22px;
    border: 1px solid var(--border);
    background: #0E1E26;
    color: #F2FBF9;
    font-size: 1.12rem;
    line-height: 1.75;
    box-shadow: var(--shadow);
}

.caption-final { color: #F2FBF9; }

.caption-partial {
    color: #7FE3D5;
    border-bottom: 2px solid rgba(127, 227, 213, .35);
}

.caption-idle { color: #5F7686; font-style: italic; }

.stTabs [data-baseweb="tab-list"] {
    gap: .5rem;
    background: rgba(255,255,255,.62);
    padding: .45rem;
    border-radius: 18px;
    border: 1px solid var(--border);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 14px;
    padding: .65rem 1rem;
    color: var(--text-muted);
    font-weight: 700;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #E4F6F3, #FDF1E6);
    color: #0E6A61;
}

.stButton > button {
    border-radius: 999px;
    border: 1px solid rgba(25, 154, 142, .28);
    background: linear-gradient(135deg, #199A8E, #2BB3A3);
    color: white;
    font-weight: 800;
    padding: .65rem 1.2rem;
    box-shadow: 0 10px 24px rgba(25, 154, 142, .22);
}

.stButton > button:hover {
    border: 1px solid rgba(25, 154, 142, .45);
    filter: brightness(1.03);
}

.stButton > button:disabled {
    background: #EEF2F7;
    color: #98A2B3;
    box-shadow: none;
}

[data-testid="stAudioInput"] {
    border-radius: 22px;
    border: 1px solid var(--border);
    background: rgba(255,255,255,.7);
    padding: .85rem;
}

[data-testid="stAlert"] { border-radius: 18px; }

hr {
    border: none;
    height: 1px;
    background: rgba(25, 154, 142, .15);
    margin: 1.4rem 0;
}

.small-muted {
    color: var(--text-muted);
    font-size: .9rem;
    line-height: 1.6;
}
</style>
"""

st.markdown(APP_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Secrets and API key
# ---------------------------------------------------------------------

def get_api_key_from_server() -> str:
    """Read the API key from Streamlit Secrets first, then the environment."""
    try:
        secret_key = st.secrets.get("OPENAI_API_KEY", "")
        if secret_key:
            return secret_key
    except Exception:
        pass

    return os.environ.get("OPENAI_API_KEY", "")


with st.sidebar:
    if LOGO.exists():
        st.markdown(
            f'<a class="logo-link" href="https://www.datacamp.com/blog" target="_blank" '
            f'rel="noopener"><img src="{logo_uri(str(LOGO))}" alt="DataCamp"/></a>'
            f'<div class="logo-sub">Built for the DataCamp blog</div>',
            unsafe_allow_html=True,
        )

    st.header("Configuration")

    server_api_key = get_api_key_from_server()

    if server_api_key:
        st.success("API key loaded from server secrets or environment.")
        api_key = server_api_key
    else:
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="For public deployment, use Streamlit Secrets instead of typing the key here.",
        )

    st.divider()

    st.markdown(
        """
        **Model**

        - Transcription: `gpt-live-transcribe`
        - Transport: WebSocket transcription session
        - Turn detection: manual commit only
        """
    )

    st.caption(
        "This model rejects server_vad and semantic_vad, so every tab commits "
        "the audio buffer manually. Tested against the live API on August 1, 2026."
    )


if not api_key:
    st.warning("Add your OpenAI API key in the sidebar or Streamlit Secrets to start.")
    st.stop()


# ---------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------

def browser_audio_to_pcm16(audio_bytes: bytes) -> Tuple[bytes, float]:
    """
    Convert Streamlit browser-recorded WAV audio to 24 kHz mono PCM16.

    Returns the raw PCM16 bytes and the duration in seconds.
    """
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        source_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width == 1:
        samples = np.frombuffer(frames, dtype=np.uint8).astype(np.int16)
        samples = (samples - 128) << 8
    elif sample_width == 2:
        samples = np.frombuffer(frames, dtype=np.int16)
    elif sample_width == 4:
        samples_32 = np.frombuffer(frames, dtype=np.int32)
        samples = (samples_32 / 2147483648.0 * 32767).astype(np.int16)
    else:
        raise ValueError(f"Unsupported sample width: {sample_width} bytes")

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1).astype(np.int16)

    if len(samples) == 0:
        raise ValueError("The recorded audio is empty.")

    duration_s = len(samples) / source_rate

    if source_rate != SAMPLE_RATE:
        target_len = max(1, int(duration_s * SAMPLE_RATE))
        old_x = np.linspace(0, duration_s, num=len(samples), endpoint=False)
        new_x = np.linspace(0, duration_s, num=target_len, endpoint=False)
        samples = np.interp(new_x, old_x, samples).astype(np.int16)

    samples = np.clip(samples, -32768, 32767).astype(np.int16)
    return samples.tobytes(), len(samples) / SAMPLE_RATE


def pcm16_to_b64(pcm16_bytes: bytes) -> str:
    """Encode raw PCM16 bytes as base64 for input_audio_buffer.append."""
    return base64.b64encode(pcm16_bytes).decode("utf-8")


def safe_json_loads(message: str) -> Dict:
    """Parse a WebSocket message without raising on malformed payloads."""
    try:
        return json.loads(message)
    except json.JSONDecodeError:
        return {"type": "invalid_json", "raw": message}


def error_message_from_event(event: Dict) -> str:
    """Pull a readable message out of a Realtime error event."""
    error = event.get("error", {})
    message = error.get("message") or event.get("message") or "Unknown API error"
    code = error.get("code")
    return f"{code}: {message}" if code else message


# ---------------------------------------------------------------------
# Session configuration
# ---------------------------------------------------------------------

FORBIDDEN_KEYWORD_CHARS = ("<", ">", "\r", "\n")


def parse_keywords(raw: str) -> List[str]:
    """Split a comma-separated keyword string and reject unsupported characters.

    The Realtime API rejects the entire session update if any keyword holds
    one of these characters, so catching it here saves a wasted round trip
    and a confusing error.
    """
    keywords = [k.strip() for k in raw.split(",") if k.strip()]

    for keyword in keywords:
        if any(char in keyword for char in FORBIDDEN_KEYWORD_CHARS):
            raise ValueError(
                f"Keyword {keyword!r} contains a character the API rejects "
                f"({', '.join(repr(c) for c in FORBIDDEN_KEYWORD_CHARS)})."
            )

    return keywords


def build_session_update(
    delay: str = "low",
    prompt: str = "",
    keywords: Optional[List[str]] = None,
    languages: Optional[List[str]] = None,
) -> Dict:
    """Build one session.update payload for a gpt-live-transcribe session."""
    transcription: Dict[str, object] = {"model": MODEL, "delay": delay}

    if prompt:
        transcription["prompt"] = prompt
    if keywords:
        transcription["keywords"] = keywords
    if languages:
        transcription["languages"] = languages

    return {
        "type": "session.update",
        "session": {
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                    "transcription": transcription,
                    # Anything other than null is rejected for this model.
                    "turn_detection": None,
                }
            },
        },
    }


# ---------------------------------------------------------------------
# Realtime transcription session
# ---------------------------------------------------------------------

def new_run_state() -> Dict:
    """Shared state a background session thread writes and the UI reads."""
    return {
        "partials": {},
        "finals": {},
        "order": [],
        "delta_count": 0,
        "first_delta_s": None,
        "final_s": None,
        "error": None,
        "done": False,
    }


def full_transcript(state: Dict) -> str:
    """Join finalized turns in the order their items were first seen.

    Completion events are not guaranteed to arrive in turn order, so this
    reconciles by item_id rather than trusting arrival order.
    """
    return " ".join(state["finals"][i] for i in state["order"] if i in state["finals"])


def live_text(state: Dict) -> Tuple[str, str]:
    """Return finalized text and the still-growing partial text."""
    final_part = full_transcript(state)
    partial_part = " ".join(
        state["partials"][i] for i in state["order"] if i in state["partials"]
    )
    return final_part, partial_part


def run_transcription_session(
    pcm16_bytes: bytes,
    session_update: Dict,
    state: Dict,
    duration_s: float,
) -> None:
    """Stream one clip through a transcription session, updating `state` live.

    Audio goes out in 100 ms chunks at real-time pace. Sending it all at once
    would only fill the server buffer early and make every delay level look
    identical, which is the whole point the benchmark tab is testing.
    """
    started = {"t": None}
    stop = threading.Event()

    def stream_audio(ws_ref) -> None:
        chunk_bytes = int(SAMPLE_RATE * (CHUNK_MS / 1000.0)) * SAMPLE_WIDTH_BYTES

        try:
            for start_idx in range(0, len(pcm16_bytes), chunk_bytes):
                if stop.is_set() or not (ws_ref.sock and ws_ref.sock.connected):
                    return

                chunk = pcm16_bytes[start_idx:start_idx + chunk_bytes]
                if not chunk:
                    break

                ws_ref.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": pcm16_to_b64(chunk),
                }))
                time.sleep(CHUNK_MS / 1000.0)

            if not stop.is_set() and ws_ref.sock and ws_ref.sock.connected:
                ws_ref.send(json.dumps({"type": "input_audio_buffer.commit"}))
        except Exception:
            return

    def on_open(ws):
        ws.send(json.dumps(session_update))
        started["t"] = time.monotonic()
        threading.Thread(target=stream_audio, args=(ws,), daemon=True).start()

    def on_message(ws, message):
        event = safe_json_loads(message)
        event_type = event.get("type", "")

        if event_type == DELTA_EVENT:
            item_id = event.get("item_id", "")
            if item_id not in state["order"]:
                state["order"].append(item_id)
            state["partials"][item_id] = state["partials"].get(item_id, "") + event.get("delta", "")
            state["delta_count"] += 1

            if state["first_delta_s"] is None and started["t"]:
                state["first_delta_s"] = round(time.monotonic() - started["t"], 3)

        elif event_type == COMPLETED_EVENT:
            item_id = event.get("item_id", "")
            # Track order here too. A receiver that only sees completed events
            # would otherwise build its transcript from an empty order list.
            if item_id not in state["order"]:
                state["order"].append(item_id)
            state["finals"][item_id] = event.get("transcript", "")
            state["partials"].pop(item_id, None)

            if started["t"]:
                state["final_s"] = round(time.monotonic() - started["t"], 3)

            stop.set()
            ws.close()

        elif event_type == "error":
            state["error"] = error_message_from_event(event)
            stop.set()
            ws.close()

    def on_error(ws, error):
        state["error"] = str(error)
        stop.set()

    app = websocket.WebSocketApp(
        WS_URL,
        header=[f"Authorization: Bearer {api_key}"],
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
    )

    watchdog = threading.Timer(duration_s + 30.0, app.close)
    watchdog.start()

    try:
        app.run_forever(ping_interval=20, ping_timeout=10)
    finally:
        watchdog.cancel()
        stop.set()
        state["done"] = True


def transcribe_blocking(
    pcm16_bytes: bytes,
    session_update: Dict,
    duration_s: float,
) -> Dict:
    """Run one session to completion and return its final state."""
    state = new_run_state()
    run_transcription_session(pcm16_bytes, session_update, state, duration_s)

    if state["error"]:
        raise RuntimeError(state["error"])

    return state


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------

brand = ""
if MARK.exists():
    brand = (
        f'<a class="brand" href="https://www.datacamp.com/blog" target="_blank" '
        f'rel="noopener" title="DataCamp">'
        f'<img src="{svg_uri(str(MARK))}" alt="DataCamp"/></a>'
    )

st.markdown(
    f"""
    <div class="hero">
        {brand}
        <div class="hero-badge">🎧 Realtime transcription demo</div>
        <h1>GPT Live Transcribe Demo</h1>
        <p>
            Record a clip, then watch it stream to <b>gpt-live-transcribe</b> at
            real-time pace. The first tab shows partial captions arriving and
            finalizing. The other two run the context and delay experiments from
            the tutorial against your own voice.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown(
        """
        <div class="soft-card">
            <h3>Live captions</h3>
            <p>Watch <b>delta</b> events grow into a finalized transcript.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_b:
    st.markdown(
        """
        <div class="soft-card">
            <h3>Context hints</h3>
            <p>Compare <b>prompt</b>, <b>keywords</b>, and <b>languages</b>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_c:
    st.markdown(
        """
        <div class="soft-card">
            <h3>Delay levels</h3>
            <p>Benchmark all five settings on one clip.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


tab1, tab2, tab3 = st.tabs([
    "🎬 Live captions",
    "🎯 Context hints",
    "⏱️ Delay benchmark",
])


# ---------------------------------------------------------------------
# Tab 1: Live captions
# ---------------------------------------------------------------------

def render_caption_stage(final_part: str, partial_part: str) -> str:
    """Render the caption panel, with in-progress text styled differently."""
    if not final_part and not partial_part:
        return '<div class="caption-stage"><span class="caption-idle">Waiting for the first delta...</span></div>'

    pieces = []
    if final_part:
        pieces.append(f'<span class="caption-final">{final_part}</span>')
    if partial_part:
        pieces.append(f'<span class="caption-partial">{partial_part}</span>')

    return f'<div class="caption-stage">{" ".join(pieces)}</div>'


with tab1:
    st.subheader("Watch a caption build itself")
    st.markdown(
        """
        Audio streams in 100 ms chunks at real-time pace, so a ten second clip
        takes ten seconds. Teal text is still provisional, coming from `delta`
        events. It turns white when a `completed` event finalizes that turn.
        """
    )

    delay_choice = st.select_slider(
        "Delay setting",
        options=DELAY_LEVELS,
        value="low",
        help="Lower settings show text sooner. Higher settings give the model more audio context first.",
    )

    audio1 = st.audio_input("Record a clip to caption", key="audio_live")

    if audio1 is not None:
        st.audio(audio1.getvalue(), format="audio/wav")

    run_t1 = st.button("Start captioning", key="run_live", disabled=audio1 is None)

    if run_t1 and audio1 is not None:
        try:
            pcm16, duration_s = browser_audio_to_pcm16(audio1.getvalue())
            state = new_run_state()

            worker = threading.Thread(
                target=run_transcription_session,
                args=(pcm16, build_session_update(delay=delay_choice), state, duration_s),
                daemon=True,
            )
            worker.start()

            stage = st.empty()
            status = st.empty()

            while not state["done"]:
                final_part, partial_part = live_text(state)
                stage.markdown(render_caption_stage(final_part, partial_part), unsafe_allow_html=True)

                if state["first_delta_s"] is not None:
                    status.caption(
                        f"First partial after {state['first_delta_s']}s  |  "
                        f"{state['delta_count']} delta events so far"
                    )
                else:
                    status.caption("Streaming audio, waiting for the first partial...")

                time.sleep(0.12)

            final_part, partial_part = live_text(state)
            stage.markdown(render_caption_stage(final_part, partial_part), unsafe_allow_html=True)

            if state["error"]:
                status.empty()
                st.error(f"Transcription failed: {state['error']}")
            else:
                status.empty()
                metric_a, metric_b, metric_c = st.columns(3)
                metric_a.metric("Time to first partial", f"{state['first_delta_s']}s"
                                if state["first_delta_s"] is not None else "n/a")
                metric_b.metric("Delta events", state["delta_count"])
                metric_c.metric("Clip length", f"{duration_s:.1f}s")

                st.markdown("### Final transcript")
                st.write(full_transcript(state) or "No speech detected.")

        except Exception as exc:
            st.error(f"Transcription failed: {exc}")

    st.caption(
        "Time to first partial is measured in this app, not reported by the model, "
        "so it includes your network round trip."
    )


# ---------------------------------------------------------------------
# Tab 2: Context hints
# ---------------------------------------------------------------------

with tab2:
    st.subheader("What each context field changes")
    st.markdown(
        """
        The same clip runs through several configurations, changing one field at
        a time. That is the only way to attribute a difference to the field you
        think caused it. Keywords are hints, not rules, so a term you list will
        not appear unless the audio actually contains it.
        """
    )

    prompt_text = st.text_area(
        "Prompt (free-form context)",
        value="A customer support call about a premium plan and account AC-42.",
        height=80,
    )

    keyword_col, language_col = st.columns(2)

    with keyword_col:
        keywords_text = st.text_input(
            "Keywords (comma separated)",
            value="premium plan, AC-42, billing",
        )

    with language_col:
        languages_text = st.text_input(
            "Languages (comma separated ISO codes)",
            value="en",
            help="Two-letter codes like en or ar. Selected ISO 639-3 codes also work.",
        )

    passes = st.slider(
        "Passes per configuration",
        min_value=1,
        max_value=3,
        value=1,
        help="The model is not deterministic on identical audio, so more passes make a difference easier to trust.",
    )

    audio2 = st.audio_input("Record a clip with a product name or an ID in it", key="audio_context")

    if audio2 is not None:
        st.audio(audio2.getvalue(), format="audio/wav")

    run_t2 = st.button("Compare configurations", key="run_context", disabled=audio2 is None)

    if run_t2 and audio2 is not None:
        try:
            keywords = parse_keywords(keywords_text)
            languages = [c.strip() for c in languages_text.split(",") if c.strip()]
            pcm16, duration_s = browser_audio_to_pcm16(audio2.getvalue())

            configurations = {
                "no_context": build_session_update(),
                "prompt_only": build_session_update(prompt=prompt_text),
                "keywords_only": build_session_update(keywords=keywords),
                "languages_only": build_session_update(languages=languages),
                "prompt_and_keywords": build_session_update(
                    prompt=prompt_text, keywords=keywords
                ),
            }

            total_runs = len(configurations) * passes
            progress = st.progress(0.0, text="Running configurations...")
            rows = []
            completed = 0

            for label, session_update in configurations.items():
                for pass_index in range(passes):
                    state = transcribe_blocking(pcm16, session_update, duration_s)
                    rows.append({
                        "Configuration": label,
                        "Pass": pass_index + 1,
                        "Transcript": full_transcript(state) or "(no speech detected)",
                    })

                    completed += 1
                    progress.progress(
                        completed / total_runs,
                        text=f"Finished {completed} of {total_runs} runs",
                    )

            progress.empty()
            st.success(f"Done. {total_runs} sessions, same audio in every one.")

            st.markdown("### Transcripts")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.download_button(
                "Download results as JSON",
                data=json.dumps(rows, indent=2, ensure_ascii=False),
                file_name="context_comparison_results.json",
                mime="application/json",
            )

        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Comparison failed: {exc}")

    st.caption(
        "A single pass can show a difference that disappears on the next run. "
        "Raise the pass count before drawing a conclusion from one transcript."
    )


# ---------------------------------------------------------------------
# Tab 3: Delay benchmark
# ---------------------------------------------------------------------

with tab3:
    st.subheader("Benchmark the five delay levels")
    st.markdown(
        """
        Each level streams the same clip at real-time pace, so a ten second clip
        takes roughly ten seconds per level. OpenAI does not publish millisecond
        figures for these settings and asks you to benchmark with your own audio,
        which is exactly what this tab does.
        """
    )

    selected_levels = st.multiselect(
        "Levels to test",
        options=DELAY_LEVELS,
        default=DELAY_LEVELS,
    )

    audio3 = st.audio_input("Record a clip for the benchmark", key="audio_benchmark")

    if audio3 is not None:
        st.audio(audio3.getvalue(), format="audio/wav")

    run_t3 = st.button(
        "Run benchmark",
        key="run_benchmark",
        disabled=audio3 is None or not selected_levels,
    )

    if run_t3 and audio3 is not None:
        try:
            pcm16, duration_s = browser_audio_to_pcm16(audio3.getvalue())
            estimate = duration_s * len(selected_levels)
            progress = st.progress(0.0, text=f"Running, roughly {estimate:.0f}s of streaming ahead...")
            results = []

            for index, level in enumerate(selected_levels, start=1):
                state = transcribe_blocking(pcm16, build_session_update(delay=level), duration_s)
                results.append({
                    "delay": level,
                    "Time to first partial (s)": state["first_delta_s"],
                    "Time to final (s)": state["final_s"],
                    "Delta events": state["delta_count"],
                    "Transcript": full_transcript(state) or "(no speech detected)",
                })

                progress.progress(
                    index / len(selected_levels),
                    text=f"Finished {level} ({index} of {len(selected_levels)})",
                )

            progress.empty()
            st.success("Done")

            frame = pd.DataFrame(results)

            chart_data = frame.set_index("delay")[["Time to first partial (s)"]]
            st.markdown("### Time to first partial")
            st.bar_chart(chart_data, color="#199A8E")

            st.markdown("### Full results")
            st.dataframe(frame, use_container_width=True, hide_index=True)

            st.download_button(
                "Download results as JSON",
                data=json.dumps(results, indent=2, ensure_ascii=False),
                file_name="delay_benchmark_results.json",
                mime="application/json",
            )

        except Exception as exc:
            st.error(f"Benchmark failed: {exc}")

    st.caption(
        "Time to final is set by when this app commits the buffer, not by the model, "
        "so read the first-partial column for the delay comparison."
    )
