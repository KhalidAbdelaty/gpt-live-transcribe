"""Shared helpers for the GPT Live Transcribe tutorial scripts.

This module holds the pieces that Test 1, Test 2, and Test 3 all reuse:
building a session.update payload, encoding PCM16 audio to base64, and
streaming a WAV file at real-time pace so repeated test runs use identical
audio. Keeping this logic in one place is what makes the Test 2 and Test 3
scripts comparable: only the field under test changes between runs.
"""

from __future__ import annotations

import asyncio
import base64
import json
import shutil
import time
import wave
from dataclasses import dataclass, field
from typing import Iterator, Optional

SAMPLE_RATE = 24_000
CHUNK_MS = 100  # size of each audio chunk sent over the WebSocket
WS_URL = "wss://api.openai.com/v1/realtime?intent=transcription"


@dataclass
class TranscriptionConfig:
    """One session.update payload for a gpt-live-transcribe session.

    Every field maps directly to session.audio.input in the Realtime
    transcription guide. Leave prompt, keywords, or languages as None to
    omit them entirely rather than sending empty values.
    """

    model: str = "gpt-live-transcribe"
    delay: str = "low"
    prompt: Optional[str] = None
    keywords: Optional[list] = None
    languages: Optional[list] = None
    turn_detection: Optional[dict] = None  # None -> manual commit

    def validate_keywords(self) -> None:
        """Reject keyword strings the Realtime API would reject outright.

        The guide states the whole session update is rejected if a keyword
        contains '<', '>', a carriage return, or a line feed. Checking this
        client-side avoids a wasted round trip and a confusing error.
        """
        if not self.keywords:
            return
        forbidden = ("<", ">", "\r", "\n")
        for kw in self.keywords:
            if any(ch in kw for ch in forbidden):
                raise ValueError(
                    f"Keyword {kw!r} contains a forbidden character "
                    f"{forbidden}. Fix it before sending session.update."
                )

    def to_session_update(self) -> dict:
        """Build the session.update event this config represents."""
        self.validate_keywords()

        transcription: dict = {"model": self.model, "delay": self.delay}
        if self.prompt:
            transcription["prompt"] = self.prompt
        if self.keywords:
            transcription["keywords"] = self.keywords
        if self.languages:
            transcription["languages"] = self.languages

        return {
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                        "transcription": transcription,
                        "turn_detection": self.turn_detection,
                    }
                },
            },
        }


def live_caption_line(prefix: str, text: str) -> str:
    """Fit a redrawing caption line inside the terminal so it never wraps.

    Partial captions are redrawn with a carriage return, which moves the
    cursor to the start of the current screen row. Once the caption is long
    enough to wrap, that row is no longer the start of the caption, so every
    update paints over the tail of the previous one and the line turns to
    mush. Keeping the text inside the terminal width keeps the redraw in one
    place, and showing the end rather than the beginning matches what a
    caption does anyway: the newest words are the ones worth reading.
    """
    width = shutil.get_terminal_size(fallback=(100, 24)).columns - 1
    room = max(20, width - len(prefix))

    if len(text) > room:
        text = "..." + text[-(room - 3):]

    # Pad so a shorter update erases whatever the longer one left behind.
    return prefix + text.ljust(room)


def short_item_id(item_id: str, keep: int = 5) -> str:
    """Shorten an item_id for display only.

    A transcription `item_id` runs to 26 characters, which pushes a caption
    line onto a second row in a normal terminal and buries the sentence you
    were trying to read. The tail is enough to tell turns apart on screen.
    Anything that needs the real value, the JSON export included, keeps it
    in full.
    """
    return "..." + item_id[-keep:] if len(item_id) > keep else item_id


def clear_live_line() -> str:
    """Blank the redrawing caption line before printing over it.

    A finalized line is usually longer than the partial it replaces, so it
    covers the old text on its own. Usually is not always: a short final
    would leave the tail of a long partial sitting on screen next to it.
    """
    width = shutil.get_terminal_size(fallback=(100, 24)).columns - 1
    return "\r" + " " * width + "\r"


def read_wav_pcm16_mono_24k(path: str) -> bytes:
    """Read a WAV file and confirm it already matches the API's audio format.

    gpt-live-transcribe expects 16-bit PCM, 24 kHz, mono, little-endian.
    Raising here beats debugging a garbled transcript later.
    """
    with wave.open(path, "rb") as wf:
        if wf.getframerate() != SAMPLE_RATE:
            raise ValueError(
                f"{path} is {wf.getframerate()} Hz, expected {SAMPLE_RATE} Hz. "
                "Resample with ffmpeg: ffmpeg -i in.wav -ar 24000 -ac 1 -sample_fmt s16 out.wav"
            )
        if wf.getnchannels() != 1:
            raise ValueError(f"{path} has {wf.getnchannels()} channels, expected 1 (mono).")
        if wf.getsampwidth() != 2:
            raise ValueError(f"{path} is not 16-bit PCM.")
        return wf.readframes(wf.getnframes())


def iter_pcm_chunks(pcm_bytes: bytes, chunk_ms: int = CHUNK_MS) -> Iterator[bytes]:
    """Split raw PCM16 audio into fixed-duration chunks for streaming."""
    bytes_per_ms = int(SAMPLE_RATE * 2 / 1000)  # 2 bytes per sample, mono
    chunk_size = bytes_per_ms * chunk_ms
    for i in range(0, len(pcm_bytes), chunk_size):
        yield pcm_bytes[i : i + chunk_size]


def encode_chunk(chunk: bytes) -> str:
    """Base64-encode one PCM16 chunk for input_audio_buffer.append."""
    return base64.b64encode(chunk).decode("utf-8")


async def stream_wav_realtime(ws, path: str, chunk_ms: int = CHUNK_MS) -> None:
    """Send a WAV file over an open WebSocket at real-time playback speed.

    Sending faster than real time does not make transcription faster, it
    just front-loads the buffer, so this sleeps between chunks to mimic a
    live microphone. Using a recorded file instead of a live mic is what
    lets Test 2 and Test 3 replay the exact same audio across configurations.
    """
    pcm = read_wav_pcm16_mono_24k(path)
    for chunk in iter_pcm_chunks(pcm, chunk_ms):
        await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": encode_chunk(chunk)}))
        # Use asyncio.sleep, not time.sleep. time.sleep blocks the whole
        # event loop, which starves any receiver task running in the same
        # loop, so incoming delta events queue up and only get processed
        # in a burst once streaming finishes. That bug made every delay
        # level look identical when this was benchmarked for real.
        await asyncio.sleep(chunk_ms / 1000)
    await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))


@dataclass
class TranscriptState:
    """Tracks partial and final transcript text keyed by item_id.

    Completion events are not guaranteed to arrive in turn order, so this
    keeps a dict rather than assuming the next completed event belongs to
    the most recently started item.
    """

    partials: dict = field(default_factory=dict)
    finals: dict = field(default_factory=dict)
    order: list = field(default_factory=list)  # item_ids in first-seen order

    def apply_delta(self, item_id: str, delta: str) -> None:
        # Check `order`, not `partials`. apply_completed clears the partial
        # for an item, so a delta arriving after that item was finalized
        # would otherwise append the same item_id a second time and make
        # full_transcript() emit its final text twice.
        if item_id not in self.order:
            self.order.append(item_id)
        self.partials[item_id] = self.partials.get(item_id, "") + delta

    def apply_completed(self, item_id: str, transcript: str) -> None:
        # A completed event can arrive for an item whose deltas were never
        # tracked locally (or never fired), so order has to be updated here
        # too, not only in apply_delta. Skipping this was a real bug caught
        # while live-testing: a receiver that only handled `completed`
        # events built a transcript from an always-empty order list.
        if item_id not in self.order:
            self.order.append(item_id)
        self.finals[item_id] = transcript
        self.partials.pop(item_id, None)

    def full_transcript(self) -> str:
        """Join finalized turns in the order items were first seen."""
        return " ".join(self.finals[i] for i in self.order if i in self.finals)
