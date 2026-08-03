"""Microphone capture for the live client (Test 1 and the full captioning app).

sounddevice runs its callback on a separate audio thread. That callback has
to return fast or the driver drops frames, so it never touches the network
directly. Instead it pushes raw PCM16 bytes onto an asyncio.Queue, and a
coroutine running in the event loop drains that queue and sends each chunk
over the WebSocket. This is the non-blocking handoff the tutorial talks about.
"""

from __future__ import annotations

import asyncio
import json
import time

import numpy as np
import sounddevice as sd

from transcribe_lib import SAMPLE_RATE, encode_chunk

BLOCK_MS = 100
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_MS / 1000)

# RMS amplitude, on the int16 scale, above which a chunk counts as speech.
# Quiet room noise sits well under 200 on the microphones I tested; normal
# speech runs into the thousands. Raise it in a noisy room.
SPEECH_RMS_THRESHOLD = 500.0


def start_microphone(loop: asyncio.AbstractEventLoop, queue: asyncio.Queue) -> sd.InputStream:
    """Open the input stream and start feeding PCM16 chunks into the queue."""

    def callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            print(f"[mic warning] {status}")
        pcm16 = (indata[:, 0] * 32767).astype(np.int16).tobytes()
        loop.call_soon_threadsafe(queue.put_nowait, pcm16)

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=BLOCK_SIZE,
        callback=callback,
    )
    stream.start()
    return stream


def chunk_rms(pcm16: bytes) -> float:
    """Root mean square amplitude of one PCM16 chunk."""
    samples = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples**2)))


class SpeechGate:
    """Decides when the audio buffer is worth committing.

    `gpt-live-transcribe` rejects server-side voice activity detection, so the
    client has to answer two questions the API will not answer for it: has the
    speaker said anything since the last commit, and have they stopped talking.

    A plain fixed-interval commit gets both wrong. It fires during silence and
    returns an empty transcript, and it cuts a sentence in half mid-word, which
    leaves the model transcribing a fragment that starts and ends nowhere. Both
    failures are visible the first time you run this against a real microphone.

    So this is a small voice activity detector, running client-side because the
    server will not run one for this model.
    """

    def __init__(
        self,
        threshold: float = SPEECH_RMS_THRESHOLD,
        # 0.8s sounded reasonable and was not: a voice dipping at the end of a
        # clause, or a breath mid-sentence, reads as a pause and commits early,
        # which hands the model half a sentence. 1.5s waits out normal speech
        # rhythm and still finalizes a caption fast enough to read live.
        silence_hold_s: float = 1.5,
        max_turn_s: float = 15.0,
    ) -> None:
        self.threshold = threshold
        self.silence_hold_s = silence_hold_s
        self.max_turn_s = max_turn_s
        self.reset()

    def reset(self) -> None:
        self._heard_speech = False
        self._last_speech_at = 0.0
        self._turn_started_at = time.monotonic()

    def observe(self, pcm16: bytes) -> None:
        """Record whether this chunk carried speech."""
        if chunk_rms(pcm16) < self.threshold:
            return

        if not self._heard_speech:
            self._heard_speech = True
            self._turn_started_at = time.monotonic()

        self._last_speech_at = time.monotonic()

    def should_commit(self) -> bool:
        """True once the speaker has said something and then paused.

        The max-turn cap keeps a single turn from growing without end when
        somebody talks continuously, which is the one thing a fixed interval
        did get right.
        """
        if not self._heard_speech:
            return False

        now = time.monotonic()
        paused = now - self._last_speech_at >= self.silence_hold_s
        too_long = now - self._turn_started_at >= self.max_turn_s
        return paused or too_long


async def send_microphone_audio(ws, queue: asyncio.Queue, gate: SpeechGate | None = None) -> None:
    """Drain the queue and forward each chunk as input_audio_buffer.append.

    Pass a `SpeechGate` if the caller needs to know when to commit. Checking
    `queue.empty()` instead does not work: this coroutine drains the queue as
    fast as the microphone fills it, so the queue is almost always empty even
    while audio is streaming steadily.
    """
    while True:
        pcm16 = await queue.get()
        await ws.send(
            json.dumps({"type": "input_audio_buffer.append", "audio": encode_chunk(pcm16)})
        )
        if gate is not None:
            gate.observe(pcm16)
