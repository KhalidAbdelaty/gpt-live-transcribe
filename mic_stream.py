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

# A chunk counts as speech when it is this many times louder than the room.
# A ratio rather than a fixed amplitude, because microphones disagree wildly:
# the same sentence measured about 200 RMS on one machine here and about 1200
# on another. Any single number is wrong on one of them.
SPEECH_NOISE_RATIO = 4.0

# Floor for the computed threshold, so a perfectly silent digital input does
# not end up treating its own dither as speech.
MIN_SPEECH_RMS = 40.0


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
    """Decides when the audio buffer is worth committing, and when to bin it.

    `gpt-live-transcribe` rejects server-side voice activity detection, so the
    client has to answer questions the API will not answer for it: has the
    speaker said anything since the last commit, have they stopped talking, and
    was it enough audio to be worth transcribing.

    A plain fixed-interval commit gets all three wrong. It fires during silence
    and returns an empty transcript, and it cuts a sentence in half mid-word,
    which leaves the model transcribing a fragment that starts and ends
    nowhere. Both failures show up the first time you run this against a real
    microphone.

    The threshold adapts instead of being configured. A number tuned on one
    microphone was wrong by a factor of five on the next one, so the gate keeps
    a running estimate of the room and calls anything several times louder than
    that speech.
    """

    def __init__(
        self,
        threshold: float | None = None,
        # 0.8s sounded reasonable and was not: a voice dipping at the end of a
        # clause, or a breath mid-sentence, reads as a pause and commits early,
        # which hands the model half a sentence. 1.5s waits out normal speech
        # rhythm and still finalizes a caption fast enough to read live.
        silence_hold_s: float = 1.5,
        max_turn_s: float = 15.0,
        min_speech_s: float = 0.4,
        noise_ratio: float = SPEECH_NOISE_RATIO,
    ) -> None:
        self.threshold = threshold  # None means adapt to the room
        self.silence_hold_s = silence_hold_s
        self.max_turn_s = max_turn_s
        self.min_speech_s = min_speech_s
        self.noise_ratio = noise_ratio
        self._noise_floor: float | None = None
        self.reset()

    def reset(self) -> None:
        """Start a fresh turn. The learned room level carries over."""
        self._heard_speech = False
        self._speech_chunks = 0
        self._last_speech_at = 0.0
        self._turn_started_at = time.monotonic()

    def _is_speech(self, rms: float) -> bool:
        """Classify one chunk, updating the room estimate only from silence.

        The floor has to learn from chunks it already believes are noise. An
        earlier version let every loud chunk nudge it upward, which sounds
        harmless and is not: through a long sentence the floor climbed toward
        the speaker's own level, the threshold climbed with it, and about six
        seconds in the gate stopped hearing the voice it had been tracking.
        It then committed mid-sentence and handed the model a fragment.
        """
        if self.threshold is not None:
            return rms >= self.threshold

        if self._noise_floor is None:
            self._noise_floor = rms
            return False

        threshold = max(self._noise_floor * self.noise_ratio, MIN_SPEECH_RMS)
        if rms >= threshold:
            return True

        # Below the threshold, so treat it as the room and adapt. Falling fast
        # and rising slowly keeps a door slam from raising the floor for the
        # rest of the session.
        rate = 0.5 if rms < self._noise_floor else 0.02
        self._noise_floor += (rms - self._noise_floor) * rate
        return False

    def observe(self, pcm16: bytes) -> None:
        """Record whether this chunk carried speech."""
        if not self._is_speech(chunk_rms(pcm16)):
            return

        if not self._heard_speech:
            self._heard_speech = True
            self._turn_started_at = time.monotonic()

        self._speech_chunks += 1
        self._last_speech_at = time.monotonic()

    def _turn_ended(self) -> bool:
        if not self._heard_speech:
            return False

        now = time.monotonic()
        paused = now - self._last_speech_at >= self.silence_hold_s
        too_long = now - self._turn_started_at >= self.max_turn_s
        return paused or too_long

    def _speech_seconds(self) -> float:
        return self._speech_chunks * BLOCK_MS / 1000

    def should_commit(self) -> bool:
        """True once the speaker has said something real and then paused."""
        return self._turn_ended() and self._speech_seconds() >= self.min_speech_s

    def should_clear(self) -> bool:
        """True when the turn ended on too little audio to transcribe.

        A door closing or a cough clears the speech threshold for a chunk or
        two. Committing that returns an empty transcript at best and an
        invented word at worst, so the buffer gets dropped with
        `input_audio_buffer.clear` instead.
        """
        return self._turn_ended() and self._speech_seconds() < self.min_speech_s


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
