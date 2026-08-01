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

import numpy as np
import sounddevice as sd

from transcribe_lib import SAMPLE_RATE, encode_chunk

BLOCK_MS = 100
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_MS / 1000)


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


async def send_microphone_audio(ws, queue: asyncio.Queue, appended: asyncio.Event | None = None) -> None:
    """Drain the queue and forward each chunk as input_audio_buffer.append.

    Pass an `appended` event if a caller needs to know whether any audio
    reached the buffer since it last committed. Checking `queue.empty()`
    instead does not work: this coroutine drains the queue as fast as the
    microphone fills it, so the queue is almost always empty even while
    audio is streaming steadily.
    """
    while True:
        pcm16 = await queue.get()
        await ws.send(
            json.dumps({"type": "input_audio_buffer.append", "audio": encode_chunk(pcm16)})
        )
        if appended is not None:
            appended.set()
