"""Test 1: a minimal live transcription client.

Connects to the Realtime API, opens a transcription session with
gpt-live-transcribe, streams microphone audio, and prints partial and final
transcripts as they arrive. Turn detection is off (manual commit) so the
event flow stays easy to follow; the "Handling Turn Detection" section of
the article swaps this for server_vad.

Run with:
    python test1_basic_client.py
"""

from __future__ import annotations

import asyncio
import json
import os

import websockets
from dotenv import load_dotenv

from mic_stream import send_microphone_audio, start_microphone
from transcribe_lib import WS_URL, TranscriptionConfig, TranscriptState

load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


async def receive_events(ws, state: TranscriptState) -> None:
    async for raw in ws:
        event = json.loads(raw)
        event_type = event.get("type")

        if event_type == "conversation.item.input_audio_transcription.delta":
            state.apply_delta(event["item_id"], event["delta"])
            print(f"\r[partial] {state.partials[event['item_id']]}", end="", flush=True)

        elif event_type == "conversation.item.input_audio_transcription.completed":
            state.apply_completed(event["item_id"], event["transcript"])
            print(f"\n[final]   {event['transcript']}  (item_id={event['item_id']})")

        elif event_type == "error":
            print(f"\n[error] {event.get('error', event)}")


async def commit_on_interval(ws, appended: asyncio.Event, seconds: float = 4.0) -> None:
    """Manually commit the buffer every few seconds.

    turn_detection is null in this test, so nothing finalizes a turn unless
    the client asks for it. A fixed interval is the simplest strategy; push-
    to-talk and VAD-based alternatives are covered later in the article.

    The `appended` event, set by the sender coroutine, is what guards against
    input_audio_buffer_commit_empty. Testing the mic queue instead would
    never commit, since the sender empties that queue continuously.
    """
    while True:
        await asyncio.sleep(seconds)
        if appended.is_set():
            appended.clear()
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))


async def main() -> None:
    config = TranscriptionConfig(delay="low", turn_detection=None)
    state = TranscriptState()
    loop = asyncio.get_running_loop()
    mic_queue: asyncio.Queue = asyncio.Queue()
    appended = asyncio.Event()

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    async with websockets.connect(WS_URL, additional_headers=headers) as ws:
        await ws.send(json.dumps(config.to_session_update()))
        print("Connected. Session configured for gpt-live-transcribe. Speak now (Ctrl+C to stop).")

        mic = start_microphone(loop, mic_queue)
        try:
            await asyncio.gather(
                send_microphone_audio(ws, mic_queue, appended),
                receive_events(ws, state),
                commit_on_interval(ws, appended),
            )
        finally:
            mic.stop()
            mic.close()
            print("\n\nFull transcript:\n" + state.full_transcript())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
