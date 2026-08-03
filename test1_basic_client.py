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

from mic_stream import SpeechGate, send_microphone_audio, start_microphone
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
            # A commit that races an already-cleared buffer is harmless noise.
            if event.get("error", {}).get("code") == "input_audio_buffer_commit_empty":
                continue
            print(f"\n[error] {event.get('error', event)}")


async def commit_on_pause(ws, gate: SpeechGate, poll_s: float = 0.1) -> None:
    """Commit once the speaker has said something and then paused.

    turn_detection is null in this test, so nothing finalizes a turn unless
    the client asks for it. Committing on a fixed timer instead looks simpler
    and behaves badly: it fires during silence and returns empty transcripts,
    and it cuts sentences mid-word. The gate watches audio energy so turns end
    where the speaker ended them.

    When the turn holds too little speech to be worth transcribing, the buffer
    is dropped with `input_audio_buffer.clear` rather than committed, so a
    cough never becomes a caption.
    """
    while True:
        await asyncio.sleep(poll_s)

        if gate.should_commit():
            gate.reset()
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        elif gate.should_clear():
            gate.reset()
            await ws.send(json.dumps({"type": "input_audio_buffer.clear"}))


async def main() -> None:
    config = TranscriptionConfig(delay="low", turn_detection=None)
    state = TranscriptState()
    loop = asyncio.get_running_loop()
    mic_queue: asyncio.Queue = asyncio.Queue()
    gate = SpeechGate()

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    async with websockets.connect(WS_URL, additional_headers=headers) as ws:
        await ws.send(json.dumps(config.to_session_update()))
        print("Connected. Session configured for gpt-live-transcribe. Speak now (Ctrl+C to stop).")

        mic = start_microphone(loop, mic_queue)
        try:
            await asyncio.gather(
                send_microphone_audio(ws, mic_queue, gate),
                receive_events(ws, state),
                commit_on_pause(ws, gate),
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
