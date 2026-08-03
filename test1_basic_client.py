"""Test 1: a minimal live transcription client.

Connects to the Realtime API, opens a transcription session with
gpt-live-transcribe, streams microphone audio, and prints partial and final
transcripts as they arrive. Turn detection is off (manual commit) so the
event flow stays easy to follow.

No context is sent by default, which is the point: it shows what the model
does on its own. If a word it is unsure of comes back in the wrong script,
name the language you are speaking and it stops guessing.

Run with:
    python test1_basic_client.py
    python test1_basic_client.py --languages en
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

import websockets
from dotenv import load_dotenv

from mic_stream import SpeechGate, send_microphone_audio, start_microphone
from transcribe_lib import (
    WS_URL,
    TranscriptionConfig,
    TranscriptState,
    clear_live_line,
    live_caption_line,
    short_item_id,
)

load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


async def receive_events(ws, state: TranscriptState) -> None:
    async for raw in ws:
        event = json.loads(raw)
        event_type = event.get("type")

        if event_type == "conversation.item.input_audio_transcription.delta":
            state.apply_delta(event["item_id"], event["delta"])
            line = live_caption_line("[partial] ", state.partials[event["item_id"]])
            print(f"\r{line}", end="", flush=True)

        elif event_type == "conversation.item.input_audio_transcription.completed":
            transcript = event.get("transcript", "").strip()
            if not transcript:
                continue
            state.apply_completed(event["item_id"], transcript)
            item = short_item_id(event["item_id"])
            print(f"{clear_live_line()}[final]   {transcript}  (item {item})")

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal GPT Live Transcribe client")
    parser.add_argument(
        "--languages",
        default=None,
        help="comma-separated ISO 639-1 codes, for example en or en,ar",
    )
    parser.add_argument("--prompt", default=None, help="free-form context about the audio")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    languages = [c.strip() for c in args.languages.split(",")] if args.languages else None
    config = TranscriptionConfig(
        delay="low",
        prompt=args.prompt,
        languages=languages,
        turn_detection=None,
    )

    state = TranscriptState()
    loop = asyncio.get_running_loop()
    mic_queue: asyncio.Queue = asyncio.Queue()
    gate = SpeechGate()

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    async with websockets.connect(WS_URL, additional_headers=headers) as ws:
        await ws.send(json.dumps(config.to_session_update()))
        hint = f", languages={','.join(languages)}" if languages else ", no context"
        print(f"Connected. Session configured for gpt-live-transcribe{hint}. Speak now (Ctrl+C to stop).")

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
