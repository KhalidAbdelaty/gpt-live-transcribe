"""The complete live captioning app.

Combines microphone capture, live partial captions, a finalized transcript
history, and a JSON export on exit. Configure delay, context, keywords,
languages, and turn detection with CLI flags instead of editing the script.

As tested against the live API on August 1, 2026, gpt-live-transcribe
rejects both server_vad and semantic_vad with "Turn detection is not
supported for this transcription model." Manual commit (turn_detection:
null) is the only mode that currently works, so it is the default here.
--turn-detection server_vad/semantic_vad are left in as options in case
that changes, and the app prints the server's error event if they are
rejected rather than failing silently.

Run with:
    python app.py --delay low --keywords "AC-42,premium plan" --languages en,ar
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPT Live Transcribe captioning app")
    parser.add_argument("--delay", default="low", choices=["minimal", "low", "medium", "high", "xhigh"])
    parser.add_argument("--prompt", default=None, help="Free-form context about the audio")
    parser.add_argument("--keywords", default=None, help="Comma-separated literal terms")
    parser.add_argument("--languages", default=None, help="Comma-separated expected language codes")
    parser.add_argument(
        "--turn-detection",
        default="manual",
        choices=["manual", "server_vad", "semantic_vad"],
        help="manual is the only mode gpt-live-transcribe currently accepts (tested August 1, 2026)",
    )
    parser.add_argument(
        "--silence-hold",
        type=float,
        default=1.5,
        help="seconds of silence after speech before the turn is committed",
    )
    parser.add_argument(
        "--max-turn",
        type=float,
        default=15.0,
        help="commit anyway after this many seconds, so a turn never grows unbounded",
    )
    parser.add_argument(
        "--speech-threshold",
        type=float,
        default=None,
        help="fixed RMS amplitude for speech; omit it and the gate learns the room instead",
    )
    parser.add_argument("--export", default="transcript_export.json")
    return parser.parse_args()


def build_turn_detection(mode: str) -> dict | None:
    if mode == "manual":
        return None
    if mode == "server_vad":
        return {"type": "server_vad", "threshold": 0.5, "silence_duration_ms": 500}
    return {"type": "semantic_vad", "eagerness": "auto"}


async def main() -> None:
    args = parse_args()
    config = TranscriptionConfig(
        delay=args.delay,
        prompt=args.prompt,
        keywords=[k.strip() for k in args.keywords.split(",")] if args.keywords else None,
        languages=[l.strip() for l in args.languages.split(",")] if args.languages else None,
        turn_detection=build_turn_detection(args.turn_detection),
    )

    state = TranscriptState()
    loop = asyncio.get_running_loop()
    mic_queue: asyncio.Queue = asyncio.Queue()
    gate = SpeechGate(
        threshold=args.speech_threshold,
        silence_hold_s=args.silence_hold,
        max_turn_s=args.max_turn,
    )
    session_started_at = time.time()
    first_delta_latency = None

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    async with websockets.connect(WS_URL, additional_headers=headers) as ws:
        await ws.send(json.dumps(config.to_session_update()))
        print(f"Session ready: delay={args.delay}, turn_detection={args.turn_detection}")
        print("Speak now. Press Ctrl+C to stop and export the transcript.\n")

        mic = start_microphone(loop, mic_queue)

        async def commit_on_pause() -> None:
            # Only meaningful in manual mode; server_vad/semantic_vad commit
            # on their own if the API ever accepts them for this model.
            if args.turn_detection != "manual":
                return
            while True:
                await asyncio.sleep(0.1)
                # The gate watches audio energy, not the mic queue, which the
                # sender keeps drained, and not the clock, which fires during
                # silence and cuts sentences in half.
                if gate.should_commit():
                    gate.reset()
                    await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                elif gate.should_clear():
                    # Too little speech to transcribe. Bin it rather than ask
                    # the model what a door closing said.
                    gate.reset()
                    await ws.send(json.dumps({"type": "input_audio_buffer.clear"}))

        async def receive() -> None:
            nonlocal first_delta_latency
            async for raw in ws:
                event = json.loads(raw)
                event_type = event.get("type")

                if event_type == "conversation.item.input_audio_transcription.delta":
                    if first_delta_latency is None:
                        first_delta_latency = round(time.time() - session_started_at, 3)
                    state.apply_delta(event["item_id"], event["delta"])
                    line = live_caption_line("[live] ", state.partials[event["item_id"]])
                    print(f"\r{line}", end="", flush=True)

                elif event_type == "conversation.item.input_audio_transcription.completed":
                    transcript = event.get("transcript", "").strip()
                    if not transcript:
                        continue
                    state.apply_completed(event["item_id"], transcript)
                    item = short_item_id(event["item_id"])
                    print(f"{clear_live_line()}[caption] {transcript}  (item {item})")

                elif event_type == "input_audio_buffer.speech_started":
                    print("\n[vad] speech started")

                elif event_type == "input_audio_buffer.speech_stopped":
                    print("[vad] speech stopped, waiting for transcript")

                elif event_type == "error":
                    err = event.get("error", event)
                    # A commit that races an already-cleared buffer is noise.
                    if err.get("code") == "input_audio_buffer_commit_empty":
                        continue
                    print(f"\n[error] {err}")
                    if err.get("param") == "session.audio.input.turn_detection":
                        print(
                            "[hint] gpt-live-transcribe rejected this turn_detection mode. "
                            "Rerun with --turn-detection manual."
                        )

        try:
            await asyncio.gather(
                send_microphone_audio(ws, mic_queue, gate), receive(), commit_on_pause()
            )
        except (asyncio.CancelledError, KeyboardInterrupt):
            pass
        finally:
            mic.stop()
            mic.close()
            export_transcript(args, state, first_delta_latency, session_started_at)


def export_transcript(args, state: TranscriptState, first_delta_latency, started_at: float) -> None:
    payload = {
        "config": {
            "delay": args.delay,
            "prompt": args.prompt,
            "keywords": args.keywords,
            "languages": args.languages,
            "turn_detection": args.turn_detection,
        },
        "session_started_at": started_at,
        "session_duration_s": round(time.time() - started_at, 3),
        "time_to_first_delta_s": first_delta_latency,
        "final_transcript": state.full_transcript(),
        "turns": [{"item_id": i, "transcript": state.finals[i]} for i in state.order if i in state.finals],
    }
    with open(args.export, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    txt_path = args.export.rsplit(".", 1)[0] + ".txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(state.full_transcript())

    print(f"\n\nExported {args.export} and {txt_path}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
