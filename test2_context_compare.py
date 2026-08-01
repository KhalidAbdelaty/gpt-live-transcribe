"""Test 2: compare no-context, prompt, keywords, and language hints.

Streams the same WAV file through each configuration, several passes each,
and prints every transcript. Using a recorded file instead of a live mic
keeps the audio identical across runs, which is the only way a context-hint
comparison means anything.

Two rules keep the comparison honest. Only one context field changes between
runs, so a difference can be attributed to that field. And each configuration
runs more than once, because the model is not deterministic on the same audio
and a single pass can show a difference that disappears on the next one.

Run with:
    python test2_context_compare.py sample_audio/technical_terms.wav --passes 3
    python test2_context_compare.py sample_audio/arabic_english_codeswitch.wav --languages ar,en

See sample_audio/README.md for how to generate or record the clips.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

import websockets
from dotenv import load_dotenv

from transcribe_lib import WS_URL, TranscriptionConfig, TranscriptState, stream_wav_realtime

load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

PROMPT = "A customer support call about a premium plan and account AC-42."
KEYWORDS = ["premium plan", "AC-42", "billing"]


def build_runs(languages: list) -> dict:
    return {
        "no_context": TranscriptionConfig(delay="low"),
        "prompt_only": TranscriptionConfig(delay="low", prompt=PROMPT),
        "keywords_only": TranscriptionConfig(delay="low", keywords=KEYWORDS),
        "languages_only": TranscriptionConfig(delay="low", languages=languages),
        "prompt_and_keywords": TranscriptionConfig(delay="low", prompt=PROMPT, keywords=KEYWORDS),
    }


async def run_once(label: str, config: TranscriptionConfig, wav_path: str) -> str:
    state = TranscriptState()
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

    async with websockets.connect(WS_URL, additional_headers=headers) as ws:
        await ws.send(json.dumps(config.to_session_update()))

        async def receive() -> None:
            async for raw in ws:
                event = json.loads(raw)
                if event["type"] == "conversation.item.input_audio_transcription.completed":
                    state.apply_completed(event["item_id"], event["transcript"])
                elif event["type"] == "error":
                    print(f"[error] {event.get('error')}")

        receiver = asyncio.create_task(receive())
        await stream_wav_realtime(ws, wav_path)
        await asyncio.sleep(3)  # give the final completed event time to arrive
        receiver.cancel()

    transcript = state.full_transcript()
    print(f"[{label}] {transcript}")
    return transcript


async def main(wav_path: str, passes: int, languages: list) -> None:
    results: dict[str, list] = {}
    for label, config in build_runs(languages).items():
        results[label] = [await run_once(label, config, wav_path) for _ in range(passes)]
        print()

    with open("context_comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Saved context_comparison_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wav_path", help="Path to a 24 kHz mono PCM16 WAV file")
    parser.add_argument("--passes", type=int, default=3, help="Runs per configuration")
    parser.add_argument(
        "--languages",
        default="en",
        help="Comma-separated codes used by the languages_only run",
    )
    args = parser.parse_args()
    asyncio.run(
        main(args.wav_path, args.passes, [c.strip() for c in args.languages.split(",")])
    )
