"""Test 3: benchmark the five delay levels on identical audio.

Measures, for each of minimal / low / medium / high / xhigh:
  - time to first delta (seconds from stream start to the first partial text)
  - time to final transcript (seconds from stream start to the completed event)
  - revision count (how many delta events arrived before the final text,
    a rough proxy for how much the partial caption changed on screen)

OpenAI's docs are explicit that exact millisecond timing varies by model
configuration and network conditions, so this script reports what it
measures on your connection rather than assuming a fixed number per level.
Run it more than once and take the median if you plan to quote a figure.

Run with:
    python test3_delay_benchmark.py sample_audio/support_call.wav --runs 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time

import websockets
from dotenv import load_dotenv

from transcribe_lib import WS_URL, TranscriptionConfig, stream_wav_realtime

load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

DELAY_LEVELS = ["minimal", "low", "medium", "high", "xhigh"]


async def benchmark_once(delay: str, wav_path: str) -> dict:
    config = TranscriptionConfig(delay=delay)
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

    start = None
    first_delta_at = None
    final_at = None
    delta_count = 0

    async with websockets.connect(WS_URL, additional_headers=headers) as ws:
        await ws.send(json.dumps(config.to_session_update()))

        async def receive() -> None:
            nonlocal first_delta_at, final_at, delta_count
            async for raw in ws:
                event = json.loads(raw)
                if event["type"] == "conversation.item.input_audio_transcription.delta":
                    delta_count += 1
                    if first_delta_at is None:
                        first_delta_at = time.monotonic()
                elif event["type"] == "conversation.item.input_audio_transcription.completed":
                    final_at = time.monotonic()

        receiver = asyncio.create_task(receive())
        start = time.monotonic()
        await stream_wav_realtime(ws, wav_path)
        await asyncio.sleep(2)  # allow the trailing completed event to land
        receiver.cancel()

    return {
        "delay": delay,
        "time_to_first_delta_s": round(first_delta_at - start, 3) if first_delta_at else None,
        "time_to_final_s": round(final_at - start, 3) if final_at else None,
        "delta_event_count": delta_count,
    }


async def main(wav_path: str, runs: int) -> None:
    all_results = {level: [] for level in DELAY_LEVELS}

    for level in DELAY_LEVELS:
        for run_index in range(runs):
            result = await benchmark_once(level, wav_path)
            all_results[level].append(result)
            print(f"{level} (run {run_index + 1}/{runs}): {result}")

    summary = []
    for level, runs_for_level in all_results.items():
        first_deltas = [r["time_to_first_delta_s"] for r in runs_for_level if r["time_to_first_delta_s"]]
        finals = [r["time_to_final_s"] for r in runs_for_level if r["time_to_final_s"]]
        summary.append(
            {
                "delay": level,
                "median_time_to_first_delta_s": statistics.median(first_deltas) if first_deltas else None,
                "median_time_to_final_s": statistics.median(finals) if finals else None,
            }
        )

    with open("delay_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump({"raw_runs": all_results, "summary": summary}, f, indent=2)

    print("\n=== Median summary (this connection, this audio) ===")
    for row in summary:
        print(row)
    print("\nSaved delay_benchmark_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wav_path", help="Path to a 24 kHz mono PCM16 WAV file")
    parser.add_argument("--runs", type=int, default=3, help="Runs per delay level")
    args = parser.parse_args()
    asyncio.run(main(args.wav_path, args.runs))
