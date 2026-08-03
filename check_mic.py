"""Measure your microphone so you can set the speech threshold from data.

`SpeechGate` decides a turn ended by comparing each audio chunk's RMS
amplitude against a threshold. The right number depends on your microphone,
your voice, and how noisy the room is, so guessing at it produces one of two
failures: a threshold set too high cuts sentences in half whenever your voice
dips, and one set too low commits on room noise and returns empty transcripts.

This script records silence, then speech, and prints a threshold that sits
between them. No API key needed; nothing is sent anywhere.

Run with:
    python check_mic.py
"""

from __future__ import annotations

import time

import numpy as np
import sounddevice as sd

from mic_stream import BLOCK_SIZE, chunk_rms
from transcribe_lib import SAMPLE_RATE

SILENCE_SECONDS = 4
SPEECH_SECONDS = 8


def measure(seconds: int, label: str) -> list:
    """Record for a fixed time, printing a live meter, and return chunk RMS values."""
    readings = []
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=BLOCK_SIZE
    )

    with stream:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            indata, _ = stream.read(BLOCK_SIZE)
            pcm16 = (indata[:, 0] * 32767).astype(np.int16).tobytes()
            rms = chunk_rms(pcm16)
            readings.append(rms)

            bar = "#" * min(50, int(rms / 100))
            remaining = deadline - time.monotonic()
            print(f"\r{label} {remaining:4.1f}s  rms {rms:7.0f} |{bar:<50}|", end="", flush=True)

    print()
    return readings


def main() -> None:
    print(f"Using input device: {sd.query_devices(kind='input')['name']}\n")

    print(f"Stay quiet for {SILENCE_SECONDS} seconds. Starting now.")
    silence = measure(SILENCE_SECONDS, "  silence")

    print(f"\nNow talk normally for {SPEECH_SECONDS} seconds, at the volume and")
    print("distance you will actually use. Starting now.")
    speech = measure(SPEECH_SECONDS, "  speech ")

    noise_floor = float(np.percentile(silence, 95))
    # The 10th percentile, not the mean: what matters is how quiet your voice
    # gets at the ends of clauses, since that is where an early commit happens.
    quiet_speech = float(np.percentile(speech, 10))
    loud_speech = float(np.percentile(speech, 90))

    print("\n--- results ---")
    print(f"room noise (95th percentile) : {noise_floor:7.0f}")
    print(f"your quietest speech (10th)  : {quiet_speech:7.0f}")
    print(f"your loudest speech (90th)   : {loud_speech:7.0f}")

    if quiet_speech <= noise_floor * 1.5:
        print(
            "\nYour quiet speech is too close to the room noise to separate reliably."
            "\nMove closer to the microphone, speak up, or record somewhere quieter,"
            "\nthen run this again."
        )
        return

    suggested = round((noise_floor + quiet_speech) / 2, -1)
    print(f"\nSuggested --speech-threshold: {suggested:.0f}")
    print(f"    python app.py --speech-threshold {suggested:.0f}")
    print(
        "\nTo use it in test1_basic_client.py, pass it to SpeechGate:"
        f"\n    gate = SpeechGate(threshold={suggested:.0f})"
    )


if __name__ == "__main__":
    main()
