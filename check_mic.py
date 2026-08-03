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

# Opening an input stream produces a burst of noise, and the keypress that
# started the script lands in the first moments too. Both are loud enough to
# drag a percentile upward, so the opening readings are thrown away.
WARMUP_SECONDS = 0.7


def measure(seconds: int, label: str) -> list:
    """Record for a fixed time, printing a live meter, and return chunk RMS values."""
    readings = []
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=BLOCK_SIZE
    )
    warmup_chunks = int(WARMUP_SECONDS * SAMPLE_RATE / BLOCK_SIZE)

    with stream:
        deadline = time.monotonic() + seconds + WARMUP_SECONDS
        index = 0

        while time.monotonic() < deadline:
            indata, _ = stream.read(BLOCK_SIZE)
            pcm16 = (indata[:, 0] * 32767).astype(np.int16).tobytes()
            rms = chunk_rms(pcm16)
            index += 1

            if index > warmup_chunks:
                readings.append(rms)

            bar = "#" * min(50, int(rms / 100))
            remaining = max(0.0, deadline - time.monotonic())
            print(f"\r{label} {remaining:4.1f}s  rms {rms:7.0f} |{bar:<50}|", end="", flush=True)

    print()
    return readings


def main() -> None:
    print(f"Using input device: {sd.query_devices(kind='input')['name']}\n")

    print(f"Stay quiet for {SILENCE_SECONDS} seconds. Starting now.")
    silence = measure(SILENCE_SECONDS, "  silence")

    print(f"\nNow talk without stopping for {SPEECH_SECONDS} seconds, at the volume")
    print("and distance you will actually use. Read something aloud rather than")
    print("speaking in sentences, since pauses would be measured as your voice.")
    print("Starting now.")
    speech = measure(SPEECH_SECONDS, "  speech ")

    # The median, not a high percentile. Four seconds is only forty chunks, so
    # one chair creak decides a 95th percentile and reports a quiet room as a
    # loud one. The median describes the level the room actually sits at.
    noise_floor = float(np.median(silence))
    noise_peak = float(np.max(silence))

    # Only chunks clearly louder than the room count as speech. Averaging the
    # whole speech recording measures the gaps between your sentences instead
    # of your voice, which reports a "quietest speech" of nearly zero on a
    # recording where you were talking the whole time.
    speech_floor = max(noise_floor * 2, 30.0)
    active = [r for r in speech if r > speech_floor]
    talking_fraction = len(active) / len(speech) if speech else 0.0

    print("\n--- results ---")
    print(f"room noise (median)          : {noise_floor:7.0f}")
    print(f"loudest thing in the quiet    : {noise_peak:7.0f}")
    print(f"you were talking for         : {talking_fraction * 100:6.0f}% of that window")

    if talking_fraction < 0.35:
        print(
            "\nMost of the speech window was quieter than the room, so there is not"
            "\nenough voice in it to measure. Run it again and talk continuously for"
            "\nthe full eight seconds, without pausing between sentences."
        )
        return

    quiet_speech = float(np.percentile(active, 20))
    loud_speech = float(np.percentile(active, 90))
    print(f"your quieter speech (20th)   : {quiet_speech:7.0f}")
    print(f"your loudest speech (90th)   : {loud_speech:7.0f}")

    if quiet_speech <= max(noise_floor, 1.0) * 3:
        print(
            f"\nYour voice only reaches about {quiet_speech / max(noise_floor, 1.0):.1f}x the room"
            "\nlevel, which is too close to separate reliably. A fan, an air conditioner,"
            "\nor microphone boost turned on in Windows will all do this. Move closer to"
            "\nthe microphone, turn off what you can, and run it again."
        )
        return

    # Sit the threshold below the quiet end of your voice rather than midway
    # between the two levels. Room noise and speech can differ by a factor of a
    # hundred, and a midpoint on that scale lands far too close to your voice,
    # cutting a turn every time you trail off.
    suggested = max(round(quiet_speech * 0.3, -1), round(noise_floor * 3, -1), 50)
    print(f"\nSuggested --speech-threshold: {suggested:.0f}")
    print(f"    python app.py --speech-threshold {suggested:.0f}")
    print(
        "\nTo use it in test1_basic_client.py, pass it to SpeechGate:"
        f"\n    gate = SpeechGate(threshold={suggested:.0f})"
    )


if __name__ == "__main__":
    main()
