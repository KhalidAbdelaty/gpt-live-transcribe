"""Generate the sample WAV clips used by Test 2 and Test 3.

OpenAI's speech endpoint can emit raw PCM at 24 kHz mono 16-bit, which is
exactly the format gpt-live-transcribe expects, so the raw bytes only need
a WAV header wrapped around them.

Run with:
    python gen_sample_audio.py
"""

from __future__ import annotations

import os
import wave
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

OUT_DIR = Path(__file__).parent / "sample_audio"
SAMPLE_RATE = 24_000

CLIPS = {
    "clean_english": (
        "alloy",
        "Good morning everyone, thanks for joining the call today. "
        "I want to walk through the quarterly numbers before we open it up for questions. "
        "The team has been working on this report for the past two weeks.",
    ),
    "technical_terms": (
        "alloy",
        "This is a customer support call about the premium plan on account AC-42. "
        "The customer says their billing statement shows an unexpected charge "
        "and wants it reviewed before the next invoice cycle.",
    ),
    "arabic_english_codeswitch": (
        "alloy",
        "طيب يا جماعة، الـ customer عايز يعرف إيه اللي حصل في الـ billing statement بتاعه. "
        "العميل عايز يراجعها بكرة قبل الـ next invoice cycle.",
    ),
    "benchmark_clip": (
        "alloy",
        "This is a customer support call about the premium plan on account AC-42. "
        "The customer says their billing statement shows an unexpected charge "
        "and wants it reviewed before the next invoice cycle. "
        "We checked the account history and found a plan upgrade applied last month, "
        "which explains the difference between the two invoices. "
        "The support agent offered to send a written summary by email "
        "and to schedule a follow up call once the next statement is generated.",
    ),
}


def write_wav(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    for name, (voice, text) in CLIPS.items():
        response = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=text,
            response_format="pcm",
        )
        pcm = response.read()
        path = OUT_DIR / f"{name}.wav"
        write_wav(path, pcm)
        duration = len(pcm) / 2 / SAMPLE_RATE
        print(f"{path.name}: {duration:.1f}s, {len(pcm)} bytes")


if __name__ == "__main__":
    main()
