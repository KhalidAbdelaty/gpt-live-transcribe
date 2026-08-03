# GPT Live Transcribe tutorial code

Companion code for the DataCamp tutorial "GPT Live Transcribe API Tutorial:
Build Real-Time Captions in Python." Every script in this folder was
syntax-checked with `python -m py_compile` and unit-tested against the
shared helpers in `transcribe_lib.py` (session config building, keyword
validation, WAV format checks, and out-of-order transcript reconciliation).

## Setup

    pip install -r requirements.txt
    cp .env.example .env   # then paste your real OPENAI_API_KEY into .env
    python gen_sample_audio.py   # writes the sample_audio/ clips

On Windows, set `PYTHONIOENCODING=utf-8` before running anything that may
print non-Latin transcripts, or the terminal raises `UnicodeEncodeError`
on Arabic output before you see it.

On macOS, `sounddevice` needs PortAudio installed at the OS level:

    brew install portaudio

On Linux:

    sudo apt-get install portaudio19-dev

## Files

- `transcribe_lib.py`: session config builder, WAV/PCM helpers, and the
  `TranscriptState` class used to reconcile out-of-order completion events.
- `mic_stream.py`: non-blocking microphone capture with `sounddevice`,
  handing audio off to asyncio through a thread-safe queue. Also holds
  `SpeechGate`, the client-side voice activity detector that decides when to
  commit a turn, since the API will not run one for this model.
- `test1_basic_client.py`: the basic live client from the "Building a Basic
  Live Transcription Client" section. Manual commits, no VAD.
- `gen_sample_audio.py`: writes the four `sample_audio/` clips through
  OpenAI's speech endpoint, already in 24 kHz mono PCM16. Run this once
  before Test 2 or Test 3.
- `check_mic.py`: records four seconds of silence and eight of speech and
  reports how far apart they are. `SpeechGate` learns the room on its own, so
  run this only when turns still end in the wrong places: it says whether the
  microphone can separate your voice from the room at all, which no tuning
  fixes if the answer is no. Needs no API key and sends nothing anywhere.
- `test2_context_compare.py`: streams one WAV file through five context
  configurations (no context, prompt-only, keywords-only, languages-only,
  prompt plus keywords), several passes each, so you can compare
  transcripts. Only one field changes per run, and `--passes` defaults to
  3 because the model is not deterministic on identical audio.
- `test3_delay_benchmark.py`: streams one WAV file across all five delay
  levels and logs time-to-first-delta and time-to-final per run.
- `app.py`: the full captioning app with CLI flags for delay, prompt,
  keywords, languages, and turn-detection mode, plus JSON/text export.
- `demo_app.py`: a Streamlit demo of the same three experiments in a
  browser. Record a clip and watch partial captions arrive and finalize,
  compare context configurations side by side, or benchmark the five delay
  levels with a chart. Run it with `streamlit run demo_app.py`.
- `assets/`: the DataCamp wordmark and mark used by the demo. Both are
  inlined as base64 data URIs, so the app needs no static file server and
  the logo survives deployment to Streamlit Community Cloud.
- `plot_delay_benchmark.py`: turns `delay_benchmark_results.json` (written
  by `test3_delay_benchmark.py`) into the delay-versus-latency chart used
  in the article. Run it after the benchmark script.
- `sample_audio/README.md`: how to record or convert WAV clips in the
  format gpt-live-transcribe expects (24 kHz, mono, 16-bit PCM).

## Notes

- All three test scripts use the same `TranscriptionConfig` and
  `TranscriptState` classes, so the only thing that changes between runs is
  the field the article is testing.
- The clips in `sample_audio/` are synthesized speech, which keeps repeated
  runs comparable but is cleaner than real input. Rerun anything you plan
  to act on against your own microphone and background noise.
- The terminal scripts capture audio with `sounddevice` and talk to the API
  through the asyncio `websockets` library. `demo_app.py` records in the
  browser instead and uses the threaded `websocket-client` library, which
  fits Streamlit's execution model better. Both send the same events.
- If you deploy `demo_app.py`, put `OPENAI_API_KEY` in Streamlit Secrets
  rather than in `.env`, and never commit `.streamlit/secrets.toml`.
- Benchmark numbers from `test3_delay_benchmark.py` are specific to your
  network connection, hardware, and audio sample. Run it against your own
  representative audio before trusting a number enough to put it in a
  production config.
