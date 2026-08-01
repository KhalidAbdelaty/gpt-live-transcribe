# Sample audio

Test 2 and Test 3 need a fixed WAV file so every run streams identical
audio. These clips are not committed to the repository. Generate them
with `python gen_sample_audio.py` from the parent folder, which calls
OpenAI's speech endpoint with `response_format="pcm"` and wraps the
result in a WAV header, so the output already matches what
`gpt-live-transcribe` expects and needs no conversion step.

Requirements: 24 kHz, mono, 16-bit PCM, little-endian, WAV container.

Clips the generator writes:

- `clean_english.wav` (about 12s): a few sentences of clear English.
- `technical_terms.wav` (about 12s): sentences containing a product name,
  an account identifier ("AC-42"), and billing vocabulary, used for the
  keyword and prompt hint comparison in Test 2.
- `arabic_english_codeswitch.wav` (about 11s): a short clip that mixes
  Arabic and English in the same sentence, matching the code-switching
  test in the article.
- `benchmark_clip.wav` (about 27s): a longer clip used for the Test 3
  delay-level benchmark, so every delay setting streams the same audio.

Synthesized speech is convenient for keeping runs identical, but it is
cleaner than real input. Rerun anything you plan to act on against your
own microphone, accents, and background noise before trusting a result.

To record your own clip instead, with `sox`:

    sox -d -r 24000 -c 1 -b 16 -e signed-integer support_call.wav

Or convert an existing recording with `ffmpeg`:

    ffmpeg -i input.mp3 -ar 24000 -ac 1 -sample_fmt s16 support_call.wav

A `noisy_meeting.wav` clip (the same script recorded with background
noise or overlapping speech) is not included; the article does not make
claims about noisy audio, so add your own if you want to test that case.
