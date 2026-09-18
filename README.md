# Live Stream Transcriber

Transcribes a live video stream (YouTube Live, Twitch, or anything
[yt-dlp](https://github.com/yt-dlp/yt-dlp) supports) in near-real-time and
writes timestamped captions to a plain text file as it goes.

## How it works

1. **yt-dlp** resolves the live page URL into a direct, playable stream URL.
2. **ffmpeg** reads that stream and decodes it to raw 16kHz mono PCM audio.
3. Audio is buffered into fixed-length chunks (8 seconds by default).
4. Each chunk is transcribed locally using **faster-whisper** (a fast
   reimplementation of OpenAI's Whisper model).
5. Each resulting caption is appended to the output file in the format:

```
[00:00:12.400 --> 00:00:15.800] This is an example caption.
[00:00:15.800 --> 00:00:19.100] Another line of transcribed speech.
```

The file is flushed after every line, so you can watch it live with:

```
tail -f transcript.txt
```
6. Each chunk is transcribed together with a small overlap (1 second by
default) of audio from the end of the previous chunk. This gives Whisper
context across the boundary so it doesn't cut words off mid-sentence;
captions from the overlapped region are only kept once, not duplicated.
7. If the stream connection drops (network hiccup, the CDN rotating the
underlying URL, etc.), the script automatically re-resolves the stream
URL and reconnects ffmpeg, up to `--max-retries` times, before giving up
and exiting cleanly with everything transcribed so far saved.
8. Some "live" streams are actually a short video looped 24/7 rather than
genuinely live content — in that case ffmpeg never disconnects, so the
script would otherwise transcribe the same content over and over
forever. To handle this, it remembers the first substantial caption as
a "signature"; once a later caption closely matches it again (after
`--loop-grace-seconds` of real playback has elapsed, so the opening
line isn't matched against itself), it's treated as the video looping
back to the start, and transcription stops there.
9. Once transcription stops — whether from loop detection, the stream
ending, `--stop-after-seconds`, or Ctrl+C — the full transcript from
this run is sent to a local Ollama model, which writes a short,
readable summary to `summary.txt`.

## Summarization (Ollama)

Summarization runs locally through [Ollama](https://ollama.com/) — no API
key, no data leaving your machine.

**One-time setup:**

```
# Install Ollama: see https://ollama.com/download
ollama serve &          # if it isn't already running as a service
ollama pull llama3.2    # or any other model you prefer
```

Summarization is on by default. After the transcript finishes, you'll see:

```
Generating summary via Ollama (model: llama3.2)...
Summary written to: /path/to/summary.txt
```

To skip it, pass `--no-summarize`. To use a different model or a
non-default Ollama host:

```
python live_transcriber.py "https://www.youtube.com/watch?v=XXXXXXXX" \
   --ollama-model mistral \
   --ollama-host http://localhost:11434
```

If Ollama isn't running or the model isn't pulled, the script prints a
clear error but still keeps your transcript — only the summary step fails.

## Setup

1. Install [ffmpeg](https://ffmpeg.org/download.html) and make sure it's on
   your PATH (`ffmpeg -version` should work in a terminal).
2. Install Python dependencies:

```
pip install -r requirements.txt
```

The first run will also download the Whisper model weights
(a few hundred MB depending on model size) — this requires internet
access once, then it's cached locally.

## Usage

```
python live_transcriber.py "https://www.youtube.com/watch?v=XXXXXXXX"
```

Common options:

```
python live_transcriber.py "https://www.twitch.tv/somechannel" \
   --output transcript.txt \
   --model small \
   --chunk-seconds 8 \
   --overlap-seconds 1 \
   --language en \
   --max-retries 5 \
   --retry-delay 5
```

Flag
Default
Description

`--output`
`transcript.txt`
Path to the output text file

`--model`
`small`
Whisper model size: `tiny`, `base`, `small`, `medium`, `large-v3`

`--chunk-seconds`
`8.0`
Length of each new audio chunk sent to the model

`--overlap-seconds`
`1.0`
Seconds of the previous chunk re-included as context, to avoid cutting words at chunk boundaries

`--language`
auto-detect
Force a language code (e.g. `en`, `es`) to skip language detection

`--device`
`cpu`
`cpu` or `cuda` (use `cuda` if you have an NVIDIA GPU + CUDA installed)

`--max-retries`
`5`
Max consecutive reconnect attempts if the stream connection drops before giving up

`--retry-delay`
`5.0`
Seconds to wait between reconnect attempts

`--stop-after-seconds`
none
Hard cutoff: stop after this many seconds of stream audio, regardless of anything else

`--no-loop-detect`
(detect on)
Disable automatic detection of looping/repeating streams (see below)

`--loop-similarity`
`0.85`
How closely a later caption must match the opening line to be treated as a loop restart (0-1)

`--loop-grace-seconds`
`15.0`
Don't check for loop repeats until this many seconds have played, so the opening line isn't self-matched

`--no-summarize`
(summary on)
Disable the end-of-run Ollama summary step

`--summary-output`
`summary.txt`
Output path for the generated summary

`--ollama-model`
`llama3.2`
Ollama model to use for summarization (must already be pulled)

`--ollama-host`
`http://localhost:11434`
Base URL of the Ollama server

Press `Ctrl+C` to stop transcribing at any time.

## Notes & tuning tips

- **Model size vs. speed**: `tiny`/`base` are fast enough to keep up with
live audio on most CPUs but less accurate. `small` is a good default
balance. `medium`/`large-v3` are more accurate but may fall behind
real-time on CPU — use `--device cuda` if you have a GPU.
- **Chunk size**: Smaller chunks (e.g. 4-5s) give lower latency captions but
can cut sentences awkwardly and cost more overhead per chunk. Larger
chunks (10-15s) give more context to the model but higher latency.
- **Twitch streams**: yt-dlp needs to be reasonably up to date to resolve
Twitch URLs reliably, since Twitch changes its player occasionally. Run
`pip install -U yt-dlp` if resolution fails.
- **Restarts**: if the stream drops, the script will automatically
re-resolve the URL and reconnect ffmpeg, up to `--max-retries` times. If it
still can't recover, it exits cleanly and your transcript file keeps
everything captured up to that point.
- **Chunk boundary quality**: if you still see occasional odd splits or
slightly overlapping timestamps in the output, try increasing
`--overlap-seconds` (e.g. to 2) for more cross-chunk context, at the cost
of a small amount of extra compute per chunk.
video, `--stop-after-seconds <N>` is the most reliable way to cut things
off precisely. Loop detection (on by default) is a good fallback when you
don't know the length ahead of time, but if the video's opening line is
`--no-loop-detect` and rely on `--stop-after-seconds` instead.
# Live Stream Transcriber
Transcribes a live video stream (YouTube Live, Twitch, or anything
[yt-dlp](https://github.com/yt-dlp/yt-dlp) supports) in near-real-time and

## How it works

2. **ffmpeg** reads that stream and decodes it to raw 16kHz mono PCM audio.
3. Audio is buffered into fixed-length chunks (8 seconds by default).
4. Each chunk is transcribed locally using **faster-whisper** (a fast
5. Each resulting caption is appended to the output file in the format:

   ```
   [00:00:12.400 --> 00:00:15.800] This is an example caption.
   [00:00:15.800 --> 00:00:19.100] Another line of transcribed speech.
   ```

   The file is flushed after every line, so you can watch it live with:

   ```bash
   tail -f transcript.txt
   ```

6. Each chunk is transcribed together with a small overlap (1 second by
   default) of audio from the end of the previous chunk. This gives Whisper
   context across the boundary so it doesn't cut words off mid-sentence;
   captions from the overlapped region are only kept once, not duplicated.
7. If the stream connection drops (network hiccup, the CDN rotating the
   underlying URL, etc.), the script automatically re-resolves the stream
   URL and reconnects ffmpeg, up to `--max-retries` times, before giving up
   and exiting cleanly with everything transcribed so far saved.
8. Some "live" streams are actually a short video looped 24/7 rather than
   genuinely live content — in that case ffmpeg never disconnects, so the
   script would otherwise transcribe the same content over and over
   forever. To handle this, it remembers the first substantial caption as
   a "signature"; once a later caption closely matches it again (after
   `--loop-grace-seconds` of real playback has elapsed, so the opening
   line isn't matched against itself), it's treated as the video looping
   back to the start, and transcription stops there.

## Setup

1. Install [ffmpeg](https://ffmpeg.org/download.html) and make sure it's on
   your PATH (`ffmpeg -version` should work in a terminal).
2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   The first run will also download the Whisper model weights
   (a few hundred MB depending on model size) — this requires internet
   access once, then it's cached locally.

## Usage

```bash
python live_transcriber.py "https://www.youtube.com/watch?v=XXXXXXXX"
```

Common options:

```bash
python live_transcriber.py "https://www.twitch.tv/somechannel" \
    --output transcript.txt \
    --model small \
    --chunk-seconds 8 \
    --overlap-seconds 1 \
    --language en \
    --max-retries 5 \
    --retry-delay 5
```

| Flag                | Default          | Description                                                                                          |
|---------------------|------------------|--------------------------------------------------------------------------------------------------------|
| `--output`          | `transcript.txt` | Path to the output text file                                                                           |
| `--model`           | `small`          | Whisper model size: `tiny`, `base`, `small`, `medium`, `large-v3`                                       |
| `--chunk-seconds`   | `8.0`            | Length of each new audio chunk sent to the model                                                       |
| `--overlap-seconds` | `1.0`            | Seconds of the previous chunk re-included as context, to avoid cutting words at chunk boundaries       |
| `--language`        | auto-detect      | Force a language code (e.g. `en`, `es`) to skip language detection                                     |
| `--device`          | `cpu`            | `cpu` or `cuda` (use `cuda` if you have an NVIDIA GPU + CUDA installed)                                 |
| `--max-retries`     | `5`              | Max consecutive reconnect attempts if the stream connection drops before giving up                      |
| `--retry-delay`     | `5.0`            | Seconds to wait between reconnect attempts                                                             |
| `--stop-after-seconds` | none          | Hard cutoff: stop after this many seconds of stream audio, regardless of anything else                 |
| `--no-loop-detect`  | (detect on)      | Disable automatic detection of looping/repeating streams (see below)                                    |
| `--loop-similarity` | `0.85`           | How closely a later caption must match the opening line to be treated as a loop restart (0-1)           |
| `--loop-grace-seconds` | `15.0`        | Don't check for loop repeats until this many seconds have played, so the opening line isn't self-matched|

Press `Ctrl+C` to stop transcribing at any time.

## Notes & tuning tips

- **Model size vs. speed**: `tiny`/`base` are fast enough to keep up with
  live audio on most CPUs but less accurate. `small` is a good default
  balance. `medium`/`large-v3` are more accurate but may fall behind
  real-time on CPU — use `--device cuda` if you have a GPU.
- **Chunk size**: Smaller chunks (e.g. 4-5s) give lower latency captions but
  can cut sentences awkwardly and cost more overhead per chunk. Larger
  chunks (10-15s) give more context to the model but higher latency.
- **Twitch streams**: yt-dlp needs to be reasonably up to date to resolve
  Twitch URLs reliably, since Twitch changes its player occasionally. Run
  `pip install -U yt-dlp` if resolution fails.
- **Restarts**: if the stream drops, the script will automatically
  re-resolve the URL and reconnect, up to `--max-retries` times. If it still
  can't recover, it exits cleanly and your transcript file keeps everything
  captured up to that point.
- **Chunk boundary quality**: if you still see occasional odd splits or
  slightly overlapping timestamps in the output, try increasing
  `--overlap-seconds` (e.g. to 2) for more cross-chunk context, at the cost
  of a small amount of extra compute per chunk.
- **Looping "live" streams**: if you know the exact length of the source
  video, `--stop-after-seconds <N>` is the most reliable way to cut things
  off precisely. Loop detection (on by default) is a good fallback when you
  don't know the length ahead of time, but if the video's opening line is
  short or generic, consider lowering `--loop-similarity` slightly, or pass
  `--no-loop-detect` and rely on `--stop-after-seconds` instead.