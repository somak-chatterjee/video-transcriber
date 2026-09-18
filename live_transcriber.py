#!/usr/bin/env python3
"""
live_transcriber.py

Transcribes a live video stream (YouTube Live, Twitch, or any URL yt-dlp
supports) in near-real-time and writes timestamped captions to a text file.

How it works:
  1. yt-dlp resolves the live page URL into a direct, playable stream URL.
  2. ffmpeg reads that stream and outputs raw 16kHz mono PCM audio.
  3. Audio is buffered in overlapping chunks (default 8s, 1s overlap) so
     Whisper has a little context around chunk boundaries and doesn't cut
     words mid-sentence.
  4. Each chunk is transcribed locally with faster-whisper.
  5. Each resulting caption is appended to the output .txt file with a
     [HH:MM:SS.mmm --> HH:MM:SS.mmm] timestamp, and the file is flushed
     immediately so it can be tailed live (`tail -f transcript.txt`).
  6. If the stream connection drops (network hiccup, CDN URL rotation,
     etc.), the script automatically re-resolves the URL and reconnects
     ffmpeg, up to --max-retries times, before giving up.
  7. Once transcription stops (loop detected, stream ended, or a manual
     cutoff is reached), the full transcript is sent to a local Ollama
     model, which writes a plain-language summary to summary.txt.

Usage:
  python live_transcriber.py "https://www.youtube.com/watch?v=XXXXXXXX" \
      --output transcript.txt \
      --model small \
      --chunk-seconds 8 \
      --overlap-seconds 1 \
      --language en \
      --ollama-model llama3.2

Requirements: see requirements.txt (pip install -r requirements.txt)
ffmpeg must also be installed and on PATH.
Ollama must be installed and running (ollama serve) with the chosen model
pulled (ollama pull llama3.2) for the summarization step; pass
--no-summarize to skip it.
"""

import argparse
import datetime as dt
import difflib
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # s16le = 16-bit little-endian


def extract_video_metadata(info: dict) -> dict:
    """Pull out the fields useful as summarization context from yt-dlp's info dict."""
    return {
        "title": info.get("title"),
        "description": info.get("description"),
        "uploader": info.get("uploader") or info.get("channel"),
        "upload_date": info.get("upload_date"),  # YYYYMMDD
        "categories": info.get("categories"),
        "tags": info.get("tags"),
    }


def resolve_stream_url(page_url: str):
    """
    Use yt-dlp to resolve a live page URL to a direct audio/video stream URL,
    and pull out video metadata (title, description, etc.) as summarization
    context. Returns (direct_url, metadata_dict).
    """
    import yt_dlp

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "format": "bestaudio/best",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(page_url, download=False)
        metadata = extract_video_metadata(info)

        if "url" in info:
            return info["url"], metadata
        formats = info.get("formats") or []
        if not formats:
            raise RuntimeError("yt-dlp could not find a playable stream URL.")
        return formats[-1]["url"], metadata


def open_ffmpeg_audio_pipe(stream_url: str) -> subprocess.Popen:
    """Launch ffmpeg to continuously decode the stream to raw PCM on stdout."""
    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-reconnect", "1",
        "-reconnect_streamed", "1",
        "-reconnect_delay_max", "5",
        "-i", stream_url,
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-",
    ]
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10 ** 7,
    )


def normalize_text(text: str) -> str:
    """Lowercase and strip punctuation/whitespace so text comparisons are robust
    to minor transcription differences between loop passes."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def loop_match_ratio(a: str, b: str) -> float:
    """
    Compare two normalized caption strings for a likely repeat, using the
    longest common substring relative to the shorter string. This is more
    tolerant than a plain SequenceMatcher ratio when the same spoken line
    gets split into differently-sized captions across loop passes (e.g. a
    chunk boundary truncates it one time but not the next) - a plain ratio
    penalizes for length differences that plain truncation shouldn't count
    against a match.
    """
    if not a or not b:
        return 0.0
    matcher = difflib.SequenceMatcher(None, a, b)
    match = matcher.find_longest_match(0, len(a), 0, len(b))
    shorter = min(len(a), len(b))
    return match.size / shorter if shorter else 0.0


def format_timestamp(seconds: float) -> str:
    td = dt.timedelta(seconds=max(0, seconds))
    total_ms = int(td.total_seconds() * 1000)
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def build_summary_prompt(text: str, metadata: dict = None) -> str:
    """
    Build the summarization prompt, folding in video metadata (title,
    description, uploader, etc.) as context when available so the model
    can ground the summary in what the video is actually about, rather
    than working from the transcript text alone.
    """
    metadata = metadata or {}
    context_lines = []

    title = metadata.get("title")
    if title:
        context_lines.append(f"Title: {title}")

    uploader = metadata.get("uploader")
    if uploader:
        context_lines.append(f"Channel/Uploader: {uploader}")

    categories = metadata.get("categories")
    if categories:
        context_lines.append(f"Category: {', '.join(categories)}")

    tags = metadata.get("tags")
    if tags:
        # Keep this short - tag lists can be long and low-signal.
        context_lines.append(f"Tags: {', '.join(tags[:10])}")

    description = metadata.get("description")
    if description:
        # Truncate long descriptions so they inform the prompt without
        # dominating it or eating too much of the model's context window.
        trimmed = description.strip()
        if len(trimmed) > 800:
            trimmed = trimmed[:800].rsplit(" ", 1)[0] + "..."
        context_lines.append(f"Video description: {trimmed}")

    context_block = ""
    if context_lines:
        context_block = (
            "Here is some context about the video this transcript comes from:\n"
            + "\n".join(context_lines) + "\n\n"
        )

    if context_lines:
        # We have real metadata, so ask for a structured summary that keeps
        # "what the video is" (from metadata) separate from "what was said"
        # (from the transcript) - otherwise models tend to blend the two, or
        # lean entirely on whichever source has punchier, easier language
        # (usually the description) rather than actually engaging with the
        # transcript content.
        instructions = (
            "Using the above, write a two-part summary with these exact headings:\n\n"
            "## Context\n"
            "In 1-3 sentences, describe what this video is / what it's about, "
            "based on the title, uploader, description, and tags above. This is "
            "background, not a summary of the transcript.\n\n"
            "## What Was Said\n"
            "Summarize the actual spoken content from the transcript below, in "
            "your own words, as a short paragraph or a few bullet points. Focus "
            "on the key points and overall message of what the speaker says. Use "
            "the Context section only to understand names, references, and "
            "subject matter, correcting for likely transcription errors - do not "
            "just restate the description here, and do not invent content that "
            "isn't actually in the transcript.\n\n"
            "Do not mention that this is a transcript or that it came from "
            "speech-to-text."
        )
    else:
        # No metadata available - fall back to a plain single-section summary.
        instructions = (
            "Write a clear, concise summary of what was said in the transcript "
            "below, in your own words, capturing the key points and overall "
            "message. Do not mention that this is a transcript or that it came "
            "from speech-to-text."
        )

    return (
        f"{context_block}"
        f"{instructions}\n\n"
        f"Transcript:\n{text}\n\nSummary:"
    )


def summarize_with_ollama(text: str, model: str, host: str, metadata: dict = None, timeout: float = 180.0) -> str:
    """
    Send transcribed text (plus optional video metadata as context) to a
    local Ollama server for summarization. Requires Ollama to be installed
    and running (`ollama serve`), with the requested model already pulled
    (`ollama pull <model>`).
    """
    import requests

    prompt = build_summary_prompt(text, metadata)

    response = requests.post(
        f"{host.rstrip('/')}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    summary = data.get("response", "").strip()
    if not summary:
        raise RuntimeError(f"Ollama returned an empty response: {data}")
    return summary


class ReconnectingStreamReader:
    """
    Wraps ffmpeg's audio pipe and transparently reconnects (by re-resolving
    the stream URL via yt-dlp and restarting ffmpeg) if the connection drops.
    """

    def __init__(self, page_url: str, max_retries: int = 5, retry_delay: float = 5.0):
        self.page_url = page_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.proc = None
        self.metadata = {}
        self._connect()

    def _connect(self):
        print(f"Resolving stream URL via yt-dlp: {self.page_url}")
        direct_url, metadata = resolve_stream_url(self.page_url)
        self.metadata = metadata
        print("Starting ffmpeg audio pipe...")
        self.proc = open_ffmpeg_audio_pipe(direct_url)

    def _terminate_current(self):
        if self.proc is None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()

    def read(self, n: int) -> bytes:
        """
        Read exactly n bytes, reconnecting on failure. Returns fewer than n
        bytes (possibly zero) only once retries are exhausted, signaling the
        caller that the stream has truly ended.
        """
        buf = b""
        retries = 0
        while len(buf) < n:
            chunk = self.proc.stdout.read(n - len(buf))
            if chunk:
                buf += chunk
                continue

            # No data came through: ffmpeg likely died or the stream dropped.
            err = self.proc.stderr.read().decode(errors="ignore") if self.proc.stderr else ""
            self._terminate_current()

            if retries >= self.max_retries:
                if err:
                    print(f"ffmpeg reported: {err.strip()}", file=sys.stderr)
                return buf

            retries += 1
            print(
                f"Stream connection dropped (reconnect attempt {retries}/{self.max_retries}). "
                f"Retrying in {self.retry_delay:.0f}s..."
            )
            if err:
                print(f"  ffmpeg said: {err.strip()}", file=sys.stderr)
            time.sleep(self.retry_delay)

            try:
                self._connect()
            except Exception as e:
                print(f"  Reconnect attempt failed: {e}", file=sys.stderr)
                continue

        return buf

    def close(self):
        self._terminate_current()


def main():
    parser = argparse.ArgumentParser(description="Transcribe a live video stream to a timestamped text file.")
    parser.add_argument("url", help="Live stream URL (YouTube Live, Twitch, etc.)")
    parser.add_argument("--output", default="transcript.txt", help="Output text file path")
    parser.add_argument(
        "--model",
        default="small",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="faster-whisper model size (bigger = more accurate, slower)",
    )
    parser.add_argument("--chunk-seconds", type=float, default=8.0, help="Length of each new audio chunk")
    parser.add_argument(
        "--overlap-seconds",
        type=float,
        default=1.0,
        help="Seconds of audio from the end of the previous chunk to re-include as context "
             "for the next chunk, to reduce words being cut at chunk boundaries",
    )
    parser.add_argument("--language", default=None, help="Force a language code (e.g. 'en'); default: auto-detect")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Inference device")
    parser.add_argument("--max-retries", type=int, default=5, help="Max consecutive reconnect attempts per stall")
    parser.add_argument("--retry-delay", type=float, default=5.0, help="Seconds to wait between reconnect attempts")
    parser.add_argument(
        "--stop-after-seconds",
        type=float,
        default=None,
        help="Hard cutoff: stop transcribing after this many seconds of stream audio, "
             "regardless of anything else. Useful if you know the video's length.",
    )
    parser.add_argument(
        "--no-loop-detect",
        dest="loop_detect",
        action="store_false",
        default=True,
        help="Disable automatic detection of looping/repeating streams (some 'live' streams "
             "are really a short VOD replayed on a loop). Enabled by default.",
    )
    parser.add_argument(
        "--loop-similarity",
        type=float,
        default=0.85,
        help="Similarity ratio (0-1) a later caption must have to the very first caption "
             "to be considered the stream looping back to the start",
    )
    parser.add_argument(
        "--loop-grace-seconds",
        type=float,
        default=15.0,
        help="Don't start checking for loop repeats until this many seconds of audio "
             "have been transcribed, so the opening lines aren't matched against themselves",
    )
    parser.add_argument(
        "--no-summarize",
        dest="summarize",
        action="store_false",
        default=True,
        help="Disable generating a summary of the transcript at the end of the run "
             "(enabled by default; requires a local Ollama server)",
    )
    parser.add_argument("--summary-output", default="summary.txt", help="Output path for the generated summary")
    parser.add_argument(
        "--ollama-model",
        default="llama3.2",
        help="Ollama model to use for summarization (must already be pulled, e.g. `ollama pull llama3.2`)",
    )
    parser.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
        help="Base URL of the Ollama server",
    )
    args = parser.parse_args()

    from faster_whisper import WhisperModel

    print(f"Loading faster-whisper model '{args.model}' on {args.device}...")
    model = WhisperModel(args.model, device=args.device, compute_type="int8" if args.device == "cpu" else "float16")

    try:
        reader = ReconnectingStreamReader(args.url, max_retries=args.max_retries, retry_delay=args.retry_delay)
    except Exception as e:
        print(f"Failed to start stream: {e}", file=sys.stderr)
        sys.exit(1)

    chunk_bytes = int(args.chunk_seconds * SAMPLE_RATE * BYTES_PER_SAMPLE)
    overlap_bytes = int(args.overlap_seconds * SAMPLE_RATE * BYTES_PER_SAMPLE)
    out_path = Path(args.output)

    elapsed = 0.0        # timestamp at which the next *new* chunk begins
    prev_tail = b""      # last overlap_bytes of the previous new chunk, used as context
    first_caption_norm = None   # normalized text of the very first caption, used as a loop signature
    loop_detected = False
    last_written_end = 0.0   # absolute stream time up to which we've actually written a transcript
    EPS = 0.05                # small tolerance for float boundary comparisons
    session_lines = []        # plain caption text (no timestamps) written this run, for summarization

    print(f"Writing live transcript to: {out_path.resolve()}")
    print("Press Ctrl+C to stop.\n")

    try:
        with out_path.open("a", encoding="utf-8") as f:
            while True:
                if args.stop_after_seconds is not None and elapsed >= args.stop_after_seconds:
                    print(f"Reached --stop-after-seconds cutoff ({args.stop_after_seconds}s). Stopping.")
                    break

                new_raw = reader.read(chunk_bytes)
                if not new_raw:
                    print("Stream ended.")
                    break

                is_final_chunk = len(new_raw) < chunk_bytes
                new_duration = len(new_raw) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
                overlap_duration = len(prev_tail) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
                chunk_start_time = elapsed - overlap_duration

                padded_new = new_raw
                if is_final_chunk:
                    padded_new = new_raw + b"\x00" * (chunk_bytes - len(new_raw))

                combined = prev_tail + padded_new
                audio = np.frombuffer(combined, dtype=np.int16).astype(np.float32) / 32768.0

                segments, _info = model.transcribe(
                    audio,
                    language=args.language,
                    vad_filter=True,
                    word_timestamps=True,
                )

                for seg in segments:
                    words = seg.words or []

                    if words:
                        # Word-level filtering: keep only the words that extend past
                        # whatever we've already written, so a chunk boundary that
                        # missed a word the first time around doesn't lose it, while
                        # words genuinely already covered aren't repeated.
                        kept = []
                        for w in words:
                            w_abs_start = chunk_start_time + w.start
                            w_abs_end = chunk_start_time + w.end
                            if w_abs_end <= last_written_end + EPS:
                                continue
                            kept.append((w_abs_start, w_abs_end, w.word))
                        if not kept:
                            continue
                        seg_abs_start = kept[0][0]
                        seg_abs_end = kept[-1][1]
                        text = "".join(w[2] for w in kept).strip()
                    else:
                        # Rare fallback if word timestamps aren't available for this segment.
                        seg_abs_start = chunk_start_time + seg.start
                        seg_abs_end = chunk_start_time + seg.end
                        if seg_abs_end <= last_written_end + EPS:
                            continue
                        text = seg.text.strip()

                    if not text:
                        continue

                    norm = normalize_text(text)

                    if args.loop_detect:
                        if first_caption_norm is None and len(norm) >= 20:
                            # Anchor the loop signature on the first substantial caption.
                            # Require a reasonably long line so we don't anchor on a short,
                            # generic phrase (e.g. "we find", "thank you") that could
                            # plausibly recur mid-video and cause a false loop detection.
                            first_caption_norm = norm
                        elif (
                            first_caption_norm is not None
                            and seg_abs_start >= args.loop_grace_seconds
                        ):
                            similarity = loop_match_ratio(norm, first_caption_norm)
                            if similarity >= args.loop_similarity:
                                print(
                                    f"\nDetected the stream looping back to its start "
                                    f"(caption at {format_timestamp(seg_abs_start)} closely matches "
                                    f"the opening line). Stopping transcription.\n"
                                )
                                loop_detected = True
                                break

                    start_ts = format_timestamp(seg_abs_start)
                    end_ts = format_timestamp(seg_abs_end)
                    line = f"[{start_ts} --> {end_ts}] {text}"
                    print(line)
                    f.write(line + "\n")
                    f.flush()
                    last_written_end = max(last_written_end, seg_abs_end)
                    session_lines.append(text)

                if loop_detected:
                    break

                elapsed += new_duration
                prev_tail = new_raw[-overlap_bytes:] if overlap_bytes > 0 else b""

                if is_final_chunk:
                    break
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        reader.close()

    if args.summarize:
        full_text = " ".join(session_lines).strip()
        if len(full_text) < 50:
            print("\nNot enough transcribed content to generate a summary; skipping.")
        else:
            print(f"\nGenerating summary via Ollama (model: {args.ollama_model})...")
            try:
                summary = summarize_with_ollama(
                    full_text, args.ollama_model, args.ollama_host, metadata=reader.metadata
                )
                summary_path = Path(args.summary_output)
                summary_path.write_text(summary + "\n", encoding="utf-8")
                print(f"Summary written to: {summary_path.resolve()}\n")
                print(summary)
            except Exception as e:
                print(f"\nFailed to generate summary via Ollama: {e}", file=sys.stderr)
                print(
                    f"Make sure Ollama is installed and running (`ollama serve`) and the "
                    f"model is pulled (`ollama pull {args.ollama_model}`). "
                    f"The transcript itself was still saved successfully.",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    main()