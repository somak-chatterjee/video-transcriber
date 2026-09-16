#!/usr/bin/env python3
"""Transcribe a live video stream to timestamped captions.

The stream is resolved with yt-dlp, decoded by ffmpeg, and transcribed
locally with faster-whisper. Audio chunks overlap slightly to reduce words
being cut at chunk boundaries. Dropped connections are automatically
re-resolved and reconnected up to --max-retries times."""

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


def resolve_stream_url(page_url: str) -> str:
    """Resolve a live page URL to a direct playable stream URL."""
    import yt_dlp

    options = {
        "quiet": True,
        "no_warnings": True,
        "format": "bestaudio/best",
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(page_url, download=False)
        if "url" in info:
            return info["url"]
        formats = info.get("formats") or []
        if not formats:
            raise RuntimeError("yt-dlp could not find a playable stream URL.")
        return formats[-1]["url"]


def open_ffmpeg_audio_pipe(stream_url: str) -> subprocess.Popen:
    """Launch ffmpeg and return a pipe containing mono 16 kHz PCM audio."""
    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-reconnect",
        "1",
        "-reconnect_streamed",
        "1",
        "-reconnect_delay_max",
        "5",
        "-i",
        stream_url,
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-",
    ]
    return subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10 ** 7,
    )


def normalize_text(text: str) -> str:
    """Lowercase and normalize text for loop detection."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def loop_match_ratio(a: str, b: str) -> float:
    """Measure similarity between two normalized caption strings."""
    if not a or not b:
        return 0.0
    matcher = difflib.SequenceMatcher(None, a, b)
    match = matcher.find_longest_match(0, len(a), 0, len(b))
    shorter = min(len(a), len(b))
    return match.size / shorter if shorter else 0.0


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm."""
    delta = dt.timedelta(seconds=max(0, seconds))
    total_ms = int(delta.total_seconds() * 1000)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


class ReconnectingStreamReader:
    """Read stream audio and reconnect after interruptions."""

    def __init__(
        self,
        page_url: str,
        max_retries: int = 5,
        retry_delay: float = 5.0,
    ):
        self.page_url = page_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.proc = None
        self._connect()

    def _connect(self):
        print(f"Resolving stream URL via yt-dlp: {self.page_url}")
        direct_url = resolve_stream_url(self.page_url)
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

    def read(self, size: int) -> bytes:
        """Read up to size bytes, reconnecting when the stream stalls."""
        buffer = b""
        retries = 0
        while len(buffer) < size:
            chunk = self.proc.stdout.read(size - len(buffer))
            if chunk:
                buffer += chunk
                continue
            error = self.proc.stderr.read().decode(errors="ignore")
            self._terminate_current()
            if retries >= self.max_retries:
                if error:
                    print(f"ffmpeg reported: {error.strip()}", file=sys.stderr)
                return buffer
            retries += 1
            print(
                f"Stream dropped; reconnecting ({retries}/"
                f"{self.max_retries}) in {self.retry_delay:.0f}s..."
            )
            if error:
                print(f"  ffmpeg said: {error.strip()}", file=sys.stderr)
            time.sleep(self.retry_delay)
            try:
                self._connect()
            except Exception as exc:
                print(f"  Reconnect failed: {exc}", file=sys.stderr)
        return buffer

    def close(self):
        """Stop the current ffmpeg process."""
        self._terminate_current()


def main():
    """Parse arguments and transcribe the live stream."""
    parser = argparse.ArgumentParser(
        description="Transcribe a live stream to a timestamped text file."
    )
    parser.add_argument("url", help="Live stream URL")
    parser.add_argument("--output", default="transcript.txt")
    parser.add_argument(
        "--model",
        default="small",
        choices=["tiny", "base", "small", "medium", "large-v3"],
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=8.0,
    )
    parser.add_argument(
        "--overlap-seconds",
        type=float,
        default=1.0,
    )
    parser.add_argument("--language", default=None)
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=5.0)
    parser.add_argument(
        "--stop-after-seconds",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--no-loop-detect",
        dest="loop_detect",
        action="store_false",
        default=True,
    )
    parser.add_argument(
        "--loop-similarity",
        type=float,
        default=0.85,
    )
    parser.add_argument(
        "--loop-grace-seconds",
        type=float,
        default=15.0,
    )
    args = parser.parse_args()

    from faster_whisper import WhisperModel

    compute_type = "int8" if args.device == "cpu" else "float16"
    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=compute_type,
    )
    try:
        reader = ReconnectingStreamReader(
            args.url,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
        )
    except Exception as exc:
        print(f"Failed to start stream: {exc}", file=sys.stderr)
        sys.exit(1)

    chunk_bytes = int(args.chunk_seconds * SAMPLE_RATE * BYTES_PER_SAMPLE)
    overlap_bytes = int(args.overlap_seconds * SAMPLE_RATE * BYTES_PER_SAMPLE)
    out_path = Path(args.output)

    elapsed = 0.0
    previous_tail = b""
    first_caption_norm = None
    loop_detected = False
    last_written_end = 0.0
    epsilon = 0.05

    try:
        with out_path.open("a", encoding="utf-8") as output:
            while True:
                if (
                    args.stop_after_seconds is not None
                    and elapsed >= args.stop_after_seconds
                ):
                    print(
                        "Reached --stop-after-seconds cutoff "
                        f"({args.stop_after_seconds}s). Stopping."
                    )
                    break

                new_raw = reader.read(chunk_bytes)
                if not new_raw:
                    print("Stream ended.")
                    break

                final_chunk = len(new_raw) < chunk_bytes
                new_duration = len(new_raw) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
                overlap_duration = len(previous_tail) / (
                    SAMPLE_RATE * BYTES_PER_SAMPLE
                )
                chunk_start = elapsed - overlap_duration

                padded_new = new_raw
                if final_chunk:
                    padded_new = new_raw + b"\x00" * (chunk_bytes - len(new_raw))

                combined = previous_tail + padded_new
                audio = np.frombuffer(
                    combined,
                    dtype=np.int16,
                ).astype(np.float32) / 32768.0

                segments, _ = model.transcribe(
                    audio,
                    language=args.language,
                    vad_filter=True,
                    word_timestamps=True,
                )

                for segment in segments:
                    words = segment.words or []
                    if words:
                        kept = []
                        for word in words:
                            word_start = chunk_start + word.start
                            word_end = chunk_start + word.end
                            if word_end <= last_written_end + epsilon:
                                continue
                            kept.append((word_start, word_end, word.word))
                        if not kept:
                            continue
                        segment_start = kept[0][0]
                        segment_end = kept[-1][1]
                        text = "".join(word[2] for word in kept).strip()
                    else:
                        segment_start = chunk_start + segment.start
                        segment_end = chunk_start + segment.end
                        if segment_end <= last_written_end + epsilon:
                            continue
                        text = segment.text.strip()
                    if not text:
                        continue

                    normalized = normalize_text(text)

                    if args.loop_detect:
                        if first_caption_norm is None and len(normalized) >= 20:
                            first_caption_norm = normalized
                        elif (
                            first_caption_norm is not None
                            and segment_start >= args.loop_grace_seconds
                        ):
                            similarity = loop_match_ratio(
                                normalized,
                                first_caption_norm,
                            )
                            if similarity >= args.loop_similarity:
                                print(
                                    "\nDetected the stream looping back to "
                                    "its start (caption at "
                                    f"{format_timestamp(segment_start)} closely "
                                    "matches the opening line). Stopping "
                                    "transcription.\n"
                                )
                                loop_detected = True
                                break

                    start_ts = format_timestamp(segment_start)
                    end_ts = format_timestamp(segment_end)
                    line = f"[{start_ts} --> {end_ts}] {text}"
                    print(line)
                    output.write(line + "\n")
                    output.flush()
                    last_written_end = max(last_written_end, segment_end)

                if loop_detected:
                    break

                elapsed += new_duration
                previous_tail = (
                    new_raw[-overlap_bytes:] if overlap_bytes > 0 else b""
                )

                if final_chunk:
                    break
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        reader.close()


if __name__ == "__main__":
    main()
