#!/usr/bin/env python3
"""Transcribe a live video stream to a timestamped text file, then summarize it.
 
See README.md for setup, usage, and architecture.
"""

import argparse
import datetime as dt
import difflib
import os
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


def configure_cuda_library_path() -> list:
    """
    Pip-installed CUDA packages (nvidia-cublas-cu12, nvidia-cudnn-cu12) drop
    their shared libraries into site-packages, but unlike a full CUDA
    Toolkit install, pip doesn't register that location on the system
    library search path. faster-whisper's GPU backend (ctranslate2) looks
    for libcublas/libcudnn via the standard dynamic linker search, so
    without this, --device cuda fails with "Library libcublas.so.12 is not
    found" even though the library is sitting right there in the venv.

    This locates those packages (if installed) and prepends their
    directories to LD_LIBRARY_PATH before faster-whisper is imported, so
    CUDA mode works out of the box without any manual environment setup.
    Returns the list of directories added, or [] if the packages aren't
    installed (e.g. CPU-only setups, or a system-wide CUDA Toolkit is
    already correctly configured).
    """
    try:
        import nvidia.cublas.lib as cublas_lib
        import nvidia.cudnn.lib as cudnn_lib
    except ImportError:
        return []

    lib_dirs = sorted({os.path.dirname(cublas_lib.__file__), os.path.dirname(cudnn_lib.__file__)})
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs + ([existing] if existing else []))
    return lib_dirs


def format_timestamp(seconds: float) -> str:
    td = dt.timedelta(seconds=max(0, seconds))
    total_ms = int(td.total_seconds() * 1000)
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def extract_context_lines(metadata: dict = None) -> list:
    """Turn a metadata dict into a list of plain-text context lines for a prompt."""
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

    return context_lines


def build_scene_context_prompt(metadata: dict) -> str:
    """
    First call of the two-call summary: a narrow prompt that only asks the
    model to describe what the video contains, using the metadata alone.

    The title and channel are NOT generated by the model - summarize_with_ollama
    copies them verbatim from the metadata into the output. Letting a small
    model write a title was observed producing a wrong film name (it took a
    phrase from the video title and presented it as the film's title), and
    facts we already hold as exact text should be inserted by code, not
    re-generated. The model is told not to state a title itself.
    """
    context_lines = extract_context_lines(metadata)
    context_block = "Here is some information about a video:\n" + "\n".join(context_lines)

    return (
        f"{context_block}\n\n"
        "In 2-3 sentences, describe what this video contains, using only the "
        "information above. The title and channel are already shown to the "
        "reader separately, so do not repeat them and do not state the name of "
        "any film or show yourself. Mention any people or characters named "
        "above exactly as written. Do not add plot details, actors, or settings "
        "that are not stated above.\n\n"
        "Description:"
    )


def build_narration_prompt(scene_context: str, text: str, title: str = None) -> str:
    """
    Second call of the two-call summary: describe what is said in the
    transcript, using the first call's output only to know who is speaking.

    Kept compact on purpose - a small model follows a short prompt with a
    few concrete rules better than a long one with many. The rules target
    observed failures: inventing scene details that aren't in the
    transcript, assuming who the speech is addressed to from one passing
    word, and describing the imagery while missing the point being made.
    Short quotes give the reader something to check each claim against.
    """
    about = f'Video title: "{title}"\n{scene_context}' if title else scene_context
    return (
        f"About the video: {about}\n\n"
        "Below is a speech-to-text transcript of what is said in it. It may "
        "contain small transcription errors or stray words.\n\n"
        "Write a summary of what the speaker says:\n"
        "1. Start with one sentence stating the speaker's main message.\n"
        "2. Then cover the supporting points in order, in your own words.\n"
        "3. Include two or three short direct quotes (under 10 words each), "
        "copied exactly from the transcript and in quotation marks, so each "
        "claim can be checked. Do not join separate parts with \"...\".\n\n"
        "Rules: everything must come from the transcript itself - use the "
        "'About the video' text only to know who is speaking, and add no scenes, "
        "objects, or dialogue that are not in the transcript. Do not say who the "
        "speech is addressed to unless the transcript makes it clear; a single "
        "mention of a name or \"mom\" does not mean the whole speech is aimed at "
        "that person. If an actor is playing a character, the speaker is the "
        "character. Do not mention that this is a transcript.\n\n"
        f"Transcript:\n{text}\n\nSummary:"
    )


def build_plain_summary_prompt(text: str) -> str:
    """Fallback single-call prompt used when no video metadata is available at all."""
    return (
        "Write a clear, concise summary of what was said in the transcript "
        "below, in your own words, capturing the key points and overall "
        "message. Do not mention that this is a transcript or that it came "
        "from speech-to-text.\n\n"
        f"Transcript:\n{text}\n\nSummary:"
    )


def _ollama_generate(prompt: str, model: str, host: str, timeout: float, temperature: float = 0.2) -> str:
    """
    Send one prompt to Ollama's /api/generate and return the response text.

    A low default temperature (0.2) biases generation toward more literal,
    conservative continuations and away from confident improvisation -
    this reduces (does not eliminate) the model's tendency to invent
    plausible-sounding but fabricated details when summarizing.
    """
    import requests

    response = requests.post(
        f"{host.rstrip('/')}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    result = data.get("response", "").strip()
    if not result:
        raise RuntimeError(f"Ollama returned an empty response: {data}")
    return result


def summarize_with_ollama(
    text: str, model: str, host: str, metadata: dict = None, timeout: float = 180.0, temperature: float = 0.2
) -> str:
    """
    Summarize transcribed text, optionally grounded in video metadata, via a
    local Ollama server. Requires Ollama to be installed and running
    (`ollama serve`), with the requested model already pulled.

    When metadata is available, this makes two separate, narrower calls
    instead of one call doing everything at once: first describing the
    scene/context from the metadata alone, then describing the narration
    using that scene description as backdrop. Splitting it this way asks
    less of the model per call (one job instead of several at once -
    format, scene description, narration, and a disambiguation rule all
    together), which matters a lot for small local models that lose
    reliability once a single prompt asks for too much simultaneously.
    """
    context_lines = extract_context_lines(metadata)

    if not context_lines:
        prompt = build_plain_summary_prompt(text)
        return _ollama_generate(prompt, model, host, timeout, temperature)

    title = (metadata or {}).get("title")
    uploader = (metadata or {}).get("uploader")

    scene_context = _ollama_generate(build_scene_context_prompt(metadata), model, host, timeout, temperature)
    narration = _ollama_generate(build_narration_prompt(scene_context, text, title), model, host, timeout, temperature)

    # Title and channel are copied from the metadata by code, not generated.
    header_lines = []
    if title:
        header_lines.append(f"Video: {title}")
    if uploader:
        header_lines.append(f"Channel: {uploader}")
    header = "\n".join(header_lines)
    context_section = f"{header}\n\n{scene_context}" if header else scene_context

    return f"## Context\n{context_section}\n\n## What Was Said\n{narration}"


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


DEFAULTS = {
    "output": "transcript.txt",
    "model": "small",
    "chunk_seconds": 8.0,
    "overlap_seconds": 1.0,
    "language": None,
    "device": "cpu",
    "max_retries": 5,
    "retry_delay": 5.0,
    "stop_after_seconds": None,
    "loop_detect": True,
    "loop_similarity": 0.85,
    "loop_grace_seconds": 15.0,
    "summarize": True,
    "summary_output": "summary.txt",
    "ollama_model": "llama3.2",
    "ollama_host": "http://localhost:11434",
    "ollama_temperature": 0.2,
}


def find_default_config_path() -> Path:
    """Look for a config.yaml next to this script or in the current directory."""
    for candidate in (Path.cwd() / "config.yaml", Path(__file__).resolve().parent / "config.yaml"):
        if candidate.exists():
            return candidate
    return None


def load_config_file(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def apply_config_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """
    Fill in any option the user didn't pass on the command line, first from
    a config file (if one is given or found), then from DEFAULTS. Options
    explicitly passed on the CLI are never overridden - they win outright.
    """
    config_path = Path(args.config) if args.config else find_default_config_path()
    config_data = {}

    if args.config and not config_path.exists():
        print(f"Warning: config file not found: {config_path}", file=sys.stderr)
    elif config_path:
        config_data = load_config_file(config_path)
        print(f"Loaded config from: {config_path}")
        unknown_keys = set(config_data) - set(DEFAULTS)
        if unknown_keys:
            print(f"Warning: ignoring unknown config keys: {', '.join(sorted(unknown_keys))}", file=sys.stderr)

    for key, default_value in DEFAULTS.items():
        if getattr(args, key, None) is None:
            resolved = config_data[key] if key in config_data else default_value
            setattr(args, key, resolved)

    return args


def main():
    parser = argparse.ArgumentParser(description="Transcribe a live video stream to a timestamped text file.")
    parser.add_argument("url", help="Live stream URL (YouTube Live, Twitch, etc.)")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a YAML config file with default values for the other options below. "
             "If omitted, looks for 'config.yaml' in the current directory or next to this script. "
             "Any flag passed explicitly on the command line always overrides the config file.",
    )
    parser.add_argument("--output", default=None, help=f"Output text file path (default: {DEFAULTS['output']})")
    parser.add_argument(
        "--model",
        default=None,
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help=f"faster-whisper model size (bigger = more accurate, slower) (default: {DEFAULTS['model']})",
    )
    parser.add_argument(
        "--chunk-seconds", type=float, default=None,
        help=f"Length of each new audio chunk (default: {DEFAULTS['chunk_seconds']})",
    )
    parser.add_argument(
        "--overlap-seconds",
        type=float,
        default=None,
        help="Seconds of audio from the end of the previous chunk to re-include as context "
             f"for the next chunk, to reduce words being cut at chunk boundaries (default: {DEFAULTS['overlap_seconds']})",
    )
    parser.add_argument("--language", default=None, help="Force a language code (e.g. 'en'); default: auto-detect")
    parser.add_argument(
        "--device", default=None, choices=["cpu", "cuda"],
        help=f"Inference device (default: {DEFAULTS['device']})",
    )
    parser.add_argument(
        "--max-retries", type=int, default=None,
        help=f"Max consecutive reconnect attempts per stall (default: {DEFAULTS['max_retries']})",
    )
    parser.add_argument(
        "--retry-delay", type=float, default=None,
        help=f"Seconds to wait between reconnect attempts (default: {DEFAULTS['retry_delay']})",
    )
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
        default=None,
        help="Disable automatic detection of looping/repeating streams (some 'live' streams "
             "are really a short VOD replayed on a loop). Enabled by default.",
    )
    parser.add_argument(
        "--loop-similarity",
        type=float,
        default=None,
        help="Similarity ratio (0-1) a later caption must have to the very first caption "
             f"to be considered the stream looping back to the start (default: {DEFAULTS['loop_similarity']})",
    )
    parser.add_argument(
        "--loop-grace-seconds",
        type=float,
        default=None,
        help="Don't start checking for loop repeats until this many seconds of audio "
             f"have been transcribed (default: {DEFAULTS['loop_grace_seconds']})",
    )
    parser.add_argument(
        "--no-summarize",
        dest="summarize",
        action="store_false",
        default=None,
        help="Disable generating a summary of the transcript at the end of the run "
             "(enabled by default; requires a local Ollama server)",
    )
    parser.add_argument(
        "--summary-output", default=None,
        help=f"Output path for the generated summary (default: {DEFAULTS['summary_output']})",
    )
    parser.add_argument(
        "--ollama-model",
        default=None,
        help=f"Ollama model to use for summarization (must already be pulled) (default: {DEFAULTS['ollama_model']})",
    )
    parser.add_argument(
        "--ollama-host",
        default=None,
        help=f"Base URL of the Ollama server (default: {DEFAULTS['ollama_host']})",
    )
    parser.add_argument(
        "--ollama-temperature",
        type=float,
        default=None,
        help="Generation temperature for summarization, 0-1. Lower is more literal/conservative, "
             f"higher is more creative but more prone to confident fabrication (default: {DEFAULTS['ollama_temperature']})",
    )
    args = parser.parse_args()
    args = apply_config_defaults(args)

    if args.device == "cuda":
        cuda_lib_dirs = configure_cuda_library_path()
        if cuda_lib_dirs:
            print(f"Configured CUDA library path from pip packages: {':'.join(cuda_lib_dirs)}")
        else:
            print(
                "Note: nvidia-cublas-cu12 / nvidia-cudnn-cu12 not found via pip; "
                "relying on a system-wide CUDA installation instead."
            )

    from faster_whisper import WhisperModel

    print(f"Loading faster-whisper model '{args.model}' on {args.device}...")
    model = WhisperModel(args.model, device=args.device, compute_type="int8" if args.device == "cpu" else "float16")

    try:
        reader = ReconnectingStreamReader(args.url, max_retries=args.max_retries, retry_delay=args.retry_delay)
    except Exception as e:
        print(f"Failed to start stream: {e}", file=sys.stderr)
        sys.exit(1)

    if reader.metadata.get("title"):
        print(f"Video title: {reader.metadata['title']}")
    if reader.metadata.get("uploader"):
        print(f"Channel: {reader.metadata['uploader']}")

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
                    full_text,
                    args.ollama_model,
                    args.ollama_host,
                    metadata=reader.metadata,
                    temperature=args.ollama_temperature,
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