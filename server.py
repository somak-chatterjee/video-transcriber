#!/usr/bin/env python3
"""
server.py

A local FastAPI web dashboard for live_transcriber.py. Runs the existing
script, UNCHANGED, as a subprocess - captures its stdout live and streams
it to a browser over a WebSocket, so you can start a transcription run,
watch the transcript scroll in, and read the final summary, all from a
page instead of a terminal.

Run:
    pip install -r requirements.txt   # now includes fastapi + uvicorn
    python server.py

Then open http://localhost:8000 in a browser.

Only one job runs at a time, matching how the CLI tool is normally used.
Starting a new job while one is already running returns an error - stop
the current one first (Stop sends the same signal as Ctrl+C, which the
script already handles gracefully: it finishes the summary for whatever
was transcribed so far, rather than dying mid-write).
"""

import asyncio
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
SCRIPT_PATH = BASE_DIR / "live_transcriber.py"

app = FastAPI(title="Live Transcriber Dashboard")


class StartRequest(BaseModel):
    url: str


class Job:
    """Tracks the single in-flight (or most recently finished) transcription job."""

    def __init__(self):
        self.process: Optional[asyncio.subprocess.Process] = None
        self.status: str = "idle"  # idle | running | done | error | stopped
        self.url: Optional[str] = None
        self.output_lines: list[str] = []
        self.summary_text: Optional[str] = None
        self.subscribers: list[WebSocket] = []
        self.started_at: float = 0.0

    def reset_for_new_run(self, url: str):
        self.process = None
        self.status = "running"
        self.url = url
        self.output_lines = []
        self.summary_text = None
        self.started_at = time.time()

    async def broadcast(self, message: dict):
        dead = []
        for ws in self.subscribers:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in self.subscribers:
                self.subscribers.remove(ws)

    async def append_line(self, line: str):
        self.output_lines.append(line)
        await self.broadcast({"type": "line", "text": line})

    async def finish(self, returncode: int):
        # If we were told to stop and the process exited nonzero (common for
        # a SIGINT'd process), still treat that as a clean stop, not an error.
        if self.status == "stopping":
            self.status = "stopped"
        else:
            self.status = "done" if returncode == 0 else "error"

        # Make failures explainable: put the exit reason in the log itself,
        # so an "Error" status is never an empty, mysterious panel.
        if self.status == "error":
            if returncode < 0:
                sig = -returncode
                hint = " (SIGKILL - commonly the OS killing the process for running out of memory)" if sig == 9 else ""
                await self.append_line(f"[server] Process was terminated by signal {sig}{hint}.")
            else:
                await self.append_line(f"[server] Process exited with error code {returncode}.")

        # Only trust summary.txt if it was written during THIS run - otherwise
        # a failed run would display a stale summary left over from an earlier one.
        summary_path = BASE_DIR / "summary.txt"
        if summary_path.exists() and summary_path.stat().st_mtime >= self.started_at:
            try:
                self.summary_text = summary_path.read_text(encoding="utf-8").strip()
            except Exception:
                self.summary_text = None

        await self.broadcast(
            {
                "type": "job_complete",
                "status": self.status,
                "returncode": returncode,
                "summary": self.summary_text,
            }
        )


job = Job()


async def _read_stream(process: asyncio.subprocess.Process):
    """Read the subprocess's combined stdout/stderr line by line and broadcast each line."""
    assert process.stdout is not None
    while True:
        raw_line = await process.stdout.readline()
        if not raw_line:
            break
        line = raw_line.decode(errors="replace").rstrip("\n")
        if line:
            await job.append_line(line)
    returncode = await process.wait()
    await job.finish(returncode)


@app.post("/start")
async def start_job(req: StartRequest):
    if job.status == "running":
        raise HTTPException(status_code=409, detail="A transcription job is already running.")

    if not req.url.strip():
        raise HTTPException(status_code=400, detail="A stream URL is required.")

    job.reset_for_new_run(req.url)

    # -u / PYTHONUNBUFFERED: when stdout is a pipe, Python block-buffers it,
    # so output would only arrive in big delayed chunks (or be lost entirely
    # if the process is killed). Unbuffered makes the dashboard truly live.
    cmd = [sys.executable, "-u", str(SCRIPT_PATH), req.url]
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(BASE_DIR),
        env=env,
    )
    job.process = process
    asyncio.create_task(_read_stream(process))

    return {"status": "started", "url": req.url}


@app.post("/stop")
async def stop_job():
    if job.status != "running" or job.process is None:
        raise HTTPException(status_code=400, detail="No job is currently running.")

    job.status = "stopping"
    # SIGINT mirrors Ctrl+C, which the script already handles as a graceful
    # stop (it still writes the summary for whatever was transcribed so far).
    job.process.send_signal(signal.SIGINT)
    return {"status": "stopping"}


@app.get("/status")
async def get_status():
    return {
        "status": job.status,
        "url": job.url,
        "line_count": len(job.output_lines),
        "summary": job.summary_text,
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    job.subscribers.append(ws)
    # Replay everything captured so far, so a client that connects mid-run
    # (or a page refresh) doesn't lose history.
    await ws.send_json({"type": "backlog", "lines": job.output_lines, "status": job.status})
    try:
        while True:
            # We don't expect incoming messages, just need to keep the
            # connection open and notice when the browser disconnects.
            await ws.receive_text()
    except WebSocketDisconnect:
        if ws in job.subscribers:
            job.subscribers.remove(ws)


# Serves static/index.html at "/" and static/app.js etc. Mounted last so it
# only catches requests the explicit routes above didn't already handle.
app.mount("/", StaticFiles(directory=str(BASE_DIR / "static"), html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)