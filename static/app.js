const urlInput = document.getElementById("urlInput");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const statusLine = document.getElementById("statusLine");
const log = document.getElementById("log");
const summarySection = document.getElementById("summarySection");
const summaryText = document.getElementById("summaryText");

let ws = null;

function setStatus(status) {
  statusLine.textContent = status.charAt(0).toUpperCase() + status.slice(1);
  statusLine.className = "status " + status;
  const running = status === "running" || status === "stopping";
  startBtn.disabled = running;
  stopBtn.disabled = !running;
}

function appendLine(text) {
  log.textContent += text + "\n";
  log.scrollTop = log.scrollHeight;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// The summary comes back as plain text with "## Heading" markdown lines
// (see build_summary_prompt in live_transcriber.py). Render those as real
// headings instead of showing the literal "##" characters.
function renderSummary(text) {
  const lines = text.split("\n");
  let html = "";
  for (const line of lines) {
    if (line.startsWith("## ")) {
      html += `<h2>${escapeHtml(line.slice(3))}</h2>`;
    } else if (line.trim() === "") {
      html += "<br>";
    } else {
      html += `<div>${escapeHtml(line)}</div>`;
    }
  }
  summaryText.innerHTML = html;
}

function showSummaryIfPresent(text) {
  if (text) {
    renderSummary(text);
    summarySection.style.display = "block";
  }
}

function connectWebSocket() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === "backlog") {
      log.textContent = "";
      msg.lines.forEach(appendLine);
      setStatus(msg.status);
    } else if (msg.type === "line") {
      appendLine(msg.text);
    } else if (msg.type === "job_complete") {
      setStatus(msg.status);
      showSummaryIfPresent(msg.summary);
    }
  };

  ws.onclose = () => {
    // The dev server can restart; keep trying to reconnect rather than
    // leaving the dashboard silently disconnected.
    setTimeout(connectWebSocket, 2000);
  };
}

startBtn.addEventListener("click", async () => {
  const url = urlInput.value.trim();
  if (!url) {
    alert("Enter a stream URL first.");
    return;
  }
  summarySection.style.display = "none";
  summaryText.innerHTML = "";
  log.textContent = "";

  const res = await fetch("/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    alert(err.detail || "Failed to start job.");
    return;
  }
  setStatus("running");
});

stopBtn.addEventListener("click", async () => {
  stopBtn.disabled = true;
  await fetch("/stop", { method: "POST" });
});

connectWebSocket();

// On page load, pick up whatever state the server already has (in case
// this is a refresh mid-job, or a job already finished before we opened
// the page).
fetch("/status")
  .then((r) => r.json())
  .then((s) => {
    setStatus(s.status);
    showSummaryIfPresent(s.summary);
  })
  .catch(() => {});