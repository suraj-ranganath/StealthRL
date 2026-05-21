const form = document.querySelector("#demoForm");
const sourceText = document.querySelector("#sourceText");
const outputText = document.querySelector("#outputText");
const outputBody = document.querySelector("#outputBody");
const outputActionNote = document.querySelector("#outputActionNote");
const originalText = document.querySelector("#originalText");
const statusLine = document.querySelector("#statusLine");
const quotaPill = document.querySelector("#quotaPill");
const metricGrid = document.querySelector("#metricGrid");
const submitButton = document.querySelector("#submitButton");
const sampleOptions = document.querySelectorAll("[data-sample]");
const apiKey = document.querySelector("#apiKey");
const temperature = document.querySelector("#temperature");
const temperatureValue = document.querySelector("#temperatureValue");
const topP = document.querySelector("#topP");
const topPValue = document.querySelector("#topPValue");
const resultPanel = document.querySelector("#resultPanel");
const progressBar = document.querySelector("#progressBar");
const detectorButtons = document.querySelectorAll("[data-detector-url]");
const feedbackRow = document.querySelector("#feedbackRow");
const feedbackButtons = document.querySelectorAll("[data-feedback-rating]");
const feedbackStatus = document.querySelector("#feedbackStatus");
const apiBaseUrl = getApiBaseUrl();

const samples = [
  "During sustained cardio exercise, the heart increases its workload and the rest of the body adjusts to support that effort. Blood vessels widen to improve circulation, leg and core muscles help push blood back toward the heart, and the lungs breathe faster to bring in oxygen while clearing waste gases such as carbon dioxide throughout the workout and recovery period.",
  "Photosynthesis converts light energy into chemical energy that plants can store inside glucose molecules. The process occurs in chloroplasts and unfolds in two linked stages: light-dependent reactions capture sunlight and split water, while the Calvin cycle uses that energy to build sugars that support plant growth, cellular repair, and long-term energy storage during changing seasons and environmental stress across ecosystems.",
  "Solar and wind power have become far cheaper over the past decade, allowing renewable energy to compete with fossil fuels in many electricity markets. Better storage systems, improved grid planning, public incentives, and larger manufacturing pipelines continue to accelerate adoption, while utilities weigh reliability, cost, emissions targets, and regional demand in their long-term investment decisions for new infrastructure projects nationwide."
];

let sampleIndex = 0;
let latestOutput = "";
let latestRequestId = "";

function setStatus(message, tone = "neutral") {
  statusLine.textContent = message;
  statusLine.dataset.tone = tone;
}

function getApiBaseUrl() {
  const meta = document.querySelector('meta[name="stealthrl-api-base-url"]');
  return (meta?.content || "").trim().replace(/\/$/, "");
}

function apiUrl(path) {
  return `${apiBaseUrl}${path}`;
}

function setOutputText(message, placeholder = false) {
  outputBody.textContent = message;
  outputText.classList.toggle("placeholder", placeholder);
}

function appendOutputText(piece) {
  outputBody.textContent += piece;
  outputText.classList.remove("placeholder");
}

function setOutputAction(message, tone = "neutral") {
  outputActionNote.textContent = message;
  outputActionNote.dataset.tone = tone;
  outputActionNote.hidden = false;
}

function clearOutputAction() {
  outputActionNote.hidden = true;
  outputActionNote.textContent = "";
  delete outputActionNote.dataset.tone;
}

function setProgress(running) {
  progressBar.classList.toggle("is-running", running);
  progressBar.style.width = running ? "100%" : "0%";
}

function formatQuota(quota) {
  if (!quota) return "quota unavailable";
  if (quota.authenticated && quota.limit === null) return "API key: unlimited";
  if (quota.authenticated) return `API key: ${quota.remaining}/${quota.limit} left`;
  return `public: ${quota.remaining}/${quota.limit} left`;
}

function updateMetrics(data) {
  const metrics = data.metrics || {};
  metricGrid.innerHTML = `
    <div><span>Input words</span><strong>${metrics.input_words ?? "-"}</strong></div>
    <div><span>Output words</span><strong>${metrics.output_words ?? "-"}</strong></div>
    <div><span>Edit rate</span><strong>${metrics.char_edit_rate ?? "-"}</strong></div>
    <div><span>Latency</span><strong>${data.latency_ms ?? "-"} ms</strong></div>
  `;
}

function setDetectorButtonsEnabled(enabled) {
  detectorButtons.forEach((button) => {
    button.disabled = !enabled;
  });
}

function setFeedbackEnabled(enabled) {
  feedbackRow.hidden = !enabled;
  feedbackButtons.forEach((button) => {
    button.disabled = !enabled;
    button.removeAttribute("aria-pressed");
  });
  feedbackStatus.textContent = "";
}

function getHeaders() {
  const headers = { "Content-Type": "application/json" };
  const key = apiKey.value.trim();
  if (key) headers.Authorization = `Bearer ${key}`;
  return headers;
}

function markActiveSample() {
  sampleOptions.forEach((button) => {
    button.setAttribute("aria-pressed", String(Number(button.dataset.sample) === sampleIndex));
  });
}

function loadSample(index) {
  sampleIndex = index % samples.length;
  sourceText.value = samples[sampleIndex];
  markActiveSample();
  sourceText.focus();
}

function startColdStartHints() {
  const timers = [
    setTimeout(() => {
      setStatus("Azure GPU is likely waking from zero. Keep this tab open; first load can take a few minutes.", "busy");
    }, 9000),
    setTimeout(() => {
      setOutputAction(
        "Still waiting? The model server may be cold-starting. You can leave this tab open and come back in a few minutes.",
        "neutral"
      );
    }, 30000)
  ];
  return () => timers.forEach((timer) => clearTimeout(timer));
}

async function refreshConfig() {
  try {
    const response = await fetch(apiUrl("/api/config"));
    if (!response.ok) throw new Error("config failed");
    const config = await response.json();
    quotaPill.textContent = formatQuota(config.public_quota);
    sourceText.maxLength = config.max_chars || 5000;
  } catch (error) {
    quotaPill.textContent = "quota unavailable";
  }
}

async function parseErrorResponse(response) {
  try {
    const data = await response.json();
    return data.detail || data.message || "Request failed";
  } catch (error) {
    return `Request failed (${response.status})`;
  }
}

function applyFinalResult(data) {
  clearOutputAction();
  setOutputText(data.output_text || "No output generated.", !data.output_text);
  latestOutput = data.output_text || "";
  latestRequestId = data.request_id || "";
  originalText.textContent = data.input_text || sourceText.value.trim();
  quotaPill.textContent = formatQuota(data.quota);
  updateMetrics(data);
  setDetectorButtonsEnabled(Boolean(latestOutput));
  setFeedbackEnabled(Boolean(latestRequestId));
  setStatus(`Completed request ${latestRequestId.slice(0, 8)}.`, "done");
}

async function handleStream(response) {
  if (!response.body) {
    throw new Error("Streaming is not available in this browser.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let sawDelta = false;
  let sawFinal = false;

  const handlePayload = (payload) => {
    if (!payload.trim()) return;
    const data = JSON.parse(payload);
    if (data.event === "status") {
      setStatus(data.message || "Running...", data.tone || "busy");
      return;
    }
    if (data.event === "delta") {
      if (!sawDelta) {
        setOutputText("");
        sawDelta = true;
      }
      appendOutputText(data.text || "");
      return;
    }
    if (data.event === "final") {
      sawFinal = true;
      applyFinalResult(data);
      return;
    }
    if (data.event === "error") {
      throw new Error(data.message || "Inference failed.");
    }
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() || "";
    for (const block of blocks) handleStreamBlock(block, handlePayload);
  }
  buffer += decoder.decode();
  if (buffer.trim()) handleStreamBlock(buffer, handlePayload);
  if (!sawFinal) throw new Error("Stream ended before the final result arrived.");
}

function handleStreamBlock(block, handlePayload) {
  const dataLines = block
    .split("\n")
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trimStart());
  if (dataLines.length) {
    handlePayload(dataLines.join("\n"));
    return;
  }

  // Local/dev fallback for newline-delimited JSON streams.
  block.split("\n").forEach((line) => handlePayload(line));
}

async function submitDemo(event) {
  event.preventDefault();
  const text = sourceText.value.trim();
  if (!text) return;

  localStorage.setItem("stealthrl_demo_api_key", apiKey.value.trim());
  resultPanel.hidden = false;
  submitButton.disabled = true;
  setDetectorButtonsEnabled(false);
  setFeedbackEnabled(false);
  latestOutput = "";
  latestRequestId = "";
  clearOutputAction();
  setOutputText("Starting StealthRL...", true);
  setStatus("Submitting request...", "busy");
  setProgress(true);
  const clearColdStartHints = startColdStartHints();

  try {
    const response = await fetch(apiUrl("/api/paraphrase/stream"), {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify({
        text,
        temperature: Number(temperature.value),
        top_p: Number(topP.value)
      })
    });

    if (!response.ok) {
      throw new Error(await parseErrorResponse(response));
    }

    await handleStream(response);
  } catch (error) {
    clearOutputAction();
    setOutputText("No output generated.", true);
    setStatus(error.message || "Something went wrong.", "error");
  } finally {
    clearColdStartHints();
    submitButton.disabled = false;
    setProgress(false);
  }
}

async function copyAndOpenDetector(event) {
  const target = event.currentTarget;
  const url = target.dataset.detectorUrl;
  const label = target.dataset.detectorName || target.textContent.trim();
  if (!url || !latestOutput) return;

  try {
    await navigator.clipboard.writeText(latestOutput);
    const message = `Copied output. Opening ${label} in a new tab.`;
    setStatus(message, "done");
    setOutputAction(message, "done");
  } catch (error) {
    const message = `Opening ${label}. Copy failed, so paste manually from the output box.`;
    setStatus(message, "error");
    setOutputAction(message, "error");
  }

  window.open(url, "_blank", "noopener,noreferrer");
}

async function submitFeedback(event) {
  const rating = event.currentTarget.dataset.feedbackRating;
  if (!latestRequestId || !rating) return;

  feedbackButtons.forEach((button) => {
    button.disabled = true;
    button.setAttribute("aria-pressed", String(button.dataset.feedbackRating === rating));
  });
  feedbackStatus.textContent = "Sending...";

  try {
    const response = await fetch(apiUrl("/api/feedback"), {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify({
        request_id: latestRequestId,
        rating
      })
    });
    if (!response.ok) throw new Error(await parseErrorResponse(response));
    feedbackStatus.textContent = "Thanks.";
  } catch (error) {
    feedbackStatus.textContent = "Could not save.";
    feedbackButtons.forEach((button) => {
      button.disabled = false;
    });
  }
}

function bindRange(input, output) {
  const update = () => {
    output.textContent = Number(input.value).toFixed(2);
  };
  input.addEventListener("input", update);
  update();
}

form.addEventListener("submit", submitDemo);
sampleOptions.forEach((button) => {
  button.addEventListener("click", () => loadSample(Number(button.dataset.sample)));
});
detectorButtons.forEach((button) => {
  button.addEventListener("click", copyAndOpenDetector);
});
feedbackButtons.forEach((button) => {
  button.addEventListener("click", submitFeedback);
});
bindRange(temperature, temperatureValue);
bindRange(topP, topPValue);

apiKey.value = localStorage.getItem("stealthrl_demo_api_key") || "";
setDetectorButtonsEnabled(false);
setFeedbackEnabled(false);
markActiveSample();
refreshConfig();
