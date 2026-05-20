const form = document.querySelector("#demoForm");
const sourceText = document.querySelector("#sourceText");
const outputText = document.querySelector("#outputText");
const outputBody = document.querySelector("#outputBody");
const outputActionNote = document.querySelector("#outputActionNote");
const originalText = document.querySelector("#originalText");
const statusLine = document.querySelector("#statusLine");
const quotaPill = document.querySelector("#quotaPill");
const backendPill = document.querySelector("#backendPill");
const metricGrid = document.querySelector("#metricGrid");
const submitButton = document.querySelector("#submitButton");
const sampleButton = document.querySelector("#sampleButton");
const sampleOptions = document.querySelectorAll("[data-sample]");
const apiKey = document.querySelector("#apiKey");
const temperature = document.querySelector("#temperature");
const temperatureValue = document.querySelector("#temperatureValue");
const topP = document.querySelector("#topP");
const topPValue = document.querySelector("#topPValue");
const resultPanel = document.querySelector("#resultPanel");
const progressBar = document.querySelector("#progressBar");
const detectorButtons = document.querySelectorAll("[data-detector-url]");
const apiBaseUrl = getApiBaseUrl();

const samples = [
  "During sustained cardio exercise, the heart increases its workload and the rest of the body adjusts to support that effort. Blood vessels widen to improve circulation, leg and core muscles help push blood back toward the heart, and the lungs breathe faster to bring in oxygen while clearing waste gases such as carbon dioxide throughout the workout and recovery period.",
  "Photosynthesis converts light energy into chemical energy that plants can store inside glucose molecules. The process occurs in chloroplasts and unfolds in two linked stages: light-dependent reactions capture sunlight and split water, while the Calvin cycle uses that energy to build sugars that support plant growth, cellular repair, and long-term energy storage during changing seasons and environmental stress across ecosystems.",
  "Solar and wind power have become far cheaper over the past decade, allowing renewable energy to compete with fossil fuels in many electricity markets. Better storage systems, improved grid planning, public incentives, and larger manufacturing pipelines continue to accelerate adoption, while utilities weigh reliability, cost, emissions targets, and regional demand in their long-term investment decisions for new infrastructure projects nationwide."
];

let sampleIndex = 0;
let latestOutput = "";

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

async function refreshConfig() {
  try {
    const response = await fetch(apiUrl("/api/config"));
    if (!response.ok) throw new Error("config failed");
    const config = await response.json();
    quotaPill.textContent = formatQuota(config.public_quota);
    backendPill.textContent = `backend: ${config.backend}`;
    sourceText.maxLength = config.max_chars || 5000;
  } catch (error) {
    quotaPill.textContent = "quota unavailable";
    backendPill.textContent = "backend: unknown";
  }
}

async function submitDemo(event) {
  event.preventDefault();
  const text = sourceText.value.trim();
  if (!text) return;

  localStorage.setItem("stealthrl_demo_api_key", apiKey.value.trim());
  resultPanel.hidden = false;
  submitButton.disabled = true;
  setDetectorButtonsEnabled(false);
  latestOutput = "";
  clearOutputAction();
  setOutputText("Running StealthRL paraphrasing...", true);
  setStatus("Submitting request...", "busy");
  setProgress(true);

  try {
    const response = await fetch(apiUrl("/api/paraphrase"), {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify({
        text,
        temperature: Number(temperature.value),
        top_p: Number(topP.value)
      })
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Request failed");
    }

    clearOutputAction();
    setOutputText(data.output_text);
    latestOutput = data.output_text;
    originalText.textContent = data.input_text;
    quotaPill.textContent = formatQuota(data.quota);
    backendPill.textContent = `backend: ${data.backend}`;
    updateMetrics(data);
    setDetectorButtonsEnabled(true);
    setStatus(`Completed request ${data.request_id.slice(0, 8)}.`, "done");
  } catch (error) {
    clearOutputAction();
    setOutputText("No output generated.", true);
    setStatus(error.message || "Something went wrong.", "error");
  } finally {
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

function bindRange(input, output) {
  const update = () => {
    output.textContent = Number(input.value).toFixed(2);
  };
  input.addEventListener("input", update);
  update();
}

form.addEventListener("submit", submitDemo);
sampleButton.addEventListener("click", () => loadSample(sampleIndex + 1));
sampleOptions.forEach((button) => {
  button.addEventListener("click", () => loadSample(Number(button.dataset.sample)));
});
detectorButtons.forEach((button) => {
  button.addEventListener("click", copyAndOpenDetector);
});
bindRange(temperature, temperatureValue);
bindRange(topP, topPValue);

apiKey.value = localStorage.getItem("stealthrl_demo_api_key") || "";
setDetectorButtonsEnabled(false);
markActiveSample();
refreshConfig();
