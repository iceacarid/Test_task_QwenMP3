const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const filenameEl = document.getElementById("filename");
const filenameText = document.getElementById("filenameText");
const processBtn = document.getElementById("processBtn");
const statusEl = document.getElementById("status");
const deviceLineEl = document.getElementById("deviceLine");
const modelLineEl = document.getElementById("modelLine");
const modelField = document.getElementById("modelField");
const modelSelect = document.getElementById("modelSelect");
const modelHint = document.getElementById("modelHint");
const deviceNotesEl = document.getElementById("deviceNotes");
const progressEl = document.getElementById("progress");
const progressSegments = document.querySelectorAll(".progress-segment");
const progressSteps = document.querySelectorAll(".progress-step");
const transcriptToggle = document.getElementById("transcriptToggle");
const transcriptEl = document.getElementById("transcript");
const downloadLink = document.getElementById("downloadLink");
const downloadText = document.getElementById("downloadText");
const formatButtons = document.querySelectorAll(".format-option");

const STORAGE_KEY = "protocolJob";
const MODEL_KEY = "protocolModel";
const POLL_MS = 1500;
const RETRY_MS = 3000;
const MAX_NETWORK_FAILS = 5;

let selectedFile = null;
let selectedFormat = "docx";
let pollTimer = null;
let currentJobId = null;
let lastStage = "queued";
let transcriptLoaded = false;
let networkFails = 0;

function storeJob(data) {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(data)); } catch (e) { /* хранилище недоступно — не критично */ }
}

function loadStoredJob() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY)); } catch (e) { return null; }
}

function clearStoredJob() {
  try { localStorage.removeItem(STORAGE_KEY); } catch (e) { /* ignore */ }
}

function setFormat(format) {
  selectedFormat = format;
  formatButtons.forEach((b) => b.classList.toggle("active", b.dataset.format === format));
}

formatButtons.forEach((btn) => {
  btn.addEventListener("click", () => setFormat(btn.dataset.format));
});

function resetResult() {
  clearTimeout(pollTimer);
  currentJobId = null;
  lastStage = "queued";
  transcriptLoaded = false;
  networkFails = 0;
  statusEl.hidden = true;
  statusEl.className = "status";
  deviceLineEl.hidden = true;
  modelLineEl.hidden = true;
  deviceNotesEl.hidden = true;
  deviceNotesEl.replaceChildren();
  progressEl.hidden = true;
  transcriptToggle.hidden = true;
  transcriptEl.hidden = true;
  transcriptEl.textContent = "";
  downloadLink.hidden = true;
}

function setFile(file) {
  resetResult();
  clearStoredJob();
  selectedFile = file;
  filenameText.textContent = file.name;
  filenameEl.hidden = false;
  processBtn.disabled = false;
}

function setStatus(text, kind = "info") {
  statusEl.textContent = text;
  statusEl.hidden = false;
  statusEl.className = `status ${kind}`;
}

const STAGE_ORDER = JobView.STAGES;

function updateProgress(stage, labels, failed = false) {
  progressEl.hidden = false;
  const stageIndex = STAGE_ORDER.indexOf(stage);

  progressSegments.forEach((el) => {
    const idx = STAGE_ORDER.indexOf(el.dataset.segment);
    el.classList.remove("active", "complete", "error");
    if (stage === "done" || idx < stageIndex) el.classList.add("complete");
    else if (idx === stageIndex) el.classList.add(failed ? "error" : "active");
  });

  progressSteps.forEach((el) => {
    const idx = STAGE_ORDER.indexOf(el.dataset.stage);
    el.classList.remove("active", "complete", "error");
    if (labels && labels[el.dataset.stage]) el.textContent = labels[el.dataset.stage];
    if (stage === "done" || idx < stageIndex) el.classList.add("complete");
    else if (idx === stageIndex) el.classList.add(failed ? "error" : "active");
  });
}

function renderNotes(view) {
  modelLineEl.hidden = !view.modelLine;
  modelLineEl.textContent = view.modelLine || "";
  deviceLineEl.hidden = !view.deviceLine;
  deviceLineEl.textContent = view.deviceLine || "";
  deviceNotesEl.replaceChildren();
  view.notes.forEach((note) => {
    const li = document.createElement("li");
    li.textContent = note;
    deviceNotesEl.appendChild(li);
  });
  deviceNotesEl.hidden = view.notes.length === 0;
}

async function loadTranscript(jobId, open) {
  const res = await fetch(`/jobs/${jobId}/transcript`);
  if (!res.ok) return;
  transcriptEl.textContent = await res.text();
  transcriptLoaded = true;
  transcriptToggle.hidden = false;
  if (open) showTranscript(true);
}

function showTranscript(visible) {
  transcriptEl.hidden = !visible;
  transcriptToggle.textContent = visible ? "Скрыть расшифровку" : "Показать расшифровку";
}

fileInput.addEventListener("change", () => {
  if (fileInput.files.length) setFile(fileInput.files[0]);
});

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("dragover");
});

dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
});

transcriptToggle.addEventListener("click", () => showTranscript(transcriptEl.hidden));

function schedulePoll(jobId, delay) {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(() => pollJob(jobId), delay);
}

function finishWithError(text) {
  setStatus(text, "error");
  processBtn.disabled = false;
}

async function pollJob(jobId) {
  if (jobId !== currentJobId) return; // пользователь уже начал другую задачу
  try {
    const res = await fetch(`/jobs/${jobId}`);
    if (res.status === 404) {
      clearStoredJob();
      finishWithError("Задача не найдена (возможно, сервер был перезапущен). Загрузите файл заново.");
      return;
    }
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const job = await res.json();
    networkFails = 0;

    const view = JobView.describeJob(job);
    if (!view.failed) lastStage = view.stage;
    updateProgress(view.failed ? lastStage : view.stage, view.stepLabels, view.failed);
    setStatus(view.text, view.kind);
    renderNotes(view);

    if (job.has_transcript && !transcriptLoaded) await loadTranscript(jobId, view.showTranscript);
    else if (view.showTranscript && transcriptEl.hidden && transcriptLoaded) showTranscript(true);

    if (!view.terminal) {
      schedulePoll(jobId, POLL_MS);
      return;
    }
    processBtn.disabled = false;
    if (job.status === "done") {
      downloadLink.href = `/jobs/${jobId}/download`;
      downloadText.textContent = view.downloadLabel;
      downloadLink.hidden = false;
    }
  } catch (err) {
    networkFails += 1;
    if (networkFails > MAX_NETWORK_FAILS) {
      finishWithError("Нет связи с сервером. Обновите страницу, чтобы продолжить отслеживание задачи.");
      return;
    }
    setStatus("Нет связи с сервером, повторяю попытку...", "warning");
    schedulePoll(jobId, RETRY_MS);
  }
}

function trackJob(jobId) {
  currentJobId = jobId;
  processBtn.disabled = true;
  schedulePoll(jobId, 0);
}

processBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  const fileName = selectedFile.name;
  resetResult();
  processBtn.disabled = true;
  setStatus("Загружаю файл...");
  updateProgress("queued", null);

  const formData = new FormData();
  formData.append("file", selectedFile);
  formData.append("output_format", selectedFormat);
  if (!modelField.hidden && modelSelect.value) formData.append("llm_model", modelSelect.value);

  try {
    const response = await fetch("/process", { method: "POST", body: formData });
    if (!response.ok) {
      const detail = await response.json().catch(() => null);
      throw new Error(detail?.detail || `Ошибка сервера: ${response.status}`);
    }

    const { job_id } = await response.json();
    storeJob({ jobId: job_id, fileName, format: selectedFormat });
    trackJob(job_id);
  } catch (err) {
    finishWithError(err.message);
  }
});

let modelInfo = { default: null, models: [] };

function updateModelHint() {
  const chosen = modelInfo.models.find((m) => m.name === modelSelect.value);
  const base = modelInfo.models.find((m) => m.name === modelInfo.default);
  const bigger = chosen && base && chosen.size_gb && base.size_gb && chosen.size_gb > base.size_gb * 1.3;
  modelHint.hidden = !bigger;
  modelHint.textContent = bigger
    ? "Крупнее и, как правило, точнее на сложных записях, но обработка заметно дольше."
    : "";
}

// Список моделей берём у бэкенда (установленные в Ollama модели Qwen). Одна модель — выбор не показываем.
async function loadModels() {
  try {
    const res = await fetch("/models");
    if (!res.ok) return;
    modelInfo = await res.json();
  } catch (e) {
    return; // без списка работаем с моделью по умолчанию
  }
  if (!modelInfo.models || modelInfo.models.length < 2) return;
  modelSelect.replaceChildren();
  modelInfo.models.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m.name;
    opt.textContent = m.name + (m.size_gb ? ` (${m.size_gb} ГБ)` : "") + (m.name === modelInfo.default ? " — по умолчанию" : "");
    modelSelect.appendChild(opt);
  });
  let saved = null;
  try { saved = localStorage.getItem(MODEL_KEY); } catch (e) { /* ignore */ }
  modelSelect.value = modelInfo.models.some((m) => m.name === saved) ? saved : modelInfo.default;
  modelField.hidden = false;
  updateModelHint();
}

modelSelect.addEventListener("change", () => {
  try { localStorage.setItem(MODEL_KEY, modelSelect.value); } catch (e) { /* ignore */ }
  updateModelHint();
});

loadModels();

// После перезагрузки страницы продолжаем следить за задачей, которую запускали до этого.
(function resumeStoredJob() {
  const saved = loadStoredJob();
  if (!saved || !saved.jobId) return;
  filenameText.textContent = saved.fileName || "";
  filenameEl.hidden = !saved.fileName;
  if (saved.format) setFormat(saved.format);
  setStatus("Восстанавливаю состояние задачи...");
  trackJob(saved.jobId);
})();
