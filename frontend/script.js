const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const filenameEl = document.getElementById("filename");
const filenameText = document.getElementById("filenameText");
const processBtn = document.getElementById("processBtn");
const statusEl = document.getElementById("status");
const progressEl = document.getElementById("progress");
const progressSegments = document.querySelectorAll(".progress-segment");
const progressSteps = document.querySelectorAll(".progress-step");
const transcriptToggle = document.getElementById("transcriptToggle");
const transcriptEl = document.getElementById("transcript");
const downloadLink = document.getElementById("downloadLink");
const downloadText = document.getElementById("downloadText");
const formatButtons = document.querySelectorAll(".format-option");

let selectedFile = null;
let selectedFormat = "docx";
let pollTimer = null;

formatButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    formatButtons.forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    selectedFormat = btn.dataset.format;
  });
});

function setFile(file) {
  selectedFile = file;
  filenameText.textContent = file.name;
  filenameEl.hidden = false;
  processBtn.disabled = false;
  statusEl.hidden = true;
  statusEl.classList.remove("error");
  progressEl.hidden = true;
  transcriptToggle.hidden = true;
  transcriptEl.hidden = true;
  downloadLink.hidden = true;
}

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.hidden = false;
  statusEl.classList.toggle("error", isError);
}

function updateProgress(stage) {
  progressEl.hidden = false;
  const order = ["transcribing", "generating", "done"];
  const stageIndex = order.indexOf(stage);

  progressSegments.forEach((el) => {
    const segIndex = order.indexOf(el.dataset.segment);
    el.classList.remove("active", "complete");
    if (stage === "done" || segIndex < stageIndex) el.classList.add("complete");
    else if (segIndex === stageIndex) el.classList.add("active");
  });

  progressSteps.forEach((el) => {
    const stepIndex = order.indexOf(el.dataset.stage);
    el.classList.remove("active", "complete");
    if (stage === "done" || stepIndex < stageIndex) el.classList.add("complete");
    else if (stepIndex === stageIndex) el.classList.add("active");
  });
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

transcriptToggle.addEventListener("click", () => {
  transcriptEl.hidden = !transcriptEl.hidden;
  transcriptToggle.textContent = transcriptEl.hidden
    ? "Показать расшифровку"
    : "Скрыть расшифровку";
});

async function pollJob(jobId) {
  try {
    const res = await fetch(`/jobs/${jobId}`);
    if (!res.ok) throw new Error("Задача не найдена.");
    const job = await res.json();

    updateProgress(job.status);

    if (job.has_transcript && transcriptToggle.hidden) {
      transcriptToggle.hidden = false;
      const transcriptRes = await fetch(`/jobs/${jobId}/transcript`);
      transcriptEl.textContent = await transcriptRes.text();
    }

    if (job.status === "error") {
      clearInterval(pollTimer);
      setStatus(job.error || "Ошибка обработки.", true);
      processBtn.disabled = false;
      return;
    }

    if (job.status === "done") {
      clearInterval(pollTimer);
      setStatus("Готово.");
      downloadLink.href = `/jobs/${jobId}/download`;
      downloadLink.download = selectedFile.name.replace(/\.[^.]+$/, "") + "_protocol." + selectedFormat;
      downloadText.textContent = `Скачать протокол (${selectedFormat.toUpperCase()})`;
      downloadLink.hidden = false;
      processBtn.disabled = false;
    }
  } catch (err) {
    clearInterval(pollTimer);
    setStatus(err.message, true);
    processBtn.disabled = false;
  }
}

processBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  processBtn.disabled = true;
  downloadLink.hidden = true;
  transcriptToggle.hidden = true;
  transcriptEl.hidden = true;
  setStatus("Загружаю файл...");
  updateProgress("queued");

  const formData = new FormData();
  formData.append("file", selectedFile);
  formData.append("output_format", selectedFormat);

  try {
    const response = await fetch("/process", { method: "POST", body: formData });
    if (!response.ok) {
      const detail = await response.json().catch(() => null);
      throw new Error(detail?.detail || `Ошибка сервера: ${response.status}`);
    }

    const { job_id } = await response.json();
    setStatus("Обрабатываю...");
    pollTimer = setInterval(() => pollJob(job_id), 1500);
  } catch (err) {
    setStatus(err.message, true);
    processBtn.disabled = false;
  }
});
