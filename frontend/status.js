// Чистая логика отображения состояния задачи (без DOM): по ответу /jobs/{id} строит текст, стадию прогресса и сообщения.
// Подключается в браузере как window.JobView и в тестах Node как module.exports.
(function (root) {
  const STAGES = ["queued", "transcribing", "generating", "done"];
  const ORDINALS = { 2: "второй", 3: "третий", 4: "четвёртый", 5: "пятый" };
  const DEVICE_NAMES = { gpu: "GPU", gpu_reduced: "GPU (экономный режим)", cpu: "CPU" };

  function ordinal(n) {
    return ORDINALS[n] || `${n}-й`;
  }

  function deviceLine(mode) {
    if (!mode || (!mode.asr && !mode.llm)) return null;
    const parts = [];
    if (mode.asr) parts.push(`распознавание — ${DEVICE_NAMES[mode.asr] || mode.asr}`);
    if (mode.llm) parts.push(`генерация протокола — ${DEVICE_NAMES[mode.llm] || mode.llm}`);
    return `Устройство: ${parts.join(", ")}`;
  }

  function describeJob(job) {
    const attempt = job.attempt || 0;
    const max = job.max_attempts || 0;
    const view = {
      stage: "queued",          // стадия прогресс-бара: queued | transcribing | generating | done
      text: "Обрабатываю...",
      kind: "info",             // info | warning | error | success
      terminal: false,          // опрос можно остановить
      failed: false,
      notes: job.device_notes || [],
      deviceLine: deviceLine(job.device_mode),
      modelLine: job.llm_model ? `Модель Qwen: ${job.llm_model}` : null,
      stepLabels: { queued: "Очередь", transcribing: "Расшифровка", generating: "Qwen", done: "Готово" },
      showTranscript: false,
      downloadLabel: null,
    };

    switch (job.status) {
      case "queued": {
        view.stage = "queued";
        const pos = job.queue_position;
        view.text = pos > 0
          ? `В очереди — позиция ${pos}${pos === 1 ? " (следующая на обработку)" : ""}`
          : "Запускаю обработку...";
        break;
      }
      case "transcribing":
        view.stage = "transcribing";
        view.text = "Распознаю речь...";
        break;
      case "generating":
        view.stage = "generating";
        view.text = "Qwen составляет протокол...";
        break;
      case "regenerating":
        view.stage = "generating";
        view.kind = "warning";
        view.text = `Протокол получился пустым — запись отправлена на ${ordinal(attempt)} проход Qwen (попытка ${attempt} из ${max})`;
        view.stepLabels.generating = `Qwen ${attempt}/${max}`; // коротко, чтобы влезало в колонку на телефоне
        break;
      case "done":
        view.stage = "done";
        view.terminal = true;
        if (job.protocol_empty) {
          view.kind = "warning";
          view.text = job.message || "Протокол составить не удалось. Ниже — расшифровка записи.";
          view.showTranscript = true;
          view.downloadLabel = "Скачать расшифровку (TXT)";
        } else {
          view.kind = "success";
          view.text = "Готово." + ((job.issues || []).length
            ? " Часть данных (имена, сроки, числа) не нашлась в расшифровке — она убрана или помечена в разделе «Требует проверки»."
            : "");
          view.downloadLabel = `Скачать протокол (${String(job.output_format || "").toUpperCase()})`;
        }
        break;
      case "error":
        view.terminal = true;
        view.failed = true;
        view.kind = "error";
        view.text = job.error || "Ошибка обработки.";
        break;
      default:
        break;
    }
    return view;
  }

  const api = { STAGES, describeJob, deviceLine, ordinal };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.JobView = api;
})(typeof window !== "undefined" ? window : globalThis);
