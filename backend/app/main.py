import os
import re
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from app.asr import transcribe_audio, format_timestamped_transcript
from app.llm import MODEL as DEFAULT_MODEL, generate_protocol, list_models, unload_model
from app.docx_builder import build_protocol_docx
from app.txt_builder import build_protocol_txt
from app.pdf_builder import build_protocol_pdf
from app.gpu_queue import GpuQueue
from app.jobs import create_job, update_job, get_job, set_device

app = FastAPI(title="ASR + Qwen Protocol Generator")

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a"}
ALLOWED_FORMATS = {"docx", "txt", "pdf"}
MEDIA_TYPES = {
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "txt": "text/plain",
    "pdf": "application/pdf",
}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

gpu_queue = GpuQueue()
MAX_QUEUE = int(os.environ.get("MAX_QUEUE", 20))
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", 500))
_UNSAFE_NAME_RE = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


def _safe_stem(filename: str) -> str:
    """Имя файла без расширения, каталогов и опасных для пути символов (попадает в имена файлов на диске).
    Разбор без pathlib: результат одинаков на Windows и Linux."""
    name = filename.replace("\\", "/").rsplit("/", 1)[-1]
    stem = name.rsplit(".", 1)[0] if "." in name else name
    return _UNSAFE_NAME_RE.sub("_", stem).strip(". ")[:100] or "audio"


_SLOW = "обработка займёт больше времени"


def _unload_llm(model: str | None = None) -> None:
    """Выгрузка Qwen освобождает VRAM для Whisper следующей задачи. При нескольких GPU-воркерах не делаем:
    выгрузка сломала бы соседнюю задачу, которая сейчас использует модель."""
    if gpu_queue.workers <= 1:
        unload_model(model)


def _device_note(part: str, mode: str, reason: str | None, model: str | None = None) -> str | None:
    """Сообщение пользователю о смене устройства; None — всё штатно."""
    what = "распознавание" if part == "asr" else "генерация протокола"
    switched = "переключено" if part == "asr" else "переключена"
    lite = f", используется облегчённая модель {model}" if part == "asr" and model and model != "large-v3" else ""
    if mode == "cpu" and reason == "no_gpu":
        return f"GPU недоступна — {what} {switched} на CPU{lite}, {_SLOW}"
    if mode == "cpu" and reason is None:
        return f"GPU не обнаружена — {what} выполняется на CPU{lite}, {_SLOW}"
    if mode == "cpu":
        return f"Недостаточно видеопамяти — {what} {switched} на CPU{lite}, {_SLOW}"
    if mode == "gpu_reduced":
        return f"Недостаточно видеопамяти — {what} идёт в экономном режиме GPU, {_SLOW}"
    return None


def _run_pipeline(job_id: str, audio_path: str, stem: str, output_format: str, llm_model: str | None = None):
    """Выполняется в отдельном потоке — блокирующие вызовы ASR/Qwen/LibreOffice не блокируют сервер."""
    try:
        update_job(job_id, status="transcribing")
        _unload_llm()  # Qwen от предыдущей задачи (keep_alive) не должен занимать VRAM во время Whisper
        def on_asr_device(mode: str, model: str, reason: str | None):
            set_device(job_id, "asr", mode, _device_note("asr", mode, reason, model))

        transcript, segments = transcribe_audio(audio_path, on_device=on_asr_device)
        if not transcript:
            update_job(job_id, status="error", error="Не удалось распознать речь в файле.")
            return

        timestamped = format_timestamped_transcript(segments)
        update_job(job_id, transcript=transcript)

        def on_attempt(attempt: int, max_attempts: int):
            update_job(job_id, status="generating" if attempt == 1 else "regenerating",
                       attempt=attempt, max_attempts=max_attempts)

        update_job(job_id, status="generating")
        def on_llm_device(mode: str, reason: str | None):
            set_device(job_id, "llm", mode, _device_note("llm", mode, reason))

        result = generate_protocol(timestamped, on_attempt=on_attempt, on_device=on_llm_device, model=llm_model)

        if result.empty:
            # Все попытки дали пустой протокол — это не ошибка: отдаём расшифровку вместо пустого документа.
            txt_path = DATA_DIR / f"{job_id}_{stem}_transcript.txt"
            txt_path.write_text(f"{result.message}\n\n{timestamped}\n", encoding="utf-8")
            update_job(job_id, status="done", protocol_empty=True, message=result.message,
                       output_path=str(txt_path), output_format="txt", download_name=f"{stem}_transcript.txt")
            return
        protocol = result.protocol
        update_job(job_id, issues=result.issues)

        base_name = f"{job_id}_{stem}_protocol"
        docx_path = DATA_DIR / f"{base_name}.docx"
        build_protocol_docx(protocol, str(docx_path))

        if output_format == "docx":
            output_path = docx_path
        elif output_format == "txt":
            output_path = DATA_DIR / f"{base_name}.txt"
            build_protocol_txt(protocol, str(output_path))
        else:
            output_path = DATA_DIR / f"{base_name}.pdf"
            build_protocol_pdf(str(docx_path), str(output_path))

        download_name = f"{stem}_protocol.{output_format}"
        update_job(
            job_id,
            status="done",
            output_path=str(output_path),
            output_format=output_format,
            download_name=download_name,
        )

    except Exception as e:
        update_job(job_id, status="error", error=str(e))
    finally:
        Path(audio_path).unlink(missing_ok=True)
        _unload_llm(llm_model)  # освобождаем VRAM для следующей задачи в очереди


@app.post("/process")
async def process_audio(file: UploadFile = File(...), output_format: str = Form("docx"), llm_model: str = Form("")):
    ext = Path((file.filename or "").replace("\\", "/")).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Неподдерживаемый формат {ext}. Разрешены: MP3, WAV, M4A.")

    output_format = output_format.lower()
    if output_format not in ALLOWED_FORMATS:
        raise HTTPException(400, f"Неподдерживаемый формат выгрузки: {output_format}")

    llm_model = llm_model.strip()
    if llm_model and llm_model != DEFAULT_MODEL:
        installed = [m["name"] for m in list_models()]
        if llm_model not in installed:
            raise HTTPException(400, f"Модель {llm_model} не установлена в Ollama. Доступны: "
                                     f"{', '.join(installed) or DEFAULT_MODEL}.")

    if gpu_queue.size() >= MAX_QUEUE:
        raise HTTPException(429, f"Очередь переполнена ({MAX_QUEUE} задач). Повторите позже.")

    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    size = 0
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        audio_path = tmp.name
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                break
            tmp.write(chunk)
    if size > max_bytes:
        Path(audio_path).unlink(missing_ok=True)
        raise HTTPException(413, f"Файл больше {MAX_UPLOAD_MB} МБ. Уменьшите размер или разделите запись.")

    job_id = create_job()
    update_job(job_id, llm_model=llm_model or DEFAULT_MODEL)
    stem = _safe_stem(file.filename or "")

    gpu_queue.submit(job_id, _run_pipeline, job_id, audio_path, stem, output_format, llm_model or None)

    return {"job_id": job_id}


@app.get("/models")
async def models():
    """Модели Qwen, установленные в Ollama (для выбора в интерфейсе), и модель по умолчанию."""
    found = list_models()
    if DEFAULT_MODEL not in [m["name"] for m in found]:  # Ollama недоступна или модель по умолчанию названа иначе
        found.insert(0, {"name": DEFAULT_MODEL, "size_gb": None})
    return {"default": DEFAULT_MODEL, "models": found}


@app.get("/jobs/{job_id}")
async def job_status(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "Задача не найдена.")
    return JSONResponse({
        "status": job["status"],
        "error": job["error"],
        "output_format": job["output_format"],
        # 0 — задачу уже забрал рабочий поток, но статус ещё не сменился (запускается)
        "queue_position": (gpu_queue.position(job_id) or 0) if job["status"] == "queued" else None,
        "has_transcript": job["transcript"] is not None,
        "attempt": job["attempt"],
        "max_attempts": job["max_attempts"],
        "protocol_empty": job["protocol_empty"],
        "message": job["message"],
        "issues": job["issues"],
        "device_mode": job["device_mode"],
        "device_notes": job["device_notes"],
        "llm_model": job["llm_model"],
    })


@app.get("/jobs/{job_id}/transcript")
async def job_transcript(job_id: str):
    job = get_job(job_id)
    if job is None or job["transcript"] is None:
        raise HTTPException(404, "Расшифровка ещё не готова.")
    return PlainTextResponse(job["transcript"])


@app.get("/jobs/{job_id}/download")
async def job_download(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "Задача не найдена.")
    if job["status"] != "done":
        raise HTTPException(409, f"Задача ещё не завершена (статус: {job['status']}).")

    output_path = Path(job["output_path"])
    return FileResponse(
        path=output_path,
        filename=job["download_name"],
        media_type=MEDIA_TYPES[job["output_format"]],
    )


# Статика фронта монтируется последней, чтобы не перекрыть /process, /jobs и /docs.
app.mount("/", StaticFiles(directory=str(PROJECT_ROOT / "frontend"), html=True), name="frontend")
