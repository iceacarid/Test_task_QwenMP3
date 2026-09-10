import asyncio
import tempfile
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from app.asr import transcribe_audio, format_timestamped_transcript
from app.llm import generate_protocol
from app.docx_builder import build_protocol_docx
from app.txt_builder import build_protocol_txt
from app.pdf_builder import build_protocol_pdf
from app.jobs import create_job, update_job, get_job

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


def _run_pipeline(job_id: str, audio_path: str, stem: str, output_format: str):
    """Выполняется в отдельном потоке — блокирующие вызовы ASR/Qwen/LibreOffice не блокируют сервер."""
    try:
        update_job(job_id, status="transcribing")
        transcript, segments = transcribe_audio(audio_path)
        if not transcript:
            update_job(job_id, status="error", error="Не удалось распознать речь в файле.")
            return

        timestamped = format_timestamped_transcript(segments)
        update_job(job_id, transcript=transcript)

        update_job(job_id, status="generating")
        protocol = generate_protocol(timestamped)

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


@app.post("/process")
async def process_audio(file: UploadFile = File(...), output_format: str = Form("docx")):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Неподдерживаемый формат {ext}. Разрешены: MP3, WAV, M4A.")

    output_format = output_format.lower()
    if output_format not in ALLOWED_FORMATS:
        raise HTTPException(400, f"Неподдерживаемый формат выгрузки: {output_format}")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        audio_path = tmp.name

    job_id = create_job()
    stem = Path(file.filename).stem

    asyncio.create_task(asyncio.to_thread(_run_pipeline, job_id, audio_path, stem, output_format))

    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
async def job_status(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "Задача не найдена.")
    return JSONResponse({
        "status": job["status"],
        "error": job["error"],
        "has_transcript": job["transcript"] is not None,
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
