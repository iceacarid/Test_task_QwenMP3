import uuid
from datetime import datetime, timezone

_jobs = {}


def create_job() -> str:
    job_id = uuid.uuid4().hex
    _jobs[job_id] = {
        "status": "queued",     # queued | transcribing | generating | regenerating | done | error
        "error": None,
        "attempt": 0,           # номер текущей попытки Qwen
        "max_attempts": 0,
        "protocol_empty": False,  # True: протокол не получился, вместо него выдана расшифровка
        "message": None,        # пояснение для пользователя
        "issues": [],           # найденные проверкой на галлюцинации проблемы
        "device_mode": {"asr": None, "llm": None},  # gpu | gpu_reduced | cpu отдельно для ASR и Qwen
        "device_notes": [],
        "llm_model": None,      # модель Qwen, выбранная для этой записи     # сообщения пользователю о переключении на CPU / экономный режим
        "transcript": None,
        "output_path": None,
        "output_format": None,
        "download_name": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return job_id


def update_job(job_id: str, **fields):
    _jobs[job_id].update(fields)


def get_job(job_id: str) -> dict | None:
    return _jobs.get(job_id)


def set_device(job_id: str, part: str, mode: str, note: str | None = None):
    """Записывает режим устройства для этапа ('asr' | 'llm') и, если есть, пояснение для пользователя."""
    job = _jobs[job_id]
    job["device_mode"][part] = mode
    if note and note not in job["device_notes"]:
        job["device_notes"].append(note)
