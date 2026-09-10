import uuid
from datetime import datetime, timezone

_jobs = {}


def create_job() -> str:
    job_id = uuid.uuid4().hex
    _jobs[job_id] = {
        "status": "queued",     # queued | transcribing | generating | done | error
        "error": None,
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
