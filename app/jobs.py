"""Tiny in-process job queue: one generation at a time, thread-safe."""
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

_jobs: dict[str, dict] = {}
_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1)


def create_job(kind: str = "video") -> dict:
    job = {
        "id": uuid.uuid4().hex[:12],
        "kind": kind,
        "status": "queued",     # queued -> running -> done | error
        "progress": 0.0,        # 0..1
        "stage": "waiting",
        "result": None,
        "error": None,
    }
    with _lock:
        _jobs[job["id"]] = job
    return job


def update(job_id: str, **fields):
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(fields)


def get_job(job_id: str) -> dict | None:
    with _lock:
        return _jobs.get(job_id)


def submit(fn, *args, **kwargs):
    _executor.submit(fn, *args, **kwargs)
