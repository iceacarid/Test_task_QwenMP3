"""Очередь GPU-задач: за раз выполняется GPU_WORKERS задач (по умолчанию 1), остальные ждут в порядке поступления.
Whisper large-v3 (~2.3 ГБ) и Qwen (~5 ГБ) не помещаются в 8 ГБ вместе с параллельными задачами."""
import logging
import threading
from collections import deque
from typing import Callable

from app.config import env_int

log = logging.getLogger("gpu_queue")


class GpuQueue:
    def __init__(self, workers: int | None = None):
        self._workers = workers if workers is not None else env_int("GPU_WORKERS", 1)
        self._waiting: deque[tuple[str, Callable, tuple]] = deque()
        self._cv = threading.Condition()
        self._threads: list[threading.Thread] = []

    @property
    def workers(self) -> int:
        return self._workers

    def submit(self, job_id: str, fn: Callable, *args) -> None:
        with self._cv:
            self._waiting.append((job_id, fn, args))
            if len(self._threads) < self._workers:
                t = threading.Thread(target=self._loop, name=f"gpu-worker-{len(self._threads)}", daemon=True)
                self._threads.append(t)
                t.start()
            self._cv.notify()

    def position(self, job_id: str) -> int | None:
        """Место в очереди ожидания: 1 — следующая на запуск; None — задача уже не ждёт."""
        with self._cv:
            for i, (jid, _, _) in enumerate(self._waiting):
                if jid == job_id:
                    return i + 1
        return None

    def size(self) -> int:
        with self._cv:
            return len(self._waiting)

    def _loop(self) -> None:
        while True:
            with self._cv:
                while not self._waiting:
                    self._cv.wait()
                job_id, fn, args = self._waiting.popleft()
            try:
                fn(*args)
            except BaseException:  # рабочий поток не должен умирать из-за одной задачи
                log.exception("задача %s завершилась исключением", job_id)
