import gc
import logging
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Callable

from faster_whisper import WhisperModel

from app.config import env_bool, env_float, env_floats, env_int, env_str
from app.repeats import collapse_loops, dedupe_segments, is_caption_hallucination

log = logging.getLogger("asr")

_MODEL_NAME = env_str("ASR_MODEL", "large-v3")
# Подписи-«галлюцинации» Whisper на музыке/шуме; список настраивается через env (регулярное выражение).
_HALLUCINATION_RE = env_str(
    "ASR_HALLUCINATION_PATTERN",
    r"^\W*((динамичная|спокойная|тихая)\s+)?(играет\s+)?музыка\W*$|субтитры|продолжение следует|спасибо за просмотр")

# Отказы, при которых имеет смысл перейти на следующий шаг цепочки (иные RuntimeError пробрасываются).
_OOM_RE = re.compile(r"out of memory|memory allocation|cudaerrormemoryallocation|failed to allocate|"
                     r"bad_alloc|not enough memory|cannot allocate", re.IGNORECASE)
_NO_GPU_RE = re.compile(r"no cuda-capable device|cuda driver version|cublas\S* is not found|cudnn\S* is not found|"
                        r"cannot be loaded|cuda.*unavailable", re.IGNORECASE)

MODE_GPU, MODE_GPU_REDUCED, MODE_CPU = "gpu", "gpu_reduced", "cpu"


class AudioError(ValueError):
    """Файл пустой, повреждён или не является аудио — сообщение показывается пользователю."""


@dataclass
class Step:
    device: str
    compute_type: str
    model: str
    mode: str


def _preprocess(audio_path: str) -> str:
    """ffmpeg: mono 16 кГц, highpass и выравнивание громкости — чтобы тихие/дальние спикеры не терялись.
    Возвращает путь к временному WAV. Если ASR_PREPROCESS=0 или ffmpeg не установлен — исходный файл.
    Если ffmpeg не смог прочитать файл — AudioError."""
    if os.path.getsize(audio_path) == 0:
        raise AudioError("Файл пустой (0 байт).")
    if not env_bool("ASR_PREPROCESS", True):
        return audio_path
    filters = env_str("ASR_AUDIO_FILTERS", "highpass=f=80,dynaudnorm=f=100:g=31:p=0.95:m=30")
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", audio_path, "-vn", "-af", filters,
             "-ac", "1", "-ar", "16000", tmp.name],
            capture_output=True, text=True, timeout=env_int("ASR_FFMPEG_TIMEOUT", 3600))
    except subprocess.TimeoutExpired:
        os.unlink(tmp.name)
        raise AudioError("Предобработка аудио заняла слишком много времени (ASR_FFMPEG_TIMEOUT).")
    except FileNotFoundError:
        log.warning("ffmpeg не найден, распознаю исходный файл без предобработки")
        os.unlink(tmp.name)
        return audio_path
    if r.returncode != 0:
        os.unlink(tmp.name)
        log.warning("ffmpeg не смог прочитать файл: %s", r.stderr.strip()[:300])
        raise AudioError("Не удалось прочитать аудио: файл повреждён или не является аудиозаписью.")
    return tmp.name


def _duration_minutes(path: str) -> float | None:
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                           capture_output=True, text=True, timeout=60)
        return float(r.stdout.strip()) / 60
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _cuda_available() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # нет DLL/драйвера — считаем, что GPU нет
        return False


def _vram_used_mb() -> int | None:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return int(r.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def plan_steps(duration_min: float | None) -> list[Step]:
    """Цепочка запуска: cuda int8_float16 → cuda int8 → cpu int8 (та же модель) → cpu int8 (меньшая модель).
    ASR_DEVICE: auto (по наличию CUDA) | cuda | cpu. Длинная запись на CPU сразу идёт на меньшую модель
    (ASR_CPU_LARGE_MAX_MINUTES), если она задана."""
    device = env_str("ASR_DEVICE", "auto").lower()
    use_cuda = device == "cuda" or (device == "auto" and _cuda_available())
    steps: list[Step] = []
    if use_cuda:
        types = [t.strip() for t in env_str("ASR_CUDA_COMPUTE_TYPES", "int8_float16,int8").split(",") if t.strip()]
        for i, t in enumerate(types):
            steps.append(Step("cuda", t, _MODEL_NAME, MODE_GPU if i == 0 else MODE_GPU_REDUCED))
    cpu_type = env_str("ASR_CPU_COMPUTE_TYPE", "int8")
    small = env_str("ASR_CPU_FALLBACK_MODEL", "medium")
    too_long = duration_min is not None and duration_min > env_float("ASR_CPU_LARGE_MAX_MINUTES", 60)
    use_small = bool(small) and small != _MODEL_NAME
    if not (too_long and use_small):  # длинную запись сразу отдаём меньшей модели, только если она есть
        steps.append(Step("cpu", cpu_type, _MODEL_NAME, MODE_CPU))
    if use_small:
        steps.append(Step("cpu", cpu_type, small, MODE_CPU))
    return steps


def _classify(err: BaseException) -> str | None:
    """'oom' | 'no_gpu' — можно переходить дальше по цепочке; None — ошибка не про ресурсы, пробрасываем."""
    if isinstance(err, MemoryError):
        return "oom"
    if isinstance(err, RuntimeError):
        text = str(err)
        if _OOM_RE.search(text):
            return "oom"
        if _NO_GPU_RE.search(text):
            return "no_gpu"
    return None


def _release_vram(before_mb: int | None) -> None:
    """После сбоя освобождаем память и проверяем по nvidia-smi, что VRAM реально вернулась."""
    gc.collect()
    if before_mb is None:
        return
    deadline = time.time() + env_float("ASR_VRAM_RELEASE_TIMEOUT", 10)
    while time.time() < deadline:
        used = _vram_used_mb()
        if used is None or used <= before_mb + 300:
            log.info("VRAM освобождена: %s МБ (до попытки было %s МБ)", used, before_mb)
            return
        time.sleep(0.5)
    log.warning("VRAM не освободилась за отведённое время: занято %s МБ, до попытки было %s МБ",
                _vram_used_mb(), before_mb)


def _run_step(wav_path: str, step: Step) -> list[dict]:
    """Загружает модель на указанное устройство, распознаёт и выгружает. Сегменты читаются здесь же:
    генератор faster-whisper ленивый, OOM/ошибка DLL может случиться уже на первом куске аудио."""
    model = None
    try:
        model = WhisperModel(step.model, device=step.device, compute_type=step.compute_type,
                             cpu_threads=env_int("ASR_CPU_THREADS", 0))
        hallucination_silence = env_float("ASR_HALLUCINATION_SILENCE_THRESHOLD", 2.0)
        raw_segments, _ = model.transcribe(
            wav_path,
            language=env_str("ASR_LANGUAGE", "ru"),
            beam_size=env_int("ASR_BEAM_SIZE", 5),
            temperature=env_floats("ASR_TEMPERATURE", "0.0,0.2,0.4,0.6"),
            condition_on_previous_text=env_bool("ASR_CONDITION_ON_PREVIOUS_TEXT", False),
            compression_ratio_threshold=env_float("ASR_COMPRESSION_RATIO_THRESHOLD", 2.4),
            log_prob_threshold=env_float("ASR_LOG_PROB_THRESHOLD", -1.0),
            no_speech_threshold=env_float("ASR_NO_SPEECH_THRESHOLD", 0.6),
            vad_filter=env_bool("ASR_VAD_FILTER", True),
            vad_parameters={
                "threshold": env_float("ASR_VAD_THRESHOLD", 0.35),
                "min_silence_duration_ms": env_int("ASR_VAD_MIN_SILENCE_MS", 500),
                "speech_pad_ms": env_int("ASR_VAD_SPEECH_PAD_MS", 400),
            },
            word_timestamps=hallucination_silence > 0,
            hallucination_silence_threshold=hallucination_silence or None,
        )
        drop_no_speech = env_float("ASR_DROP_NO_SPEECH_PROB", 0.6)
        drop_logprob = env_float("ASR_DROP_AVG_LOGPROB", -1.0)
        segments = []
        for s in raw_segments:
            text = collapse_loops(s.text.strip())
            if not text or is_caption_hallucination(text, _HALLUCINATION_RE):
                continue
            # Уверенно «не речь» и низкая уверенность модели — типичная галлюцинация на музыке/шуме.
            if s.no_speech_prob > drop_no_speech and s.avg_logprob < drop_logprob:
                log.info("drop hallucination [%.1f] %r", s.start, text[:60])
                continue
            segments.append({"start": round(s.start, 1), "end": round(s.end, 1), "text": text})
        return segments
    finally:
        model = None
        gc.collect()


def transcribe_audio(audio_path: str, on_device: Callable[[str, str, str | None], None] | None = None):
    """Распознаёт речь в файле.
    Возвращает (full_text, segments) — полный текст для Qwen и список сегментов
    с таймкодами [{start, end, text}] для отображения пользователю.
    Модель грузится и выгружается внутри функции, чтобы не держать VRAM занятой пока работает Qwen.
    При нехватке видеопамяти/отсутствии GPU идёт по цепочке plan_steps; on_device(mode, model, reason)
    вызывается при каждом выборе устройства (reason: None | 'oom' | 'no_gpu')."""
    wav_path = _preprocess(audio_path)
    try:
        duration = _duration_minutes(wav_path)
        max_minutes = env_float("ASR_MAX_AUDIO_MINUTES", 240)
        if duration is not None and duration > max_minutes:
            raise AudioError(f"Запись длиннее допустимых {max_minutes:g} минут ({duration:.0f} мин). "
                             "Разделите её на части.")
        steps = plan_steps(duration)
        reason = None
        segments = None
        while steps:
            step = steps.pop(0)
            if on_device:
                on_device(step.mode, step.model, reason)
            log.info("ASR: %s/%s model=%s mode=%s", step.device, step.compute_type, step.model, step.mode)
            before = _vram_used_mb() if step.device == "cuda" else None
            failure = None
            try:
                segments = _run_step(wav_path, step)
            except (RuntimeError, MemoryError) as e:
                failure = _classify(e)
                # CUDA-рантайм недоступен для всего процесса: повтор на GPU бесполезен (в тесте зависал в encode)
                remaining = [x for x in steps if x.device != "cuda"] if failure == "no_gpu" else steps
                if failure is None or not remaining:
                    raise
                steps = remaining
                log.warning("ASR шаг %s/%s (%s) не удался [%s]: %s", step.device, step.compute_type,
                            step.model, failure, str(e)[:200])
            if failure is None:
                break
            reason = failure
            _release_vram(before)  # вне блока except: иначе traceback держит модель и память не освободится
    finally:
        if wav_path != audio_path:
            os.unlink(wav_path)

    segments = dedupe_segments(segments or [], similarity=env_float("ASR_DEDUPE_SIMILARITY", 0.9))
    full_text = " ".join(s["text"] for s in segments)
    return full_text.strip(), segments


def format_timestamped_transcript(segments) -> str:
    """Форматирует сегменты в читаемый текст с таймкодами вида [MM:SS] текст."""
    lines = []
    for s in segments:
        start_min, start_sec = divmod(int(s["start"]), 60)
        lines.append(f"[{start_min:02d}:{start_sec:02d}] {s['text']}")
    return "\n".join(lines)
