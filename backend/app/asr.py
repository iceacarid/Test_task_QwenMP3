import gc
from faster_whisper import WhisperModel

_MODEL_NAME = "large-v3"


def transcribe_audio(audio_path: str):
    """Распознаёт речь в файле.
    Возвращает (full_text, segments) — полный текст для Qwen и список сегментов
    с таймкодами [{start, end, text}] для отображения пользователю.
    Модель грузится и выгружается внутри функции, чтобы не держать VRAM
    занятой пока работает Qwen (GPU 8Гб, места на обе модели одновременно нет)."""
    model = WhisperModel(_MODEL_NAME, device="cuda", compute_type="int8_float16")

    raw_segments, _ = model.transcribe(audio_path, language="ru")
    segments = [
        {"start": round(s.start, 1), "end": round(s.end, 1), "text": s.text.strip()}
        for s in raw_segments
    ]
    full_text = " ".join(s["text"] for s in segments)

    del model
    gc.collect()

    return full_text.strip(), segments


def format_timestamped_transcript(segments) -> str:
    """Форматирует сегменты в читаемый текст с таймкодами вида [MM:SS] текст."""
    lines = []
    for s in segments:
        start_min, start_sec = divmod(int(s["start"]), 60)
        lines.append(f"[{start_min:02d}:{start_sec:02d}] {s['text']}")
    return "\n".join(lines)
