"""Чтение настроек из переменных окружения. Все пороги ASR/LLM вынесены сюда, а не в код."""
import os


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def env_floats(name: str, default: str) -> list[float]:
    return [float(x) for x in os.environ.get(name, default).split(",") if x.strip()]
