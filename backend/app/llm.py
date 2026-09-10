import json
import os
import requests
from pathlib import Path

MODEL = os.environ.get("QWEN_MODEL", "qwen2.5:7b-instruct")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "protocol_system.txt"


def generate_protocol(transcript: str) -> dict:
    """Отправляет расшифровку в Qwen (через Ollama), возвращает разобранный JSON-протокол."""
    system_prompt = _PROMPT_PATH.read_text(encoding="utf-8")

    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Расшифровка совещания:\n\n{transcript}"}
        ],
        "format": "json",
        "stream": False
    }, timeout=300)
    response.raise_for_status()

    content = response.json()["message"]["content"]
    return json.loads(content)
