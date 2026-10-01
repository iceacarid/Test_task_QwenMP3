import json
import logging
import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable

import requests

from app.config import env_float, env_int, env_str
from app.grounding import confirm_speakers, ground_protocol
from app.repeats import normalize

log = logging.getLogger("llm")

MODEL = env_str("QWEN_MODEL", "qwen2.5:7b-instruct")
OLLAMA_URL = env_str("OLLAMA_URL", "http://localhost:11434/api/chat")

_PROMPTS = Path(__file__).resolve().parents[2] / "prompts"

_NULLABLE_STR = {"type": ["string", "null"]}
PROTOCOL_SCHEMA = {
    "type": "object",
    "properties": {
        "meta": {
            "type": "object",
            "properties": {"time": _NULLABLE_STR, "place": _NULLABLE_STR,
                           "participants": {"type": "array", "items": {"type": "string"}}},
            "required": ["time", "place", "participants"],
        },
        "agenda_items": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "speaker": _NULLABLE_STR,
                "topic_summary": {"type": "string"},
                "audio_timestamp": _NULLABLE_STR,
                "resolved": {"type": "boolean"},
                "decisions": {"type": "array", "items": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}, "responsible": _NULLABLE_STR, "deadline": _NULLABLE_STR},
                    "required": ["text", "responsible", "deadline"]}},
                "needs_review": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["speaker", "topic_summary", "audio_timestamp", "resolved", "decisions", "needs_review"],
        }},
    },
    "required": ["meta", "agenda_items"],
}


class LLMError(RuntimeError):
    """Ошибка обращения к Ollama; текст показывается пользователю."""


class LLMMemoryError(LLMError):
    """Ollama не хватило памяти (VRAM/RAM) или упал runner модели — можно повторить с меньшей нагрузкой на GPU."""


class LLMUnavailable(LLMError):
    """Ollama недоступна (процесс перезапускается или остановлен)."""


_MEM_RE = re.compile(r"out of memory|requires more system memory|unable to allocate|cudamalloc|cuda error|"
                     r"runner process has terminated|runner terminated|exit status|failed to load model|"
                     r"insufficient memory|not enough memory|cannot allocate", re.IGNORECASE)

MODE_GPU, MODE_GPU_REDUCED, MODE_CPU = "gpu", "gpu_reduced", "cpu"


@dataclass
class Settings:
    temperature: float
    num_ctx_max: int
    reserve_out: int
    chars_per_token: float
    chunk_overlap_chars: int
    chunk_chars: int
    max_attempts: int
    timeout: int
    min_transcript_chars: int
    min_diversity: float
    merge_similarity: float
    merge_ts_window: int
    use_schema: bool
    gpu_layers_reduced: int
    timeout_reduced: int
    timeout_cpu: int
    conn_retries: int
    retry_delay: float
    ladder: list[str]
    speaker_window: int
    max_output_tokens: int
    seed: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            temperature=env_float("LLM_TEMPERATURE", 0.0),
            num_ctx_max=env_int("LLM_NUM_CTX_MAX", 12288),
            reserve_out=env_int("LLM_RESERVE_OUT_TOKENS", 2048),
            chars_per_token=env_float("LLM_CHARS_PER_TOKEN", 2.5),
            chunk_overlap_chars=env_int("LLM_CHUNK_OVERLAP_CHARS", 1500),
            chunk_chars=env_int("LLM_CHUNK_CHARS", 9000),
            max_attempts=env_int("LLM_MAX_ATTEMPTS", 3),
            timeout=env_int("LLM_TIMEOUT", 900),
            min_transcript_chars=env_int("LLM_MIN_TRANSCRIPT_CHARS", 200),
            min_diversity=env_float("LLM_MIN_DIVERSITY", 0.15),
            merge_similarity=env_float("LLM_MERGE_SIMILARITY", 0.55),
            merge_ts_window=env_int("LLM_MERGE_TS_WINDOW", 45),
            use_schema=env_str("LLM_USE_SCHEMA", "1") == "1",
            gpu_layers_reduced=env_int("LLM_GPU_LAYERS_REDUCED", 14),
            timeout_reduced=env_int("LLM_TIMEOUT_REDUCED", 1800),
            timeout_cpu=env_int("LLM_TIMEOUT_CPU", 3600),
            conn_retries=env_int("LLM_CONN_RETRIES", 2),
            retry_delay=env_float("LLM_RETRY_DELAY", 5),
            ladder=[m.strip() for m in env_str("LLM_FALLBACK_LADDER", "gpu,gpu_reduced,cpu").split(",")
                    if m.strip() in (MODE_GPU, MODE_GPU_REDUCED, MODE_CPU)] or [MODE_GPU],
            speaker_window=env_int("LLM_SPEAKER_MAX_AGE", 600),
            max_output_tokens=env_int("LLM_MAX_OUTPUT_TOKENS", 3072),
            seed=env_int("LLM_SEED", 42),
        )


@dataclass
class ProtocolResult:
    protocol: dict | None
    empty: bool
    message: str | None = None
    attempts: list[dict] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    device_mode: str = MODE_GPU


_TS_PREFIX_RE = re.compile(r"^\[\d{1,3}:\d{2}\]\s*", re.MULTILINE)


def _diversity(transcript: str, window: int = 300) -> float:
    """Доля различных слов среди первых `window` слов: у осмысленной речи ~0.4–0.7, у мусора из
    повторяющихся слогов — единицы процентов."""
    words = normalize(_TS_PREFIX_RE.sub("", transcript)).split()[:window]
    return len(set(words)) / len(words) if words else 0.0


@dataclass
class GpuState:
    """Текущая ступень лестницы GPU→CPU; сохраняется между запросами задачи, чтобы не повторять неудачу."""
    settings: Settings
    index: int = 0

    @property
    def mode(self) -> str:
        return self.settings.ladder[self.index]

    @property
    def num_gpu(self) -> int | None:
        return {MODE_GPU: None, MODE_GPU_REDUCED: self.settings.gpu_layers_reduced, MODE_CPU: 0}[self.mode]

    @property
    def timeout(self) -> int:
        s = self.settings
        return {MODE_GPU: s.timeout, MODE_GPU_REDUCED: s.timeout_reduced, MODE_CPU: s.timeout_cpu}[self.mode]

    def advance(self) -> bool:
        if self.index + 1 >= len(self.settings.ladder):
            return False
        self.index += 1
        return True


def _chat(system: str, user: str, temperature: float, num_ctx: int, s: Settings,
          num_gpu: int | None = None, timeout: int | None = None, model: str | None = None) -> dict:
    timeout = timeout or s.timeout
    # num_predict ограничивает длину ответа: без него зациклившаяся или замедленная генерация держит очередь до таймаута
    options = {"num_ctx": num_ctx, "temperature": temperature, "num_predict": s.max_output_tokens}
    if s.seed >= 0:  # фиксированное зерно: одна и та же запись даёт один и тот же протокол (-1 — случайное)
        options["seed"] = s.seed
    if num_gpu is not None:
        options["num_gpu"] = num_gpu
    payload = {
        "model": model or MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "format": PROTOCOL_SCHEMA if s.use_schema else "json",
        "stream": False,
        "options": options,
        "keep_alive": env_str("LLM_KEEP_ALIVE", "10m"),  # держим на время этапа; после него выгружаем явно (unload_model)
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        r.raise_for_status()
    except requests.ConnectionError as e:
        raise LLMUnavailable("Сервис Ollama недоступен. Проверьте, что он запущен.") from e
    except requests.Timeout as e:
        raise LLMError(f"Ollama не ответила за {timeout} с.") from e
    except requests.HTTPError as e:
        body = e.response.text[:300]
        cls = LLMMemoryError if _MEM_RE.search(body) else LLMError
        raise cls(f"Ollama вернула ошибку {e.response.status_code}: {body[:200]}") from e
    try:
        j = r.json()
    except ValueError as e:
        raise LLMError("Ollama вернула некорректный ответ (не JSON).") from e
    if isinstance(j, dict) and j.get("error"):
        cls = LLMMemoryError if _MEM_RE.search(str(j["error"])) else LLMError
        raise cls(f"Ollama сообщила об ошибке: {str(j['error'])[:200]}")
    return j


def unload_model(model: str | None = None) -> bool:
    """Просит Ollama выгрузить Qwen из памяти (keep_alive=0), чтобы освободить VRAM под Whisper. Ошибки не критичны."""
    url = OLLAMA_URL.rsplit("/api/", 1)[0] + "/api/generate"
    try:
        r = requests.post(url, json={"model": model or MODEL, "keep_alive": 0}, timeout=env_int("LLM_UNLOAD_TIMEOUT", 60))
        return r.ok
    except requests.RequestException as e:
        log.warning("не удалось выгрузить модель из Ollama: %s", e)
        return False


def list_models() -> list[dict]:
    """Установленные в Ollama модели семейства Qwen: [{'name', 'size_gb'}]. Ошибка связи — пустой список."""
    url = OLLAMA_URL.rsplit("/api/", 1)[0] + "/api/tags"
    try:
        r = requests.get(url, timeout=env_int("LLM_TAGS_TIMEOUT", 5))
        r.raise_for_status()
        items = r.json().get("models", [])
    except (requests.RequestException, ValueError, AttributeError):
        return []
    models = [{"name": m["name"], "size_gb": round(m.get("size", 0) / 1024 ** 3, 1)}
              for m in items if isinstance(m, dict) and str(m.get("name", "")).lower().startswith("qwen")]
    return sorted(models, key=lambda m: m["size_gb"])


def _chat_adaptive(system: str, user: str, temperature: float, num_ctx: int, s: Settings, state: GpuState,
                   on_device: Callable[[str, str | None], None] | None, model: str | None = None) -> dict:
    """Запрос с лестницей нагрузки: нехватка памяти → меньше слоёв на GPU → чистый CPU (num_gpu=0);
    недоступность Ollama (перезапуск) → несколько повторов с паузой."""
    retries = 0
    while True:
        try:
            return _chat(system, user, temperature, num_ctx, s, num_gpu=state.num_gpu, timeout=state.timeout, model=model)
        except LLMMemoryError as e:
            if not state.advance():
                raise
            log.warning("Ollama: нехватка памяти (%s), режим %s (num_gpu=%s)", e, state.mode, state.num_gpu)
            if on_device:
                on_device(state.mode, "oom")
        except LLMUnavailable:
            if retries >= s.conn_retries:
                raise
            retries += 1
            log.warning("Ollama недоступна, повтор %d/%d через %s с", retries, s.conn_retries, s.retry_delay)
            time.sleep(s.retry_delay)


# Пересказ мнения («X отметил, что …») — не решение: такие «решения» 14B выдумывает на записях без решений.
_REPORTED_SPEECH_RE = re.compile(
    r"\b(?:отметил[аи]?|сказал[аи]?|считает|считают|рассказал[аи]?|подчеркнул[аи]?|заявил[аи]?|утверждает|утверждают|"
    r"полагает|полагают|упомянул[аи]?|объяснил[аи]?|говорит|говорят)\s*,?\s*что\b", re.IGNORECASE)
# Пометки про личность говорящего или без смысла: разделения по голосам нет, проверить по тексту нечего.
_USELESS_NOTE_RE = re.compile(
    r"кто\s+(?:именно\s+)?(?:говорит|спрашива|отвеча|выступа)|в каком контексте|как это влияет на смысл|чей\s+голос",
    re.IGNORECASE)
_LEADING_NOISE = {"ваша", "ваш", "наша", "наш", "моя", "мой", "уважаемый", "уважаемая", "господин", "госпожа"}


def _clean_person(name: object) -> str | None:
    """«Ваша Екатерина» (так её услышал ASR) → «Екатерина»: притяжательные слова перед именем отбрасываем."""
    words = str(name or "").split()
    while words and words[0].lower().strip(",.") in _LEADING_NOISE:
        words.pop(0)
    return " ".join(words) or None


def _clean_decisions(raw_decisions: list) -> list[dict]:
    """Оставляет настоящие решения: убирает пересказ мнений без срока и роли вместо ответственного."""
    out = []
    for d in raw_decisions:
        if not isinstance(d, dict) or not str(d.get("text") or "").strip():
            continue
        text = str(d["text"]).strip()
        responsible, deadline = d.get("responsible") or None, d.get("deadline") or None
        if responsible and set(normalize(str(responsible)).split()) <= _NOT_A_NAME:
            responsible = None  # «Спикер», «кандидат» — не ответственный
        if deadline is None and _REPORTED_SPEECH_RE.search(text[:160]):
            continue
        out.append({"text": text, "responsible": responsible, "deadline": deadline})
    return out


def _concrete_notes(review: object) -> list[str]:
    """Оставляет только конкретные пометки (с цитатой в кавычках) без дублей: общие фразы вроде
    «некоторые фразы неясны» пользователю не помогают."""
    notes: list[str] = []
    for x in review if isinstance(review, list) else []:
        text = str(x).strip()
        if re.search(r"[«\"“].+?[»\"”]", text) and not _USELESS_NOTE_RE.search(text):
            text = text if len(text) <= 180 else text[:177] + "…"
            if text not in notes:
                notes.append(text)
    return notes[:2]  # больше двух пометок на пункт читателя только утомляют


def _norm_timestamp(ts: object) -> str | None:
    """Модель иногда пишет ЧЧ:ММ:СС — переводим в ММ:СС (как в расшифровке)."""
    if not ts:
        return None
    m = re.match(r"^\s*(\d{1,2}):(\d{2}):(\d{2})\s*$", str(ts))
    return f"{int(m.group(1)) * 60 + int(m.group(2)):02d}:{m.group(3)}" if m else str(ts).strip()


def _clean(raw: object) -> dict | None:
    """Приводит ответ модели к схеме. None — если это не протокол."""
    if not isinstance(raw, dict) or not isinstance(raw.get("agenda_items"), list):
        return None
    meta = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
    parts = meta.get("participants")
    protocol = {"meta": {"time": meta.get("time") or None, "place": meta.get("place") or None,
                         "participants": [c for c in (_clean_person(p) for p in parts if isinstance(p, str)) if c]
                         if isinstance(parts, list) else []},
                "agenda_items": []}
    for it in raw["agenda_items"]:
        if not isinstance(it, dict):
            continue
        raw_decisions = it.get("decisions") if isinstance(it.get("decisions"), list) else []
        decisions = _clean_decisions(raw_decisions)
        review = it.get("needs_review")
        protocol["agenda_items"].append({
            "speaker": _clean_person(it.get("speaker")),
            "topic_summary": str(it.get("topic_summary") or "").strip(),
            "audio_timestamp": _norm_timestamp(it.get("audio_timestamp")),
            "resolved": bool(it.get("resolved")) and bool(decisions),
            "decisions": decisions,
            "needs_review": _concrete_notes(review),
        })
    return protocol


def empty_reason(protocol: dict | None) -> str | None:
    """Почему протокол считается пустым; None — протокол содержательный."""
    if protocol is None:
        return "JSON не разобран или нет ключа agenda_items"
    items = protocol["agenda_items"]
    if not items:
        return "agenda_items пуст"
    if all(not i["topic_summary"] for i in items):
        return "topic_summary пуст во всех пунктах"
    return None


def split_chunks(transcript: str, chunk_chars: int, overlap_chars: int) -> list[str]:
    """Режет расшифровку по строкам на куски ≤ chunk_chars; соседние куски перекрываются на ~overlap_chars."""
    lines: list[str] = []
    for line in transcript.splitlines():
        while len(line) > chunk_chars:
            lines.append(line[:chunk_chars])
            line = line[chunk_chars:]
        lines.append(line)
    chunks: list[str] = []
    cur: list[str] = []
    size = 0
    for line in lines:
        if cur and size + len(line) + 1 > chunk_chars:
            chunks.append("\n".join(cur))
            tail: list[str] = []
            t = 0
            for prev in reversed(cur):
                if t >= overlap_chars:
                    break
                tail.insert(0, prev)
                t += len(prev) + 1
            while tail and t + len(line) + 1 > chunk_chars:  # перекрытие не должно раздувать кусок сверх лимита
                t -= len(tail.pop(0)) + 1
            cur, size = tail, t
        cur.append(line)
        size += len(line) + 1
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def _ts_seconds(ts: str | None) -> int | None:
    m = re.match(r"^(\d{1,3}):(\d{2})$", ts or "")
    return int(m.group(1)) * 60 + int(m.group(2)) if m else None


def _is_same_topic(a: dict, b: dict, s: Settings) -> bool:
    sim = SequenceMatcher(None, normalize(a["topic_summary"]), normalize(b["topic_summary"])).ratio()
    ta, tb = _ts_seconds(a["audio_timestamp"]), _ts_seconds(b["audio_timestamp"])
    if ta is not None and tb is not None and abs(ta - tb) <= s.merge_ts_window and sim >= 0.3:
        return True
    return sim >= s.merge_similarity


_NOT_A_NAME = {"спикер", "докладчик", "ведущий", "ведущая", "модератор", "участник", "участники", "коллеги", "докладчики", "интервьюируемый", "интервьюируемая", "интервьюер", "кандидат", "соискатель", "журналист", "корреспондент", "гость", "гости", "другие", "остальные", "присутствующие", "прочие", "все", "рекрутер"}


def _same_word(a: str, b: str) -> bool:
    """Слова совпадают или отличаются на ошибку распознавания (Казарян ~ Газарян)."""
    return a == b or (min(len(a), len(b)) >= 5 and SequenceMatcher(None, a, b).ratio() >= 0.8)


def _covers(big: set[str], small: set[str]) -> bool:
    return all(any(_same_word(w, x) for x in big) for w in small)


def _add_person(people: list[str], name: str) -> None:
    """Добавляет участника без дублей: «Анжелика» покрыта «Анжелика Шавский», а «Карен Газарян» — «Карен Казарян».
    Обезличенные слова («Спикер», «Докладчик») не считаются именами."""
    words = set(normalize(name).split())
    if not words or words <= _NOT_A_NAME:
        return
    for i, other in enumerate(people):
        ow = set(normalize(other).split())
        if _covers(ow, words):
            return
        if _covers(words, ow):
            people[i] = name
            return
    people.append(name)


def collapse_repeated_items(protocol: dict, similarity: float = 0.9, min_chars: int = 60) -> None:
    """Модель иногда выдаёт несколько пунктов с одной и той же сутью (особенно на длинных записях). Читателю повторы
    бесполезны, поэтому почти одинаковые по тексту пункты схлопываются в первый (самый ранний): решения и пометки
    объединяются, докладчик и таймкод остаются от первого. Короткие формулировки не трогаем."""
    kept: list[dict] = []
    for it in protocol["agenda_items"]:
        norm = normalize(it["topic_summary"])
        dup = None
        if len(norm) >= min_chars:
            dup = next((k for k in kept
                        if SequenceMatcher(None, normalize(k["topic_summary"]), norm).ratio() >= similarity), None)
        if dup is None:
            kept.append(it)
            continue
        for d in it["decisions"]:
            if not any(SequenceMatcher(None, normalize(d["text"]), normalize(x["text"])).ratio() >= 0.8
                       for x in dup["decisions"]):
                dup["decisions"].append(d)
        dup["resolved"] = dup["resolved"] or it["resolved"]
        dup["needs_review"] += [x for x in it["needs_review"] if x not in dup["needs_review"]]
        dup["needs_review"] = dup["needs_review"][:2]
    protocol["agenda_items"] = kept


def merge_protocols(parts: list[dict], s: Settings) -> dict:
    """Объединяет протоколы кусков без дублей: пункты с одной темой из перекрытия сливаются."""
    merged: dict = {"meta": {"time": None, "place": None, "participants": []}, "agenda_items": []}
    prev_items: list[dict] = []
    for part in parts:
        for key in ("time", "place"):
            merged["meta"][key] = merged["meta"][key] or part["meta"][key]
        for p in part["meta"]["participants"]:
            _add_person(merged["meta"]["participants"], p)
        added: list[dict] = []
        for item in part["agenda_items"]:
            dup = next((k for k in prev_items if _is_same_topic(k, item, s)), None)
            if dup is None:
                merged["agenda_items"].append(item)
                added.append(item)
                continue
            if len(item["topic_summary"]) > len(dup["topic_summary"]):
                dup["topic_summary"] = item["topic_summary"]
            dup["speaker"] = dup["speaker"] or item["speaker"]
            dup["resolved"] = dup["resolved"] or item["resolved"]
            for d in item["decisions"]:
                if not any(SequenceMatcher(None, normalize(d["text"]), normalize(x["text"])).ratio() >= 0.8
                           for x in dup["decisions"]):
                    dup["decisions"].append(d)
            dup["needs_review"] += [x for x in item["needs_review"] if x not in dup["needs_review"]]
        prev_items = added
    merged["agenda_items"].sort(key=lambda i: (_ts_seconds(i["audio_timestamp"]) is None,
                                               _ts_seconds(i["audio_timestamp"]) or 0))
    collapse_repeated_items(merged)
    return merged


def canonicalize_speakers(protocol: dict) -> None:
    """Приводит докладчиков к полным именам из participants: «Тарен» (ошибка ASR вместо «Карен») → «Карен Казарян»,
    «Максим» → «Максим Бустовой». Заменяет только при однозначном совпадении; обезличенные слова («Спикер») убирает."""
    people = protocol["meta"]["participants"]
    for item in protocol["agenda_items"]:
        words = set(normalize(item.get("speaker") or "").split())
        if not words:
            continue
        if words <= _NOT_A_NAME:
            item["speaker"] = None
            continue
        matches = [p for p in people if _covers(set(normalize(p).split()), words)]
        if len(matches) == 1:
            item["speaker"] = matches[0]


def _num_ctx(system: str, user: str, s: Settings) -> int:
    need = int((len(system) + len(user)) / s.chars_per_token) + s.reserve_out + 256
    return max(4096, min(s.num_ctx_max, -(-need // 1024) * 1024))


def _plan_chunks(transcript: str, system: str, attempt: int, s: Settings) -> list[str]:
    """Попытка 1 — весь текст, если влезает в контекст, иначе куски; следующие попытки режут мельче."""
    budget_tokens = s.num_ctx_max - int(len(system) / s.chars_per_token) - s.reserve_out - 512
    budget_chars = min(int(budget_tokens * s.chars_per_token), s.chunk_chars)
    if attempt == 1:
        return [transcript] if len(transcript) <= budget_chars else \
            split_chunks(transcript, budget_chars, s.chunk_overlap_chars)
    size = max(4000, budget_chars // (2 ** (attempt - 1)))
    if len(transcript) <= size:
        return [transcript]
    return split_chunks(transcript, size, min(s.chunk_overlap_chars, size // 4))


def _run_attempt(transcript: str, attempt: int, s: Settings, state: GpuState,
                 on_device: Callable[[str, str | None], None] | None,
                 model: str | None = None) -> tuple[dict | None, dict]:
    name = "protocol_retry.txt" if attempt > 1 else "protocol_system.txt"
    system = (_PROMPTS / name).read_text(encoding="utf-8")
    temperature = round(s.temperature + 0.15 * (attempt - 1), 2)
    chunks = _plan_chunks(transcript, system, attempt, s)
    info = {"attempt": attempt, "mode": "chunked" if len(chunks) > 1 else "single", "chunks": len(chunks),
            "temperature": temperature, "prompt_tokens": 0, "eval_tokens": 0, "num_ctx": 0,
            "truncated": False, "empty_reason": None, "response_head": None, "seconds": 0}
    started = time.time()
    parts: list[dict] = []
    for n, chunk in enumerate(chunks, 1):
        if len(chunks) == 1:
            user = f"Расшифровка:\n\n{chunk}"
        else:
            user = (f"Фрагмент {n} из {len(chunks)} расшифровки (соседние фрагменты частично перекрываются). "
                    f"Опиши только то, что обсуждается в этом фрагменте.\n\n{chunk}")
        num_ctx = _num_ctx(system, user, s)
        j = _chat_adaptive(system, user, temperature, num_ctx, s, state, on_device, model)
        content = j.get("message", {}).get("content", "")
        info["num_ctx"] = max(info["num_ctx"], num_ctx)
        info["prompt_tokens"] += j.get("prompt_eval_count", 0)
        info["eval_tokens"] += j.get("eval_count", 0)
        info["response_head"] = info["response_head"] or content[:200]
        if j.get("prompt_eval_count", 0) >= num_ctx - 8:
            info["truncated"] = True
            log.warning("attempt %d chunk %d: prompt %d tokens >= num_ctx %d, context truncated",
                        attempt, n, j.get("prompt_eval_count", 0), num_ctx)
        try:
            cleaned = _clean(json.loads(content))
        except json.JSONDecodeError:
            cleaned = None
        if cleaned is not None:
            parts.append(cleaned)
    protocol = merge_protocols(parts, s) if parts else None
    info["empty_reason"] = empty_reason(protocol)
    info["device_mode"] = state.mode
    info["seconds"] = round(time.time() - started)
    log.info("LLM attempt %s", json.dumps(info, ensure_ascii=False))
    return protocol, info


def generate_protocol(transcript: str, on_attempt: Callable[[int, int], None] | None = None,
                      on_device: Callable[[str, str | None], None] | None = None,
                      model: str | None = None) -> ProtocolResult:
    """Строит протокол по расшифровке с таймкодами. Пустой результат приводит к повторным проходам
    (чанкинг, уточнённый промпт, другая температура) до LLM_MAX_ATTEMPTS. Если все пустые — result.empty=True."""
    s = Settings.from_env()
    if len(_TS_PREFIX_RE.sub("", transcript).strip()) < s.min_transcript_chars:
        return ProtocolResult(None, True, "В записи слишком мало речи, чтобы составить протокол.")
    if _diversity(transcript) < s.min_diversity:
        return ProtocolResult(None, True, "Речь в записи не удалось разобрать (в расшифровке одни повторяющиеся "
                                          "слоги и междометия), протокол составить нельзя.")
    attempts: list[dict] = []
    state = GpuState(s)
    if on_device:
        on_device(state.mode, None)
    for attempt in range(1, s.max_attempts + 1):
        if on_attempt:
            on_attempt(attempt, s.max_attempts)
        protocol, info = _run_attempt(transcript, attempt, s, state, on_device, model)
        attempts.append(info)
        if info["empty_reason"] is None:
            canonicalize_speakers(protocol)
            issues = ground_protocol(protocol, transcript) + confirm_speakers(protocol, transcript, s.speaker_window)
            if issues:
                log.warning("grounding: %s", issues)
            return ProtocolResult(protocol, False, None, attempts, issues, state.mode)
    return ProtocolResult(None, True,
                          f"Протокол получился пустым после {len(attempts)} попыток. Ниже приложена расшифровка записи.",
                          attempts, device_mode=state.mode)
