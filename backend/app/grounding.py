"""Проверка протокола на галлюцинации: имена, сроки и цифры должны встречаться в расшифровке."""
import re
from difflib import SequenceMatcher

from app.repeats import normalize

_TS_RE = re.compile(r"^\d{1,3}[:.]\d{2}$")  # модель иногда пишет таймкод как 05.00
_AUDIO_TS_RE = re.compile(r"^\d{1,3}:\d{2}$")  # таймкод записи строго ММ:СС
_LINE_TS_RE = re.compile(r"^\[(\d{1,3}):(\d{2})\]", re.MULTILINE)
_STOP = {"для", "что", "как", "это", "или", "при", "над", "под", "без", "его", "все", "так"}


class Transcript:
    """Индекс слов расшифровки для быстрой проверки «слово (или его форма) есть в тексте»."""

    def __init__(self, text: str):
        words = normalize(text).split()
        self.words = set(words)
        self.by_prefix: dict[str, list[str]] = {}
        for w in self.words:
            self.by_prefix.setdefault(w[:4], []).append(w)
        self.numbers = {w for w in words if w.isdigit()}
        secs = [int(m) * 60 + int(s) for m, s in _LINE_TS_RE.findall(text)]
        self.max_seconds = max(secs) if secs else None

    def has_number(self, n: str) -> bool:
        """Число есть в тексте; год «2023» считается подтверждённым записью «23-го года»."""
        return n in self.numbers or (len(n) == 4 and n.startswith("20") and n[2:] in self.numbers)

    def has_word(self, w: str) -> bool:
        """Слово есть в тексте с точностью до окончания (Иванов ~ Иванову ~ Ивановым)."""
        if w in self.words:
            return True
        if len(w) < 4:
            return False
        for t in self.by_prefix.get(w[:4], ()):
            if min(len(t), len(w)) >= 4 and _common_prefix(w, t) >= max(4, min(len(w), len(t)) - 2):
                return True
        return False


_NOT_A_NAME_WORDS = {"спикер", "докладчик", "ведущий", "ведущая", "модератор", "участник", "коллеги", "интервьюируемый", "интервьюируемая", "интервьюер", "кандидат", "соискатель", "журналист", "корреспондент", "гость", "гости", "другие", "остальные", "присутствующие", "прочие", "все", "рекрутер"}


def _name_mentioned(word: str, tokens: list[str]) -> bool:
    """Имя встречается среди слов окна: точно, с другим падежом (Иванов ~ Иванову) или с ошибкой распознавания
    (Карен ~ Тарену)."""
    for t in tokens:
        if t == word:
            return True
        short = min(len(t), len(word))
        if short >= 4 and _common_prefix(word, t) >= max(4, short - 2):
            return True
        if short >= 5 and SequenceMatcher(None, word, t[:len(word) + 1]).ratio() >= 0.7:
            return True
    return False


def _person_words(name: str) -> list[str]:
    """Слова имени человека. Латиница (DSL, ArenaData) — это названия компаний и продуктов, а не имена в русской
    расшифровке, поэтому в логике очередности выступающих не участвует."""
    return [w for w in normalize(name).split()
            if len(w) >= 3 and w not in _NOT_A_NAME_WORDS and re.search("[а-яё]", w)]


def _is_address_form(word: str, token: str) -> bool:
    """Форма имени, которой ведущий передаёт слово или представляет человека: «Александр.» (звательный/именительный),
    «к Максиму» (дательный), «Анжелике». Родительный, винительный и творительный («знаю Максима», «с Максимом»)
    — просто упоминание внутри речи, слова не передаёт."""
    if token == word:
        return True
    if token in (word + "у", word + "ю"):                      # Максим → Максиму
        return True
    if word.endswith("а") and token == word[:-1] + "е":        # Анжелика → Анжелике
        return True
    if len(token) == len(word) >= 5 and SequenceMatcher(None, word, token).ratio() >= 0.8:
        return True                                            # ошибка ASR в одной букве: Карен ~ Тарен
    if len(word) >= 4 and token.endswith(("у", "ю")) and SequenceMatcher(None, word, token[:-1]).ratio() >= 0.8:
        return True                                            # то же в дательном: Карен ~ Тарену
    return False


def _last_address(words: list[str], lines: list[tuple[int, list[str]]], lo: int, hi: int) -> int | None:
    """Время последней реплики в [lo, hi], где любое из слов имени стоит в форме обращения."""
    found = None
    for sec, toks in lines:
        if lo <= sec <= hi and any(_is_address_form(w, t) for w in words for t in toks):
            found = sec if found is None else max(found, sec)
    return found


def confirm_speakers(protocol: dict, transcript: str, max_age_sec: int = 600, after_sec: int = 30) -> list[str]:
    """Докладчика определяет модель по одному тексту (диаризации нет), поэтому имя легко угадать неверно.
    Ведущий представляет участника или передаёт ему слово, и дальше говорит он, пока слово не передадут другому.
    Поэтому докладчик остаётся, если его имя прозвучало в форме обращения не раньше чем за max_age_sec до
    начала пункта (или до after_sec после, когда человек называет себя сам) И после этого до начала пункта
    слово не передавали другому участнику. Иначе «докладчик не указан».
    max_age_sec=0 — проверка выключена; пункт без таймкода не проверяется."""
    if max_age_sec <= 0:
        return []
    lines = [(int(m) * 60 + int(s), normalize(text).split())
             for m, s, text in re.findall(r"^\[(\d{1,3}):(\d{2})\]\s*(.*)$", transcript, re.MULTILINE)]
    if not lines:
        return []
    participants = [_person_words(p) for p in protocol.get("meta", {}).get("participants", [])]
    issues: list[str] = []
    for item in protocol.get("agenda_items", []):
        speaker, ts = item.get("speaker"), item.get("audio_timestamp")
        if not speaker or not ts or not _AUDIO_TS_RE.match(str(ts)):
            continue
        names = _person_words(speaker)
        if not names:
            continue
        m, s = str(ts).split(":")
        start = int(m) * 60 + int(s)
        own = _last_address(names, lines, start - max_age_sec, start + after_sec)
        if own is None:
            issues.append(f"speaker {speaker!r} at [{ts}]: name not heard in the {max_age_sec}s before, dropped")
            item["speaker"] = None
            continue
        # другие участники: только те, чьи слова не совпадают с именем докладчика; учитываем обращения до начала пункта
        later = [t for words in participants if words and not set(words) & set(names)
                 for t in [_last_address(words, lines, start - max_age_sec, start)] if t is not None and t > own]
        if later:
            issues.append(f"speaker {speaker!r} at [{ts}]: floor was passed to another participant, dropped")
            item["speaker"] = None
    return issues


def _common_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def is_grounded(value: str | None, tr: Transcript, min_ratio: float = 0.8) -> bool:
    """Значение подтверждено расшифровкой: цифры встречаются буквально, слова — почти дословно."""
    if not value or not value.strip():
        return True
    words = normalize(value).split()
    digits = [w for w in words if w.isdigit()]
    if any(not tr.has_number(d) for d in digits):
        return False
    content = [w for w in words if not w.isdigit() and len(w) >= 3 and w not in _STOP]
    if not content:
        return True
    hit = sum(1 for w in content if tr.has_word(w))
    return hit / len(content) >= min_ratio


def _note(item: dict, text: str) -> None:
    item.setdefault("needs_review", []).append(text)


def ground_protocol(protocol: dict, transcript: str) -> list[str]:
    """Обнуляет непроверяемые responsible/deadline/speaker/participants и неверные таймкоды,
    добавляя пометки в needs_review. Возвращает список найденных проблем (для лога и отчёта)."""
    tr = Transcript(transcript)
    issues: list[str] = []

    meta = protocol.get("meta") or {}
    if isinstance(meta.get("participants"), list):
        kept = []
        for p in meta["participants"]:
            if isinstance(p, str) and is_grounded(p, tr):
                kept.append(p)
            else:
                issues.append(f"participant not in transcript: {p!r}")
        meta["participants"] = kept
    for key in ("time", "place"):
        if meta.get(key) and not is_grounded(meta[key], tr):
            issues.append(f"meta.{key} not in transcript: {meta[key]!r}")
            meta[key] = None

    for item in protocol.get("agenda_items", []):
        if item.get("speaker") and not is_grounded(item["speaker"], tr):
            issues.append(f"speaker not in transcript: {item['speaker']!r}")
            item["speaker"] = None
        ts = item.get("audio_timestamp")
        if ts is not None:
            ok = isinstance(ts, str) and _AUDIO_TS_RE.match(ts) is not None
            if ok and tr.max_seconds is not None:
                m, s = ts.split(":")
                ok = int(m) * 60 + int(s) <= tr.max_seconds + 5
            if not ok:
                issues.append(f"bad audio_timestamp: {ts!r}")
                item["audio_timestamp"] = None
        for d in item.get("decisions", []):
            for key, label in (("responsible", "Ответственный"), ("deadline", "Срок")):
                val = d.get(key)
                if val and _TS_RE.match(str(val).strip()):
                    issues.append(f"{key} looks like a timestamp, dropped: {val!r}")  # путаница формата, не выдумка
                    d[key] = None
                elif val and not is_grounded(val, tr):
                    issues.append(f"{key} not in transcript: {val!r}")
                    _note(item, f"Не подтверждено расшифровкой: {label.lower()} «{val}» — проверьте по записи")
                    d[key] = None
        summary_nums = [w for w in normalize(item.get("topic_summary") or "").split() if w.isdigit()]
        bad = [n for n in summary_nums if not tr.has_number(n)]
        if bad:
            issues.append(f"numbers in summary not in transcript: {bad}")
            _note(item, f"В сути обсуждения есть числа, которых нет в расшифровке: {', '.join(bad)} — проверьте по записи")
    return issues
