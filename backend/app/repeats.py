"""Удаление и обнаружение петель повторов в результатах распознавания."""
import re
from difflib import SequenceMatcher

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def normalize(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.lower()))


def is_caption_hallucination(text: str, pattern: str) -> bool:
    """True, если сегмент — типичная подпись Whisper на музыке/шуме («Играет музыка», «Субтитры ...»)."""
    return bool(pattern) and re.search(pattern, text, re.IGNORECASE) is not None


def collapse_loops(text: str, min_repeats: int = 3, max_ngram: int = 12) -> str:
    """Схлопывает подряд идущие повторы одной и той же фразы внутри текста:
    'а б в а б в а б в' -> 'а б в'. Повтор считается петлёй, если фраза идёт min_repeats раз подряд."""
    words = text.split()
    changed = True
    while changed:
        changed = False
        for n in range(1, max_ngram + 1):
            i = 0
            while i + n * min_repeats <= len(words):
                gram = [w.lower().strip(".,!?…;:") for w in words[i:i + n]]
                reps = 1
                while i + (reps + 1) * n <= len(words) and \
                        [w.lower().strip(".,!?…;:") for w in words[i + reps * n:i + (reps + 1) * n]] == gram:
                    reps += 1
                if reps >= min_repeats:
                    del words[i + n:i + reps * n]
                    changed = True
                i += 1
    return " ".join(words)


def dedupe_segments(segments: list[dict], similarity: float = 0.9, window: int = 3) -> list[dict]:
    """Убирает сегмент, если он (почти) совпадает с одним из последних `window` оставленных."""
    kept: list[dict] = []
    for seg in segments:
        norm = normalize(seg["text"])
        if not norm:
            continue
        if any(SequenceMatcher(None, norm, normalize(k["text"])).ratio() >= similarity
               for k in kept[-window:]):
            continue
        kept.append(seg)
    return kept


def repetition_report(segments: list[dict], ngram: int = 6, max_ngram_count: int = 4,
                      similarity: float = 0.9, window: int = 3) -> dict:
    """Автоматическая проверка на повторы после распознавания.
    - near_dup_segments: сегменты, почти совпадающие с одним из предыдущих `window`;
    - max_ngram_count: сколько раз чаще всего встречается одна и та же n-грамма слов (петля даёт десятки).
    ok=True, если дублей нет и ни одна n-грамма не встречается чаще max_ngram_count раз."""
    norms = [normalize(s["text"]) for s in segments]
    near_dup = 0
    for i, n in enumerate(norms):
        if n and any(SequenceMatcher(None, n, m).ratio() >= similarity for m in norms[max(0, i - window):i] if m):
            near_dup += 1
    words = " ".join(norms).split()
    counts: dict[tuple[str, ...], int] = {}
    for i in range(len(words) - ngram + 1):
        g = tuple(words[i:i + ngram])
        counts[g] = counts.get(g, 0) + 1
    worst_gram, worst = max(counts.items(), key=lambda kv: kv[1], default=((), 0))
    return {
        "near_dup_segments": near_dup,
        "max_ngram_count": worst,
        "worst_ngram": " ".join(worst_gram),
        "ok": near_dup == 0 and worst <= max_ngram_count,
    }
