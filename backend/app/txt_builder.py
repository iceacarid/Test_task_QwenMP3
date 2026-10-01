def build_protocol_txt(protocol: dict, output_path: str):
    """Собирает протокол в виде обычного текстового файла из того же JSON, что и DOCX."""
    meta = protocol.get("meta") or {}
    lines = [
        "УТВЕРЖДАЮ",
        "[Должность]",
        "[Наименование организации]",
        "______________ [Ф.И.О.]",
        "«____» ____________ 20__ г.",
        "",
        "П Р О Т О К О Л",
        "совещания при [должность / Ф.И.О.]",
        "№___от __.__.20__",
        "[город / населённый пункт]",
        "",
    ]

    lines.append(f"Время проведения: {meta.get('time') or 'не указано в записи'}")
    lines.append(f"Место проведения: {meta.get('place') or 'не указано в записи'}")
    participants = meta.get("participants") or []
    lines.append(f"Присутствуют: {', '.join(participants) if participants else 'не указаны в записи'}")
    lines.append("")

    agenda_items = protocol.get("agenda_items") or []
    n_decisions = sum(len(item.get("decisions") or []) for item in agenda_items)
    has_decisions = n_decisions > 0
    show_undecided = n_decisions >= 2  # «Решение не принято» по пунктам имеет смысл, только если в записи есть настоящие решения
    if agenda_items:
        lines.append("Заслушали:")
        lines.append("")

    for item in agenda_items:
        speaker = item.get("speaker") or "докладчик не указан"
        timestamp = item.get("audio_timestamp")
        lines.append(f"{speaker}  [{timestamp}]" if timestamp else speaker)
        lines.append(item.get("topic_summary") or "")

        decisions = item.get("decisions") or []
        if decisions:
            lines.append("Решение:")
            for d in decisions:
                line = f"– {d.get('text') or ''}"
                extra = []
                if d.get("responsible"):
                    extra.append(f"ответственный: {d['responsible']}")
                if d.get("deadline"):
                    extra.append(f"срок: {d['deadline']}")
                if extra:
                    line += " (" + ", ".join(extra) + ")"
                lines.append(line)
        elif show_undecided:
            lines.append("Решение не принято.")

        needs_review = item.get("needs_review") or []
        if needs_review:
            lines.append("Требует проверки:")
            for fragment in needs_review:
                lines.append(f"— {fragment}")

        lines.append("")

    if agenda_items and not has_decisions:
        lines.append("Решения и поручения в записи не зафиксированы.")
        lines.append("")

    lines.append("Протокол вёл:")
    lines.append("[Должность]")
    lines.append("______________")
    lines.append("[Ф.И.О.]")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
