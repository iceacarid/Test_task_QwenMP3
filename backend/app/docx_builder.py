import copy
import json
import os
import sys
from pathlib import Path

from docx import Document
from docx.opc.exceptions import PackageNotFoundError
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.text.paragraph import Paragraph


def add_approval_block(doc):
    """Блок УТВЕРЖДАЮ — административные поля не извлекаются из аудио,
    оставлены плейсхолдерами в квадратных скобках, как в шаблоне."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p.add_run("УТВЕРЖДАЮ").bold = True

    for text in ("[Должность]", "[Наименование организации]", "______________ [Ф.И.О.]", "«____» ____________ 20__ г."):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        p.add_run(text)


def add_title(doc):
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("П Р О Т О К О Л")
    run.bold = True
    run.font.size = Pt(16)

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.add_run("совещания при [должность / Ф.И.О.]")

    number_line = doc.add_paragraph()
    number_line.alignment = WD_ALIGN_PARAGRAPH.CENTER
    number_line.add_run("№___от __.__.20__")

    city_line = doc.add_paragraph()
    city_line.alignment = WD_ALIGN_PARAGRAPH.CENTER
    city_line.add_run("[город / населённый пункт]")


def add_closing_block(doc):
    """Подвал "Протокол вёл" — как и УТВЕРЖДАЮ, поле не определяется из аудио."""
    doc.add_paragraph()
    doc.add_paragraph().add_run("Протокол вёл:").bold = True
    doc.add_paragraph("[Должность]")
    doc.add_paragraph("______________")
    doc.add_paragraph("[Ф.И.О.]")


def add_meta(doc, meta):
    doc.add_paragraph()
    time = meta.get("time") or "не указано в записи"
    place = meta.get("place") or "не указано в записи"
    participants = meta.get("participants") or []
    participants_str = ", ".join(participants) if participants else "не указаны в записи"

    p = doc.add_paragraph()
    p.add_run("Время проведения: ").bold = True
    p.add_run(time)

    p = doc.add_paragraph()
    p.add_run("Место проведения: ").bold = True
    p.add_run(place)

    p = doc.add_paragraph()
    p.add_run("Присутствуют: ").bold = True
    p.add_run(participants_str)


def add_agenda_items(doc, agenda_items):
    heading = doc.add_paragraph()
    heading.add_run("Заслушали:").bold = True
    # Если решений нет во всей записи (доклады, интервью, дискуссия), «Решение не принято» под каждым пунктом
    # только вводит в заблуждение: пишем одну итоговую строку в конце.
    n_decisions = sum(len(item.get("decisions") or []) for item in agenda_items)
    has_decisions = n_decisions > 0
    show_undecided = n_decisions >= 2  # «Решение не принято» по пунктам имеет смысл, только если в записи есть настоящие решения

    for item in agenda_items:
        doc.add_paragraph()
        speaker = item.get("speaker") or "докладчик не указан"

        speaker_p = doc.add_paragraph()
        speaker_p.add_run(speaker).bold = True
        timestamp = item.get("audio_timestamp")
        if timestamp:
            ts_run = speaker_p.add_run(f"  [{timestamp}]")
            ts_run.italic = True

        doc.add_paragraph(item.get("topic_summary") or "")

        decisions = item.get("decisions") or []
        if decisions:
            p = doc.add_paragraph()
            p.add_run("Решение:").bold = True
            for d in decisions:
                line = f"– {d.get('text') or ''}"
                extra = []
                if d.get("responsible"):
                    extra.append(f"ответственный: {d['responsible']}")
                if d.get("deadline"):
                    extra.append(f"срок: {d['deadline']}")
                if extra:
                    line += " (" + ", ".join(extra) + ")"
                doc.add_paragraph(line)
        elif show_undecided:
            p = doc.add_paragraph()
            run = p.add_run("Решение не принято.")
            run.italic = True

        needs_review = item.get("needs_review") or []
        if needs_review:
            p = doc.add_paragraph()
            run = p.add_run("Требует проверки:")
            run.italic = True
            run.bold = True
            for fragment in needs_review:
                p2 = doc.add_paragraph()
                r2 = p2.add_run(f"— {fragment}")
                r2.italic = True

    if agenda_items and not has_decisions:
        doc.add_paragraph()
        run = doc.add_paragraph().add_run("Решения и поручения в записи не зафиксированы.")
        run.italic = True


TEMPLATE_PATH = Path(os.environ.get("PROTOCOL_TEMPLATE", Path(__file__).resolve().parents[2] / "templates" / "protocol_template.docx"))
NO_DECISIONS_LINE = "Решения и поручения в записи не зафиксированы."


def _find_paragraph(doc, prefix: str):
    for p in doc.paragraphs:
        if p.text.strip().startswith(prefix):
            return p
    raise ValueError(f"в шаблоне нет абзаца «{prefix}…»")


def _set_text(paragraph, text: str, italic: bool | None = None, bold: bool | None = None) -> None:
    """Заменяет текст абзаца, сохраняя оформление первого фрагмента (шрифт, размер)."""
    runs = paragraph.runs
    if not runs:
        paragraph.add_run(text)
        runs = paragraph.runs
    runs[0].text = text
    for extra in runs[1:]:
        extra._r.getparent().remove(extra._r)
    if italic is not None:
        runs[0].italic = italic
    if bold is not None:
        runs[0].bold = bold


def _build_from_template(protocol: dict, output_path: str) -> None:
    """Заполняет шаблон протокола: таблицы «УТВЕРЖДАЮ» и «Протокол вёл», заголовок и поля остаются как в шаблоне;
    блок докладчика (имя, суть, «Решение:», «– …») клонируется из шаблона для каждого вопроса."""
    doc = Document(str(TEMPLATE_PATH))
    meta = protocol.get("meta") or {}
    items = protocol.get("agenda_items") or []

    _set_text(_find_paragraph(doc, "Время проведения"), f"Время проведения: {meta.get('time') or 'не указано в записи'}.")
    _set_text(_find_paragraph(doc, "Место проведения"), f"Место проведения: {meta.get('place') or 'не указано в записи'}.")
    pres = _find_paragraph(doc, "Присутствуют")
    people = ", ".join(meta.get("participants") or []) or "не указаны в записи"
    pres.runs[1].text = f"{people}."

    heading = _find_paragraph(doc, "Заслушали")
    # прототипы блока берём из первого блока шаблона, затем убираем все заготовки до таблицы «Протокол вёл»
    body = list(doc.element.body)
    start = body.index(heading._p) + 1
    block = body[start:start + 4]
    if len(block) < 4:
        raise ValueError("в шаблоне нет блока докладчика")
    proto_speaker, proto_summary, proto_dec_head, proto_dec_line = (copy.deepcopy(e) for e in block)
    end = next(i for i, e in enumerate(body) if i > start and e.tag.endswith("}tbl"))
    for e in body[start:end - 1]:  # оставляем пустой абзац перед таблицей подписи
        doc.element.body.remove(e)
    anchor = heading._p

    def add(proto, text, italic=None, bold=None, space_before=None):
        nonlocal anchor
        el = copy.deepcopy(proto)
        anchor.addnext(el)
        anchor = el
        par = Paragraph(el, heading._parent)
        _set_text(par, text, italic=italic, bold=bold)
        if space_before is not None:
            par.paragraph_format.space_before = Pt(space_before)
        return par

    n_decisions = sum(len(i.get("decisions") or []) for i in items)
    has_decisions = n_decisions > 0
    show_undecided = n_decisions >= 2  # «Решение не принято» по пунктам имеет смысл, только если в записи есть настоящие решения
    for n, item in enumerate(items):
        sp = add(proto_speaker, item.get("speaker") or "докладчик не указан", space_before=10 if n else None)
        if item.get("audio_timestamp"):
            sp.add_run(f"  [{item['audio_timestamp']}]").italic = True
        add(proto_summary, item.get("topic_summary") or "")
        decisions = item.get("decisions") or []
        if decisions:
            add(proto_dec_head, "Решение:")
            for d in decisions:
                line = f"– {d.get('text') or ''}"
                extra = []
                if d.get("responsible"):
                    extra.append(f"ответственный: {d['responsible']}")
                if d.get("deadline"):
                    extra.append(f"срок: {d['deadline']}")
                add(proto_dec_line, line + (" (" + ", ".join(extra) + ")" if extra else ""))
        elif show_undecided:
            add(proto_dec_line, "Решение не принято.", italic=True)
        notes = item.get("needs_review") or []
        if notes:
            add(proto_dec_head, "Требует проверки:", italic=True)
            for fragment in notes:
                add(proto_dec_line, f"— {fragment}", italic=True)
    if items and not has_decisions:
        add(proto_summary, NO_DECISIONS_LINE, italic=True, space_before=10)

    for sec in doc.sections:  # пометка «Тестовый шаблон» относится к самому шаблону, а не к готовому протоколу
        for p in sec.footer.paragraphs:
            if "Тестовый шаблон" in p.text:
                _set_text(p, "")
    doc.save(output_path)


def build_protocol_docx(protocol: dict, output_path: str):
    """Собирает DOCX-протокол из словаря (результат generate_protocol) и сохраняет по указанному пути.
    Основа — templates/protocol_template.docx; если шаблон недоступен или не соответствует ожидаемой структуре,
    документ строится заново с тем же оформлением."""
    try:
        _build_from_template(protocol, output_path)
        return
    except (OSError, PackageNotFoundError, ValueError, StopIteration, IndexError, KeyError) as e:
        print(f"Шаблон не использован ({type(e).__name__}: {e}), строю документ с нуля", file=sys.stderr)
    doc = Document()
    add_approval_block(doc)
    add_title(doc)
    add_meta(doc, protocol.get("meta") or {})
    add_agenda_items(doc, protocol.get("agenda_items") or [])
    add_closing_block(doc)
    doc.save(output_path)


if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else "protocol.json"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "protocol.docx"
    with open(input_path, "r", encoding="utf-8") as f:
        protocol_data = json.load(f)
    build_protocol_docx(protocol_data, output_path)
    print(f"Готово: {output_path}")
