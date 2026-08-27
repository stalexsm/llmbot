"""Скиллы: markdown-файлы с инструкциями, доступные агенту через индекс.

Раскладка — один каталог на скилл: ``<скиллы>/<имя>/SKILL.md``. Шапка файла —
YAML frontmatter с полями ``name`` и ``description``: имя нужно модели, чтобы
найти файл в каталоге скиллов, описание — чтобы понять, когда скилл нужен.
Полный текст модель читает сама через exec по необходимости. Добавление
скилла — добавление каталога с файлом, правки кода не требуются: индекс
собирается заново при каждом запуске. Запасной вариант без frontmatter —
строка ``Description:``; имя тогда берётся из имени каталога.
"""

import re
from dataclasses import dataclass
from pathlib import Path

# Описание в индексе обрезается: это оглавление, а не сам скилл.
_MAX_DESCRIPTION_CHARS = 160

_DESCRIPTION_PREFIX = "description:"
_FIRST_SENTENCE_RE = re.compile(r"[.!?](?:\s|$)")


@dataclass(frozen=True)
class SkillEntry:
    """Один скилл из индекса: имя, краткое описание и путь к SKILL.md."""

    name: str
    description: str
    file: Path


def parse_skill(file: Path, text: str) -> SkillEntry | None:
    """Разобрать текст ``SKILL.md`` скилла.

    Имя и описание берутся из frontmatter (без шапки — запасная строка
    ``Description:``, а имя каталога). Имя из шапки обязано совпадать
    с именем каталога: именно по имени модель ищет файл среди скиллов,
    поэтому расходящееся имя в индекс не попадает (``None``).
    """
    meta = _parse_frontmatter(text)
    declared = meta.get("name")
    name = declared or file.parent.name
    description = meta.get("description") or _plain_description(text)
    if declared is not None and declared != file.parent.name:
        return None
    if not description:
        return None
    return SkillEntry(name=name, description=description, file=file)


def load_skills(directory: Path) -> tuple[SkillEntry, ...]:
    """Собрать индекс скиллов из каталога, отсортированный по имени.

    Ищет ``<каталог>/<имя>/SKILL.md``; файлы без описания пропускаются.
    Несуществующий каталог — пустой индекс (бот работает без скиллов).
    """
    entries = []
    for file in sorted(directory.glob("*/SKILL.md")):
        entry = parse_skill(file, file.read_text(encoding="utf-8"))
        if entry is not None:
            entries.append(entry)
    return tuple(entries)


def render_skills_index(entries: tuple[SkillEntry, ...]) -> str:
    """Отрендерить секцию индекса скиллов для системного промпта.

    В индекс попадает только первое предложения описания: это оглавление,
    полный текст модель читает из файла. Пустой индекс — пустая секция:
    промпт не упоминает скиллы вовсе.
    """
    if not entries:
        return ""
    listed = "\n".join(f"- {entry.name}: {_first_sentence(entry.description)}" for entry in entries)
    example = entries[0].file
    return (
        "<skills>\n"
        "Файлы скиллов лежат в каталоге скиллов под именем из индекса:\n"
        f"{listed}\n"
        "Задача из индекса решается только через скилл: первым же вызовом execute_command "
        "прочитай его файл командой cat skills/<имя>/SKILL.md и выполняй инструкцию "
        "как написано, не пропуская шагов. Полный пример такой команды: "
        f"cat {example}.\n"
        "Для обычных вопросов, не требующих живых данных, отвечай сразу из своих знаний.\n"
        "</skills>"
    )


def _parse_frontmatter(text: str) -> dict[str, str]:
    """Разобрать YAML-шапку ``---\\nключ: значение…\\n---`` из начала файла."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    meta: dict[str, str] = {}
    for line in lines[1:]:
        stripped = line.strip()
        if stripped == "---":
            return meta
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if key in ("name", "description") and value:
            meta[key] = value
    return {}


def _plain_description(text: str) -> str | None:
    """Найти запасную строку ``Description: …`` вне frontmatter."""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.lower().startswith(_DESCRIPTION_PREFIX):
            continue
        description = stripped[len(_DESCRIPTION_PREFIX) :].strip()
        return description or None
    return None


def _first_sentence(description: str) -> str:
    """Первое предложение описания (со знаком препинания), до лимита индекса."""
    match = _FIRST_SENTENCE_RE.search(description)
    sentence = description[: match.end()].strip() if match else description
    if len(sentence) <= _MAX_DESCRIPTION_CHARS:
        return sentence
    return sentence[: _MAX_DESCRIPTION_CHARS - 1].rstrip() + "…"
