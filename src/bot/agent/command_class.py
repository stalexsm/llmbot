"""Классификация команд exec: чистые функции без побочных эффектов.

Класс команды — метка для метрик расхода токенов: по полю ``tool_name``
события ``tool_call`` строится профиль «какие классы команд сколько съели».
"""

import json
import re
from enum import StrEnum


class CommandClass(StrEnum):
    """Крупные классы команд exec для профиля расхода токенов."""

    GIT = "git"
    PYTHON = "python"
    RG = "rg"
    CAT = "cat"
    LS = "ls"
    OTHER = "other"


# Класс определяет первое слово команды (после присваиваний окружения).
_CLASS_BY_PROGRAM: dict[str, CommandClass] = {
    "git": CommandClass.GIT,
    "python": CommandClass.PYTHON,
    "python3": CommandClass.PYTHON,
    "pytest": CommandClass.PYTHON,
    # uv — питонья обвязка этого проекта (uv run pytest, uv sync).
    "uv": CommandClass.PYTHON,
    "rg": CommandClass.RG,
    "grep": CommandClass.RG,
    "cat": CommandClass.CAT,
    "sed": CommandClass.CAT,
    "ls": CommandClass.LS,
}

# Ведущие присваивания окружения (FOO=bar cmd) класс не меняют.
_ENV_ASSIGNMENT_PREFIX = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")


def command_from_arguments(arguments: str) -> str | None:
    """Извлечь команду из JSON-аргументов вызова инструмента.

    None — аргументы невалидны: не JSON, не объект или без строкового
    непустого поля ``command``.
    """
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    command = parsed.get("command")
    if not isinstance(command, str) or not command.strip():
        return None
    return command


def classify_command(command: str) -> CommandClass:
    """Класс команды по первому значимому слову (чистая функция).

    Ведущие присваивания окружения пропускаются; каналы и списки не
    разбираются — класс определяет первое слово (``cat f | grep x`` — cat).
    """
    words = command.split()
    while words and _ENV_ASSIGNMENT_PREFIX.match(words[0]):
        words = words[1:]
    if not words:
        return CommandClass.OTHER
    return _CLASS_BY_PROGRAM.get(words[0], CommandClass.OTHER)


def classify_call_arguments(arguments: str) -> CommandClass:
    """Класс команды из сырых аргументов вызова; невалидные аргументы — прочее."""
    command = command_from_arguments(arguments)
    if command is None:
        return CommandClass.OTHER
    return classify_command(command)
