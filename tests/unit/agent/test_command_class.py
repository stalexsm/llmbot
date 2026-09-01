"""Unit tests for the exec command classifier (pure functions)."""

import json

import pytest

from bot.agent.command_class import (
    CommandClass,
    classify_call_arguments,
    classify_command,
    command_from_arguments,
)


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("git status", CommandClass.GIT),
        ("git diff --stat HEAD~1", CommandClass.GIT),
        ("   git log --oneline", CommandClass.GIT),
        ("python -c 'print(1)'", CommandClass.PYTHON),
        ("python3 --version", CommandClass.PYTHON),
        ("pytest -q tests/unit", CommandClass.PYTHON),
        ("uv run pytest", CommandClass.PYTHON),
        ("uv sync", CommandClass.PYTHON),
        ("rg pattern src/", CommandClass.RG),
        ("grep -rn todo .", CommandClass.RG),
        ("cat skills/weather/SKILL.md", CommandClass.CAT),
        ("sed -n 1,10p file.txt", CommandClass.CAT),
        ("ls", CommandClass.LS),
        ("ls -la src/", CommandClass.LS),
        ("echo привет", CommandClass.OTHER),
        ("curl https://example.com", CommandClass.OTHER),
        ("find . -name '*.py'", CommandClass.OTHER),
        # Каналы не разбираются: класс определяет первое слово.
        ("cat file.txt | grep x", CommandClass.CAT),
        ("git status && ls", CommandClass.GIT),
        # Ведущие присваивания окружения класс не меняют.
        ("FOO=bar pytest -q", CommandClass.PYTHON),
        ("PATH=/usr/bin git status", CommandClass.GIT),
        ("", CommandClass.OTHER),
        ("   ", CommandClass.OTHER),
    ],
)
def test_classify_command(command: str, expected: CommandClass) -> None:
    assert classify_command(command) is expected


def test_command_from_arguments_returns_command() -> None:
    arguments = json.dumps({"command": "git status"}, ensure_ascii=False)
    assert command_from_arguments(arguments) == "git status"


@pytest.mark.parametrize(
    "arguments",
    [
        "не json",
        "[1, 2]",
        json.dumps({"cmd": "git status"}),
        json.dumps({"command": ""}),
        json.dumps({"command": "   "}),
        json.dumps({"command": 42}),
    ],
)
def test_command_from_arguments_invalid_returns_none(arguments: str) -> None:
    assert command_from_arguments(arguments) is None


def test_classify_call_arguments_classifies_command() -> None:
    arguments = json.dumps({"command": "rg foo ."}, ensure_ascii=False)
    assert classify_call_arguments(arguments) is CommandClass.RG


def test_classify_call_arguments_invalid_is_other() -> None:
    assert classify_call_arguments("не json") is CommandClass.OTHER
