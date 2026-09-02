"""Проверки ответов: каждая задача принимает верный ответ и отвергает неверный."""

import subprocess
from pathlib import Path

from bot.benchmark.tasks import benchmark_tasks

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Верные ответы на задачи; для привязанных к репозиторию — вычисленные
# по текущему состоянию репозитория тем же способом, что и проверки.
_GOLDEN_ANSWERS = {
    "k01": "команда cat выводит файл",
    "k02": "Это Париж.",
    "k03": "Очередь.",
    "k04": "pip",
    "k05": "60 минут",
    "e01": "Команда вывела benchmark-ok",
    "e02": "Ядро: Darwin",
    "e03": str(len((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8").splitlines())),
    "e04": "Версия проекта 0.1.0",
    "e05": "NewType импортируется из typing",
    "r01": "Подкаталоги: "
    + ", ".join(sorted(p.name for p in (_REPO_ROOT / "src" / "bot").iterdir() if p.is_dir())),
    "r02": "Это dataclass AgentRun",
    "r03": "Имя инструмента — execute_command",
    "r04": str(
        len(
            (_REPO_ROOT / "src" / "bot" / "agent" / "exec.py")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    ),
    "r05": "Библиотека structlog",
    "r06": "Интерфейсы — typing.Protocol",
    "m01": "Нет, этот скилл про погоду, а не про конвертацию единиц.",
    "m02": str(
        sum(
            len(p.read_text(encoding="utf-8").splitlines())
            for p in (
                _REPO_ROOT / "src" / "bot" / "main.py",
                _REPO_ROOT / "src" / "bot" / "domain" / "ids.py",
            )
        )
    ),
    "m03": subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip(),
    "m04": "Python 3.13 установлен",
}

# Заведомо неверные ответы на те же задачи.
_WRONG_ANSWERS = {
    "k01": "сортировка",
    "k02": "Это Лондон.",
    "k03": "стек",
    "k04": "cargo",
    "k05": "100 минут",
    "e01": "Команда вывела hello",
    "e02": "Ядро: Windows",
    "e03": "99999 строк",
    "e04": "Версия 9.9.9",
    "e05": "Из модуля os",
    "r01": "Каталоги: alpha, beta",
    "r02": "Это dataclass AgentLoop",
    "r03": "Имя инструмента — run_shell",
    "r04": "1 строка",
    "r05": "Библиотека logging",
    "r06": "Интерфейсы — abc.ABC",
    "m01": "Да, скилл универсальный.",
    "m02": "Итого 7 строк",
    "m03": "Хеш: deadbee",
    "m04": "Установлен Python 2",
}


def test_every_checker_accepts_correct_answer() -> None:
    for task in benchmark_tasks():
        assert task.checker(_GOLDEN_ANSWERS[task.task_id]), task.task_id


def test_every_checker_rejects_wrong_answer() -> None:
    for task in benchmark_tasks():
        assert not task.checker(_WRONG_ANSWERS[task.task_id]), task.task_id
