"""Фиксированный набор из 20 offline-детерминированных задач бенчмарка.

Задач четыре вида: 5 «из знаний», 5 одношаговых exec, 6 многошаговых по
репозиторию и 4 смешанных. Сетевых задач нет: сравнение до/после должно
зависеть только от наших изменений, а не от внешних сервисов.

Проверка ответа — чистая предикат-функция от финального текста ответа.
Части проверок читают файлы репозитория на момент проверки: набор остаётся
воспроизводимым для фиксированного состояния репозитория, но не ломается
от каждой правки строк.
"""

import subprocess
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

Checker = Callable[[str], bool]


class TaskKind(StrEnum):
    """Вид задачи бенчмарка: какой навык агента она нагружает."""

    KNOWLEDGE = "knowledge"
    EXEC = "exec"
    REPO = "repo"
    MIXED = "mixed"


REPO_ROOT = Path(__file__).resolve().parents[3]

# Целевой состав набора: 5 «из знаний», 5 одношаговых exec,
# 6 многошаговых по репозиторию, 4 смешанных (инвариант спеки).
EXPECTED_KIND_COUNTS: dict[TaskKind, int] = {
    TaskKind.KNOWLEDGE: 5,
    TaskKind.EXEC: 5,
    TaskKind.REPO: 6,
    TaskKind.MIXED: 4,
}


@dataclass(frozen=True)
class BenchmarkTask:
    """Одна задача бенчмарка: промпт для агента и проверка финального ответа.

    ``checker`` — детерминированный предикат: ответ верен или нет. Он не
    влияет на критерий success (тот — про предел шагов и отсутствие ошибки),
    а отделяет «дешевле» от «глупее» при сравнении до/после оптимизаций.
    """

    task_id: str
    kind: TaskKind
    prompt: str
    checker: Checker


def _contains_any(*phrases: str) -> Checker:
    """Ответ должен содержать хотя бы одну из фраз (без учёта регистра)."""
    lowered = tuple(phrase.lower() for phrase in phrases)

    def check(answer: str) -> bool:
        text = answer.lower()
        return any(phrase in text for phrase in lowered)

    return check


def _contains_all(*phrases: str) -> Checker:
    """Ответ должен содержать каждую из фраз (без учёта регистра)."""
    lowered = tuple(phrase.lower() for phrase in phrases)

    def check(answer: str) -> bool:
        text = answer.lower()
        return all(phrase in text for phrase in lowered)

    return check


def _file_line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def _mentions_line_count(path: Path) -> Checker:
    """Ответ должен называть актуальное число строк файла."""

    def check(answer: str) -> bool:
        return str(_file_line_count(path)) in answer

    return check


def _mentions_project_version() -> Checker:
    """Ответ должен называть версию проекта из pyproject.toml."""

    def check(answer: str) -> bool:
        with open(REPO_ROOT / "pyproject.toml", "rb") as file:
            version = tomllib.load(file)["project"]["version"]
        return version in answer

    return check


def _mentions_sum_of_line_counts(*paths: Path) -> Checker:
    """Ответ должен называть сумму строк указанных файлов."""

    def check(answer: str) -> bool:
        total = sum(_file_line_count(path) for path in paths)
        return str(total) in answer

    return check


def _mentions_directory_names(directory: Path) -> Checker:
    """Ответ должен перечислить все имена каталогов в указанном каталоге.

    Артефакты сборки вроде ``__pycache__`` не считаются: их наличие зависит
    от режима запуска, а не от репозитория — воспроизводимость важнее.
    """
    names = sorted(
        entry.name
        for entry in directory.iterdir()
        if entry.is_dir() and not entry.name.endswith((".pycache",)) and entry.name != "__pycache__"
    )
    return _contains_all(*names)


def _mentions_head_commit() -> Checker:
    """Ответ должен называть короткий хеш текущего HEAD."""

    def check(answer: str) -> bool:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return head.lower() in answer.lower()

    return check


def _mentions_file_names(directory: Path) -> Checker:
    """Ответ должен упомянуть каждое имя файла в указанном каталоге."""
    names = sorted(entry.name for entry in directory.iterdir() if entry.is_file())
    return _contains_all(*names)


_TASKS: tuple[BenchmarkTask, ...] = (
    # --- 5 задач «из знаний» (без инструментов) ---
    BenchmarkTask(
        task_id="k01",
        kind=TaskKind.KNOWLEDGE,
        prompt="Какая утилита Unix выводит содержимое файла на экран? Назови одну команду.",
        checker=_contains_any("cat"),
    ),
    BenchmarkTask(
        task_id="k02",
        kind=TaskKind.KNOWLEDGE,
        prompt="Столица Франции — какой город?",
        checker=_contains_any("париж", "paris"),
    ),
    BenchmarkTask(
        task_id="k03",
        kind=TaskKind.KNOWLEDGE,
        prompt=(
            "Как называется структура данных «первым пришёл — первым вышел»? Ответь одним словом."
        ),
        checker=_contains_any("очеред", "queue", "fifo"),
    ),
    BenchmarkTask(
        task_id="k04",
        kind=TaskKind.KNOWLEDGE,
        prompt="Какой пакетный менеджер используется в Python-проектах? Назови один инструмент.",
        checker=_contains_any("pip", "uv", "poetry"),
    ),
    BenchmarkTask(
        task_id="k05",
        kind=TaskKind.KNOWLEDGE,
        prompt="Сколько минут в одном часе? Ответь числом.",
        checker=_contains_any("60"),
    ),
    # --- 5 одношаговых exec-задач ---
    BenchmarkTask(
        task_id="e01",
        kind=TaskKind.EXEC,
        prompt="Выполни команду echo benchmark-ok и скажи, что она вывела.",
        checker=_contains_any("benchmark-ok"),
    ),
    BenchmarkTask(
        task_id="e02",
        kind=TaskKind.EXEC,
        prompt="Выполни команду uname -s и назови ядро операционной системы, которое она вывела.",
        checker=_contains_any("darwin", "linux"),
    ),
    BenchmarkTask(
        task_id="e03",
        kind=TaskKind.EXEC,
        prompt="Посчитай число строк в файле pyproject.toml командой wc -l и назови число.",
        checker=_mentions_line_count(REPO_ROOT / "pyproject.toml"),
    ),
    BenchmarkTask(
        task_id="e04",
        kind=TaskKind.EXEC,
        prompt="Какая версия проекта указана в pyproject.toml в поле version? Найди и назови.",
        checker=_mentions_project_version(),
    ),
    BenchmarkTask(
        task_id="e05",
        kind=TaskKind.EXEC,
        prompt=(
            "Выполни cat src/bot/domain/ids.py и скажи, из какого модуля стандартной "
            "библиотеки импортируется NewType."
        ),
        checker=_contains_any("typing"),
    ),
    # --- 6 многошаговых задач по репозиторию ---
    BenchmarkTask(
        task_id="r01",
        kind=TaskKind.REPO,
        prompt="Перечисли все подкаталоги каталога src/bot и назови их имена.",
        checker=_mentions_directory_names(REPO_ROOT / "src" / "bot"),
    ),
    BenchmarkTask(
        task_id="r02",
        kind=TaskKind.REPO,
        prompt=(
            "Открой src/bot/agent/loop.py и найди имя dataclass, описывающего результат "
            "одного агентного цикла. Назови его."
        ),
        checker=_contains_any("AgentRun"),
    ),
    BenchmarkTask(
        task_id="r03",
        kind=TaskKind.REPO,
        prompt=(
            "Найди в src/bot/agent/exec.py, какое имя инструмента (ToolSpec name) "
            "у exec-инструмента, и назови его."
        ),
        checker=_contains_any("execute_command"),
    ),
    BenchmarkTask(
        task_id="r04",
        kind=TaskKind.REPO,
        prompt="Посчитай число строк в файле src/bot/agent/exec.py и назови число.",
        checker=_mentions_line_count(REPO_ROOT / "src" / "bot" / "agent" / "exec.py"),
    ),
    BenchmarkTask(
        task_id="r05",
        kind=TaskKind.REPO,
        prompt=(
            "Какая библиотека используется в этом проекте для структурированного "
            "логирования? Проверь по импортам в src/bot и назови пакет."
        ),
        checker=_contains_any("structlog"),
    ),
    BenchmarkTask(
        task_id="r06",
        kind=TaskKind.REPO,
        prompt=(
            "Прочитай docs/code-style.md: какой инструмент из typing используется "
            "в проекте для интерфейсов? Назови его."
        ),
        checker=_contains_any("protocol"),
    ),
    # --- 4 смешанные задачи ---
    BenchmarkTask(
        task_id="m01",
        kind=TaskKind.MIXED,
        prompt=(
            "Прочитай skills/weather/SKILL.md и скажи: годится ли этот скилл для задачи "
            "«конвертировать 5 км в мили»? Ответь «да» или «нет» и объясни "
            "в одном предложении."
        ),
        checker=lambda answer: "нет" in answer.lower(),
    ),
    BenchmarkTask(
        task_id="m02",
        kind=TaskKind.MIXED,
        prompt=(
            "Посчитай суммарное число строк в файлах src/bot/main.py и "
            "src/bot/domain/ids.py (wc -l по обоим) и назови одно итоговое число."
        ),
        checker=_mentions_sum_of_line_counts(
            REPO_ROOT / "src" / "bot" / "main.py",
            REPO_ROOT / "src" / "bot" / "domain" / "ids.py",
        ),
    ),
    BenchmarkTask(
        task_id="m03",
        kind=TaskKind.MIXED,
        prompt="Выполни git rev-parse --short HEAD и назови короткий хеш последнего коммита.",
        checker=_mentions_head_commit(),
    ),
    BenchmarkTask(
        task_id="m04",
        kind=TaskKind.MIXED,
        prompt="Выполни python3 --version и назови версию Python, которую вывела команда.",
        checker=_contains_all("python", "3."),
    ),
)


def benchmark_tasks() -> tuple[BenchmarkTask, ...]:
    """Фиксированный набор задач; порядок стабилен между прогонами."""
    return _TASKS


def task_kind_counts(tasks: tuple[BenchmarkTask, ...]) -> dict[TaskKind, int]:
    """Число задач по видам — самопроверка целевого состава набора."""
    counts: dict[TaskKind, int] = dict.fromkeys(TaskKind, 0)
    for task in tasks:
        counts[task.kind] += 1
    return counts
