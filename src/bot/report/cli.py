"""CLI-dashboard: ``uv run python -m bot.report`` [``--task <id>``].

Без аргументов печатает агрегат по всем запускам; с ``--task <id>`` —
пошаговый timeline одного запуска. Неизвестный id — сообщение в stderr
и код выхода 1; пустое или отсутствующее хранилище — нулевой отчёт.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from bot.domain.ids import RequestId
from bot.metrics.recorder import EVENTS_FILENAME, METRICS_DIRECTORY
from bot.report.aggregate import build_dashboard
from bot.report.events import read_events
from bot.report.render import render_dashboard, render_timeline
from bot.report.timeline import build_timeline

DEFAULT_METRICS_FILE = METRICS_DIRECTORY / EVENTS_FILENAME


def main(argv: Sequence[str] | None = None, *, events_file: Path | None = None) -> int:
    """Собрать и напечатать отчёт; возвращает код выхода процесса."""
    parser = argparse.ArgumentParser(
        prog="python -m bot.report",
        description="Текстовый dashboard по расходу токенов агента (JSONL-метрики).",
    )
    parser.add_argument(
        "--task",
        metavar="ID",
        help="timeline одного запуска агента (request_id из метрик)",
    )
    args = parser.parse_args(argv)
    path = events_file if events_file is not None else DEFAULT_METRICS_FILE

    events = read_events(path)
    if args.task is not None:
        timeline = build_timeline(events, RequestId(args.task))
        if timeline is None:
            sys.stderr.write(f"Нет событий для запуска {args.task}\n")
            return 1
        sys.stdout.write(render_timeline(timeline))
        return 0

    sys.stdout.write(render_dashboard(build_dashboard(events)))
    return 0
